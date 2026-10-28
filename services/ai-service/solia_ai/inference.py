"""Serve predictions from a trained checkpoint and export the ONNX artifact.

This is the inference side of the model (docs/ARCHITECTURE.md §5, docs/API-CONTRACTS.md):

* **The output contract**, per requested horizon: ``pump_prob``, ``rug_prob``, ``expected_return``
  and a derived ``confidence`` ∈ [0, 100] — plus the ``model_version`` the caller echoes back for
  traceability. The model is a *signal* model: nothing here decides a trade (that is the policy
  layer's job, see docs/ML-MODEL.md).
* **Confidence is derived, not learned** (:func:`derive_confidence`): a monotonic map of the
  probability margin. An optional temperature (the calibration finalized later this sprint) is
  applied to the logits, so the served ``pump_prob``/``rug_prob`` *and* the confidence derived
  from them are both calibrated. A default temperature of 1.0 leaves the probabilities untouched.
* **The ONNX export** (:func:`export_onnx`) is the artifact the latency-critical .NET fast path
  runs in-process. It emits the whole contract — confidence included, with the temperature baked
  in — from the same torch code as :class:`Predictor`, so .NET never re-implements the formula.
  It is exported at a fixed window length (``seq_len``) with a dynamic batch and always ships with
  a parity check: ONNXRuntime must match PyTorch within tolerance, or the export fails rather than
  handing the pipeline a graph that quietly diverges.

The .NET fast path left-pads a token's recent rows to ``seq_len`` and passes the padding mask, so
the graph's masked mean scores a young token (a handful of snapshots) exactly as PyTorch would. A
graph cannot raise, so a row that is *entirely* padding (a token with no snapshot yet) scores the
neutral "no signal" output — 0.5 / 0.5 / 0.0 return / 0 confidence — instead of NaN.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .dataset import pad_window
from .model import SoliaTransformer

# Pinned so the pipeline's ONNXRuntime always sees the same operator set the export was tested
# against; bump deliberately (and re-check parity) rather than tracking torch's default. 18 is the
# floor the dynamo exporter implements — a lower request only triggers a lossy down-conversion.
DEFAULT_OPSET = 18
# ONNXRuntime vs PyTorch: float32 matmul reassociation across runtimes drifts at ~1e-6; 1e-4 is a
# comfortable ceiling that still catches a genuinely wrong graph (missing sigmoid, wrong mask).
PARITY_TOL = 1e-4

INPUT_NAMES = ("features", "padding_mask")
OUTPUT_NAMES = ("pump_prob", "rug_prob", "expected_return", "confidence")


# --------------------------------------------------------------------------- confidence


def _check_temperature(temperature: float) -> float:
    if not temperature > 0:  # also rejects NaN
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return float(temperature)


def _confidence(pump: torch.Tensor, rug: torch.Tensor) -> torch.Tensor:
    """The one confidence formula, shared by :class:`Predictor` and the ONNX graph."""
    margin = torch.maximum((pump - 0.5).abs(), (rug - 0.5).abs())
    return (200.0 * margin).clamp(0.0, 100.0)


def contract_outputs(logits: dict[str, torch.Tensor], temperature: float = 1.0) -> dict[str, Any]:
    """Model logits -> the serving contract: temperature-scaled probabilities, expected return,
    and the confidence derived from those *same* (calibrated) probabilities."""
    pump = torch.sigmoid(logits["pump_logits"] / temperature)
    rug = torch.sigmoid(logits["rug_logits"] / temperature)
    return {
        "pump_prob": pump,
        "rug_prob": rug,
        "expected_return": logits["expected_return"],
        "confidence": _confidence(pump, rug),
    }


def derive_confidence(
    pump_prob: np.ndarray | float,
    rug_prob: np.ndarray | float,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Confidence ∈ [0, 100] from the pump/rug probabilities (docs/ML-MODEL.md "Confidence").

    How decisive the dominant signal is: ``200 * max(|pump-0.5|, |rug-0.5|)``, so a coin-flip
    (0.5) scores 0 and a near-certain 0/1 scores 100. It is monotonic with the probability margin,
    hence with historical hit-rate once the probabilities are calibrated. ``temperature`` (>0,
    default 1.0 = identity) applies that calibration in logit space before the margin is taken.
    """
    temperature = _check_temperature(temperature)
    pump = torch.as_tensor(np.asarray(pump_prob, dtype=np.float64))
    rug = torch.as_tensor(np.asarray(rug_prob, dtype=np.float64))
    if temperature != 1.0:
        eps = 1e-7
        pump = torch.sigmoid(torch.logit(pump, eps=eps) / temperature)
        rug = torch.sigmoid(torch.logit(rug, eps=eps) / temperature)
    return _confidence(pump, rug).numpy()


# --------------------------------------------------------------------------- checkpoints

NO_CHAMPION = "no champion is registered; train and promote a model first"


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint written by :func:`solia_ai.train.build_checkpoint`.

    ``weights_only``: a checkpoint only holds tensors and plain containers, and its path comes
    from the registry — never unpickle arbitrary objects from it.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"{path} is not a Solia checkpoint (no 'model_state')")
    return checkpoint


def resolve_checkpoint_path(path: str | Path) -> Path:
    """The registered checkpoint ``path`` if it exists here, else the same file name in
    ``$MODEL_DIR``.

    The registry stores the absolute path where training wrote the checkpoint, which is only
    valid on that machine: a champion trained on the host (``./models``) and served from the
    container (``/app/models`` volume) is found by name in the serving process's model directory.
    """
    registered = Path(path)
    if registered.is_file():
        return registered
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        # Split on both separators: a Windows-registered path read on Linux has no "/" in it.
        candidate = Path(model_dir) / re.split(r"[\\/]", str(path))[-1]
        if candidate.is_file():
            return candidate
    return registered  # missing everywhere: let load_checkpoint raise on the registered path


def build_model(checkpoint: dict[str, Any]) -> SoliaTransformer:
    """Rebuild the model from a checkpoint and load its weights, in eval mode.

    The architecture comes from the checkpoint's recorded ``hyperparams``; loading the state dict
    re-checks (via the model's extra state) that its horizons and feature spec match the code —
    a checkpoint trained against a different feature set is refused rather than silently mis-fed.
    """
    model = SoliaTransformer.from_config(checkpoint["hyperparams"])
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


# --------------------------------------------------------------------------- predictions


@dataclass(frozen=True)
class HorizonPrediction:
    """One horizon's slice of the output contract."""

    horizon: str
    pump_prob: float
    rug_prob: float
    expected_return: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "pump_prob": self.pump_prob,
            "rug_prob": self.rug_prob,
            "expected_return": self.expected_return,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class Prediction:
    """A token's prediction: the model version plus one entry per requested horizon."""

    model_version: str
    predictions: list[HorizonPrediction]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "predictions": [p.as_dict() for p in self.predictions],
        }


class Predictor:
    """A loaded checkpoint ready to score token windows into the output contract."""

    def __init__(
        self,
        model: SoliaTransformer,
        *,
        model_version: str,
        temperature: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.eval()
        self.model_version = model_version
        self.temperature = _check_temperature(temperature)
        # The checkpoint's non-weight fields (metrics, created_at, train_end_ts, ...).
        self.metadata = dict(metadata or {})

    @property
    def horizons(self) -> tuple[str, ...]:
        return self.model.horizons

    @property
    def seq_len(self) -> int:
        return self.model.seq_len

    @property
    def min_seq_len(self) -> int:
        """The shortest window the champion was trained on: the checkpoint's recorded
        ``hyperparams.model.min_seq_len`` (which defaults to ``seq_len``, as in ``config.py``), or 1
        when the checkpoint records no model hyperparams."""
        model = (self.metadata.get("hyperparams") or {}).get("model")
        if not model:
            return 1
        return int(model.get("min_seq_len", model.get("seq_len", self.seq_len)))

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, temperature: float = 1.0) -> Predictor:
        checkpoint = load_checkpoint(path)
        version = checkpoint.get("version") or Path(path).stem
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state"}
        return cls(
            build_model(checkpoint),
            model_version=version,
            temperature=temperature,
            metadata=metadata,
        )

    @classmethod
    def from_record(cls, record: Any, *, temperature: float = 1.0) -> Predictor:
        """Load a registry record's checkpoint (see :func:`resolve_checkpoint_path`), stamped with
        the registry version (the traceable identity callers echo back)."""
        predictor = cls.from_checkpoint(
            resolve_checkpoint_path(record.path), temperature=temperature
        )
        predictor.model_version = record.version
        return predictor

    @classmethod
    def from_champion(cls, engine: Any, *, temperature: float = 1.0) -> Predictor:
        """Load the registry's champion checkpoint (raises if none is promoted yet)."""
        from .registry import read_champion

        record = read_champion(engine)
        if record is None:
            raise LookupError(NO_CHAMPION)
        return cls.from_record(record, temperature=temperature)

    def _horizon_indices(self, horizons: Sequence[str] | None) -> list[int]:
        if horizons is None:
            return list(range(len(self.horizons)))
        if isinstance(horizons, str):  # a bare "5m" would otherwise iterate per character
            raise TypeError(f"horizons must be a sequence of names, got the string {horizons!r}")
        duplicates = sorted({h for h in horizons if list(horizons).count(h) > 1})
        if duplicates:
            raise ValueError(f"duplicate horizon(s): {', '.join(duplicates)}")
        unknown = [h for h in horizons if h not in self.horizons]
        if unknown:
            raise ValueError(
                f"unknown horizon(s): {', '.join(unknown)} "
                f"(the model serves {', '.join(self.horizons)})"
            )
        index = {h: i for i, h in enumerate(self.horizons)}
        return [index[h] for h in horizons]

    def _to_batch(self, windows: Sequence[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        xs, masks = [], []
        for window in windows:
            array = np.asarray(window, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != self.model.n_features:
                raise ValueError(
                    f"each window must be [L, {self.model.n_features}], got shape {array.shape}"
                )
            x, mask = pad_window(array, self.seq_len)
            xs.append(x)
            masks.append(mask)
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(masks))

    def predict(
        self, windows: np.ndarray | Iterable[np.ndarray], horizons: Sequence[str] | None = None
    ) -> Prediction | list[Prediction]:
        """Score one window (a ``[L, F]`` array) or a batch of them into the output contract.

        A single 2-D array returns one :class:`Prediction`; an iterable of windows returns a list,
        one per window. ``horizons`` optionally restricts and orders the returned horizons (a
        subset of the model's); the default is all of them, in the model's order.
        """
        single = isinstance(windows, np.ndarray) and windows.ndim == 2
        batch = [windows] if single else list(windows)
        if not batch:
            return []
        indices = self._horizon_indices(horizons)
        names = self.horizons if horizons is None else list(horizons)

        x, mask = self._to_batch(batch)
        with torch.no_grad():
            out = contract_outputs(self.model(x, mask), self.temperature)
        pump = out["pump_prob"].numpy()
        rug = out["rug_prob"].numpy()
        ret = out["expected_return"].numpy()
        confidence = out["confidence"].numpy()

        results = [
            Prediction(
                model_version=self.model_version,
                predictions=[
                    HorizonPrediction(
                        horizon=name,
                        pump_prob=float(pump[b, h]),
                        rug_prob=float(rug[b, h]),
                        expected_return=float(ret[b, h]),
                        confidence=float(confidence[b, h]),
                    )
                    for name, h in zip(names, indices, strict=True)
                ],
            )
            for b in range(len(batch))
        ]
        return results[0] if single else results


# --------------------------------------------------------------------------- ONNX export


class _OnnxInferenceModule(nn.Module):
    """Wraps the model so the exported graph emits the full serving contract (calibrated
    probabilities, return, confidence), not raw logits — via :func:`contract_outputs`, the same
    code :class:`Predictor` runs, so the .NET fast path never re-implements the formula."""

    def __init__(self, model: SoliaTransformer, temperature: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.temperature = _check_temperature(temperature)

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # encode_logits (not forward): skip the data-dependent validation branches torch.export
        # can't capture. forward's "entirely padding" guard can't raise inside a graph, so such a
        # row is un-masked for the math (no 0/0 -> NaN) and its outputs overwritten as neutral.
        has_rows = (~padding_mask).any(dim=1, keepdim=True)
        safe_mask = padding_mask & has_rows
        out = contract_outputs(self.model.encode_logits(features, safe_mask), self.temperature)
        return (
            torch.where(has_rows, out["pump_prob"], 0.5),
            torch.where(has_rows, out["rug_prob"], 0.5),
            torch.where(has_rows, out["expected_return"], 0.0),
            torch.where(has_rows, out["confidence"], 0.0),
        )


def _example_inputs(
    model: SoliaTransformer, batch: int, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.normal(size=(batch, model.seq_len, model.n_features)).astype("f4"))
    mask = torch.zeros(batch, model.seq_len, dtype=torch.bool)
    return x, mask


def onnx_parity(
    model: SoliaTransformer,
    onnx_path: str | Path,
    *,
    temperature: float = 1.0,
    tol: float = PARITY_TOL,
    seed: int = 0,
) -> float:
    """Max absolute output difference between ONNXRuntime and :class:`Predictor`'s PyTorch path
    over a few random batches (including partially padded windows; ``confidence`` is compared on
    its 0–1 scale so ``tol`` means the same thing for every output). Raises :class:`ValueError`
    if it exceeds ``tol``."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model.eval()
    rng = np.random.default_rng(seed)
    worst = 0.0
    for batch in (1, 5):
        x, mask = _example_inputs(model, batch, seed=seed + batch)
        # Left-pad some rows on part of the batch so the masked path is exercised, never fully.
        for b in range(batch):
            pad = int(rng.integers(0, model.seq_len))
            mask[b, :pad] = True
        with torch.no_grad():
            expected = contract_outputs(model(x, mask), temperature)
        feeds = {"features": x.numpy(), "padding_mask": mask.numpy()}
        actual = session.run(list(OUTPUT_NAMES), feeds)
        for name, got in zip(OUTPUT_NAMES, actual, strict=True):
            scale = 100.0 if name == "confidence" else 1.0
            diff = np.abs(got - expected[name].numpy()) / scale
            worst = max(worst, float(diff.max()))
    if worst > tol:
        raise ValueError(f"ONNX parity failed: max |onnx - torch| = {worst:.2e} > tol {tol:.2e}")
    return worst


def export_onnx(
    model: SoliaTransformer,
    path: str | Path,
    *,
    opset: int = DEFAULT_OPSET,
    model_version: str | None = None,
    temperature: float = 1.0,
    verify: bool = True,
    tol: float = PARITY_TOL,
) -> Path:
    """Export the model to ONNX at ``path`` and (by default) verify ONNX↔PyTorch parity.

    Exported at a fixed window length (``seq_len``) with a dynamic batch, taking ``features``
    ``[B, seq_len, F]`` and a bool ``padding_mask`` ``[B, seq_len]`` and returning
    ``pump_prob`` / ``rug_prob`` / ``expected_return`` / ``confidence`` ``[B, n_horizons]`` — the
    fast-path contract, with ``temperature`` baked in (and recorded in the metadata alongside
    ``model_version``). When ``verify`` is set, a graph that fails its parity check is removed and
    the :class:`ValueError` re-raised.
    """
    path = Path(path)
    wrapper = _OnnxInferenceModule(model, temperature).eval()
    # A batch > 1 example: torch.export specializes a size-1 dim to a constant, which would bake
    # the example's batch into the graph and defeat the dynamic batch axis.
    x, mask = _example_inputs(model, 2)
    path.parent.mkdir(parents=True, exist_ok=True)

    batch = torch.export.Dim("batch")
    with warnings.catch_warnings():
        # Both inputs tie dim 0 to the same `batch`; the exporter notes it keeps one name for the
        # shared constraint — expected, not a problem.
        warnings.filterwarnings("ignore", message=".*shares the same shape constraints.*")
        torch.onnx.export(
            wrapper,
            (x, mask),
            str(path),
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_shapes=({0: batch}, {0: batch}),
            opset_version=opset,
            dynamo=True,
        )

    import onnx

    metadata = {"temperature": repr(wrapper.temperature)}
    if model_version is not None:
        metadata["model_version"] = model_version
    onnx_model = onnx.load(str(path))
    for key, value in metadata.items():
        onnx_model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))
    onnx.save(onnx_model, str(path))

    if verify:
        try:
            onnx_parity(model, path, temperature=wrapper.temperature, tol=tol)
        except ValueError:
            # Only a genuine parity failure condemns the graph; an interrupt or a broken
            # onnxruntime install says nothing about it, so the file is kept in those cases.
            path.unlink(missing_ok=True)
            raise
    return path


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a Solia checkpoint to ONNX.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="path to a .pt checkpoint to export")
    source.add_argument(
        "--champion", action="store_true", help="export the registry's champion checkpoint"
    )
    parser.add_argument("--out", required=True, help="output .onnx path")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="calibration temperature baked into graph"
    )
    parser.add_argument("--no-verify", action="store_true", help="skip the ONNX parity check")
    args = parser.parse_args(argv)

    # The same loaders /predict uses, so the ONNX file is stamped with the exact model_version
    # Python serves (the registry's, for the champion) — one source of truth for traceability.
    if args.champion:
        from .db import make_engine

        try:
            predictor = Predictor.from_champion(make_engine(), temperature=args.temperature)
        except LookupError as exc:
            raise SystemExit(str(exc)) from None
    else:
        predictor = Predictor.from_checkpoint(args.checkpoint, temperature=args.temperature)

    out = export_onnx(
        predictor.model,
        args.out,
        opset=args.opset,
        model_version=predictor.model_version,
        temperature=predictor.temperature,
        verify=not args.no_verify,
    )
    print(f"exported {predictor.model_version} -> {out} (opset {args.opset})")


if __name__ == "__main__":
    main()
