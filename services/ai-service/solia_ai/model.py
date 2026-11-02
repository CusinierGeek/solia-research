"""The Solia signal model: a time-series Transformer.

Attends over a token's recent feature sequence and predicts, per horizon, from one shared
representation: ``pump`` / ``rug`` (classification, returned as logits) and ``expected_return``
(regression). It outputs predictions only — buy/sell/sizing/TP-SL are the policy layer's job
(see docs/ML-MODEL.md).

* **Normalization lives in the model** (:class:`FeatureNormalizer`): a signed ``log1p`` tames
  scales from ~1e-8 (price) to ~1e7 (market cap), then per-feature standardization with
  statistics fitted on the training rows. They are buffers, so checkpoints and the ONNX export
  apply exactly the transform training used.
* **Variable-length sequences**: ``1 .. seq_len`` rows, left-padded with a padding mask (see
  :func:`solia_ai.dataset.pad_window`). Attention ignores padding and the context is a masked mean,
  so a young token with a handful of snapshots can be scored without padding skewing it. Positions
  are relative to the most recent row.
* **The horizons and feature spec travel with the weights** (module extra state): a checkpoint
  refuses to load into a model with different horizons or features.
* **``expected_return`` is bounded by construction**: the return head emits ``softplus(x) - 1``,
  always ``>= -1`` (a long position cannot lose more than 100 %). A checkpoint saved before that
  head (no ``return_head`` in its extra state) loads with the legacy unbounded ``Linear`` output, so
  its weights keep their meaning; the serving contract clamps it (see
  :func:`solia_ai.inference.bound_return`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import load_config, model_shape_error
from .features import BASELINE_GROUPS, FEATURE_COLUMNS, column_mask, group_columns

# The return-head parametrisation recorded in a checkpoint's extra state. A checkpoint without the
# key predates it and carries the legacy raw ``Linear`` output.
BOUNDED_RETURN_HEAD = "softplus_minus_one"


def _sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Classic fixed positional encoding, shape ``[seq_len, d_model]``."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe


def _head(d_model: int, n_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, n_out),
    )


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    """``sign(x) * log(1 + |x|)``: monotonic, keeps sign and zero, compresses huge magnitudes."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


class IncompatibleCheckpointError(ValueError):
    """A checkpoint trained against another feature spec (or horizons) than this code's.

    Not a corrupt file: the model is fine, the code moved on. Serving it would mis-feed every
    column, so it is refused; the fix is a retrain on the current spec (re-replaying the features
    first when columns were added) and a promotion.
    """


class FeatureNormalizer(nn.Module):
    """``(signed_log1p(x) - mean) / std`` per feature, with ``mean``/``std`` stored as buffers.

    Identity-after-log until :meth:`fit` is called on training rows; :attr:`fitted` records it.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(
        self, rows: torch.Tensor | np.ndarray, *, index: Sequence[int] | None = None
    ) -> FeatureNormalizer:
        """Fit on ``[R, F]`` training rows (not validation: that would leak).

        With ``index``, ``rows`` is ``[R, len(index)]``: those features only, every other one taken
        as constant 0 (mean 0, std 1) — the same fit as on the full rows with those columns 0,
        without materializing them.
        """
        data = torch.as_tensor(rows, dtype=torch.float32)
        width = self.mean.numel() if index is None else len(index)
        if data.ndim != 2 or data.shape[1] != width or data.shape[0] == 0:
            raise ValueError(f"expected non-empty [R, {width}] rows, got {tuple(data.shape)}")
        if index is not None and sorted(set(index)) != sorted(index):
            raise ValueError(f"normalizer index has duplicates: {list(index)}")
        if index is not None and not all(0 <= int(i) < self.mean.numel() for i in index):
            raise ValueError(f"normalizer index out of range [0, {self.mean.numel()}): {index}")
        logged = signed_log1p(data)
        if not torch.isfinite(logged).all():
            raise ValueError("normalizer rows contain non-finite values")
        mean = logged.mean(dim=0)
        # Constant features (std 0) are left unscaled rather than divided by ~0.
        std = logged.std(dim=0, unbiased=False)
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
        if index is None:
            self.mean.copy_(mean)
            self.std.copy_(std)
        else:
            at = torch.as_tensor(list(index), dtype=torch.long)
            self.mean.zero_().index_copy_(0, at, mean)
            self.std.fill_(1.0).index_copy_(0, at, std)
        self.fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (signed_log1p(x) - self.mean) / self.std


class SoliaTransformer(nn.Module):
    """Transformer encoder with per-horizon pump/rug/return heads."""

    def __init__(
        self,
        *,
        n_features: int,
        horizons: Sequence[str],
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.1,
        seq_len: int = 64,
        feature_columns: Sequence[str] | None = None,
        enabled_columns: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        horizons = tuple(horizons)
        if not horizons:
            raise ValueError("at least one horizon is required")
        if problem := model_shape_error(
            d_model=d_model, n_heads=n_heads, seq_len=seq_len, min_seq_len=1
        ):
            raise ValueError(problem)
        columns = tuple(FEATURE_COLUMNS if feature_columns is None else feature_columns)
        if len(columns) != n_features:
            raise ValueError(f"{len(columns)} feature columns for n_features={n_features}")
        mask = column_mask(columns, columns if enabled_columns is None else enabled_columns)

        self.n_features = n_features
        self.horizons = horizons
        self.feature_columns = columns
        self.seq_len = seq_len

        # Disabled columns (feature groups not enabled) are zeroed before anything else, in training
        # and inference alike: no signal. Persisted, so a checkpoint keeps the mask it trained with.
        self.register_buffer("feature_mask", torch.from_numpy(mask))
        self.normalizer = FeatureNormalizer(n_features)
        self.input_proj = nn.Linear(n_features, d_model)
        self.register_buffer(
            "positional_encoding", _sinusoidal_encoding(seq_len, d_model), persistent=False
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model if ff_dim is None else ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # enable_nested_tensor=False: incompatible with norm_first and would just warn.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.pump_head = _head(d_model, len(horizons), dropout)
        self.rug_head = _head(d_model, len(horizons), dropout)
        self.return_head = _head(d_model, len(horizons), dropout)
        # False only for a legacy checkpoint loaded into this model (see set_extra_state).
        self.bounded_return = True

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)

    # --- checkpoint metadata (saved in state_dict via nn.Module extra state) ---

    def get_extra_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "horizons": list(self.horizons),
            "feature_columns": list(self.feature_columns),
        }
        if self.bounded_return:
            state["return_head"] = BOUNDED_RETURN_HEAD
        return state

    def set_extra_state(self, state: dict[str, Any]) -> None:
        for key, mine in (("horizons", self.horizons), ("feature_columns", self.feature_columns)):
            theirs = tuple(state.get(key, ()))
            if theirs != mine:
                raise IncompatibleCheckpointError(
                    f"checkpoint {key} {list(theirs)} don't match the model's {list(mine)}"
                )
        head = state.get("return_head")
        if head not in (None, BOUNDED_RETURN_HEAD):
            raise IncompatibleCheckpointError(f"unknown return head {head!r} in checkpoint")
        self.bounded_return = head == BOUNDED_RETURN_HEAD

    # --- forward ---

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """``x``: ``[B, L, n_features]``, ``L >= 1``; ``padding_mask``: ``[B, L]`` bool, ``True`` on
        (left) padding. Longer inputs keep their most recent ``seq_len`` rows. Returns a dict of
        ``[B, n_horizons]`` tensors."""
        if x.ndim != 3:
            raise ValueError(
                f"expected a [batch, length, features] tensor, got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {x.shape[-1]}")
        if x.shape[1] == 0:
            raise ValueError("expected at least one row per sequence")
        if padding_mask is not None:
            if padding_mask.shape != x.shape[:2]:
                raise ValueError(
                    f"padding_mask shape {tuple(padding_mask.shape)} != {tuple(x.shape[:2])}"
                )
            padding_mask = padding_mask.to(torch.bool)
        if x.shape[1] > self.seq_len:
            x = x[:, -self.seq_len :]
            padding_mask = None if padding_mask is None else padding_mask[:, -self.seq_len :]
        if padding_mask is not None and padding_mask.all(dim=1).any():
            raise ValueError("a sequence is entirely padding")
        return self.encode_logits(x, padding_mask)

    def encode_logits(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """The forward math on already-validated inputs (``L <= seq_len``, no fully-padded row).

        Split out so the ONNX export can trace the computation without :meth:`forward`'s
        data-dependent validation branches, which ``torch.export`` cannot capture — both paths
        share this one implementation, so the graph can never drift from training.
        """
        length = x.shape[1]
        # Right-aligned positions: the most recent row always gets the last encoding.
        hidden = (
            self.input_proj(self.normalizer(self.mask_features(x)))
            + self.positional_encoding[-length:]
        )
        if padding_mask is not None:
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)

        if padding_mask is None:
            context = encoded.mean(dim=1)
        else:
            keep = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
            context = (encoded * keep).sum(dim=1) / keep.sum(dim=1)
        expected_return = self.return_head(context)
        if self.bounded_return:
            # > -1 by construction; training regresses this same value on the (>= -1) target.
            expected_return = F.softplus(expected_return) - 1.0
        return {
            "pump_logits": self.pump_head(context),
            "rug_logits": self.rug_head(context),
            "expected_return": expected_return,
        }

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Inference-facing outputs: probabilities (sigmoid) + expected return. Runs in eval mode
        and restores the previous train/eval mode afterwards."""
        was_training = self.training
        self.eval()
        try:
            out = self.forward(x, padding_mask)
        finally:
            self.train(was_training)
        return {
            "pump_prob": torch.sigmoid(out["pump_logits"]),
            "rug_prob": torch.sigmoid(out["rug_logits"]),
            "expected_return": out["expected_return"],
        }

    def fit_normalizer(
        self, rows: torch.Tensor | np.ndarray, *, columns: Sequence[str] | None = None
    ) -> SoliaTransformer:
        """Fit the input normalization on ``[R, F]`` training rows (call before training).

        With ``columns`` (a narrow matrix, :attr:`solia_ai.dataset.Samples.columns`), ``rows`` holds
        only those columns, the others being 0: the fit is the same as on the widened rows.
        """
        data = torch.as_tensor(rows, dtype=torch.float32)
        if columns is None:
            self.normalizer.fit(self.mask_features(data))
            return self
        if data.ndim != 2 or data.shape[1] != len(columns) or data.shape[0] == 0:
            raise ValueError(
                f"expected non-empty [R, {len(columns)}] rows, got {tuple(data.shape)}"
            )
        unknown = [c for c in columns if c not in self.feature_columns]
        if unknown:
            raise ValueError(f"column(s) not among the model's features: {', '.join(unknown)}")
        index = [self.feature_columns.index(c) for c in columns]
        mask = self.feature_mask[torch.as_tensor(index, dtype=torch.long)].to(data.device)
        masked = torch.where(mask, data, torch.zeros((), dtype=data.dtype, device=data.device))
        self.normalizer.fit(masked, index=index)
        return self

    def mask_features(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (``[..., F]``) with disabled columns set to 0 — whatever they held, NaN/inf
        included (a product with the mask would keep them: ``nan * 0`` is NaN)."""
        mask = self.feature_mask.to(x.device)
        return torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        feature_columns: Sequence[str] | None = None,
    ) -> SoliaTransformer:
        """Build from a config. ``feature_columns`` (default :data:`FEATURE_COLUMNS`) rebuilds a
        checkpoint trained on an earlier, narrower spec."""
        cfg = load_config() if config is None else config
        m = cfg["model"]
        groups = cfg.get("features", {}).get("groups", BASELINE_GROUPS)
        columns = list(FEATURE_COLUMNS if feature_columns is None else feature_columns)
        return cls(
            n_features=len(columns),
            horizons=cfg["horizons"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            n_layers=m["n_layers"],
            ff_dim=m.get("ff_dim"),
            dropout=m["dropout"],
            seq_len=m["seq_len"],
            feature_columns=columns,
            enabled_columns=[c for c in group_columns(groups) if c in columns],
        )
