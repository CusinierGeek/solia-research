import io

import numpy as np
import pytest
import torch

from solia_ai.config import CONFIG_ENV, load_config
from solia_ai.dataset import pad_window
from solia_ai.features import FEATURE_COLUMNS
from solia_ai.model import FeatureNormalizer, SoliaTransformer, signed_log1p

N_FEATURES = len(FEATURE_COLUMNS)
HORIZONS = ("30s", "1m", "5m")


def tiny(seq_len: int = 8, horizons=HORIZONS, dropout: float = 0.1) -> SoliaTransformer:
    return SoliaTransformer(
        n_features=N_FEATURES,
        horizons=horizons,
        d_model=16,
        n_heads=2,
        n_layers=1,
        ff_dim=32,
        dropout=dropout,
        seq_len=seq_len,
    )


@pytest.mark.parametrize("batch", [1, 4])
def test_forward_output_shapes(batch):
    model = tiny().eval()

    out = model(torch.randn(batch, 8, N_FEATURES))

    assert set(out) == {"pump_logits", "rug_logits", "expected_return"}
    for tensor in out.values():
        assert tensor.shape == (batch, 3)
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_from_config_builds_from_the_packaged_config(monkeypatch):
    monkeypatch.delenv(CONFIG_ENV, raising=False)  # independent of a developer override
    model = SoliaTransformer.from_config(load_config())

    assert model.n_horizons == 5
    assert model.horizons == ("30s", "1m", "5m", "15m", "1h")
    assert model.encoder.layers[0].linear1.out_features == load_config()["model"]["ff_dim"]
    assert model(torch.randn(2, model.seq_len, N_FEATURES))["pump_logits"].shape == (2, 5)


def test_from_config_rejects_an_empty_mapping_instead_of_loading_the_global_config():
    with pytest.raises(KeyError):
        SoliaTransformer.from_config({})


# --- predict keeps the caller's mode ---


def test_predict_probabilities_in_range():
    pred = tiny().predict(torch.randn(3, 8, N_FEATURES))

    assert torch.all((pred["pump_prob"] >= 0) & (pred["pump_prob"] <= 1))
    assert torch.all((pred["rug_prob"] >= 0) & (pred["rug_prob"] <= 1))
    assert pred["expected_return"].shape == (3, 3)


def test_predict_restores_training_mode():
    model = tiny().train()

    model.predict(torch.randn(2, 8, N_FEATURES))

    assert model.training
    assert all(m.training for m in model.modules())  # dropout stays active for training


def test_predict_keeps_eval_mode_and_is_deterministic():
    model = tiny().eval()
    x = torch.randn(2, 8, N_FEATURES)

    first = model.predict(x)["pump_prob"]

    assert not model.training
    assert torch.equal(first, model.predict(x)["pump_prob"])


def test_deterministic_under_seed():
    x = torch.randn(2, 8, N_FEATURES)
    torch.manual_seed(0)
    a = tiny().eval()
    torch.manual_seed(0)
    b = tiny().eval()

    assert torch.allclose(a(x)["pump_logits"], b(x)["pump_logits"])


# --- variable length & padding ---


def test_short_sequences_are_accepted_without_padding():
    out = tiny(seq_len=8).eval()(torch.randn(2, 3, N_FEATURES))

    assert out["pump_logits"].shape == (2, 3)


def test_padding_does_not_change_the_prediction():
    model = tiny(seq_len=8).eval()
    rows = np.random.default_rng(0).normal(size=(3, N_FEATURES)).astype(np.float32)
    x, mask = pad_window(rows, 8)

    padded = model(torch.from_numpy(x)[None], torch.from_numpy(mask)[None])["pump_logits"]
    unpadded = model(torch.from_numpy(rows)[None])["pump_logits"]

    assert torch.allclose(padded, unpadded, atol=1e-5)


def test_padding_values_are_ignored():
    model = tiny(seq_len=8).eval()
    rows = torch.randn(1, 8, N_FEATURES)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, :5] = True
    garbage = rows.clone()
    garbage[:, :5] = 1e6

    assert torch.allclose(
        model(rows, mask)["rug_logits"], model(garbage, mask)["rug_logits"], atol=1e-5
    )


def test_mixed_lengths_in_one_batch_match_individual_predictions():
    model = tiny(seq_len=8).eval()
    rng = np.random.default_rng(1)
    windows = [rng.normal(size=(n, N_FEATURES)).astype(np.float32) for n in (2, 8, 5)]
    xs, masks = zip(*(pad_window(w, 8) for w in windows), strict=True)

    batch = model(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(masks)))["pump_logits"]

    for k, w in enumerate(windows):
        alone = model(torch.from_numpy(w)[None])["pump_logits"][0]
        assert torch.allclose(batch[k], alone, atol=1e-5)


def test_longer_sequences_keep_the_most_recent_rows():
    model = tiny(seq_len=8).eval()
    x = torch.randn(1, 12, N_FEATURES)

    assert torch.allclose(model(x)["pump_logits"], model(x[:, -8:])["pump_logits"])


def test_fully_padded_sequence_is_rejected():
    model = tiny()
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[1] = True

    with pytest.raises(ValueError, match="entirely padding"):
        model(torch.randn(2, 8, N_FEATURES), mask)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((8, N_FEATURES), r"\[batch, length, features\]"),
        ((2, 8, N_FEATURES + 1), f"expected {N_FEATURES} features"),
        ((2, 0, N_FEATURES), "at least one row"),
    ],
)
def test_bad_input_shapes_are_rejected(shape, message):
    with pytest.raises(ValueError, match=message):
        tiny()(torch.randn(*shape))


def test_mismatched_padding_mask_is_rejected():
    with pytest.raises(ValueError, match="padding_mask shape"):
        tiny()(torch.randn(2, 8, N_FEATURES), torch.zeros(2, 7, dtype=torch.bool))


# --- normalization ---


def test_signed_log1p_keeps_sign_and_compresses():
    out = signed_log1p(torch.tensor([-1e7, -1.0, 0.0, 1e-8, 1e7]))

    assert out[2] == 0
    assert out[0] == -out[4]
    assert out.abs().max() < 17  # 1e7 -> ~16.1


def test_normalizer_fits_log_space_statistics():
    rng = np.random.default_rng(0)
    rows = np.column_stack([rng.uniform(1e-9, 1e-7, 500), rng.uniform(1e3, 1e7, 500)]).astype(
        np.float32
    )
    norm = FeatureNormalizer(2).fit(rows)

    out = norm(torch.from_numpy(rows))

    assert bool(norm.fitted)
    assert torch.allclose(out.mean(dim=0), torch.zeros(2), atol=1e-4)
    assert torch.allclose(out.std(dim=0, unbiased=False), torch.tensor([0.0, 1.0]), atol=1e-3)
    # tiny prices are below log precision -> constant -> left unscaled instead of dividing by ~0


def test_normalizer_rejects_bad_rows():
    norm = FeatureNormalizer(3)
    with pytest.raises(ValueError, match=r"\[R, 3\]"):
        norm.fit(np.zeros((4, 2)))
    with pytest.raises(ValueError, match=r"\[R, 3\]"):
        norm.fit(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        norm.fit(np.array([[1.0, np.inf, 0.0]]))


def test_huge_raw_feature_scales_stay_finite_through_the_model():
    model = tiny().eval()
    x = torch.zeros(4, 8, N_FEATURES)
    x[..., FEATURE_COLUMNS.index("price")] = 1e-8
    x[..., FEATURE_COLUMNS.index("market_cap")] = 1e7
    x[..., FEATURE_COLUMNS.index("token_age_sec")] = 1e6
    model.fit_normalizer(x.reshape(-1, N_FEATURES))

    out = model(x)

    assert all(torch.isfinite(v).all() for v in out.values())


# --- checkpoints ---


def roundtrip(state: dict) -> dict:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def test_checkpoint_round_trip_keeps_weights_normalizer_and_horizons():
    model = tiny().eval()
    model.fit_normalizer(torch.randn(50, N_FEATURES) * 100)
    x = torch.randn(2, 8, N_FEATURES)

    restored = tiny().eval()
    restored.load_state_dict(roundtrip(model.state_dict()))

    assert torch.equal(restored(x)["pump_logits"], model(x)["pump_logits"])
    assert bool(restored.normalizer.fitted)
    assert restored.horizons == HORIZONS


def test_checkpoint_with_other_horizons_is_refused():
    state = roundtrip(tiny(horizons=("1m", "30s", "5m")).state_dict())

    with pytest.raises(ValueError, match="horizons"):
        tiny().load_state_dict(state)


def test_checkpoint_with_another_feature_spec_is_refused():
    state = tiny().state_dict()
    state["_extra_state"] = {**state["_extra_state"], "feature_columns": ["price"] * N_FEATURES}

    with pytest.raises(ValueError, match="feature_columns"):
        tiny().load_state_dict(state)


# --- construction ---


def test_rejects_d_model_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        SoliaTransformer(n_features=N_FEATURES, horizons=HORIZONS, d_model=10, n_heads=3)


def test_rejects_no_horizons_and_mismatched_feature_columns():
    with pytest.raises(ValueError, match="horizon"):
        SoliaTransformer(n_features=N_FEATURES, horizons=())
    with pytest.raises(ValueError, match="feature columns"):
        SoliaTransformer(n_features=3, horizons=HORIZONS)
