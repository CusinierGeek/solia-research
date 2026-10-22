import numpy as np
import pytest
import torch
from torch.nn import functional as F

from solia_ai.config import CONFIG_ENV, load_config
from solia_ai.features import FEATURE_COLUMNS
from solia_ai.loss import SoliaLoss, focal_bce_loss
from solia_ai.model import SoliaTransformer

N_FEATURES = len(FEATURE_COLUMNS)
HORIZONS = ("30s", "1m", "5m")


def tiny_model(seq_len: int = 8) -> SoliaTransformer:
    return SoliaTransformer(
        n_features=N_FEATURES,
        horizons=HORIZONS,
        d_model=16,
        n_heads=2,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
        seq_len=seq_len,
    )


def all_labeled(pump, rug, ret):
    return {
        "pump": pump,
        "rug": rug,
        "expected_return": ret,
        "mask": torch.ones_like(pump),
    }


# --- focal_bce_loss ---


def test_focal_with_alpha_one_and_gamma_zero_is_plain_bce():
    logits = torch.tensor([[-2.0, 0.5, 3.0]])
    target = torch.tensor([[0.0, 1.0, 1.0]])

    out = focal_bce_loss(logits, target, alpha=1.0, gamma=0.0)

    assert torch.allclose(out, F.binary_cross_entropy_with_logits(logits, target, reduction="none"))


def test_focal_downweights_easy_examples_relative_to_bce():
    logits = torch.tensor([[4.0]])  # confident and correct
    target = torch.tensor([[1.0]])

    bce = focal_bce_loss(logits, target, alpha=1.0, gamma=2.0)
    focal = focal_bce_loss(logits, target, alpha=0.0, gamma=2.0)

    assert focal.item() < bce.item()  # the focal term shrinks the loss on an easy positive


def test_no_pos_weight_matches_a_unit_pos_weight():
    torch.manual_seed(0)
    logits, target = torch.randn(4, 3), torch.randint(0, 2, (4, 3)).float()

    plain = focal_bce_loss(logits, target, alpha=0.3, gamma=2.0)
    unit = focal_bce_loss(logits, target, alpha=0.3, gamma=2.0, pos_weight=torch.ones(3))

    assert torch.allclose(plain, unit)


def test_label_weight_is_mask_times_horizon_weights():
    loss_fn = SoliaLoss(
        HORIZONS, alpha=0.5, focal_gamma=2.0, horizon_weights={"30s": 2.0, "5m": 0.5}
    )
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    weight = loss_fn.label_weight({"mask": mask})

    assert torch.equal(weight, torch.tensor([[2.0, 0.0, 0.5], [0.0, 1.0, 0.5]]))


def test_pos_weight_scales_the_positive_class():
    logits = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[1.0, 0.0]])

    out = focal_bce_loss(logits, target, alpha=1.0, gamma=0.0, pos_weight=torch.tensor([3.0, 3.0]))

    # positive entry is 3x its unit-weight BCE (log 2); negative entry is unchanged.
    assert out[0, 0].item() == pytest.approx(3 * np.log(2), rel=1e-5)
    assert out[0, 1].item() == pytest.approx(np.log(2), rel=1e-5)


# --- SoliaLoss masking & weighting ---


def test_perfect_predictions_give_near_zero_loss():
    loss_fn = SoliaLoss(HORIZONS, alpha=0.5, focal_gamma=2.0)
    big = torch.full((2, 3), 20.0)
    outputs = {
        "pump_logits": big.clone(),
        "rug_logits": -big.clone(),
        "expected_return": torch.zeros(2, 3),
    }
    targets = all_labeled(torch.ones(2, 3), torch.zeros(2, 3), torch.zeros(2, 3))

    out = loss_fn(outputs, targets)

    assert out["total"].item() == pytest.approx(0.0, abs=1e-6)


def test_masked_horizons_do_not_affect_the_loss():
    loss_fn = SoliaLoss(HORIZONS, alpha=0.5, focal_gamma=2.0)
    outputs = {
        "pump_logits": torch.zeros(1, 3),
        "rug_logits": torch.zeros(1, 3),
        "expected_return": torch.zeros(1, 3),
    }
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    targets = {
        "pump": torch.tensor([[1.0, 0.0, 1.0]]),
        "rug": torch.zeros(1, 3),
        "expected_return": torch.tensor([[0.0, 0.0, 5.0]]),
        "mask": mask,
    }

    baseline = loss_fn(outputs, targets)["total"].item()
    # Rewrite everything under the masked-out horizon; the loss must not move.
    targets["pump"][0, 2] = 0.0
    targets["expected_return"][0, 2] = -99.0

    assert loss_fn(outputs, targets)["total"].item() == pytest.approx(baseline)


def test_all_masked_is_zero_and_stays_differentiable():
    loss_fn = SoliaLoss(HORIZONS, alpha=0.5, focal_gamma=2.0)
    logits = torch.zeros(1, 3, requires_grad=True)
    outputs = {"pump_logits": logits, "rug_logits": logits, "expected_return": logits}
    targets = {
        "pump": torch.ones(1, 3),
        "rug": torch.ones(1, 3),
        "expected_return": torch.ones(1, 3),
        "mask": torch.zeros(1, 3),
    }

    total = loss_fn(outputs, targets)["total"]
    total.backward()

    assert total.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros(1, 3))


def test_horizon_weights_reweight_the_mean():
    outputs = {
        "pump_logits": torch.zeros(1, 2),
        "rug_logits": torch.zeros(1, 2),
        "expected_return": torch.tensor([[1.0, 3.0]]),
    }
    targets = all_labeled(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2))

    uniform = SoliaLoss(("30s", "1m"), alpha=1.0, focal_gamma=0.0)(outputs, targets)["return"]
    skewed = SoliaLoss(("30s", "1m"), alpha=1.0, focal_gamma=0.0, horizon_weights={"1m": 3.0})(
        outputs, targets
    )["return"]

    # Huber(1)=0.5, Huber(3)=2.5. Uniform mean = 1.5; weighting 1m by 3x pulls it toward 2.5.
    assert uniform.item() == pytest.approx(1.5)
    assert skewed.item() == pytest.approx((0.5 * 1 + 2.5 * 3) / 4)


def test_task_weights_combine_the_components():
    loss_fn = SoliaLoss(
        HORIZONS, alpha=0.5, focal_gamma=2.0, task_weights={"pump": 2.0, "rug": 0.0, "return": 3.0}
    )
    outputs = {
        "pump_logits": torch.randn(2, 3),
        "rug_logits": torch.randn(2, 3),
        "expected_return": torch.randn(2, 3),
    }
    targets = all_labeled(
        (torch.rand(2, 3) > 0.5).float(), (torch.rand(2, 3) > 0.5).float(), torch.randn(2, 3)
    )

    out = loss_fn(outputs, targets)

    expected = 2.0 * out["pump"] + 0.0 * out["rug"] + 3.0 * out["return"]
    assert out["total"].item() == pytest.approx(expected.item())


# --- from_config ---


def test_from_config_maps_strategy_and_regression_weights(monkeypatch):
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    cfg = load_config()

    loss_fn = SoliaLoss.from_config(cfg)

    assert loss_fn.horizons == tuple(cfg["horizons"])
    assert loss_fn.task_weights["pump"] == cfg["loss"]["strategy_weight"]
    assert loss_fn.task_weights["rug"] == cfg["loss"]["strategy_weight"]
    assert loss_fn.task_weights["return"] == cfg["loss"]["regression_weight"]


def test_from_config_passes_pos_weight():
    loss_fn = SoliaLoss.from_config(
        load_config(), pos_weight={"pump": torch.full((5,), 4.0), "rug": torch.ones(5)}
    )

    assert torch.equal(loss_fn.pump_pos_weight, torch.full((5,), 4.0))


# --- validation ---


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 1.5, "focal_gamma": 2.0}, "alpha"),
        ({"alpha": 0.5, "focal_gamma": -1.0}, "focal_gamma"),
        ({"alpha": 0.5, "focal_gamma": 2.0, "task_weights": {"nope": 1.0}}, "unknown task"),
        ({"alpha": 0.5, "focal_gamma": 2.0, "horizon_weights": {"30s": -1.0}}, "non-negative"),
        ({"alpha": 0.5, "focal_gamma": 2.0, "horizon_weights": {"1hr": 2.0}}, "unknown horizon"),
    ],
)
def test_invalid_construction_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SoliaLoss(HORIZONS, **kwargs)


def test_pos_weight_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="pos_weight shape"):
        SoliaLoss(HORIZONS, alpha=0.5, focal_gamma=2.0, pos_weight={"pump": torch.ones(2)})


# --- the gradient actually trains ---


def test_one_training_step_reduces_the_loss():
    torch.manual_seed(0)
    model = tiny_model().train()
    model.fit_normalizer(torch.randn(64, N_FEATURES))
    loss_fn = SoliaLoss(HORIZONS, alpha=0.5, focal_gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

    x = torch.randn(16, 8, N_FEATURES)
    targets = all_labeled(
        (torch.rand(16, 3) > 0.5).float(), (torch.rand(16, 3) > 0.7).float(), torch.randn(16, 3)
    )

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(model(x), targets)["total"]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
