"""#205: the GBDT (LightGBM) benchmark on the Transformer's own walk-forward windows."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from test_train import make_features_df, small_config

import solia_ai.ablation as ablation
import solia_ai.benchmark_gbdt as bench
import solia_ai.config
import solia_ai.db
from solia_ai.ablation import walk_forward_windows
from solia_ai.benchmark_gbdt import (
    GbdtSettings,
    MissingDependencyError,
    require_lightgbm,
    run_benchmark,
    tabular_columns,
    tabular_features,
)
from solia_ai.dataset import config_samples, embargo_seconds
from solia_ai.features import BASELINE_GROUPS, FEATURE_COLUMNS, group_columns

pytest.importorskip("lightgbm")  # CI installs the benchmark extra; the missing case is faked below

FAST = GbdtSettings(
    params={"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5},
    early_stopping_rounds=5,
    n_jobs=1,
)
COLUMNS = group_columns(BASELINE_GROUPS)


def samples():
    return config_samples(make_features_df(n_tokens=8), small_config())


def bench_run(**kwargs):
    kwargs = {"n_windows": 1, "seeds": [0], "gbdt": FAST, "device": "cpu", **kwargs}
    return run_benchmark(make_features_df(n_tokens=8), small_config(), **kwargs)


# --- tabular inputs ---


def test_tabular_inputs_summarize_each_windows_real_rows():
    s = samples()
    tab = tabular_features(s, COLUMNS)
    index = [FEATURE_COLUMNS.index(c) for c in COLUMNS]
    k = len(COLUMNS)

    assert tab.shape == (len(s), 5 * k + 1) and tab.dtype == np.float32
    for i in (0, len(s) // 2, len(s) - 1):
        rows = s.window(i)[:, index].astype(np.float64)
        expected = np.concatenate(
            [rows[-1], rows.mean(0), rows.min(0), rows.max(0), rows[-1] - rows[0], [len(rows)]]
        )
        np.testing.assert_allclose(tab[i], expected.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_tabular_inputs_of_a_subset_and_in_chunks_are_the_same(monkeypatch):
    s = samples()
    whole = tabular_features(s, COLUMNS)
    monkeypatch.setattr(bench, "_CHUNK", 7)
    picks = np.arange(len(s))[::5]
    np.testing.assert_array_equal(tabular_features(s, COLUMNS, indices=picks), whole[picks])


def test_tabular_column_names_follow_the_layout():
    names = tabular_columns(["price", "volume"])
    assert names[:2] == ["last:price", "last:volume"] and names[-1] == "window_len"
    assert len(names) == 5 * 2 + 1


@pytest.mark.parametrize("columns", [[], ["not_a_column"]])
def test_tabular_inputs_need_feature_columns(columns):
    with pytest.raises(ValueError, match="feature columns"):
        tabular_features(samples(), columns)


# --- same windows, no leakage ---


def test_both_models_run_on_the_ablations_windows_and_the_gbdt_never_sees_the_slice(monkeypatch):
    s = samples()
    expected = walk_forward_windows(
        s.end_ts, n_windows=2, eval_fraction=0.3, val_fraction=0.2,
        embargo=embargo_seconds(s.horizons),
    )  # fmt: skip
    tab = tabular_features(s, COLUMNS)
    seen_tr, seen_gb, current = [], [], {}
    real_tr, real_fit, real_run = ablation.train_and_score, bench.fit_head, bench.gbdt_run

    def spy_run(tab, samples, window, **kw):
        current["window"] = window.index
        return real_run(tab, samples, window, **kw)

    def spy_tr(train_ds, val_ds, eval_ds, cfg, **kw):
        seen_tr.append((train_ds.indices, val_ds.indices, eval_ds.indices))
        return real_tr(train_ds, val_ds, eval_ds, cfg, **kw)

    def spy_fit(x_train, y_train, x_val, y_val, **kw):
        seen_gb.append((current["window"], x_train, x_val))
        return real_fit(x_train, y_train, x_val, y_val, **kw)

    monkeypatch.setattr(bench, "train_and_score", spy_tr)
    monkeypatch.setattr(bench, "fit_head", spy_fit)
    monkeypatch.setattr(bench, "gbdt_run", spy_run)
    bench_run(n_windows=2, eval_fraction=0.3)

    assert len(seen_tr) == 2
    for (tr, va, ev), window in zip(seen_tr, expected, strict=True):
        np.testing.assert_array_equal(tr, window.train_idx)
        np.testing.assert_array_equal(va, window.val_idx)
        np.testing.assert_array_equal(ev, window.eval_idx)

    def rows(idx):
        return {r.tobytes() for r in tab[idx]}

    allowed_train = [rows(w.train_idx) for w in expected]
    allowed_val = [rows(w.val_idx) for w in expected]
    scored = [rows(w.eval_idx) for w in expected]
    assert seen_gb
    assert {k for k, _, _ in seen_gb} == {0, 1}
    for k, x_train, x_val in seen_gb:
        assert {r.tobytes() for r in x_train} <= allowed_train[k]
        assert {r.tobytes() for r in x_val} <= allowed_val[k]  # early stopping: the val split
        assert not ({r.tobytes() for r in x_train} | {r.tobytes() for r in x_val}) & scored[k]


def test_the_gbdt_is_scored_on_the_same_labels_as_the_transformer(monkeypatch):
    s = samples()
    window = walk_forward_windows(s.end_ts, n_windows=1, eval_fraction=0.3, val_fraction=0.2,
                                  embargo=embargo_seconds(s.horizons))[0]  # fmt: skip
    seen = []
    real = bench.classification_metrics

    def spy(y, prob, **kw):
        seen.append(np.asarray(y))
        return real(y, prob, **kw)

    monkeypatch.setattr(bench, "classification_metrics", spy)
    bench.gbdt_run(tabular_features(s, COLUMNS), s, window, seed=0, settings=FAST)

    labeled = [window.eval_idx[s.mask[window.eval_idx, h] == 1] for h in range(len(s.horizons))]
    expected = [
        getattr(s, head)[idx, h] for h, idx in enumerate(labeled) for head in ("pump", "rug")
    ]
    assert len(seen) == len(expected)
    for got, want in zip(seen, expected, strict=True):
        np.testing.assert_array_equal(got, want)


# --- the report ---


def test_the_report_has_the_expected_shape(tmp_path):
    report = bench_run(seeds=[0, 1])
    horizons = small_config()["horizons"]
    keys = [f"{hd}/{h}/{m}" for hd in ("pump", "rug") for h in horizons
            for m in ("roc_auc", "pr_auc", "ece")]  # fmt: skip

    assert set(report.summary) == {"transformer", "gbdt"}
    assert list(report.summary["gbdt"]) == keys and list(report.lift) == keys
    assert {r["model"] for r in report.runs} == {"transformer", "gbdt"} and len(report.runs) == 4
    assert all(lift["verdict"] in ("better", "worse", "not significant", "n/a")
               for lift in report.lift.values())  # fmt: skip
    assert report.importance and set(report.importance[0]) == {"column", "gain", "share"}
    assert sum(i["share"] for i in report.importance) <= 1.0 + 1e-9
    assert report.settings["columns"] == COLUMNS and len(report.windows) == 1

    json_path, text_path = report.write(tmp_path / "gbdt.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["settings"]["gbdt"]["early_stopping_rounds"] == 5
    text = text_path.read_text(encoding="utf-8")
    assert "GBDT (LightGBM) vs Transformer" in text and "Top" in text and "pump/1m/pr_auc" in text


def test_bad_settings_are_refused_before_training():
    with pytest.raises(ValueError, match="n_windows"):
        bench_run(n_windows=0)
    with pytest.raises(ValueError, match="top_features"):
        bench_run(top_features=0)


# --- the optional dependency ---


@pytest.fixture
def no_lightgbm(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)  # import lightgbm -> ImportError


def test_a_missing_lightgbm_names_the_extra(no_lightgbm):
    with pytest.raises(MissingDependencyError, match=r'pip install -e "\.\[benchmark\]"'):
        require_lightgbm()


def test_the_benchmark_refuses_to_start_without_lightgbm(no_lightgbm, monkeypatch):
    monkeypatch.setattr(bench, "config_samples", lambda *a, **k: pytest.fail("must not build"))
    with pytest.raises(MissingDependencyError):
        bench_run()


def test_the_cli_exits_with_the_hint_before_reading(no_lightgbm, monkeypatch):
    monkeypatch.setattr(solia_ai.db, "read_training_features",
                        lambda *a, **k: pytest.fail("must not read"))  # fmt: skip
    with pytest.raises(SystemExit, match="benchmark"):
        bench.main([])


def test_the_cli_reads_the_bounded_data_and_writes_both_reports(monkeypatch, tmp_path):
    df = make_features_df(n_tokens=8)
    reads = []

    class Bounds:
        @staticmethod
        def describe():
            return "fake read"

        @staticmethod
        def as_dict():
            return {"rows_read": len(df)}

    def fake_read(engine, settings, end=None, columns=None):
        reads.append((settings.max_rows, columns))
        return df, Bounds()

    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())
    monkeypatch.setattr(solia_ai.db, "make_engine", lambda *a, **k: object())
    monkeypatch.setattr(solia_ai.db, "read_training_features", fake_read)
    monkeypatch.setattr(bench, "GbdtSettings", lambda **kw: FAST)
    out = tmp_path / "reports" / "gbdt.json"

    bench.main(["--windows", "1", "--seeds", "0", "--max-rows", "500", "--device", "cpu",
                "--out", str(out)])  # fmt: skip

    assert reads == [(500, None)]
    assert json.loads(out.read_text(encoding="utf-8"))["data"] == {"rows_read": len(df)}
    assert out.with_suffix(".txt").exists()
