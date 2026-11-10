"""Winning wallets (#269): parsing, positions, profiles, persistence, copy test, partitions, CLI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from solia_ai import wallet_study as w
from solia_ai.survivor_study import Costs

T0 = 1_793_000_000_000  # ms
FREE = Costs(size=1e-9, network_fee=0.0, account_rent=0.0, fee_pct={"default": 0.0}, virtual_sol={})


def frame(**e) -> bytes:
    return json.dumps(e, separators=(",", ":")).encode()  # the archives are compact


def trade(
    action,
    mint,
    wallet,
    ts,
    tokens,
    sol,
    *,
    block=1,
    reserve=100.0,
    price=None,
    sig=None,
    pool="pump-amm",
    quote=w.WSOL,
    fee=0.0,
):
    return frame(
        action=action,
        mint=mint,
        txSigner=wallet,
        timestamp=ts,
        block=block,
        tokenAmount=tokens,
        quoteAmount=sol,
        quoteInPool=reserve,
        price=price if price is not None else sol / tokens,
        pool=pool,
        quoteMint=quote,
        priorityFee=fee,
        signature=sig or f"{wallet}{ts}{action}",
    )


# --------------------------------------------------------------------------- parsing


def test_parse_keeps_sol_trades_creates_and_token_transfers():
    lines = [
        trade("buy", "M", "a", T0, 100.0, 1.0, fee=0.001),
        trade("sell", "M", "a", T0 + 1000, 100.0, 2.0),
        trade("buy", "U", "a", T0, 100.0, 1.0, quote="USDC"),  # not SOL-quoted
        frame(
            action="create",
            mint="N",
            txSigner="c",
            timestamp=T0,
            block=5,
            initialBuy=1000.0,
            quoteAmount=0.5,
            vQuoteInBondingCurve=30.5,
            quoteInPool=0.5,
            pool="pump",
            signature="s1",
        ),
        frame(
            action="transfer",
            txSigner="a",
            timestamp=T0,
            block=1,
            transfers=[
                {"from": "a", "to": "b", "amount": 5, "isSolana": False, "mint": "M"},
                {"from": "a", "to": "b", "amount": 1, "isSolana": True, "mint": "So1"},
            ],
        ),
        frame(
            action="transfer",
            txSigner="a",
            timestamp=T0,
            transfers=[{"from": "a", "to": "b", "amount": 1, "isSolana": True, "mint": "So1"}],
        ),
        b"not json",
        frame(action="leaders", mint="M"),
    ]
    trades, creates, transfers = w.parse_lines(lines)
    assert list(trades["side"]) == ["buy", "sell", "buy"]
    assert trades.loc[trades["mint"] == "N", "reserve"].iloc[0] == 30.5  # the curve's virtual SOL
    assert trades.loc[trades["mint"] == "N", "venue"].iloc[0] == "pumpfun"
    assert trades["priority_fee"].iloc[0] == 0.001
    assert list(creates["creator"]) == ["c"] and creates["block"].iloc[0] == 5
    assert transfers[["from", "to", "mint"]].values.tolist() == [["a", "b", "M"]]


def test_a_create_dev_buy_is_not_counted_twice():
    lines = [
        frame(
            action="create",
            mint="N",
            txSigner="c",
            timestamp=T0,
            block=5,
            initialBuy=1000.0,
            quoteAmount=0.5,
            pool="pump",
            signature="s1",
        ),
        trade("buy", "N", "c", T0, 1000.0, 0.5, sig="s1"),
    ]
    trades, _, _ = w.parse_lines(lines)
    assert len(trades) == 1


def test_extract_hour_partitions_by_token(tmp_path):
    import zstandard

    day = tmp_path / "archives" / "2026" / "10" / "29"
    day.mkdir(parents=True)
    lines = [trade("buy", f"M{i}", "a", T0 + i, 10.0, 1.0) for i in range(20)]
    (day / "03.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(b"\n".join(lines)))
    out = tmp_path / "out"
    msgs = w.extract(tmp_path / "archives", out, jobs=1, parts=3)
    assert "20 trades" in msgs[0]
    parts = w.partitions(out)
    assert len(parts) == 3
    loaded = [w.load_period(p, w._utc("2026-10-29"), w._utc("2026-10-30")) for p in parts]
    mints = [set(x["trades"]["mint"].astype(str)) for x in loaded]
    assert sum(len(m) for m in mints) == 20 and not (mints[0] & mints[1])
    assert "already" in w.extract(tmp_path / "archives", out, parts=3)[0]
    with pytest.raises(ValueError, match="no .jsonl.zst"):
        w.extract(tmp_path / "nothing", out)


def test_hour_of_archive_path(tmp_path):
    path = tmp_path / "2026" / "11" / "04" / "07.jsonl.zst"
    assert w.hour_of(path) == w._utc("2026-11-04 07:00")


# --------------------------------------------------------------------------- positions


def trades_frame(rows) -> pd.DataFrame:
    cols = [
        "ts",
        "block",
        "mint",
        "wallet",
        "side",
        "tokens",
        "sol",
        "reserve",
        "price",
        "venue",
        "priority_fee",
    ]
    return pd.DataFrame(rows, columns=cols).astype({"block": "Int64"})


def t(ts, mint, wallet, side, tokens, sol, block=10, fee=0.0, reserve=100.0):
    return (
        T0 + ts * 1000,
        block,
        mint,
        wallet,
        side,
        tokens,
        sol,
        reserve,
        sol / tokens,
        "pumpswap",
        fee,
    )


EMPTY_X = pd.DataFrame(columns=["ts", "mint", "from", "to", "amount"])


def test_positions_statuses_and_pnl():
    tr = trades_frame(
        [
            t(0, "M", "win", "buy", 100, 1.0, fee=0.01),
            t(60, "M", "win", "sell", 100, 1.5),
            t(0, "M", "hold", "buy", 100, 1.0),
            t(30, "M", "hold", "sell", 50, 0.5),
            t(10, "M", "dump", "sell", 10, 0.1),  # sells without buying
            t(0, "M", "moved", "buy", 100, 1.0),
            t(90, "M", "last", "buy", 1, 0.02),  # sets the last price 0.02 / token
        ]
    )
    x = pd.DataFrame([(T0, "M", "moved", "other", 100.0)], columns=EMPTY_X.columns)
    creates = pd.DataFrame([(T0 - 2000, 8, "M", "dev")], columns=["ts", "block", "mint", "creator"])
    p = w.positions(tr, creates, x).set_index("wallet")
    assert p.loc["win", "status"] == "closed"
    assert p.loc["win", "pnl"] == pytest.approx(0.49)
    assert p.loc["win", "hold_s"] == 60 and p.loc["win", "entry_delay_s"] == 2.0
    assert p.loc["win", "entry_delay_blocks"] == 2
    assert p.loc["hold", "status"] == "open"
    assert p.loc["hold", "pnl"] == pytest.approx(0.5 - 1.0 + 50 * 0.02)  # marked at last price
    assert p.loc["dump", "status"] == "unbacked" and np.isnan(p.loc["dump", "pnl"])
    assert p.loc["moved", "status"] == "transferred"
    assert not p["is_creator"].any()


def test_creator_flag_and_same_block_share():
    tr = trades_frame(
        [
            t(0, "M", "dev", "buy", 100, 1.0, block=8),
            t(5, "M", "dev", "sell", 100, 3.0, block=9),
            t(0, "M", "fast", "buy", 100, 1.0, block=8),
            t(5, "M", "fast", "sell", 100, 0.5),
            t(0, "OLD", "fast", "buy", 100, 1.0, block=50),  # created before the data
            t(5, "OLD", "fast", "sell", 100, 1.1, block=51),
        ]
    )
    creates = pd.DataFrame([(T0, 8, "M", "dev")], columns=["ts", "block", "mint", "creator"])
    p = w.positions(tr, creates, EMPTY_X)
    prof = w.wallet_profiles(p)
    assert prof.loc["dev", "creator_share"] == 1.0 and prof.loc["fast", "creator_share"] == 0.0
    # same-block share among the positions whose create is in the data (OLD's isn't)
    assert prof.loc["fast", "same_block"] == 1.0 and prof.loc["fast", "create_seen"] == 0.5
    assert prof.loc["dev", "win_rate"] == 1.0 and prof.loc["fast", "win_rate"] == 0.5


def profiles(pnls: dict[str, list[float]]) -> pd.DataFrame:
    rows = [
        {
            "wallet": k,
            "mint": f"{k}{i}",
            "status": "closed",
            "pnl": v,
            "buy_sol": 1.0,
            "hold_s": 60.0,
            "entry_delay_s": 1.0,
            "entry_delay_blocks": 0.0,
            "is_creator": False,
        }
        for k, vs in pnls.items()
        for i, v in enumerate(vs)
    ]
    return w.wallet_profiles(pd.DataFrame(rows))


def test_top_wallets_need_enough_positions():
    prof = profiles({"big_once": [50.0], "steady": [1.0] * 10, "loser": [-1.0] * 10})
    assert list(w.top_wallets(prof, n=2)) == ["steady", "loser"]


def test_persistence_compares_top_and_rest_on_b():
    a = profiles({f"t{i}": [1.0] * 10 for i in range(3)} | {f"r{i}": [-0.1] * 10 for i in range(3)})
    b = profiles({"t0": [0.5] * 6, "t1": [-0.2] * 6, "r0": [-0.5] * 6})
    old = w.TOP_N
    w.TOP_N = 3
    try:
        out = w.persistence(a, b, samples=50)
    finally:
        w.TOP_N = old
    assert out["eligible_in_a"] == 6
    assert out["top"]["active_in_b"] == 2 and out["top"]["share_profitable"] == 0.5
    assert out["top"]["pnl_total"] == pytest.approx(0.5 * 6 - 0.2 * 6)
    assert out["rest"]["active_in_b"] == 1 and out["rest"]["pnl_total"] == pytest.approx(-3.0)
    assert out["wallets_in_both"] == 3


def test_group_stats_without_activity():
    assert w.group_stats(profiles({"x": [1.0]}), pd.Index(["nobody"])) == {
        "wallets": 1,
        "active_in_b": 0,
    }


def test_profile_by_pnl_buckets():
    prof = profiles({f"w{i:03d}": [float(i)] * 10 for i in range(200)})
    out = w.profile_by_pnl(prof)
    assert out["top1%"]["wallets"] == 2 and out["bottom10%"]["wallets"] == 20
    assert out["top1%"]["pnl_total"] > out["middle"]["pnl_total"] / 160


# --------------------------------------------------------------------------- copy test


def test_copy_enters_after_the_latency_and_exits_after_the_leader_sells():
    tr = trades_frame(
        [
            t(0, "M", "lead", "buy", 100, 1.0),  # price 0.01
            t(1, "M", "x", "buy", 100, 1.2),  # 0.012: inside the 3 s latency
            t(4, "M", "x", "buy", 100, 1.5),  # 0.015: our entry
            t(30, "M", "lead", "sell", 100, 2.5),  # leader exits
            t(31, "M", "y", "buy", 100, 2.4),  # inside the 2 s exit latency
            t(33, "M", "y", "sell", 100, 3.0),  # 0.03: our exit
            t(0, "N", "lead", "buy", 100, 1.0),  # no trade after the latency: not copied
        ]
    )
    out = w.copy_trades(tr, pd.Index(["lead"]), 3.0, FREE)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["exit_kind"] == "sell"
    assert row["net"] == pytest.approx(0.03 / 0.015 - 1, abs=1e-6)


def test_copy_without_a_leader_sell_exits_at_the_last_trade():
    tr = trades_frame(
        [
            t(0, "M", "lead", "buy", 100, 1.0),
            t(5, "M", "x", "buy", 100, 1.0),
            t(50, "M", "x", "sell", 100, 0.5),
        ]
    )
    out = w.copy_trades(tr, pd.Index(["lead"]), 3.0, FREE)
    assert out["exit_kind"].iloc[0] == "end"
    assert out["net"].iloc[0] == pytest.approx(-0.5, abs=1e-6)


def test_copy_pays_costs_and_summary():
    tr = trades_frame(
        [
            t(0, "M", "lead", "buy", 100, 1.0),
            t(5, "M", "x", "buy", 100, 1.0, reserve=20.0),
            t(9, "M", "lead", "sell", 100, 1.0),
            t(12, "M", "x", "sell", 100, 1.0, reserve=20.0),
        ]
    )
    out = w.copy_trades(tr, pd.Index(["lead"]), 3.0, Costs(virtual_sol={}))
    assert out["net"].iloc[0] < -0.08  # flat price, 1 SOL in a 20 SOL pool + fees
    s = w.copy_summary(out, samples=20)
    assert s["copies"] == 1 and s["win_rate"] == 0.0 and s["exited_on_leader_sell"] == 1.0
    assert w.copy_summary(out.iloc[:0]) == {"copies": 0}


# --------------------------------------------------------------------------- study & CLI


def period(seed: int) -> dict[str, pd.DataFrame]:
    """Wallet `smart` buys before a rise and sells at the top on every token; `dumb` the reverse."""
    rows = []
    for i in range(12):
        m, base = f"M{seed}{i}", i * 600
        rows += [
            t(base, m, "smart", "buy", 100, 1.0),
            t(base + 1, m, "dumb", "buy", 100, 1.0),
            t(base + 5, m, "crowd", "buy", 100, 1.5),
            t(base + 20, m, "crowd", "buy", 100, 2.0),
            t(base + 40, m, "smart", "sell", 100, 2.0),
            t(base + 45, m, "x", "buy", 100, 2.2),
            t(base + 50, m, "dumb", "sell", 100, 0.5),
            t(base + 55, m, "x", "sell", 100, 2.1),
        ]
    tr = trades_frame(rows)
    return {
        "trades": tr,
        "creates": pd.DataFrame(columns=["ts", "block", "mint", "creator"]),
        "transfers": EMPTY_X,
    }


def test_study_end_to_end():
    report = w.study(period(1), period(2), costs=FREE, samples=50)
    per = report["persistence"]
    assert per["eligible_in_a"] == 3  # smart, dumb, x (crowd never sells: open positions)
    top = report["top_wallets_a"][0]
    assert top["wallet"] == "smart" and top["b_pnl"] > 0
    copy = report["copy"]["3s"]["top"]
    assert copy["copies"] > 0
    text = w.render(report)
    assert "Persistence" in text and "Copy test" in text


def test_partitioned_study_matches(tmp_path):
    for k, seed in ((0, 1), (1, 2)):
        for _name, p, stamp in (
            ("A", period(seed), "2026102903"),
            ("B", period(seed + 10), "2026110303"),
        ):
            d = tmp_path / f"p{k}"
            d.mkdir(exist_ok=True)
            for kind, f in p.items():
                f.to_parquet(d / f"{kind}-{stamp}.parquet", index=False)
    report = w.study_partitions(tmp_path, w.PERIOD_A, w.PERIOD_B, costs=FREE, samples=20)
    assert report["periods"]["A"]["tokens"] == 24
    assert report["top_wallets_a"][0]["wallet"] == "smart"
    with pytest.raises(ValueError, match="no token partitions"):
        w.study_partitions(tmp_path / "none", w.PERIOD_A, w.PERIOD_B)


def test_cli_arguments():
    args = w.parser().parse_args(["analyze", "--data", "d"])
    assert (args.a, args.b) == (w.PERIOD_A, w.PERIOD_B)
    args = w.parser().parse_args(["extract", "--archives", "a", "--out", "o"])
    assert args.parts == 8 and args.jobs == 1
    with pytest.raises(SystemExit):
        w.main(["extract", "--archives", "a", "--out", "o", "--jobs", "0"])
    with pytest.raises(SystemExit, match="period A must end"):
        w.main(["analyze", "--data", "d", "--a", "2026-11-01/2026-11-05"])
    with pytest.raises(SystemExit):
        w.parser().parse_args(["analyze", "--data", "d", "--b", "bad"])
