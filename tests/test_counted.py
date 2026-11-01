"""Tests for dcbias.counted and dcbias.counted_bands.

Run from the repository root:  python -m pytest tests/test_counted.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from dcbias import counted as C
from dcbias import generate as G
from dcbias import physics as V


@pytest.fixture(scope="module")
def cfg():
    return V.load_config()


@pytest.fixture(scope="module")
def row(cfg):
    return next(r for r in G.design(cfg) if r["split"] == "test_id" and int(r["M"]) == 16)


def test_oracle_stream_reproduces_mc_curve(cfg, row):
    """One batch on STREAM_ORACLE is the oracle's realisation: same-sample semi BER equals mc_curve."""
    st, dev, _ = V.build(cfg, row)
    b = V.bias_grid(cfg, st["I_rated"], st["C"])[[10, 40, 80]]
    d = V.draws(cfg, V.rng_for(cfg, V.STREAM_ORACLE, row["state_id"]), V.P(cfg, "ofdm.R_oracle"), st["M"])
    want = V.mc_curve(cfg, st, dev, b, d, V.quadrature(cfg))[0]
    err, nbits, semi = C.count_row((cfg, row, b, 1, V.STREAM_ORACLE))
    np.testing.assert_allclose(semi, want, rtol=0, atol=1e-9)
    Yn, _ = V.mc_receive(cfg, st, dev, b, d, V.quadrature(cfg))
    for j in range(len(b)):
        assert err[j] / nbits == pytest.approx(V.counted_ber(Yn[j], d["idx"], st["M"]), abs=1e-15)


def test_batches_pool_errors_and_evm(cfg, row):
    """nb = 2 pools both batches: errors add, semi BER comes from the pooled EVM."""
    st, dev, _ = V.build(cfg, row)
    b = np.array([row["O_L1"]]) if "O_L1" in row else V.bias_grid(cfg, st["I_rated"], st["C"])[[40]]
    quad, bits, M = V.quadrature(cfg), V.gray_bits(st["M"]), st["M"]
    err, e2 = 0, 0.0
    for k in range(2):
        d = V.draws(cfg, V.rng_for(cfg, C.STREAM_COUNTED_SHIFT, row["state_id"], k), V.P(cfg, "ofdm.R_oracle"), M)
        Yn, _ = V.mc_receive(cfg, st, dev, b, d, quad)
        err += np.count_nonzero(bits[V.hard_decide(Yn[0], M)] != bits[d["idx"]])
        e2 = e2 + np.sum(np.abs(Yn[0] - d["S"]) ** 2, axis=0)
    got_err, nbits, semi = C.count_row((cfg, row, b, 2, C.STREAM_COUNTED_SHIFT))
    assert got_err[0] == err
    assert nbits == 2 * V.P(cfg, "ofdm.R_oracle") * V.P(cfg, "ofdm.Nu") * 4
    np.testing.assert_allclose(semi[0], V.log10_ber(2 * V.P(cfg, "ofdm.R_oracle") / e2[None], M)[0], rtol=0, atol=1e-12)


def test_n_batches(cfg):
    bits16 = V.P(cfg, "ofdm.R_oracle") * V.P(cfg, "ofdm.Nu") * 4
    assert C.n_batches(-3.0, 16, cfg) == 1
    assert C.n_batches(-6.0, 16, cfg) == int(np.ceil(C.TARGET_ERRORS / (1e-6 * bits16)))
    assert C.n_batches(-9.0, 16, cfg) == C.MAX_BATCHES


def test_regrets_sign_and_zero_counts():
    err = np.array([[100.0, 1000.0, 0.0]])
    nbits = np.array([1e6])
    semi = np.array([[-4.0, -3.0, -4.5]])
    r = C.regrets(err, nbits, semi)
    np.testing.assert_allclose(r["counted"], [[1.0, np.log10(0.5 / 100)]])
    np.testing.assert_allclose(r["semi"], [[1.0, -0.5]])  # signed: no floor at 0


def test_counted_matches_semi_where_gaussian(cfg, row):
    """At a mildly clipped bias with ~200 expected errors, counted and semi BER agree within 0.1 dec."""
    st, dev, _ = V.build(cfg, row)
    b = V.bias_grid(cfg, st["I_rated"], st["C"])
    d = V.draws(cfg, V.rng_for(cfg, V.STREAM_ORACLE, row["state_id"]), V.P(cfg, "ofdm.R_oracle"), st["M"])
    lb = V.mc_curve(cfg, st, dev, b, d, V.quadrature(cfg))[0]
    j = int(np.argmin(lb))
    nb = C.n_batches(max(lb[j], -6.0), st["M"], cfg)
    err, nbits, semi = C.count_row((cfg, row, b[[j]], nb, C.STREAM_COUNTED_SHIFT))
    assert err[0] >= 50
    assert abs(np.log10(err[0] / nbits) - semi[0]) <= 0.1


def test_oracle_stream_single_batch_only(cfg, row):
    with pytest.raises(AssertionError):
        C.count_row((cfg, row, [1.0], 2, V.STREAM_ORACLE))


def test_summarise_toy():
    """2 units x 2 states, hand-computed: cost = twin - identity, counted - semi for R1, R2 corrected by
    counted - PCHIP on S, R3 sign; the off-S row only enters the full-set mean and the S share."""
    k = "A"
    units = np.array([0, 0, 1, 1])
    tw = {"counted": {k: np.array([0.30, 0.30, 0.30, 0.30]), "b": np.array([0.1, 0.1, 0.1, 0.1])},
          "semi": {k: np.array([0.31, 0.31, 0.31, 0.31]), "b": np.array([0.1, 0.1, 0.1, 0.1])},
          "pchip": {k: np.array([0.32, 0.32, 0.32, 0.32]), "b": np.array([0.1, 0.1, 0.1, 0.1])}}
    idn = {m: {k: np.full(4, 0.10), "b": np.full(4, 0.05)} for m in ("counted", "semi", "pchip")}
    full = {k: np.array([0.22, 0.22, 0.22, 0.22, 0.42]), "b": np.full(5, 0.05)}  # 5th row outside S
    inS = np.array([True, True, True, True, False])
    lbt = np.array([-2.0, -3.5, -5.0, -6.5])
    e = C.summarise(units, tw, idn, full, inS, lbt, np.array([0, 0, 1, 1, 1]), keys=(k,), contrast=(k, "b"))
    c = e["cost"][k]
    assert c["counted"]["point"] == pytest.approx(0.20)
    assert c["counted_minus_semi"]["point"] == pytest.approx(-0.01)
    assert e["R1"][k] == "PASS"
    assert c["pchip_full_set_mean"] == pytest.approx(0.26)
    assert c["corrected_full_set"]["point"] == pytest.approx(0.26 - 0.02)
    assert e["R2"][k] == "above" and e["R2_on_S"][k]["counted"] == "above"
    assert c["share_of_full_set_cost_in_S"] == pytest.approx(0.88 / 1.30)
    assert [v["n"] for v in e["R1_bands"][k].values()] == [1, 1, 1, 1]
    assert e["contrast_twin"]["counted"]["point"] == pytest.approx(0.20)
    assert e["R3"] == "resolved"
    tw["semi"][k] = tw["semi"][k] + 0.05  # counted - semi = -0.06 on every row -> outside +/-0.03
    assert C.summarise(units, tw, idn, full, inS, lbt, np.array([0, 0, 1, 1, 1]), keys=(k,), contrast=(k, "b"))["R1"][k] == "FAIL"


def test_corrected_cost_joint_bootstrap():
    """The R2 CI resamples units jointly for the full-set mean and the S shift, so it is the quantile of
    the summed draws and at least as wide as the shift-only CI; the sensitivities have the stated points."""
    from dcbias import evaluate as E
    rng = np.random.default_rng(5)
    uf = np.repeat(np.arange(8), 10)
    full = rng.normal(0.12, 0.03, uf.size) + 0.02 * uf / 8
    inS = np.tile(np.arange(10) < 7, 8)
    d = rng.normal(-0.01, 0.02, inS.sum())
    tail = np.tile(np.arange(7) >= 5, 8)
    r = C.corrected_cost(uf, full, inS, d, tail)
    fd = E.cluster_boot(uf, full, B=10000, seed=0)[1]
    dd = E.cluster_boot(uf[inS], d, B=10000, seed=0)[1]
    s = fd + dd
    assert r["primary"]["point"] == pytest.approx(full.mean() + d.mean())
    np.testing.assert_allclose(r["primary"]["ci"], np.quantile(s, [0.025, 0.975]))
    assert np.diff(r["primary"]["ci"])[0] >= np.diff(np.quantile(dd, [0.025, 0.975]))[0]
    assert r["no_shift_off_S"]["point"] == pytest.approx(full.mean() + inS.mean() * d.mean())
    assert r["tail_shift_off_S"]["point"] == pytest.approx(full.mean() + inS.mean() * d.mean() + (1 - inS.mean()) * d[tail].mean())


def test_bands_toy():
    """2 units, 3 S rows in two bands + 1 off-S row, hand-computed band means, % of semi, floor, shares."""
    from dcbias import counted_bands as CB
    k, b = "A", "b"
    tw = {"counted": {k: np.array([0.3, 0.5, 0.2]), b: np.zeros(3)}, "semi": {k: np.array([0.2, 0.4, 0.2]), b: np.zeros(3)},
          "pchip": {k: np.array([0.25, 0.45, 0.2]), b: np.zeros(3)}}
    idn = {m: {k: np.full(3, 0.1), b: np.zeros(3)} for m in ("counted", "semi", "pchip")}
    full = {k: np.array([0.15, 0.35, 0.1, 0.4])}
    inS = np.array([True, True, True, False])
    lbt = np.array([-2.0, -2.5, -6.5])
    e = CB.bands(tw, idn, full, inS, lbt, np.array([0, 1, 1]), np.array([0, 1, 1, 0]), keys=(k,), contrast=(k, b))[k]
    top = e["[-3.0,inf)"]
    assert top["n"] == 2 and top["share_of_full_set_pchip_cost"] == pytest.approx(0.5 / 1.0)
    assert top["counted"]["point"] == pytest.approx(0.3) and top["semi"]["point"] == pytest.approx(0.2)
    assert top["counted_minus_semi_pct_of_semi"] == pytest.approx(50.0)
    assert top["contrast_twin"]["counted"]["point"] == pytest.approx(0.4)
    assert e["[-7.0,-6.0)"]["n"] == 1 and "[-6.0,-4.5)" not in e
    assert e["off_S"]["n"] == 1 and e["off_S"]["share_of_full_set_pchip_cost"] == pytest.approx(0.4)
    assert e["off_S"]["pchip"]["point"] == pytest.approx(0.4)
    assert e["S_only_floor"]["point"] == pytest.approx((0.2 + 0.4 + 0.1) / 4)
