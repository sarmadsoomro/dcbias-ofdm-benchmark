"""Unit tests for the simulator, controllers, evaluation and reporting code.

Run from the repository root:  python -m pytest tests/test_physics.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.model_selection import GroupKFold

from dcbias import evaluate as E
from dcbias import generate as G
from dcbias import physics as V
from dcbias import zhang_check as Z


@pytest.fixture(scope="module")
def cfg():
    return V.load_config()


@pytest.fixture(scope="module")
def quad(cfg):
    return V.quadrature(cfg)


def dev_of(cfg, I_rated=1.0, **kw):
    return V.nominal_device(cfg, I_rated) | kw


def state(I_rated=1.0, sigma=0.12, T_hs=310.0, snr_db=20.0, M=16, C=1.0):
    return {"I_rated": I_rated, "sigma": sigma, "T_hs": T_hs, "snr_db": snr_db, "M": M, "C": C}


# ---- 1. Hermitian symmetry
def test_hermitian_symbol(cfg):
    N, Nu = V.P(cfg, "ofdm.N"), V.P(cfg, "ofdm.Nu")
    assert Nu == N // 2 - 1 == 255
    _, S, s = V.ofdm_symbols(np.random.default_rng(0), 64, 16, N, Nu)
    X = np.fft.fft(s, axis=1, norm="ortho")
    assert np.max(np.abs(X[:, 0])) < 1e-12 and np.max(np.abs(X[:, N // 2])) < 1e-12
    np.testing.assert_allclose(X[:, 1 : Nu + 1] / S, np.sqrt(N / (2 * Nu)), rtol=1e-10)
    assert abs(np.mean(s**2) - 1) < 0.02  # analytic normalisation, E[s^2] = 1


# ---- 2. EVM floor: no clipping, linear LED, 120 dB -> EVM < -60 dB
def test_evm_floor(cfg, quad):
    c = V.load_config(overrides={"noise.rho": 0.0})
    st = state(sigma=0.05, snr_db=120.0, C=10.0, M=16)
    dev = dev_of(c, kt=0.0, delta=0.0, Rth=0.1)
    d = V.draws(c, np.random.default_rng(1), 32, 16)
    assert V.evm_db(c, st, dev, 0.5, d, V.quadrature(c))[0] < -60


# ---- 3. symmetry theorem: kappa = rho = delta = 0, flat channel -> L1 = (i_min + C)/2 +/- 1 step
@pytest.mark.parametrize("snr", [10.0, 20.0, 30.0])
@pytest.mark.parametrize("i_min", [0.0, 0.10])
def test_symmetry_theorem(snr, i_min):
    c = V.load_config(overrides={"noise.rho": 0.0, "led.f3dB_over_B": None, "led.i_min_frac": i_min})
    q = V.quadrature(c)
    st = state(sigma=0.15, snr_db=snr, C=0.8)
    dev = dev_of(c, kt=0.0, delta=0.0)
    b = V.bias_grid(c, 1.0, st["C"])
    assert b[0] == pytest.approx(i_min)
    step, mid = V.P(c, "ofdm.grid_step_frac"), (i_min + 0.8) / 2
    an = b[np.argmin(V.analytic_curve(c, st, dev, b, q)["lb"])]
    d = V.draws(c, np.random.default_rng(2), 256, 16)
    mc = b[np.argmin(V.mc_curve(c, st, dev, b, d, q)[0])]
    assert abs(an - mid) <= step + 1e-12
    assert abs(mc - mid) <= step + 1e-12


# ---- 4. droop: halving eta in the noise-limited regime costs 6.02 dB of SINR
def test_droop_halving(cfg, quad):
    c = V.load_config(overrides={"noise.rho": 0.0})
    q = V.quadrature(c)
    st = state(sigma=0.05, snr_db=0.0, C=1.0)
    b = np.array([0.5])
    full = V.analytic_curve(c, st, dev_of(c, delta=0.0, kt=0.0), b, q)["sinr"]
    half = V.analytic_curve(c, st, dev_of(c, delta=0.0, kt=0.0, eta0=0.5 * V.P(c, "led.eta0_nom")), b, q)["sinr"]
    db = 10 * np.log10(full / half)
    assert np.all(np.abs(db - 6.0) <= 0.1)


# ---- 5. foldback: exact values at the knee, the slope midpoint and the floor
def test_foldback(cfg):
    knee, s, fmin = V.P(cfg, "driver.knee_K"), V.P(cfg, "driver.slope_per_K"), V.P(cfg, "driver.f_min")
    assert V.foldback_ceiling(cfg, 0.8, knee - 10) == 0.8
    assert V.foldback_ceiling(cfg, 0.8, knee) == 0.8
    mid = knee + 0.5 * (1 - fmin) / s
    assert V.foldback_ceiling(cfg, 0.8, mid) == pytest.approx(0.8 * (1 - 0.5 * (1 - fmin)), abs=1e-12)
    assert V.foldback_ceiling(cfg, 0.8, knee + 10 / s) == pytest.approx(0.8 * fmin, abs=1e-12)


# ---- 6. self-heating: exact fixed point for nominal parameters; runaway raises
def test_self_heating(cfg, quad):
    dev = dev_of(cfg)
    b = np.linspace(0.05, 0.95, 7)
    th = V.thermal(b, 0.12, 1.0, 330.0, dev, quad)
    T0 = V.T0
    Ei = th["i"] @ quad[1]
    Eq = Ei - dev["kt"] * (th["i"] ** 2 @ quad[1]) / dev["I_rated"]
    eta = dev["eta0"] * (1 - dev["delta"] * (th["Tj"] - T0))
    P_elec = dev["V0"] * Ei + dev["Rs"] * (th["i"] ** 2 @ quad[1]) + dev["KV"] * (th["Tj"] - T0) * Ei
    np.testing.assert_allclose(th["Tj"], 330.0 + dev["Rth"] * (P_elec - eta * Eq), atol=1e-9)
    assert np.all(th["Tj"] > 330.0)
    with pytest.raises(V.ThermalRunaway):
        V.thermal(b, 0.12, 1.0, 330.0, dev_of(cfg, KV=1.0, Rth=100.0), quad)


# ---- 7. CRN: identical draws at every bias; same seed -> bit-identical curve; chunking irrelevant
def test_crn(cfg, quad):
    st, dev = state(), dev_of(cfg)
    b = V.bias_grid(cfg, 1.0, 1.0)[10:60]
    d = V.draws(cfg, np.random.default_rng(3), 32, 16)
    full = V.mc_curve(cfg, st, dev, b, d, quad)[0]
    parts = np.concatenate([V.mc_curve(cfg, st, dev, b[j : j + 1], d, quad)[0] for j in range(len(b))])
    np.testing.assert_array_equal(full, parts)
    d2 = V.draws(cfg, np.random.default_rng(3), 32, 16)
    np.testing.assert_array_equal(full, V.mc_curve(cfg, st, dev, b, d2, quad)[0])


# ---- 8. MC vs analytic labels within 2 grid steps at R = 256 (nominal quadratic and Tsonev cubic)
@pytest.mark.parametrize("model", ["quadratic", "tsonev_cubic"])
@pytest.mark.parametrize("k", range(5))
def test_mc_vs_analytic(k, model):
    cfg = V.load_config(overrides={"led.model": model})
    quad = V.quadrature(cfg)
    rng = np.random.default_rng(10 + k)
    Ir = rng.uniform(0.7, 1.0)
    T = rng.uniform(298, 400)
    st = state(I_rated=Ir, sigma=rng.uniform(0.08, 0.2) * Ir, T_hs=T, snr_db=rng.uniform(5, 35),
               M=int(rng.choice([4, 16, 64])), C=float(V.foldback_ceiling(cfg, Ir, T)))
    dev = dev_of(cfg, Ir)
    b = V.bias_grid(cfg, Ir, st["C"])
    d = V.draws(cfg, rng, 256, st["M"])
    ja = np.argmin(V.analytic_curve(cfg, st, dev, b, quad)["lb"])
    jm = np.argmin(V.mc_curve(cfg, st, dev, b, d, quad)[0])
    assert abs(int(ja) - int(jm)) <= 2


# ---- 9. boundary-label share <= 0.5 % on a 200-state smoke run (label curves only)
def test_boundary_share(cfg, quad):
    rows = [r for r in G.design(cfg) if r["split"] == "train"][:200]
    hits = 0
    for r in rows:
        st, dev, _ = V.build(cfg, r)
        b = V.bias_grid(cfg, st["I_rated"], st["C"])
        d = V.draws(cfg, V.rng_for(cfg, V.STREAM_LABEL, r["state_id"]), 64, st["M"])
        hits += V.labels_from_curve(b, V.mc_curve(cfg, st, dev, b, d, quad)[0], cfg)["boundary"]
    assert hits / len(rows) <= 0.005


# ---- 10. unit sets disjoint: train vs test, and across GroupKFold folds
def test_unit_disjoint(cfg):
    rows = pd.DataFrame(G.design(cfg))
    units = {s: set(g["unit_id"]) for s, g in rows.groupby("split")}
    splits = list(units)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            assert not units[a] & units[b]
    tr = rows[rows["split"] == "train"]
    for f_tr, f_va in GroupKFold(5).split(tr, groups=tr["unit_id"]):
        assert not set(tr["unit_id"].iloc[f_tr]) & set(tr["unit_id"].iloc[f_va])


# ---- 11. cluster bootstrap resamples units, not rows
def test_cluster_bootstrap_units():
    units = np.repeat([0, 1], 100)
    vals = np.repeat([0.0, 1.0], 100)
    _, draws = E.cluster_boot(units, vals, B=500)
    assert set(np.round(draws, 12)) <= {0.0, 0.5, 1.0}


# ---- 12. regret(b_o) == 0 and regret >= -1e-9 everywhere
def test_regret(cfg, quad):
    st, dev = state(), dev_of(cfg)
    b = V.bias_grid(cfg, 1.0, 1.0)
    lb = V.analytic_curve(cfg, st, dev, b, quad)["lb"]
    cur = [E.Curve(b, lb)]
    assert E.regret([b[np.argmin(lb)]], cur, 1e-9)[0] == 0.0
    xs = np.random.default_rng(4).uniform(-0.2, 1.2, 200)
    assert np.all(E.regret(xs, cur * len(xs), 1e-9) >= -1e-9)


# ---- 13. B2: Zhang's stated objective is minimised at the window midpoint; Table I not reproducible
def test_zhang_midpoint():
    for r in Z.rows():
        assert r["argmin_clip_noise"] == pytest.approx(r["midpoint"], abs=1e-4)
        assert not r["eq15_has_root"]
    X = pd.DataFrame({"C_hat": [0.4, 0.9], "i_min": [0.1, 0.0]})
    m = E.nonml_methods(V.load_config(), lambda idx: [])
    np.testing.assert_array_equal(m["B2_zhang_midpoint"][0]().predict(X), m["B3_centre"][0]().predict(X))
    np.testing.assert_allclose(m["B3_centre"][0]().predict(X), [0.25, 0.45])  # midpoint of [i_min, C_hat]


# ---- 14. determinism: the first 3 rows regenerate bit-exactly
def test_determinism():
    c = V.load_config(overrides={"ofdm.R_label": 32, "ofdm.R_probe": 4})
    rows = G.design(c)[:3]
    a = [G.simulate((c, r)) for r in rows]
    b = [G.simulate((c, r)) for r in rows]
    for (ra, ca), (rb, cb) in zip(a, b):
        assert ra == rb
        for k in ca:
            np.testing.assert_array_equal(ca[k], cb[k])


# ---- 15. LED monotonicity assertion (both models)
def test_monotone_assertion(cfg):
    V.check_monotone(dev_of(cfg, kt=0.49))
    with pytest.raises(ValueError):
        V.check_monotone(dev_of(cfg, kt=0.5))
    with pytest.raises(ValueError):
        V.draw_unit(V.load_config(overrides={"led.kappa_tilde": 0.45}), 0)
    cub = V.nominal_device(V.load_config(overrides={"led.model": "tsonev_cubic"}), 1.0)
    V.check_monotone(cub)
    with pytest.raises(ValueError):
        V.check_monotone(cub | {"cubic": np.array([0.0, -1.0, 1.0, 0.0])})  # slope 1 - 2x < 0 above 0.5
    with pytest.raises(ValueError):
        V.nominal_device(V.load_config(overrides={"led.model": "abc"}), 1.0)


# ---- helpers and regression tests
def test_sd_ddof1():
    assert E.sd1([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.13809, abs=1e-5)


def test_y_scaler_train_only():
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    y = 3 + 2 * rng.normal(size=200)
    m = E.ML("mlp", ["a", "b"], {"alpha": 1e-3}).fit(X.iloc[:150], y[:150])
    assert m.m.transformer_.mean_[0] == pytest.approx(y[:150].mean(), abs=1e-12)
    assert m.m.transformer_.scale_[0] == pytest.approx(y[:150].std(), abs=1e-12)


def test_no_silent_fallbacks(cfg):
    b = np.linspace(0, 1, 5)
    with pytest.raises(FloatingPointError):
        V.labels_from_curve(b, np.array([-1, -2, np.nan, -1, 0.0]), cfg)
    X = pd.DataFrame({"x": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0], "z": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]})
    lut = E.LUT(["x", "z"], 2).fit(X, np.arange(6.0))
    assert lut.n_empty == 2 and lut.n_cells == 4
    lut.predict(pd.DataFrame({"x": [0.0, 1.0], "z": [1.0, 1.0]}))
    np.testing.assert_array_equal(lut.last_filled, [True, False])  # per-row flag for filled cells
    with pytest.raises(ValueError):
        V.bias_grid(cfg, 1.0, 0.001)


def test_quadrature_clipped_mean(cfg, quad):
    b, sig, C = np.array([0.1, 0.5, 0.9]), 0.2, 1.0
    dev = dev_of(cfg)
    lo = dev["i_min"]
    assert lo == pytest.approx(0.1)  # the clip is applied in thermal(), not only in the grid
    th = V.thermal(b, sig, C, 300.0, dev, quad)
    a, z = (lo - b) / sig, (C - b) / sig
    exact = lo * norm.cdf(a) + b * (norm.cdf(z) - norm.cdf(a)) + sig * (norm.pdf(a) - norm.pdf(z)) + C * norm.sf(z)
    np.testing.assert_allclose(th["i"] @ quad[1], exact, rtol=1e-5)  # uniform nodes: O(h^2) error at the clip kinks


def test_exact_ber():
    from scipy.stats import norm
    g = 10 ** (np.array([0.0, 6.0, 12.0, 18.0]) / 10)
    np.testing.assert_allclose(10 ** V.log10_ber(g[:, None], 4), norm.sf(np.sqrt(g)), rtol=1e-12)  # QPSK = Q(sqrt g)
    rng = np.random.default_rng(7)
    for M in (16, 64):  # exact formula vs counted bits on pure AWGN, within 4 Poisson SDs
        idx = rng.integers(0, M, (2000, 255))
        S = V.constellation(M)[idx]
        for gdb in (6.0, 12.0, 18.0):
            gg = 10 ** (gdb / 10)
            n = (rng.normal(size=S.shape) + 1j * rng.normal(size=S.shape)) * np.sqrt(1 / (2 * gg))
            f = 10 ** V.log10_ber(np.full((1, 255), gg), M)[0]
            nb = idx.size * np.log2(M)
            assert abs(V.counted_ber(S + n, idx, M) - f) * nb <= 4 * np.sqrt(f * nb * np.log2(M))


def test_gray_and_counted():
    for M in (4, 16, 64):
        bits, m = V.gray_bits(M), int(np.sqrt(M))
        for ix in range(m - 1):  # adjacent levels on each axis differ in exactly one bit
            assert np.sum(bits[ix * m] != bits[(ix + 1) * m]) == 1
            assert np.sum(bits[ix] != bits[ix + 1]) == 1
        idx = np.random.default_rng(6).integers(0, M, (8, 255))
        assert V.counted_ber(V.constellation(M)[idx], idx, M) == 0.0


def test_lut_discrete_column():  # M gets one cell per value, centred on the value
    X = pd.DataFrame({"M": [4, 16, 64] * 4, "x": np.linspace(0, 1, 12)})
    y = np.array([1.0, 2.0, 3.0] * 4)
    np.testing.assert_allclose(E.LUT(["M"], 6).fit(X, y).predict(pd.DataFrame({"M": [4, 16, 64]})), [1, 2, 3])


def test_decision_rule():
    m = 0.10
    assert E.decide(0.2, [0.12, 0.3], True, True, True, m).startswith("(a)")
    assert E.decide(0.2, [0.12, 0.3], True, False, True, m).startswith("(a')")
    assert E.decide(0.2, [0.12, 0.3], True, True, False, m).startswith("UNMAPPED")
    assert E.decide(0.0, [-0.05, 0.05], False, False, True, m).startswith("(b) equivalence")
    assert E.decide(-0.3, [-0.4, -0.2], False, False, True, m).startswith("(b) negative")
    assert E.decide(0.05, [-0.02, 0.15], False, False, True, m).startswith("(b) inconclusive")


# ---- Regression tests for mutations of the data-generation and evaluation rules

def test_c2_intersection_union():  # (i) C2 needs every OOD CI lower bound > 0; C1 needs both conditions
    ok = {"point": 0.2, "ci": [0.05, 0.3]}
    res = {"test_id": ok, "ood_T": ok, "ood_SNR": ok, "ood_sigma": ok}
    assert E.c1_c2(res, 0.10) == (True, True)
    assert E.c1_c2(res | {"ood_SNR": {"point": 0.2, "ci": [-0.01, 0.4]}}, 0.10) == (True, False)
    assert E.c1_c2({k: v for k, v in res.items() if k != "ood_T"}, 0.10) == (True, False)


def test_decide_overlap():  # (ii) equivalence takes precedence; direction reported when the CI excludes 0
    m = 0.10
    neg = E.decide(-0.05, [-0.08, -0.01], False, False, True, m)
    pos = E.decide(0.05, [0.01, 0.08], False, False, True, m)
    assert neg.startswith("(b) equivalence") and neg.endswith("direction: non-ML lower regret")
    assert pos.startswith("(b) equivalence") and pos.endswith("direction: ML lower regret")
    assert E.decide(-0.3, [-0.4, -0.2], False, False, True, m) == "(b) negative result: non-ML better; direction: non-ML lower regret"
    assert "direction" not in E.decide(0.0, [-0.05, 0.05], False, False, True, m)


def test_c1_requires_ci_lower_bound():  # (iii) point >= margin but CI_lo <= 0 -> C1 false
    res = {"test_id": {"point": 0.15, "ci": [-0.02, 0.32]}}
    assert E.c1_c2(res, 0.10)[0] is False
    res = {"test_id": {"point": 0.15, "ci": [0.01, 0.32]}}
    assert E.c1_c2(res, 0.10)[0] is True
    res = {"test_id": {"point": 0.09, "ci": [0.01, 0.2]}}
    assert E.c1_c2(res, 0.10)[0] is False


def test_paired_cluster_bootstrap():  # (iv) a shared unit effect cancels only when a and b are paired
    rng = np.random.default_rng(8)
    units = np.repeat(np.arange(40), 25)
    shared = rng.normal(0, 1.0, 40)[units]
    a = shared + 0.1 + rng.normal(0, 0.05, units.size)
    b = shared + rng.normal(0, 0.05, units.size)
    paired, _ = E.cluster_boot(units, a, b, B=2000)
    perm = b[np.argsort(rng.permutation(40)[units] * 1000 + np.arange(units.size) % 25)]
    unpaired, _ = E.cluster_boot(units, a, perm, B=2000)
    assert np.diff(paired["ci"])[0] < 0.2 * np.diff(unpaired["ci"])[0]
    assert paired["ci"][0] > 0


def _obs_table(cfg, n=6):
    rows = [r for r in G.design(V.load_config(overrides={"ofdm.R_label": 16, "ofdm.R_probe": 4})) if r["split"] == "test_id"][:n]
    c = V.load_config(overrides={"ofdm.R_label": 16, "ofdm.R_probe": 4, "ofdm.R_oracle": 16})
    return pd.DataFrame([G.simulate((c, r))[0] for r in rows])


def test_observe_lambda(cfg):  # (v) lambda = 0 is exact; Tj_Rth_hat is not a noise-free latent channel at lambda_T = 1
    df = _obs_table(cfg)
    o = V.observe(cfg, df, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(o["T_ntc_hat"], df["T_ntc"], atol=1e-12)
    np.testing.assert_allclose(o["Tj_Vf_hat"], df["Tj_p"], atol=1e-9)
    np.testing.assert_allclose(o["Tj_Rth_hat"], df["Tj_p"], atol=1e-9)
    np.testing.assert_allclose(o["SNR_hat"], df["SNR_ref_dB"], atol=1e-12)
    Rth = V.P(cfg, "thermal.Rth")
    resid = [V.observe(cfg, df, 1.0, 1.0, 1.0, rep=k)["Tj_Rth_hat"].to_numpy() - V.observe(cfg, df, 1.0, 1.0, 1.0, rep=k)["T_ntc_hat"].to_numpy()
             - Rth * df["P_diss_p"].to_numpy() for k in range(2)]
    assert np.all(np.abs(resid[0] - resid[1]) > 1e-6)  # sampling noise, not a function of latents


def test_noise_calibration(cfg):  # (vi) linear LED, unclipped: SINR = SNR_ref N/(2 Nu) +/- 0.05 dB
    c = V.load_config(overrides={"noise.rho": 0.0, "led.f3dB_over_B": None, "led.i_min_frac": 0.0})
    q = V.quadrature(c)
    N, Nu = V.P(c, "ofdm.N"), V.P(c, "ofdm.Nu")
    for snr in (10.0, 25.0):
        st = state(sigma=0.05, snr_db=snr, C=1.0)
        sinr = V.analytic_curve(c, st, dev_of(c, kt=0.0, delta=0.0), np.array([0.5]), q)["sinr"]
        assert np.all(np.abs(10 * np.log10(sinr) - (snr + 10 * np.log10(N / (2 * Nu)))) <= 0.05)


def test_shot_noise_proportional():  # (vii) rho = 1: noise variance proportional to E[p]
    c = V.load_config(overrides={"noise.rho": 1.0})
    Ep = np.array([0.1, 0.2, 0.4, 0.8])
    v = V.noise_var(c, state(), Ep)
    np.testing.assert_allclose(v / Ep, v[0] / Ep[0], rtol=1e-12)


def test_cv_regret_unit_disjoint():  # (viii) through cv_regret itself, with a recording factory
    rng = np.random.default_rng(9)
    X = pd.DataFrame({"a": rng.normal(size=60)})
    groups = np.repeat(np.arange(12), 5)
    seen = []

    class Rec:
        def fit(self, X_, y_):
            self.units = set(groups[X_.index])
            return self

        def predict(self, X_):
            seen.append((self.units, set(groups[X_.index])))
            return np.zeros(len(X_))

    b = np.linspace(0, 1, 11)
    cur = [E.Curve(b, (b - 0.5) ** 2)] * 60
    E.cv_regret(Rec, X, np.zeros(60), groups, lambda idx: [cur[i] for i in idx], 4, 1e-9)
    assert len(seen) == 4 and all(not (tr & va) for tr, va in seen)
    assert set().union(*(va for _, va in seen)) == set(range(12))


def test_streams_differ(cfg):  # (ix) label, oracle and probe draws differ for one state
    d = [V.draws(cfg, V.rng_for(cfg, s, 7), 4, 16)["s"] for s in (V.STREAM_LABEL, V.STREAM_ORACLE, V.STREAM_PROBE)]
    assert not np.allclose(d[0], d[1]) and not np.allclose(d[0], d[2]) and not np.allclose(d[1], d[2])


@pytest.mark.parametrize("C", [0.8, 0.80013, 0.6337])
@pytest.mark.parametrize("j", [0, 30, -1])
def test_refine_spacing(cfg, C, j):  # (x) no near-duplicate oracle nodes
    b = V.bias_grid(cfg, 1.0, C)
    bo = V.refine_grid(cfg, b, j % len(b), 1.0, C)
    fine = V.P(cfg, "ofdm.refine_step_frac")
    assert np.all(np.diff(bo) >= 0.5 * fine - 1e-15)
    assert np.all(np.isin(b, bo)) and bo[0] == b[0] and bo[-1] == C
    assert len(bo) > len(b)


def test_b1b_branches():  # (xi) inside the window: i_min + k sigma; window too narrow: midpoint
    X = pd.DataFrame({"sigma_I": [0.1, 0.2], "C_hat": [1.0, 0.6], "i_min": [0.1, 0.1]})
    np.testing.assert_allclose(E.b1b(X, 2.0), [0.3, 0.35])


def test_regret_floor():  # (xii) below the floor, regret is 0; above it, the decade difference
    b = np.linspace(0, 1, 11)
    cur = [E.Curve(b, -12 + 20 * (b - 0.5) ** 2)]  # min 1e-12, crosses 1e-9 at |b - 0.5| = 0.387
    assert E.regret([0.6], cur, 1e-9)[0] == 0.0
    assert E.regret([1.0], cur, 1e-9)[0] == pytest.approx(-12 + 5 + 9)
    assert E.regret([1.0], cur, 1e-12)[0] == pytest.approx(5.0)


def test_cubic_led_in_thermal():  # one LED function: thermal() E[p] uses the cubic when selected
    c = V.load_config(overrides={"led.model": "tsonev_cubic"})
    q = V.quadrature(c)
    dev = V.nominal_device(c, 1.0)
    th = V.thermal(np.array([0.4]), 0.1, 1.0, 300.0, dev, q)
    i = np.clip(0.4 + 0.1 * q[0], dev["i_min"], 1.0)
    np.testing.assert_allclose(th["Ep"], th["eta"] * (np.polyval(dev["cubic"], i) @ q[1]), rtol=1e-12)
    np.testing.assert_allclose(np.polyval(np.polyder(dev["cubic"]), 0.5), 1 - V.P(c, "led.kappa_tilde"), rtol=1e-12)


def test_design_strata(cfg):  # Option B: half of each non-OOD-T unit's states in the foldback stratum
    rows = pd.DataFrame(G.design(cfg))
    s_lo, s_hi = V.P(cfg, "id_region.stratum_T_hs_K")
    idt = rows[rows["split"].isin(["train", "test_id"])]
    assert idt.groupby("unit_id")["stratum"].apply(lambda g: (g == "foldback").mean()).eq(0.5).all()
    f = idt[idt["stratum"] == "foldback"]["T_hs"]
    assert f.between(s_lo, s_hi).all() and (idt[idt["stratum"] == "base"]["T_hs"] < s_lo).all()
    assert (rows[rows["split"] == "ood_T"]["T_hs"] > s_hi).all()


# ---- Regression tests for mutations of the sensing and B7 rules

@pytest.fixture(scope="module")
def obs_tab(cfg):
    return _obs_table(cfg)


def test_tj_rth_exact_at_lambda1(cfg, obs_tab):  # (v') latent-free residual, exactly sigma_P noise (m11)
    df = obs_tab
    o = V.observe(cfg, df, 1.0, 1.0, 1.0)
    nw = V.P(cfg, "sensing.window_symbols") * V.P(cfg, "ofdm.N")
    z4 = np.array([V.rng_for(cfg, V.STREAM_SENSE, s, 0).standard_normal(5 + nw)[4] for s in df["state_id"]])
    p_nom = [V.p_opt_nominal(cfg, r["I_rated"], r["sigma_I"], r["b_p"], r["C_hat"], r["T_ntc_hat"]) for r in o.to_dict("records")]
    np.testing.assert_allclose(o["P_opt_nom_est"], p_nom, rtol=1e-12)
    Rth = V.P(cfg, "thermal.Rth")
    resid = o["Tj_Rth_hat"] - o["T_ntc_hat"] - Rth * (df["P_elec_p"] - o["P_opt_nom_est"])
    np.testing.assert_allclose(resid, Rth * df["P_elec_p"] * V.P(cfg, "sensing.sigma_P_rel") * z4, atol=1e-9)


def test_imin_window_and_waveform(cfg, obs_tab):  # observe i_min (m14); waveform clipped at i_min, not 0
    o = V.observe(cfg, obs_tab, 0.0, 0.0, 0.0)
    assert V.P(cfg, "led.i_min_frac") == 0.10
    np.testing.assert_allclose(o["i_min"], 0.10 * obs_tab["I_rated"], rtol=1e-12)
    assert np.all(o["w_min"] >= o["i_min"] - 1e-12)
    assert np.any(np.abs(o["w_min"] - o["i_min"]) < 1e-12)  # power: at least one row hits the clip


def test_bp_floor_and_tj_ok(cfg):  # b_p >= i_min (m13); tj_ok judged at Tj_L1, not at the probe
    c = V.load_config(overrides={"ofdm.R_label": 16, "ofdm.R_probe": 4, "states.probe_frac": [0.01, 0.02]})
    row = [r for r in G.design(c) if r["split"] == "train"][0]
    out, _ = G.simulate((c, row))
    assert out["b_p"] == pytest.approx(V.P(c, "led.i_min_frac") * out["I_rated"])
    thr = (out["Tj_L1"] + out["Tj_p"]) / 2
    assert abs(out["Tj_L1"] - out["Tj_p"]) > 1e-6
    out2, _ = G.simulate((V.load_config(overrides={"ofdm.R_label": 16, "ofdm.R_probe": 4, "states.probe_frac": [0.01, 0.02],
                                                   "led.Tj_max_K": thr}), row))
    assert out2["tj_ok"] == (out2["Tj_L1"] <= thr)


def test_load_drops_tj_rows(cfg, tmp_path):  # load excludes tj_ok = False rows
    df = pd.DataFrame({"state_id": [0, 1, 2], "tj_ok": [True, False, True]})
    np.savez(tmp_path / "curves.npz", state_id=df["state_id"].to_numpy(), oracle_b=np.zeros((3, 2)))
    df.to_csv(tmp_path / "states.csv", index=False)
    (tmp_path / "provenance.json").write_text('{"config": {}}')
    d, cv, _ = E.load(tmp_path)
    assert d["state_id"].tolist() == [0, 2] and len(cv["oracle_b"]) == 2


def test_b7_bias_shifts_posterior(cfg, obs_tab):  # B7 subtracts the mean residual
    o = V.observe(cfg, obs_tab, 1.0, 1.0, 1.0)
    sig = {k: 1.0 for k in E.B7_OBS}
    zero = {k: 0.0 for k in E.B7_OBS}
    rows = o.to_dict("records")
    a = [E._b7_row((cfg, r, sig, zero, E.b7_latents(cfg))) for r in rows]
    b = [E._b7_row((cfg, r, sig, zero | {"Prx_hat": 3.0, "EVM_p": -3.0}, E.b7_latents(cfg))) for r in rows]
    assert a != b


def test_b7_spec_calibration(cfg, obs_tab):  # n = 0: SDs from sensor specs, bias 0, no training rows read
    sV = V.P(cfg, "sensing.sigma_V") / abs(V.P(cfg, "sensing.K_V_nom"))
    sd = E.spec_sig(cfg, {"Tj_Vf_hat": V.T0, "P_elec": 2.0})
    ntc2 = V.P(cfg, "sensing.sigma_T_ntc") ** 2 + V.P(cfg, "unit_spread.ntc_offset_sd") ** 2
    assert sd["EVM_p"] == pytest.approx(np.hypot(E.EVM_PROBE_SD_DB, V.P(cfg, "sensing.snr_sd_dB")))
    assert sd["Tj_Vf_hat"] == pytest.approx(np.sqrt(sV**2 + ntc2)) and sd["Prx_hat"] == V.P(cfg, "sensing.prx_sd_dB")
    assert sd["Tj_Rth_hat"] == pytest.approx(2.0 * V.P(cfg, "thermal.Rth") * V.P(cfg, "sensing.sigma_P_rel"))
    assert E.spec_sig(cfg, {"Tj_Vf_hat": V.T0 + 100, "P_elec": 2.0})["Tj_Vf_hat"] > np.sqrt(sV**2 + ntc2)
    m = E.B7(cfg, calib="spec").fit(None)
    assert m.sig is None and set(m.bias.values()) == {0.0}
    o = V.observe(cfg, obs_tab, 1.0, 1.0, 1.0)
    bh = m.predict(o)
    for b, r in zip(bh, o.to_dict("records")):
        assert np.any(np.isclose(V.bias_grid(cfg, r["I_rated"], r["C_hat"]), b, rtol=0, atol=1e-12))
    with pytest.raises(ValueError):
        E.B7(cfg, ceiling="wave", calib="spec")


def test_b7_spec_z(cfg):  # spec SDs describe the residuals at the true latents
    z, zm = E.spec_z(cfg, V.observe(cfg, _obs_table(cfg, 30), 1.0, 1.0, 1.0))  # 0.068 dB EVM SD gave z ~ 6
    assert set(z) == set(zm) == set(E.B7_OBS)
    for k in ("EVM_p", "Prx_hat", "Tj_Vf_hat"):
        assert 0.3 < z[k] <= E.B7CAL_ZMAX, (k, z[k])


def test_model_grid_m_axis(cfg):  # ModelGrid tabulates over distinct M
    c = V.load_config(overrides={"evaluation.model_grid": {"I_rated_A": 2, "sigma_frac": 2, "T_hs_K": 2, "SNR_ref_dB": 2}})
    g = E.ModelGrid(c).fit()
    np.testing.assert_array_equal(g.axes[-1], np.asarray(V.P(c, "states.M"), float))
    assert len(set(g.axes[-1])) == 3


def test_score_test_strata(cfg):  # per-stratum subsets carry their own rows
    b = np.linspace(0.1, 1.0, 10)
    lb = -3 + (b - 0.5) ** 2
    obs = pd.DataFrame({"split": ["test_id"] * 4, "stratum": ["base", "base", "foldback", "foldback"], "n_local_min": [1] * 4,
                        "O_L1": [0.5] * 4, "i_min": [0.1] * 4})
    cv = {"oracle_b": np.tile(b, (4, 1)), "oracle_lb": np.tile(lb, (4, 1))}
    out, _ = E.score_test(obs, cv, {"m": np.array([0.5, 0.6, 0.05, 0.7])}, cfg)
    assert out["test_id/base"]["m"]["n"] == 2 and out["test_id/foldback"]["m"]["n"] == 2
    assert out["test_id/foldback"]["m"]["below_i_min_share"] == 0.5


def test_binding_one_step(cfg):  # exactly one grid step is not binding (m10)
    step = V.P(cfg, "ofdm.grid_step_frac") * 2.0
    assert not G.binding(cfg, 1.0, 1.0 + step, 2.0)
    assert G.binding(cfg, 1.0, 1.0 + 1.01 * step, 2.0)


# ---- Waveform ceiling, back-off and stress columns

def test_c3_boot_degenerate():  # zero paired differences -> degenerate, no CI or p
    u = np.repeat(np.arange(5), 4)
    z = np.zeros(20, bool)
    d = E.c3_boot(u, z, z, 200)
    assert d["degenerate"] and d["ci"] is None and d["p_le0"] is None and d["violations_ml"] == 0
    same = z.copy()
    same[[0, 7]] = True  # both arms violate on the same rows: still no paired difference
    assert E.c3_boot(u, same, same, 200)["degenerate"]
    v = z.copy()
    v[3] = True
    nd = E.c3_boot(u, v, z, 200)
    assert not nd["degenerate"] and nd["ci"][1] > 0 and nd["violations_ml"] == 1


def test_wave_ceiling_branches(cfg):  # clipped top -> w_max - off; unclipped -> max(C_hat, w_max); clamp
    q = {16: E.unclipped_max_q(cfg, 16)}
    assert 2.0 < q[16] < 4.5  # max of 4 x 512 near-Gaussian samples
    top = 0.3 + 0.1 * q[16]
    X = pd.DataFrame({"M": [16] * 5, "b_p": [0.3] * 5, "sigma_I": [0.1] * 5, "I_rated": [1.0] * 5, "i_min": [0.1] * 5,
                      "w_max": [0.45, top + 0.05, top + 0.05, 0.11, 0.55],
                      "C_hat": [0.46, 0.9, 0.5, 0.115, 0.95]})
    C = E.wave_ceiling(cfg, X, q, k=0.02, off=0.001)
    lo = 0.1 + 2 * V.P(cfg, "ofdm.grid_step_frac")
    # row 5: w_max below q01 but far below C_hat -> an unclipped window, C_hat stands
    np.testing.assert_allclose(C, [0.449, 0.9, top + 0.05, lo, 0.95])
    with pytest.raises(ValueError):
        E.wave_clipped(X.assign(M=4), q, 0.02)  # no quantile for M = 4


def test_backoff_beta():  # feasible rows only; never negative; no silent 0
    short = np.array([0.01, 0.02, 0.03, 0.5, 0.6])
    feas = np.array([True, True, True, False, False])
    assert E.backoff_beta(short, feas, 1.0) == pytest.approx(0.03)
    assert E.backoff_beta(-short, feas, 0.95) == 0.0
    with pytest.raises(ValueError):
        E.backoff_beta(np.array([np.nan, 0.1]), np.array([True, True]), 0.95)
    with pytest.raises(ValueError):
        E.backoff_beta(short, np.zeros(5, bool), 0.95)


def test_pick_best_pool():  # selection pool: only selection keys; secondary B7w cannot enter
    sel = {"B7_bayes": 0.2, "B5_lut2d_Ir_T": 0.3, "ML_mlp_state_obs": 0.25, "ML_hgb_state": 0.1}
    assert E.pick_best(sel) == ("B7_bayes", "ML_mlp_state_obs")


def test_fec_relevant_pairing(cfg):  # secondary Delta pairs rows and units on the relevant subset
    b = np.linspace(0.1, 1.0, 10)
    good, bad = -3 + (b - 0.5) ** 2, -1 + (b - 0.5) ** 2
    n = 8
    obs = pd.DataFrame({"split": ["test_id"] * n, "stratum": ["base"] * n, "n_local_min": [1] * n, "O_L1": [0.5] * n,
                        "i_min": [0.1] * n, "unit_id": [0, 0, 1, 1, 2, 2, 3, 3]})
    cv = {"oracle_b": np.tile(b, (n, 1)), "oracle_lb": np.vstack([good, bad] * 4)}
    bh = {"B7_bayes": np.array([0.6, 0.9, 0.7, 0.9, 0.6, 0.9, 0.8, 0.9]), "ML_mlp_state_obs": np.full(n, 0.5)}
    _, per_row = E.score_test(obs, cv, bh, cfg)
    dec = E.bootstrap_decision(obs, cv, bh, per_row, "B7_bayes", "ML_mlp_state_obs", cfg)
    fr = dec["delta_fec_relevant_secondary"]["test_id"]
    assert fr["n"] == 4
    assert fr["point"] == pytest.approx(np.mean(np.array([0.1, 0.2, 0.1, 0.3]) ** 2))  # regret on good rows = (b - 0.5)^2
    assert dec["C3_degenerate"] and dec["C3"]  # neither arm violates on the feasible rows: degenerate, holds


def test_score_test_fec_relevant(cfg):  # subset holds only states whose oracle minimum meets HD-FEC
    b = np.linspace(0.1, 1.0, 10)
    good, bad = -3 + (b - 0.5) ** 2, -1 + (b - 0.5) ** 2
    obs = pd.DataFrame({"split": ["test_id"] * 3, "stratum": ["base"] * 3, "n_local_min": [1] * 3, "O_L1": [0.5] * 3, "i_min": [0.1] * 3})
    cv = {"oracle_b": np.tile(b, (3, 1)), "oracle_lb": np.vstack([good, bad, good])}
    out, rows = E.score_test(obs, cv, {"m": np.array([0.5, 0.5, 0.7])}, cfg)
    assert out["test_id/fec_relevant"]["m"]["n"] == 2
    np.testing.assert_allclose(rows[("test_id/fec_relevant", "m")], [0.0, 0.04], atol=1e-12)


def test_macros_render(tmp_path):  # every leaf becomes a macro, lists by index, positional numbers
    from dcbias import macros as Mc
    (tmp_path / "results_main.json").write_text('{"a": {"x": 1.234e-5, "ci": [0.1, 0.2]}, "ok": true, "name": "B7_bayes"}')
    tex = Mc.render(tmp_path)
    assert r"\csname dcb@main/a/x\endcsname{0.00001234}" in tex
    assert r"\csname dcb@main/a/x@2\endcsname{0.000012}" in tex
    assert r"\csname dcb@main/a/ci/1\endcsname{0.2000}" in tex
    assert r"\csname dcb@main/ok\endcsname{true}" in tex and r"{B7\_bayes}" in tex
    (tmp_path / "results_x.json").write_text('{"neg": -0.00196, "nul": null}')
    tex = Mc.render(tmp_path)
    assert r"\csname dcb@x/neg@2\endcsname{\ensuremath{-}0.0020}" in tex  # sig zeros kept, typographic minus
    assert "dcb@x/nul" not in tex  # null leaf -> no macro -> citing it stops the build


def test_sweep_summary(tmp_path):  # sweep comparisons reach the paper through one summary JSON
    import json
    from dcbias import sweep_summary as S
    for name, pt, ci in (("a=1", 0.001, [0.0005, 0.002]), ("b=2", -0.0004, [-0.0009, 0.0001])):
        (tmp_path / f"sw-{name}").mkdir()
        (tmp_path / f"sw-{name}" / "results_main.json").write_text(json.dumps(
            {"best_non_ml": "B7_bayes", "best_ml_state_obs": "M", "_eval_provenance": {"sha": "s"},
             "test": {"test_id": {"B7_bayes": {"mean_regret": 0.002}, "M": {"mean_regret": 0.001}}},
             "confirmatory": {"framing": "(b) eq", "delta": {"test_id": {"point": pt, "ci": ci}}}}))
    out = S.summarise(tmp_path, "sw-")
    assert set(out["by_sweep"]) == {"a=1", "b=2"}
    r = out["range"]
    assert (r["delta_point_min"], r["delta_point_max"], r["ci_abs_max"]) == (-0.0004, 0.001, 0.002)
    assert (r["n_framing_b"], r["n_best_non_ml_B7"], r["n_ci_above_0"], r["n_ci_below_0"]) == (2, 2, 1, 0)


# ---- Selection pool, macros and reporting details

class _Const:
    def __init__(self, c=0.0):
        self.c = c

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.c)


def test_l2_betas_feasible_wiring():  # beta comes from the {target}_feasible rows only
    X = pd.DataFrame({"I_rated": [1.0] * 10, "L2_feasible": [True] * 5 + [False] * 5})
    y = np.array([0.01] * 5 + [0.5] * 5)
    m = {"fit": _Const(), "fixed": _Const()}
    beta = E.l2_betas(X, y, "L2", {"fit": _Const}, m, np.repeat(np.arange(5), 2), 5)
    assert beta == {"fit": pytest.approx(0.01), "fixed": pytest.approx(0.01)}


def test_pick_best_rejects_secondary():  # B7w can never enter the selection pool
    with pytest.raises(ValueError):
        E.pick_best({"B7_bayes": 0.2, "B7w_bayes_wave": 0.1, "ML_mlp_state_obs": 0.3})


def test_wave_tolerance():  # monotone prefix, not the largest isolated win
    assert E.wave_tolerance([(0, True), (1, True), (8, False), (64, True)]) == 1
    assert E.wave_tolerance([(64, True), (0, False)]) is None
    assert E.wave_tolerance([(0, True), (16, True)]) == 16


def test_macros_nonfinite_and_escapes(tmp_path):
    from dcbias import macros as Mc
    assert Mc.fmt(-0.0, 4) == "0.000" and Mc.fmt("a$b^c~{d}", 4) == r"a\$b\textasciicircum{}c\textasciitilde{}\{d\}"
    (tmp_path / "results_main.json").write_text('{"bad": NaN, "inf": Infinity, "ok": 1.5}')
    tex = Mc.render(tmp_path)
    assert "dcb@main/bad" not in tex and "dcb@main/inf" not in tex and "dcb@main/ok" in tex
    (tmp_path / "results_main.json").write_text('{"a~b": 1.0}')
    with pytest.raises(ValueError):
        Mc.render(tmp_path)


def test_calibrate_wave():  # k from |C_hat - C|, offset from guarded-clipped rows
    q = {16: 3.0}
    X = pd.DataFrame({"M": [16] * 4, "I_rated": [1.0] * 4, "b_p": [0.3] * 4, "sigma_I": [0.1] * 4,
                      "C": [0.5, 0.5, 0.9, 0.9], "C_hat": [0.51, 0.49, 0.9, 0.9], "w_max": [0.502, 0.502, 0.55, 0.62]})
    k, off = E.calibrate_wave(X, q)
    assert k == pytest.approx(np.quantile([0.01, 0.01, 0, 0], E.WAVE_CLIP_K_Q))
    # rows 1-2 clipped (w_max < 0.6, within k of C_hat): off = 0.002; row 3 below 0.6 but far below C_hat: not clipped
    assert off == pytest.approx(0.002)


# ---- Probe cap, seed spread and provenance

def test_probe_cap_uses_ntc_reading(cfg):  # the cap reads C_hat (observe's lambda_T = 1 draw), never true C
    row = {"state_id": 123, "probe_frac": 0.95}
    Ir, T_ntc = 1.0, 395.0
    nw = V.P(cfg, "sensing.window_symbols") * V.P(cfg, "ofdm.N")
    z0 = V.rng_for(cfg, V.STREAM_SENSE, 123, 0).standard_normal(5 + nw)[0]
    C_hat = float(V.foldback_ceiling(cfg, Ir, T_ntc + V.P(cfg, "sensing.sigma_T_ntc") * z0))
    C = float(V.foldback_ceiling(cfg, Ir, T_ntc))
    assert abs(C_hat - C) > 1e-6
    assert G.probe_bias(cfg, row, Ir, C, T_ntc, 0.1) == pytest.approx(min(0.95, 0.9 * C_hat, C))
    assert G.probe_bias(cfg, row | {"probe_frac": 0.3}, Ir, C, T_ntc, 0.1) == pytest.approx(0.3)  # uncapped
    assert G.probe_bias(cfg, row, Ir, 0.5 * C_hat, T_ntc, 0.1) == pytest.approx(0.5 * C_hat)  # driver clips at true C


def test_seed_sd_all_splits(cfg):  # seed spread reported on every test split present
    rng = np.random.default_rng(0)
    b = np.linspace(0.1, 1.0, 10)
    n = 12
    obs = pd.DataFrame({c: rng.uniform(0.2, 1.0, n) for c in V.FEATURES["state"]})
    obs["split"] = ["test_id"] * 6 + ["ood_T"] * 6
    cv = {"oracle_b": np.tile(b, (n, 1)), "oracle_lb": np.tile(-3 + (b - 0.5) ** 2, (n, 1))}
    X = obs.copy()
    rep = {"hyperparameters": {"ML_ridge_state": {"chosen": {"alpha": 1.0}}}}
    out = E.seed_sd(obs, cv, cfg, rep, X, rng.uniform(0.3, 0.7, n))
    assert set(out["ML_ridge_state"]) == {"test_id", "ood_T"}
    assert len(out["ML_ridge_state"]["ood_T"]["by_seed"]) == V.P(cfg, "evaluation.model_seeds")


def test_macros_record_keys(tmp_path):  # list records keyed by identifying fields; duplicates refused
    from dcbias import macros as Mc
    (tmp_path / "results_sensing.json").write_text('{"cells": [{"lambda_T": 4, "lambda_W": 16, "r": 0.5}, {"lambda_T": 4, "lambda_W": 0, "r": 0.25}]}')
    tex = Mc.render(tmp_path)
    assert r"\csname dcb@sensing/cells/lambda_T=4,lambda_W=16/r\endcsname{0.5000}" in tex
    (tmp_path / "results_sensing.json").write_text('{"cells": [{"lambda_T": 4, "r": 0.5}, {"lambda_T": 4, "r": 0.25}]}')
    with pytest.raises(ValueError):
        Mc.render(tmp_path)


def test_misspec_changes_controller_led_only(cfg):  # every declared variant moves the controller's q(i); data cfg untouched
    i = np.linspace(0.1, 1.0, 50)
    for true_model, variants in ((k, dict(v)) for k, v in E.MISSPEC.items()):
        data = E.misspec_cfg(cfg, {"led.model": true_model})
        before = V.config_hash(data)
        q_true = V.led_q(i, V.nominal_device(data, 1.0))
        assert variants.pop("true_form") == {}
        for over in variants.values():
            c = E.misspec_cfg(data, over)
            assert np.max(np.abs(V.led_q(i, V.nominal_device(c, 1.0)) - q_true)) > 1e-3
            assert {k for k in over} <= {"led.model", "led.kappa_tilde"}
        assert V.config_hash(data) == before


def test_observe_belief(cfg, obs_tab):  # device-shift arm
    base = V.observe(cfg, obs_tab, 1.0, 1.0, 1.0)
    pd.testing.assert_frame_equal(base, V.observe(cfg, obs_tab, 1.0, 1.0, 1.0, belief=cfg))
    other = V.load_config(overrides={"led.model": "tsonev_cubic", "led.i_min_frac": 0.05})
    o = V.observe(cfg, obs_tab, 1.0, 1.0, 1.0, belief=other)
    moved = {"P_opt_nom_est", "Tj_Rth_hat", "i_min"}
    for k in moved:
        assert not np.allclose(o[k], base[k]), k
    pd.testing.assert_frame_equal(o.drop(columns=list(moved)), base.drop(columns=list(moved)))


def test_generate_splits_twins(tmp_path):  # test_id-only twins pair row-for-row with the full design
    tiny = {"design.n_units_id": 4, "design.n_train_units": 2, "design.n_units_ood": 1, "design.states_per_unit": 2,
            "ofdm.R_label": 16, "ofdm.R_oracle": 16}
    full, _ = G.generate(V.load_config(overrides=tiny), tmp_path / "full", splits=["train", "test_id"])
    twin, _ = G.generate(V.load_config(overrides=tiny | {"led.kappa_tilde": 0.27}), tmp_path / "twin", splits=["test_id"])
    ref = full[full["split"] == "test_id"].reset_index(drop=True)
    assert len(twin) == len(ref) == 4
    same = ["state_id", "unit_id", "I_rated", "sigma_frac", "T_hs", "SNR_ref_dB", "M", "probe_frac", "T_ntc", "C"]
    same += [c for c in ref if c.startswith("u_") and c != "u_kt"]
    pd.testing.assert_frame_equal(twin[same], ref[same], check_exact=True)
    assert not np.allclose(twin["u_kt"], ref["u_kt"])  # the shifted LED latent does move


# ---- Cubic LED form with per-unit spread
def test_cubic_spread(cfg):
    cub, spr = V.load_config(overrides={"led.model": "tsonev_cubic"}), V.load_config(overrides={"led.model": "tsonev_cubic_spread"})
    k0 = V.P(cfg, "led.kappa_tilde")
    # tsonev_cubic ignores a unit's kappa_tilde (unchanged data); the spread form equals it at the nominal
    assert np.array_equal(V.nominal_device(cub, 1.0, 0.33)["cubic"], V.nominal_device(cub, 1.0)["cubic"])
    assert np.allclose(V.nominal_device(spr, 1.0, k0)["cubic"], V.nominal_device(cub, 1.0)["cubic"], rtol=0, atol=1e-15)
    lo, hi = (V.nominal_device(spr, 1.0, k0 * f) for f in (0.85, 1.15))
    assert not np.allclose(lo["cubic"], hi["cubic"])
    for d in (lo, hi):  # slope 1 - kt at I_rated / 2, monotone across the unit range
        assert np.isclose(np.polyval(np.polyder(d["cubic"]), 0.5), 1 - d["kt"])
        V.check_monotone(d)
    # build() gives each unit its own curve under the spread form only
    row = {"unit_id": 3, "I_rated": 1.0, "sigma_frac": 0.1, "T_hs": 300.0, "SNR_ref_dB": 20.0, "M": 16}
    u = V.draw_unit(spr, 3)
    assert np.allclose(V.build(spr, row)[1]["cubic"], V.nominal_device(spr, 1.0, u["kt"])["cubic"])
    assert np.array_equal(V.build(cub, row)[1]["cubic"], V.nominal_device(cub, 1.0)["cubic"])


def test_b7_wide_latents(cfg):
    base, wide = E.b7_latents(cfg), E.b7_latents(cfg, E.B7_WIDE_KT)
    assert len(wide) == len(base) // V.P(cfg, "evaluation.b7_grid_points") * len(E.B7_WIDE_KT)
    assert {l["kt"] for l in wide} == set(E.B7_WIDE_KT) and max(E.B7_WIDE_KT) < 0.5
    assert {(l["delta"], l["eta0"]) for l in wide} == {(l["delta"], l["eta0"]) for l in base}


def test_penalty_db():
    p = E.penalty_db(np.array([0.0, 0.2, 0.2, -0.0]), np.array([0.0, 0.5, 0.001, 0.3]))
    assert p[0] == 0 and np.isclose(p[1], 0.4) and np.isnan(p[2]) and p[3] == 0
    s = E.penalty_summary(p)
    assert s["n_unresolvable"] == 1 and np.isclose(s["mean"], 0.4 / 3)


def test_old_twin_regenerates():
    from dcbias import sweeps as S
    out = G.RUNS_DIR / "twin-led.kappa_tilde=0.42"
    if not out.exists():
        pytest.skip("twin data not present")
    assert S.rows_reproduce(out, V.load_config(overrides=E.SHIFT["led.kappa_tilde=0.42"]), n=2)
