"""Evaluation: baselines B0-B8, ML, BER regret, cluster bootstrap and the decision rule.

Reads runs/<run>/{states.csv, curves.npz}; writes results_<task>.json there.

Tasks:
  main      all methods at nominal sensing noise; C1-C3 and the framing; L2 secondary metrics;
            model-seed SD; cost (latency, memory, simulator calls)
  sensing   thermal-family vs waveform-family lambda map (deployment-matched + train-clean)
  learning  learning curves over training units
  b8        online golden-section sweep on the live link (cost-regret frontier)

Selection rule: best ML and best non-ML are chosen by validation regret inside the
training units (GroupKFold by unit), never on test. Regret on training rows uses each row's
R_label curve; regret on test rows uses the independent oracle curve.

Usage: python -m dcbias.evaluate <run_dir> main [--workers 8]
"""
import argparse
import copy
import io
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from dcbias import generate as G
from dcbias import physics as V

TEST_SPLITS = ("test_id", "ood_T", "ood_SNR", "ood_sigma")
# Additions kept out of config.json so the stored provenance config stays identical to the config file.
L2_BACKOFF_Q = 0.95  # back-off = this quantile of the out-of-fold training L2 shortfall
WAVE_CLIP_K_Q = 0.95  # clip test needs w_max within this |C_hat - C| quantile of C_hat
LAMBDA_W_STRESS = (8, 16, 32, 64)  # extra lambda_W columns, ~7 to 4 effective ADC bits at 10-bit nominal
# Pre-specified hyperparameter grids.
GRIDS = {
    "ridge": [{"alpha": a} for a in (1e-3, 1e-2, 1e-1, 1.0, 10.0)],
    "rf": [{"min_samples_leaf": m} for m in (1, 5, 20)],
    "hgb": [{"learning_rate": lr, "max_leaf_nodes": n} for lr in (0.05, 0.1) for n in (15, 31, 63)],
    "mlp": [{"alpha": a} for a in (1e-4, 1e-3, 1e-2)],
}


# --------------------------------------------------------------------------- data and regret

def load(run_dir):
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "states.csv")
    cv = dict(np.load(run_dir / "curves.npz"))
    if not np.array_equal(cv["state_id"], df["state_id"].to_numpy()):
        raise ValueError("curves.npz rows are not aligned with states.csv")
    prov = json.loads((run_dir / "provenance.json").read_text())
    keep = df["tj_ok"].to_numpy(bool)  # Tj above the LED limit at L1: excluded, share in provenance
    return df[keep].reset_index(drop=True), {k: v[keep] for k, v in cv.items()}, prov["config"]


class Curve:
    """One state's log10 BER-vs-bias curve with a PCHIP interpolant (shape-preserving: exact at the
    nodes, no undershoot below the minimum node, so regret(b_o) = 0 and regret >= 0)."""

    def __init__(self, b, lb):
        self.b, self.lb, self.min = b, lb, float(lb.min())
        self.f = PchipInterpolator(b, lb, extrapolate=False)

    def __call__(self, x):
        """The driver command window is [i_min, C], the curve's span: a request outside it is clamped,
        so a bias below i_min is scored at i_min's BER (kinder than the physics; score_test reports
        the share of such requests)."""
        return self.f(np.clip(x, self.b[0], self.b[-1]))


def curve_rows(cv, kind, rows):
    """Curve per row index, NaN padding removed."""
    out = []
    for r in rows:
        b, lb = cv[f"{kind}_b"][r], cv[f"{kind}_lb"][r]
        ok = ~np.isnan(b)
        if not ok.any():
            raise ValueError(f"row {r} has no {kind} curve")
        out.append(Curve(b[ok], lb[ok]))
    return out


def lb_at(bhat, curves):
    return np.array([float(c(x)) for x, c in zip(bhat, curves)])


def regret(bhat, curves, floor):
    fl = np.log10(floor)
    best = np.array([c.min for c in curves])
    return np.maximum(lb_at(bhat, curves), fl) - np.maximum(best, fl)


# --------------------------------------------------------------------------- statistics

def sd1(x):
    """Sample SD with ddof = 1."""
    return float(np.std(x, ddof=1))


def cluster_boot(units, a, b=None, B=10000, seed=0):
    """Paired cluster bootstrap over units: returns point, 95 % percentile CI, draws of
    mean(a) - mean(b) (or mean(a)). Resamples units, not rows."""
    d = np.asarray(a, float) - (0.0 if b is None else np.asarray(b, float))
    u, inv = np.unique(units, return_inverse=True)
    sums, cnts = np.bincount(inv, d), np.bincount(inv)
    pick = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    draws = sums[pick].sum(1) / cnts[pick].sum(1)
    return {"point": float(d.mean()), "ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "p_le0": float((1 + np.sum(draws <= 0)) / (B + 1))}, draws


def c3_boot(units, v_ml, v_non, B):
    """C3 violation-rate difference ML - non-ML. When no paired difference is non-zero the bootstrap
    has zero variance: report it as degenerate with counts, no CI or p.
    A degenerate C3 holds trivially (difference 0 <= margin) and the flag travels with it."""
    v_ml, v_non = np.asarray(v_ml, bool), np.asarray(v_non, bool)
    counts = {"n": int(len(v_ml)), "violations_ml": int(v_ml.sum()), "violations_non_ml": int(v_non.sum())}
    if not np.any(v_ml != v_non):
        return {"degenerate": True, "point": 0.0, "ci": None, "p_le0": None} | counts
    return cluster_boot(units, v_ml, v_non, B)[0] | {"degenerate": False} | counts


def c1_c2(res, margin):
    """Decision rule. C1: point >= margin and CI lower bound > 0 on test_id. C2: the CI lower bound is
    > 0 on every OOD set; an intersection-union test, so no multiplicity adjustment."""
    c1 = res["test_id"]["point"] >= margin and res["test_id"]["ci"][0] > 0
    ood = ("ood_T", "ood_SNR", "ood_sigma")
    return bool(c1), bool(all(s in res and res[s]["ci"][0] > 0 for s in ood))


def decide(c1_point, c1_ci, c1, c2, c3, margin):
    """Decision table. Cases the table does not map are returned as UNMAPPED, never guessed.
    Equivalence takes precedence over the negative row; the direction is appended whenever the 95 %
    CI excludes 0."""
    if c1 and c3:
        return "(a) ML practically better, including out of regime" if c2 else "(a') ML better in-distribution only; OOD failure goes in the abstract"
    if c1 and not c3:
        return "UNMAPPED: C1 holds but C3 (constraint non-inferiority) fails; decide with the user"
    if -margin < c1_ci[0] and c1_ci[1] < margin:
        out = "(b) equivalence: model-based or LUT control suffices"
    elif c1_ci[1] < 0:
        out = "(b) negative result: non-ML better"
    else:
        out = "(b) inconclusive ML comparison; no ML-superiority claim"
    if c1_ci[0] > 0:
        return out + "; direction: ML lower regret"
    if c1_ci[1] < 0:
        return out + "; direction: non-ML lower regret"
    return out


# --------------------------------------------------------------------------- non-ML baselines

class LUT:
    """Binned means on quantile edges, multilinear interpolation between bin centres, coordinates
    clamped to the outermost centres. A column with no more distinct values than bins (M) gets one
    cell per value, centred on the value. Empty cells are filled from the nearest filled cell and
    counted; predict() flags rows whose nearest cell was filled in `last_filled`."""

    def __init__(self, cols, bins):
        self.cols, self.bins = cols, bins

    def fit(self, X, y):
        X = X[self.cols].to_numpy(float)
        self.edges, centres = [], []
        for d in range(X.shape[1]):
            u = np.unique(X[:, d])
            if len(u) <= self.bins:
                self.edges.append(np.concatenate([[u[0]], (u[:-1] + u[1:]) / 2, [u[-1]]]) if len(u) > 1 else np.array([u[0], u[0]]))
                centres.append(u)
            else:
                e = np.unique(np.quantile(X[:, d], np.linspace(0, 1, self.bins + 1)))
                self.edges.append(e)
                centres.append((e[:-1] + e[1:]) / 2)
        idx = tuple(np.clip(np.searchsorted(e, X[:, d], side="right") - 1, 0, len(e) - 2) for d, e in enumerate(self.edges))
        shape = tuple(len(e) - 1 for e in self.edges)
        s, n = np.zeros(shape), np.zeros(shape)
        np.add.at(s, idx, y)
        np.add.at(n, idx, 1)
        self.n_cells, self.n_empty = int(n.size), int((n == 0).sum())
        table = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        empty = (n == 0).astype(float)
        if self.n_empty:
            filled = np.argwhere(n > 0)
            for cell in np.argwhere(n == 0):
                table[tuple(cell)] = table[tuple(filled[np.argmin(((filled - cell) ** 2).sum(1))])]
        self.centres = centres
        axes = [c if len(c) > 1 else np.array([c[0] - 1, c[0] + 1]) for c in self.centres]
        for d, c in enumerate(self.centres):
            if len(c) == 1:
                table, empty = np.repeat(table, 2, axis=d), np.repeat(empty, 2, axis=d)
        self.interp = RegularGridInterpolator(axes, table)
        self.empty = RegularGridInterpolator(axes, empty, method="nearest")
        self.lo, self.hi = [a[0] for a in axes], [a[-1] for a in axes]
        return self

    def predict(self, X):
        q = np.clip(X[self.cols].to_numpy(float), self.lo, self.hi)
        self.last_filled = self.empty(q) > 0.5
        return self.interp(q)


class Rule:
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(self.fn(X), float)


def window(X):
    """Clipping window [i_min, C_hat] a controller knows: the LED turn-on clip and its ceiling reading."""
    return X["i_min"].to_numpy(), X["C_hat"].to_numpy()


def midpoint(X):
    lo, C = window(X)
    return (lo + C) / 2


def b1b(X, k):
    """k sigma above the turn-on clip when the window holds +/- k sigma, else the window midpoint."""
    lo, C = window(X)
    kb = lo + k * X["sigma_I"].to_numpy()
    return np.where(kb + k * X["sigma_I"].to_numpy() <= C, kb, (lo + C) / 2)


class FitK:
    """B1c: one k for the ceiling-clamped k*sigma rule, fitted on training regret."""

    def __init__(self, curves_of, floor):
        self.curves_of, self.floor = curves_of, floor

    def fit(self, X, y):
        cur, fl = self.curves_of(X.index), self.floor
        ks = np.linspace(0.5, 6.0, 111)
        self.k = float(ks[np.argmin([regret(b1b(X, k), cur, fl).mean() for k in ks])])
        return self

    def predict(self, X):
        return b1b(X, self.k)


class Static:
    """B0: one constant bias (amps) minimising mean training regret."""

    def __init__(self, curves_of, floor):
        self.curves_of, self.fl = curves_of, np.log10(floor)

    def fit(self, X, y):
        cur = self.curves_of(X.index)
        cand = np.linspace(0, X["I_rated"].max(), 401)
        lbs = np.array([c(cand) for c in cur])
        best = np.array([c.min for c in cur])
        r = np.maximum(lbs, self.fl) - np.maximum(best, self.fl)[:, None]
        self.c = float(cand[np.argmin(r.mean(0))])
        return self

    def predict(self, X):
        return np.full(len(X), self.c)


class ModelGrid:
    """B6b: L1 (or L2) precomputed from the nominal-device analytic simulator on a regular grid over
    (I_rated, sigma_frac, T_ntc, SNR, M), multilinear interpolation (M enters L1 through the
    non-flat channel). Spans the full state range including OOD, so it extrapolates by construction.
    n_calls = simulator-call budget."""

    def __init__(self, cfg, target="L1"):
        self.cfg, self.target = cfg, target

    def fit(self, X=None, y=None):
        cfg, n = self.cfg, V.P(self.cfg, "evaluation.model_grid")
        ax = [np.linspace(*V.P(cfg, f"states.{k}"), n[k]) for k in ("I_rated_A", "sigma_frac", "T_hs_K", "SNR_ref_dB")]
        Ms = V.P(cfg, "states.M")
        quad = V.quadrature(cfg)
        tab = np.empty([len(a) for a in ax] + [len(Ms)])
        for idx in np.ndindex(*tab.shape):
            Ir, f, T, S = (ax[d][idx[d]] for d in range(4))
            C = float(V.foldback_ceiling(cfg, Ir, T))
            st = {"I_rated": Ir, "sigma": f * Ir, "T_hs": T, "snr_db": S, "M": Ms[idx[4]], "C": C}
            b = V.bias_grid(cfg, Ir, C)
            lab = V.labels_from_curve(b, V.analytic_curve(cfg, st, V.nominal_device(cfg, Ir), b, quad)["lb"], cfg)
            tab[idx] = lab[self.target]
        self.n_calls = int(tab.size)
        self.axes = ax + [np.asarray(Ms, float)]
        self.interp = RegularGridInterpolator(self.axes, tab)
        return self

    def predict(self, X):
        q = np.column_stack([X["I_rated"], X["sigma_I"] / X["I_rated"], X["T_ntc_hat"], X["SNR_hat"], X["M"]])
        lo, hi = [a[0] for a in self.axes], [a[-1] for a in self.axes]
        return self.interp(np.clip(q, lo, hi))


# --------------------------------------------------------------------------- B7 Bayesian calibrate-then-optimise

B7_OBS = ("EVM_p", "Prx_hat", "Tj_Vf_hat", "Tj_Rth_hat")


def b7_latents(cfg, kt_axis=None):
    """kt_axis replaces the kappa_tilde axis (wide-prior B7)."""
    n = V.P(cfg, "evaluation.b7_grid_points")
    on = V.P(cfg, "design.unit_spread_on")
    axes = []
    for key, sp in (("led.kappa_tilde", "kappa_tilde"), ("led.delta", "delta"), ("led.eta0_nom", "eta0")):
        w = V.P(cfg, f"unit_spread.{sp}") if on else 0.0
        axes.append(V.P(cfg, key) * np.linspace(1 - w, 1 + w, n if w > 0 else 1))
    if kt_axis is not None:
        axes[0] = np.asarray(kt_axis, float)
    return [dict(kt=a, delta=b, eta0=c) for a in axes[0] for b in axes[1] for c in axes[2]]


def _b7_curves(cfg, row, latent, b):
    """Analytic log10 BER over b and predicted observables at b_p, for one latent triple, using only
    what the controller knows: I_rated, sigma_I, M, b_p, T_ntc_hat (as T_hs), SNR_hat, C_hat."""
    st = {"I_rated": row["I_rated"], "sigma": row["sigma_I"], "T_hs": row["T_ntc_hat"], "snr_db": row["SNR_hat"],
          "M": int(row["M"]), "C": row["C_hat"]}
    dev = V.nominal_device(cfg, row["I_rated"]) | latent
    an = V.analytic_curve(cfg, st, dev, np.append(b, row["b_p"]), V.quadrature(cfg))
    # Tj_Rth_hat follows the observation model in V.observe: T_ntc_hat + Rth_nom (P_elec - P_opt_nom_est)
    pred = {"EVM_p": 10 * np.log10(np.mean(1 / an["sinr"][-1])), "Prx_hat": 10 * np.log10(an["Ep"][-1]), "Tj_Vf_hat": an["Tj"][-1],
            "Tj_Rth_hat": row["T_ntc_hat"] + V.P(cfg, "thermal.Rth") * (an["P_elec"][-1] - row["P_opt_nom_est"]),
            "P_elec": an["P_elec"][-1]}  # not an observable: spec_sig() needs it
    return an["lb"][:-1], pred


# Probe-EVM spread at fixed state and device, from independent probe redraws: median 0.137 dB over 20 test_id
# rows x 40 redraws at the true devices. The EVM SD is 0.505-0.577 dB over that whole range, dominated by
# snr_sd_dB, so the source rows do not matter. Clipping is shared across one symbol's subcarriers, so
# 10/ln10/sqrt(R_probe * Nu) = 0.068 dB understates it. A declared constant of the controller.
EVM_PROBE_SD_DB = 0.14


def spec_sig(cfg, pred):
    """Likelihood SDs from sensor specifications alone, with no characterised unit (n = 0;
    "sensor-spec calibration, model error unmodelled").
    EVM: probe spread (EVM_PROBE_SD_DB) and the SNR_hat error (snr_sd_dB; B7 predicts EVM from SNR_hat,
         slope taken as 1 dB/dB, the noise-limited upper bound).
    P_rx: prx_sd_dB.
    Tj_Vf: V noise and K-factor error at the predicted Tj, plus the NTC noise and unit offset, since
         the prediction takes T_ntc_hat as T_hs.
    Tj_Rth: electrical-power error at the predicted P_elec (V.observe: Rth_nom is used and T_ntc_hat
         cancels). Unit-to-unit model error is in no spec, so it is absent."""
    sV = V.P(cfg, "sensing.sigma_V") / abs(V.P(cfg, "sensing.K_V_nom"))
    ntc = np.hypot(V.P(cfg, "sensing.sigma_T_ntc"), V.P(cfg, "unit_spread.ntc_offset_sd"))
    return {"EVM_p": float(np.hypot(EVM_PROBE_SD_DB, V.P(cfg, "sensing.snr_sd_dB"))),
            "Prx_hat": V.P(cfg, "sensing.prx_sd_dB"),
            "Tj_Vf_hat": float(np.sqrt(sV**2 + (V.P(cfg, "unit_spread.epsK_sd") * (pred["Tj_Vf_hat"] - V.T0)) ** 2 + ntc**2)),
            "Tj_Rth_hat": max(V.P(cfg, "thermal.Rth") * V.P(cfg, "sensing.sigma_P_rel") * pred["P_elec"], 1e-6)}


def spec_z(cfg, X):
    """SD over rows of (observed - predicted) / spec_sig at the true latents, per observable: near 1
    when spec_sig describes the residual. Same row sample as B7.fit."""
    z = {k: [] for k in B7_OBS}
    for r in X.sample(min(len(X), 1000), random_state=0).to_dict("records"):
        b = V.bias_grid(cfg, r["I_rated"], r["C_hat"])
        _, pred = _b7_curves(cfg, r, {"kt": r["u_kt"], "delta": r["u_delta"], "eta0": r["u_eta0"]}, b[:2])
        sd = spec_sig(cfg, pred)
        for k in B7_OBS:
            z[k].append((r[k] - pred[k]) / sd[k])
    return {k: sd1(v) for k, v in z.items()}, {k: float(np.mean(v)) for k, v in z.items()}


def _b7_row(args):
    cfg, row, sig, bias, latents = args
    b = V.bias_grid(cfg, row["I_rated"], row["C_hat"])
    lbs, ll = [], []
    for lat in latents:
        lb, pred = _b7_curves(cfg, row, lat, b)
        lbs.append(lb)
        if sig is None:  # sensor-spec B7, bias 0; per-latent SDs, so the log-SD terms stay
            sd = spec_sig(cfg, pred)
            ll.append(sum(-0.5 * ((row[k] - pred[k]) / sd[k]) ** 2 - np.log(sd[k]) for k in B7_OBS))
        else:
            ll.append(-0.5 * sum(((row[k] - pred[k] - bias[k]) / sig[k]) ** 2 for k in B7_OBS))
    ll = np.array(ll)
    w = np.exp(ll - ll.max())
    with np.errstate(all="ignore"):  # macOS Accelerate matmul raises spurious FP flags
        e = (w / w.sum()) @ np.array(lbs)
    if not np.all(np.isfinite(e)):
        raise FloatingPointError(f"B7 posterior-expected log10 BER not finite (state {row.get('state_id')})")
    return b[np.argmin(e)]


def unclipped_max_q(cfg, M, q=0.01, draws=2000):
    """q-quantile of max(s) over one sensing window of unclipped unit-power DCO-OFDM.
    A dedicated seed, never a simulation stream."""
    W, N, Nu = V.P(cfg, "sensing.window_symbols"), V.P(cfg, "ofdm.N"), V.P(cfg, "ofdm.Nu")
    rng = np.random.default_rng(20261016 + int(M))
    return float(np.quantile([V.ofdm_symbols(rng, W, int(M), N, Nu)[2].max() for _ in range(draws)], q))


def wave_clipped(X, q, k):
    """True where the top of the probe waveform is clipped, so w_max reads the ceiling (plus ADC noise):
    w_max sits below the 1 % quantile of the unclipped window maximum, b_p + sigma * q[M], AND within
    k * I_rated below the NTC ceiling reading C_hat. The second condition stops the ~1 % of unclipped
    windows whose maximum falls below q01 from being read as a ceiling far below C.
    k = a training quantile of |C_hat - C| / I_rated."""
    qm = X["M"].map(q).to_numpy(float)
    if np.isnan(qm).any():
        raise ValueError(f"no unclipped-maximum quantile for M in {sorted(set(X['M']) - set(q))}")
    w = X["w_max"].to_numpy()
    return (w < X["b_p"].to_numpy() + X["sigma_I"].to_numpy() * qm) & (w >= X["C_hat"].to_numpy() - k * X["I_rated"].to_numpy())


def wave_ceiling(cfg, X, q, k, off=0.0):
    """Ceiling estimate from the drive-current tap: w_max - off * I_rated where the
    waveform is clipped (off = the ADC-noise excess of w_max over C in I_rated units; the LSB scales
    with I_rated, calibrated on training rows by B7.fit), else
    max(C_hat, w_max), since the ceiling is at least the largest sample seen. q, k: see wave_clipped; q maps M to the
    unclipped-maximum quantile (unclipped_max_q, a stored constant of the controller). Clamped to
    [i_min + 2 grid steps, I_rated] so the bias grid exists."""
    w, C_hat, Ir = X["w_max"].to_numpy(), X["C_hat"].to_numpy(), X["I_rated"].to_numpy()
    C = np.where(wave_clipped(X, q, k), w - off * Ir, np.maximum(C_hat, w))
    lo = X["i_min"].to_numpy() + 2 * V.P(cfg, "ofdm.grid_step_frac") * Ir
    return np.clip(C, lo, Ir)


def calibrate_wave(X, q):
    """B7w constants from training rows with known C: k = WAVE_CLIP_K_Q quantile of |C_hat - C| / I_rated,
    off = median ADC excess (w_max - C) / I_rated on rows the guarded test reads as clipped (0 if none)."""
    Ir = X["I_rated"].to_numpy()
    k = float(np.quantile(np.abs(X["C_hat"] - X["C"]).to_numpy() / Ir, WAVE_CLIP_K_Q))
    clip = wave_clipped(X, q, k)
    return k, float(np.median(((X["w_max"] - X["C"]).to_numpy() / Ir)[clip])) if clip.any() else 0.0


class B7:
    """Posterior over (kappa_tilde, delta, eta0) on a 6^3 grid; uniform prior (the unit distribution
    is uniform on these ranges); Gaussian likelihood of (EVM, P_rx, Tj_Vf, Tj_Rth) at the probe bias
    with per-observable mean residual and SD estimated on training rows at the true latents; outputs
    the bias minimising posterior-expected log10 BER. Other unit parameters stay nominal (model
    misspecification is part of the comparison). The analytic model has no per-state waveform, so
    waveform statistics do not enter the likelihood. They do carry the true clip level C through
    w_max: ceiling="wave" replaces C_hat by wave_ceiling() in fit and predict
    (secondary comparator B7w, outside the selection pool)."""

    def __init__(self, cfg, workers=1, ceiling="ntc", kt_axis=None, calib="units"):
        """calib="spec": no characterised units; SDs from spec_sig(), mean residual 0."""
        if calib not in ("units", "spec") or (calib == "spec" and ceiling != "ntc"):
            raise ValueError(f"calib={calib!r} with ceiling={ceiling!r}")
        self.cfg, self.workers, self.ceiling, self.calib = cfg, workers, ceiling, calib
        self.latents = b7_latents(cfg, kt_axis)
        self.q = {M: unclipped_max_q(cfg, M) for M in V.P(cfg, "states.M")} if ceiling == "wave" else None
        self.c_off, self.c_k = 0.0, 0.0

    def _x(self, X):
        return X.assign(C_hat=wave_ceiling(self.cfg, X, self.q, self.c_k, self.c_off)) if self.ceiling == "wave" else X

    def fit(self, X, y=None):
        self.n_calls_per_state = len(self.latents)
        if self.calib == "spec":  # X is not read
            self.sig, self.bias = None, {k: 0.0 for k in B7_OBS}
            return self
        if self.ceiling == "wave":  # training rows know the true C, as they know the latents below
            self.c_k, self.c_off = calibrate_wave(X, self.q)
        X = self._x(X)
        rows = X.sample(min(len(X), 1000), random_state=0).to_dict("records")
        res = {k: [] for k in B7_OBS}
        for r in rows:
            b = V.bias_grid(self.cfg, r["I_rated"], r["C_hat"])
            _, pred = _b7_curves(self.cfg, r, {"kt": r["u_kt"], "delta": r["u_delta"], "eta0": r["u_eta0"]}, b[:2])
            for k in B7_OBS:
                res[k].append(r[k] - pred[k])
        self.sig = {k: max(sd1(v), 1e-6) for k, v in res.items()}
        self.bias = {k: float(np.mean(v)) for k, v in res.items()}
        return self

    def predict(self, X):
        jobs = [(self.cfg, r, self.sig, self.bias, self.latents) for r in self._x(X).to_dict("records")]
        if self.workers > 1:
            with Pool(self.workers) as pool:
                return np.array(pool.map(_b7_row, jobs, chunksize=8))
        return np.array([_b7_row(j) for j in jobs])


# --------------------------------------------------------------------------- ML

def make_ml(name, hp, seed):
    if name == "ridge":
        return make_pipeline(StandardScaler(), PolynomialFeatures(3), StandardScaler(), Ridge(**hp))
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, max_features=0.5, n_jobs=-1, random_state=seed, **hp)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=500, early_stopping=False, random_state=seed, **hp)
    if name == "mlp":
        # y scaler fitted on training rows only: TransformedTargetRegressor refits it per fit().
        net = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, early_stopping=False, random_state=seed, **hp))
        return TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
    raise ValueError(name)


class ML:
    def __init__(self, name, cols, hp, seed=0):
        self.name, self.cols, self.hp, self.seed = name, cols, hp, seed

    def fit(self, X, y):
        self.m = make_ml(self.name, self.hp, self.seed).fit(X[self.cols].to_numpy(float), y)
        return self

    def predict(self, X):
        return self.m.predict(X[self.cols].to_numpy(float))


# --------------------------------------------------------------------------- selection

def cv_regret(make, X, y, groups, curves_of, folds, floor):
    """Mean validation regret over GroupKFold by unit (unit sets disjoint)."""
    pred = np.empty(len(X))
    for tr, va in GroupKFold(folds).split(X, y, groups):
        pred[va] = make().fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
    return float(regret(pred, curves_of(X.index), floor).mean())


def tune_ml(name, cols, X, y, groups, curves_of, folds, floor):
    scores = [cv_regret(lambda hp=hp: ML(name, cols, hp), X, y, groups, curves_of, folds, floor) for hp in GRIDS[name]]
    j = int(np.argmin(scores))
    return GRIDS[name][j], scores


def nonml_methods(cfg, curves_of, target="L1"):
    """name -> (factory, needs_cv). needs_cv: fitted on training rows, so scored by GroupKFold."""
    fl = V.P(cfg, "ber.regret_floor")
    m = {
        "B0_static": (lambda: Static(curves_of, fl), True),
        "B2_zhang_midpoint": (lambda: Rule(midpoint), False),
        "B3_centre": (lambda: Rule(midpoint), False),
        "B4_lut1d_C": (lambda: LUT(["C_hat"], V.P(cfg, "evaluation.lut1d_bins")), True),
        "B5_lut2d_Ir_T": (lambda: LUT(["I_rated", "T_ntc_hat"], V.P(cfg, "evaluation.lut2d_bins")), True),
        "B6a_lut_state": (lambda: LUT(V.FEATURES["state"] if target != "L1" else V.FEATURES["state"][:-1],
                                      V.P(cfg, "evaluation.lut_state_bins")), True),
    }
    if target == "L1":
        for k in V.P(cfg, "evaluation.k_grid"):
            m[f"B1a_k{k}"] = (lambda k=k: Rule(lambda X: k * X["sigma_I"]), False)
            m[f"B1b_k{k}"] = (lambda k=k: Rule(lambda X: b1b(X, k)), False)
        m["B1c_kfit"] = (lambda: FitK(curves_of, fl), True)
    return m


# --------------------------------------------------------------------------- tasks

def prepare(run_dir, lam=(1.0, 1.0, 1.0), rep=0, belief=None):
    df, cv, cfg = load(run_dir)
    obs = V.observe(cfg, df, *lam, rep=rep, belief=belief)
    tr = np.flatnonzero(obs["split"] == "train")
    label_curves = curve_rows(cv, "label", tr)
    lookup = dict(zip(obs.index[tr], label_curves))

    def curves_of(index):
        return [lookup[i] for i in index]

    return obs, cv, cfg, tr, curves_of


def score_test(obs, cv, bhat_by_method, cfg):
    """Per-method regret on each test split (and on test_id per T stratum) against the oracle curves,
    with secondary metrics. MAE / R2 are also given on unimodal curves only."""
    out, per_row = {}, {}
    fec = np.log10(V.P(cfg, "ber.fec_hd"))
    sets = [(s, obs["split"] == s) for s in TEST_SPLITS]
    sets += [(f"test_id/{st}", (obs["split"] == "test_id") & (obs["stratum"] == st)) for st in ("base", "foldback")]
    # Secondary, post hoc: states whose oracle minimum meets HD-FEC
    relevant = np.nanmin(cv["oracle_lb"], axis=1) <= fec
    sets += [(f"{s}/fec_relevant", (obs["split"] == s).to_numpy() & relevant) for s in TEST_SPLITS]
    for split, mask in sets:
        rows = np.flatnonzero(mask)
        if not len(rows):
            continue
        uni = obs["n_local_min"].to_numpy()[rows] == 1
        cur = curve_rows(cv, "oracle", rows)
        b0 = obs["O_L1"].to_numpy()[rows]
        feasible = np.array([c.min <= fec for c in cur])
        out[split] = {}
        for name, bh in bhat_by_method.items():
            r = regret(bh[rows], cur, V.P(cfg, "ber.regret_floor"))
            per_row[(split, name)] = r
            lbh = lb_at(bh[rows], cur)
            e = bh[rows] - b0
            out[split][name] = {
                "n": len(rows), "mean_regret": float(r.mean()), "p95_regret": float(np.quantile(r, 0.95)),
                "below_i_min_share": float(np.mean(bh[rows] < obs["i_min"].to_numpy()[rows])),
                "regret_floor_sensitivity": {str(f): float(regret(bh[rows], cur, f).mean()) for f in V.P(cfg, "ber.regret_floor_sensitivity")},
                "fec_violation_rate": float(np.mean(lbh[feasible] > fec)) if feasible.any() else None,
                "mae_A": float(np.abs(e).mean()), "rmse_A": float(np.sqrt((e**2).mean())),
                "r2_vs_oracle": float(1 - (e**2).sum() / ((b0 - b0.mean()) ** 2).sum()),
                "unimodal_share": float(uni.mean()), "mae_A_unimodal": float(np.abs(e[uni]).mean()) if uni.any() else None,
                "r2_vs_oracle_unimodal": float(1 - (e[uni] ** 2).sum() / ((b0[uni] - b0[uni].mean()) ** 2).sum()) if uni.sum() > 1 else None,
            }
    return out, per_row


def bootstrap_decision(obs, cv, bhat, per_row, best_non, best_ml, cfg):
    B, margin = V.P(cfg, "evaluation.bootstrap_B"), V.P(cfg, "evaluation.margin_decades")
    res = {}
    for split in TEST_SPLITS:
        if (split, best_ml) not in per_row:
            continue
        units = obs.loc[obs["split"] == split, "unit_id"].to_numpy()
        res[split], _ = cluster_boot(units, per_row[(split, best_non)], per_row[(split, best_ml)], B)
    c1, c2 = c1_c2(res, margin)
    lo, hi = res["test_id"]["ci"]
    # violation-rate difference ML - non-ML among oracle-feasible ID states, cluster bootstrap.
    rows = np.flatnonzero(obs["split"] == "test_id")
    fec = np.log10(V.P(cfg, "ber.fec_hd"))
    cur = curve_rows(cv, "oracle", rows)
    feas = np.array([c.min <= fec for c in cur])
    v_ml = (lb_at(bhat[best_ml][rows], cur) > fec)[feas]
    v_non = (lb_at(bhat[best_non][rows], cur) > fec)[feas]
    c3s = c3_boot(obs["unit_id"].to_numpy()[rows][feas], v_ml, v_non, B)
    c3 = c3s["degenerate"] or c3s["ci"][1] <= V.P(cfg, "evaluation.c3_margin")
    # Secondary, post hoc: the same paired Delta on the FEC-relevant states
    fec_rel = {}
    for split in TEST_SPLITS:
        key = f"{split}/fec_relevant"
        if (key, best_ml) in per_row and len(per_row[(key, best_ml)]):
            m = (obs["split"] == split).to_numpy() & (np.nanmin(cv["oracle_lb"], axis=1) <= fec)
            fec_rel[split], _ = cluster_boot(obs["unit_id"].to_numpy()[m], per_row[(key, best_non)], per_row[(key, best_ml)], B)
            fec_rel[split]["n"] = int(m.sum())
    return {
        "delta": res, "delta_fec_relevant_secondary": fec_rel,
        "C3_violation_diff": c3s, "C1": c1, "C2": c2, "C3": bool(c3), "C3_degenerate": c3s["degenerate"],
        "observed_equivalence_bound_decades": float(max(abs(lo), abs(hi))),  # descriptive only
        "framing": decide(res["test_id"]["point"], res["test_id"]["ci"], c1, c2, c3, margin),
    }


def pick_best(selection):
    """Pre-specified selection: best non-ML among B* keys, best ML among *_state_obs keys of
    the selection dict only. Secondary comparators are never written into it."""
    if any(k.startswith("B7w") for k in selection):
        raise ValueError("secondary comparator B7w must not enter the selection pool")
    non = [k for k in selection if k.startswith("B")]
    mls = [k for k in selection if k.endswith("_state_obs")]
    return min(non, key=selection.get), min(mls, key=selection.get)


def task_main(run_dir, workers):
    obs, cv, cfg, tr, curves_of = prepare(run_dir)
    X, y, groups = obs.iloc[tr], obs["L1"].to_numpy()[tr], obs["unit_id"].to_numpy()[tr]
    folds, fl = V.P(cfg, "evaluation.cv_folds"), V.P(cfg, "ber.regret_floor")
    report = {"selection": {}, "hyperparameters": {}, "lut_empty_cells": {}, "lut_filled_rows_share": {}, "budget": {}}
    fitted = {}

    for name, (make, needs_cv) in nonml_methods(cfg, curves_of).items():
        score = cv_regret(make, X, y, groups, curves_of, folds, fl) if needs_cv else float(regret(make().predict(X), curves_of(X.index), fl).mean())
        report["selection"][name] = score
        fitted[name] = make().fit(X, y)
        if isinstance(fitted[name], LUT):
            report["lut_empty_cells"][name] = [fitted[name].n_empty, fitted[name].n_cells]
    b2, b3 = fitted["B2_zhang_midpoint"].predict(obs), fitted["B3_centre"].predict(obs)
    if not np.array_equal(b2, b3):
        raise AssertionError("B2 (Zhang midpoint) must equal B3 for the window [i_min, C_hat]")

    t = time.time()
    fitted["B6b_model_grid"] = ModelGrid(cfg).fit()
    report["budget"]["B6b_simulator_calls"] = fitted["B6b_model_grid"].n_calls
    report["selection"]["B6b_model_grid"] = float(regret(fitted["B6b_model_grid"].predict(X), curves_of(X.index), V.P(cfg, "ber.regret_floor")).mean())
    # B7 selection score out of sample: residual SDs from 80 % of the training units,
    # scored on states of the held-out 20 %. The final B7 is refitted on all training rows.
    units = np.unique(groups)
    hold = np.random.default_rng(0).choice(units, max(1, len(units) // 5), replace=False)
    in_hold = np.isin(groups, hold)
    sub = X[in_hold].groupby("unit_id", group_keys=False).head(V.P(cfg, "evaluation.b7_select_states_per_unit"))
    b7_sel = B7(cfg, workers).fit(X[~in_hold])
    r7 = regret(b7_sel.predict(sub), curves_of(sub.index), fl)
    report["selection"]["B7_bayes"] = float(r7.mean())
    # B7 is scored on fewer rows than the other non-ML candidates: report its SE
    report["selection_b7_rows"] = len(sub)
    report["selection_b7_unit_boot_se"] = float(np.std(cluster_boot(sub["unit_id"].to_numpy(), r7, B=V.P(cfg, "evaluation.bootstrap_B"))[1], ddof=1))
    report["budget"]["B7_selection_holdout_units"] = hold.tolist()
    b7 = B7(cfg, workers).fit(X)
    report["budget"]["B7_simulator_calls_per_state"] = b7.n_calls_per_state
    report["budget"]["B7_obs_sd"], report["budget"]["B7_obs_mean_residual"] = b7.sig, b7.bias
    fitted["B7_bayes"] = b7
    # Secondary comparator B7w: B7 with the waveform-informed ceiling. Same
    # held-out selection protocol, reported under "secondary"; it does not enter the selection pool.
    b7w_sel = B7(cfg, workers, "wave").fit(X[~in_hold])
    report["secondary"] = {"B7w_selection": float(regret(b7w_sel.predict(sub), curves_of(sub.index), fl).mean())}
    b7w = fitted["B7w_bayes_wave"] = B7(cfg, workers, "wave").fit(X)
    # Diagnostic (true C used for scoring only): rows read as clipped although C exceeds w_max by > 3 LSB
    lsb = V.P(cfg, "sensing.adc_fs_frac") * obs["I_rated"].to_numpy() / 2 ** V.P(cfg, "sensing.adc_bits")
    false_clip = wave_clipped(obs, b7w.q, b7w.c_k) & (obs["C"].to_numpy() - obs["w_max"].to_numpy() > 3 * lsb)
    report["secondary"] |= {"B7w_clip_k_frac_I_rated": b7w.c_k, "B7w_adc_offset_frac_I_rated": b7w.c_off,
                            "B7w_false_clip_share": {s: float(false_clip[(obs["split"] == s).to_numpy()].mean()) for s in TEST_SPLITS}}
    report["runtime_model_based_s"] = time.time() - t

    for fs in ("state", "state_obs"):
        for name in GRIDS:
            hp, scores = tune_ml(name, V.FEATURES[fs], X, y, groups, curves_of, folds, fl)
            key = f"ML_{name}_{fs}"
            report["hyperparameters"][key] = {"chosen": hp, "cv_regret_by_grid": scores}
            report["selection"][key] = float(min(scores))
            fitted[key] = ML(name, V.FEATURES[fs], hp).fit(X, y)

    test = np.flatnonzero(obs["split"].isin(TEST_SPLITS))
    bhat = {}
    for k, m in fitted.items():
        bhat[k] = np.full(len(obs), np.nan)  # B7 costs 216 analytic curves per row: test rows only
        bhat[k][test] = m.predict(obs.iloc[test])
        if isinstance(m, LUT):
            report["lut_filled_rows_share"][k] = {s: float(m.last_filled[(obs["split"].to_numpy()[test] == s)].mean())
                                                  for s in TEST_SPLITS if (obs["split"] == s).any()}
    best_non, best_ml = pick_best(report["selection"])
    report["best_non_ml"], report["best_ml_state_obs"] = best_non, best_ml
    report["best_ml_state_only"] = min((k for k in report["selection"] if k.endswith("_state")), key=report["selection"].get)

    report["test"], per_row = score_test(obs, cv, bhat, cfg)
    report["confirmatory"] = bootstrap_decision(obs, cv, bhat, per_row, best_non, best_ml, cfg)
    # paired Delta = regret - regret(other), other = best ML and B7 (positive = other better), per split
    for other in (best_ml, best_non):
        report["secondary"][f"delta_B7w_bayes_wave_minus_{other}"] = {
            s: cluster_boot(obs.loc[obs["split"] == s, "unit_id"].to_numpy(), per_row[(s, "B7w_bayes_wave")], per_row[(s, other)],
                            V.P(cfg, "evaluation.bootstrap_B"))[0] for s in TEST_SPLITS if (s, other) in per_row}
    report["variance_decomposition_r2_test_id"] = {k: report["test"]["test_id"][k]["r2_vs_oracle"]
                                                   for k in ("B3_centre", "B4_lut1d_C", "B5_lut2d_Ir_T", "B6b_model_grid")}
    report["model_seed_sd"] = seed_sd(obs, cv, cfg, report, X, y)
    report["L2"] = task_l2(obs, cv, cfg, tr, report["hyperparameters"]["ML_hgb_state_obs"]["chosen"])
    report["L2_38"] = task_l2(obs, cv, cfg, tr, report["hyperparameters"]["ML_hgb_state_obs"]["chosen"], "L2_38", "ber.fec_alt")
    report["cost"] = cost(fitted, obs, cfg)
    write(run_dir, "main", report)
    return report


def seed_sd(obs, cv, cfg, report, X, y):
    """Mean regret of each ML model over model seeds on every test split; SD with ddof = 1 (all
    splits). The confirmatory Delta uses seed 0."""
    fl = V.P(cfg, "ber.regret_floor")
    sets = {s: np.flatnonzero(obs["split"] == s) for s in TEST_SPLITS if (obs["split"] == s).any()}
    cur = {s: curve_rows(cv, "oracle", r) for s, r in sets.items()}
    out = {}
    for key, h in report["hyperparameters"].items():
        _, name, fs = key.split("_", 2)
        r = {s: [] for s in sets}
        for seed in range(V.P(cfg, "evaluation.model_seeds")):
            m = ML(name, V.FEATURES[fs], h["chosen"], seed).fit(X, y)
            for s, rows in sets.items():
                r[s].append(float(regret(m.predict(obs.iloc[rows]), cur[s], fl).mean()))
        out[key] = {s: {"mean": float(np.mean(v)), "sd_ddof1": sd1(v), "min": min(v), "max": max(v), "by_seed": v} for s, v in r.items()}
    return out


def backoff_beta(shortfall, feasible, q):
    """Back-off in I_rated units: the q-quantile of the out-of-fold shortfall (label - prediction) /
    I_rated over training rows whose label is feasible (where L2 is infeasible it is the argmin, a
    different target). Never negative; raises instead of returning a silent 0."""
    s = np.asarray(shortfall, float)[np.asarray(feasible, bool)]
    if not len(s) or not np.all(np.isfinite(s)):
        raise ValueError(f"back-off needs finite shortfalls on feasible rows ({len(s)} rows)")
    return float(max(0.0, np.quantile(s, q)))


def l2_betas(X, y, target, makers, methods, groups, folds):
    """backoff_beta per method from out-of-fold training predictions (methods in `makers` are refitted
    per GroupKFold fold; the others, B6b, see no training data) on the `{target}_feasible` rows."""
    feas = X[f"{target}_feasible"].to_numpy(bool)
    beta = {}
    for name, m in methods.items():
        pred = np.empty(len(X))
        if name in makers:
            for a, b in GroupKFold(folds).split(X, y, groups):
                pred[b] = makers[name]().fit(X.iloc[a], y[a]).predict(X.iloc[b])
        else:
            pred = m.predict(X)
        beta[name] = backoff_beta((y - pred) / X["I_rated"].to_numpy(), feas, L2_BACKOFF_Q)
    return beta


def task_l2(obs, cv, cfg, tr, hgb_hp, target="L2", fec_key="ber.fec_hd"):
    """Secondary label L2 (minimum bias meeting HD-FEC 4.5e-3): excess electrical power vs the oracle
    L2 bias and FEC violation rate, for the state-based LUT, B6b and HGB (state_obs). HGB reuses the
    hyperparameters tuned for L1 (no separate L2 tuning: L2 is secondary). target="L2_38" with
    fec_key="ber.fec_alt" gives the 3.8e-3 sensitivity column.

    Back-off variant: L2 is the
    FEC boundary, so a prediction just below it violates FEC. Each method also commands
    prediction + beta * I_rated, beta = the L2_BACKOFF_Q quantile of the training shortfall
 / I_rated on L2-feasible training rows (backoff_beta), out of fold (GroupKFold
    by unit) for the fitted methods."""
    X, y = obs.iloc[tr], obs[target].to_numpy()[tr]
    groups, folds = obs["unit_id"].to_numpy()[tr], V.P(cfg, "evaluation.cv_folds")
    makers = {
        "B6a_lut_state": lambda: LUT(V.FEATURES["state"], V.P(cfg, "evaluation.lut_state_bins")),
        "ML_hgb_state_obs": lambda: ML("hgb", V.FEATURES["state_obs"], hgb_hp),
    }
    methods = {k: f().fit(X, y) for k, f in makers.items()}
    methods["B6b_model_grid"] = ModelGrid(cfg, target).fit()
    beta = l2_betas(X, y, target, makers, methods, groups, folds)
    fec = np.log10(V.P(cfg, fec_key))
    out = {"fec": V.P(cfg, fec_key), "B6a_lut_empty_cells": [methods["B6a_lut_state"].n_empty, methods["B6a_lut_state"].n_cells], "hgb_hyperparameters": hgb_hp,
           "backoff_quantile": L2_BACKOFF_Q, "backoff_frac_I_rated": beta}
    for split in TEST_SPLITS:
        rows = np.flatnonzero(obs["split"] == split)
        if not len(rows):
            continue
        cur = curve_rows(cv, "oracle", rows)
        pel = [(c.b, cv["oracle_pelec"][r][~np.isnan(cv["oracle_b"][r])]) for r, c in zip(rows, cur)]
        feas = obs[f"O_{target}_feasible"].to_numpy()[rows].astype(bool)
        p0 = np.array([np.interp(o, b, p) for o, (b, p) in zip(obs[f"O_{target}"].to_numpy()[rows], pel)])
        out[split] = {}
        Ir = obs["I_rated"].to_numpy()[rows]

        def metrics(bh):
            ph = np.array([np.interp(np.clip(x, b[0], b[-1]), b, p) for x, (b, p) in zip(bh, pel)])
            return {"excess_power_pct_feasible": float(100 * np.mean((ph - p0)[feas] / p0[feas])) if feas.any() else None,
                    "fec_violation_rate_feasible": float(np.mean(lb_at(bh, cur)[feas] > fec)) if feas.any() else None}

        for name, m in methods.items():
            bh = m.predict(obs.iloc[rows])
            out[split][name] = {"lut_filled_rows_share": float(m.last_filled.mean()) if isinstance(m, LUT) else None} | metrics(bh)
            out[split][name]["backoff"] = metrics(bh + beta[name] * Ir)
    return out


def cost(fitted, obs, cfg):
    """Single-threaded batch-1 latency (median, P99) and serialized size."""
    n = V.P(cfg, "evaluation.latency_calls")
    one = obs.iloc[[0]]
    out = {}
    with threadpool_limits(1):
        for name, m in fitted.items():
            if isinstance(m, ML) and hasattr(m.m, "n_jobs"):
                m.m.n_jobs = 1
            calls = 50 if isinstance(m, B7) else n
            if isinstance(m, B7):
                m.workers = 1
            for _ in range(min(100, calls)):
                m.predict(one)
            ts = []
            for _ in range(calls):
                t = time.perf_counter()
                m.predict(one)
                ts.append(time.perf_counter() - t)
            buf = io.BytesIO()
            # serialise what a deployed controller would store: the model, the table, or its fitted constants
            state = m.m if isinstance(m, ML) else m.interp if hasattr(m, "interp") else {
                k: v for k, v in vars(m).items() if k in ("c", "k", "sig", "bias", "ceiling", "q", "c_off", "c_k")}
            joblib.dump(state, buf)
            out[name] = {"latency_median_us": 1e6 * float(np.median(ts)), "latency_p99_us": 1e6 * float(np.quantile(ts, 0.99)),
                         "calls": calls, "bytes": buf.getbuffer().nbytes}
            if isinstance(m, LUT):
                out[name]["lut_bytes_float32"] = 4 * m.n_cells
    out["_cpu"] = os.uname().machine + " / " + os.uname().sysname
    return out


def wave_tolerance(wins):
    """The largest lambda_W such that the wave family matches or beats thermal at it and at every
    smaller lambda_W (stress columns included); None if it loses at the smallest. wins: (lambda_W, bool)."""
    out = None
    for lw, ok in sorted(wins):
        if not ok:
            break
        out = lw
    return out


def task_sensing(run_dir, workers):
    """6x6 lambda_T x lambda_W map plus the LAMBDA_W_STRESS columns; per cell retrain and refit on noisy rows (deployment-matched) and
    the train-clean / test-noisy secondary. Best per family by holdout regret on 20 % of the
    training units. Hyperparameters fixed to those chosen in results_main.json."""
    main = json.loads((Path(run_dir) / "results_main.json").read_text())
    df, cv, cfg = load(run_dir)
    grid = V.P(cfg, "sensing.lambda_grid")
    if grid[0] != 0:
        raise ValueError("sensing.lambda_grid must start at 0: the (0, 0) cell provides the train-clean fit")
    tr_all = np.flatnonzero(df["split"] == "train")
    units = np.unique(df["unit_id"].to_numpy()[tr_all])
    hold_units = np.random.default_rng(0).choice(units, max(1, len(units) // 5), replace=False)
    in_hold = np.isin(df["unit_id"].to_numpy()[tr_all], hold_units)
    fit_rows, hold_rows = tr_all[~in_hold], tr_all[in_hold]
    test_rows = np.flatnonzero(df["split"] == "test_id")
    lab = curve_rows(cv, "label", hold_rows)
    ora = curve_rows(cv, "oracle", test_rows)
    hp = {k: v["chosen"] for k, v in main["hyperparameters"].items()}

    def families(o_fit):
        X, y = o_fit.iloc[fit_rows], o_fit["L1"].to_numpy()[fit_rows]
        th = {f"ML_{n}_thermal": ML(n, V.FEATURES["thermal_family"], hp[f"ML_{n}_state_obs"]).fit(X, y) for n in ("hgb", "ridge")}
        th["B3_centre"] = Rule(midpoint)
        th["B5_lut2d_Ir_T"] = LUT(["I_rated", "T_ntc_hat"], V.P(cfg, "evaluation.lut2d_bins")).fit(X, y)
        th["B6a_lut_state"] = LUT(V.FEATURES["state"][:-1], V.P(cfg, "evaluation.lut_state_bins")).fit(X, y)
        wa = {f"ML_{n}_wave": ML(n, V.FEATURES["wave_family"], hp[f"ML_{n}_state_obs"]).fit(X, y) for n in ("hgb", "ridge")}
        return th, wa

    # On probe-capped rows b_p = 0.9 x the lambda_T = 1 NTC ceiling reading, so at lambda_T
    # not in {0, 1} b_p and T_ntc_hat together recover the true T_ntc. Capped shares and uncapped-row
    # regrets are reported per cell as the sensitivity.
    capped = df["b_p"].to_numpy() < df["probe_frac"].to_numpy() * df["I_rated"].to_numpy() - 1e-12
    capped_test = capped[test_rows]
    cells, clean_fit = [], None
    for lT in grid:
        for lW in (*grid, *LAMBDA_W_STRESS):  # stress columns beyond the pre-specified grid
            o = V.observe(cfg, df, lT, lW, 1.0)
            if lT == 0 and lW == 0:
                clean_fit = families(o)
            cell = {"lambda_T": lT, "lambda_W": lW}
            for mode, fams in (("matched", families(o)), ("train_clean", clean_fit)):
                if fams is None:
                    continue
                best = {}
                for fam, ms in zip(("thermal", "wave"), fams):
                    hold = {k: regret(m.predict(o.iloc[hold_rows]), lab, V.P(cfg, "ber.regret_floor")).mean() for k, m in ms.items()}
                    k = min(hold, key=hold.get)
                    best[fam] = (k, regret(ms[k].predict(o.iloc[test_rows]), ora, V.P(cfg, "ber.regret_floor")))
                boot, _ = cluster_boot(o["unit_id"].to_numpy()[test_rows], best["wave"][1], best["thermal"][1], V.P(cfg, "evaluation.bootstrap_B"))
                cell[mode] = ({f"best_{f}": best[f][0] for f in best} | {f"regret_{f}": float(best[f][1].mean()) for f in best}
                              | {f"regret_{f}_uncapped": float(best[f][1][~capped_test].mean()) for f in best} | {"wave_minus_thermal": boot})
            cells.append(cell)
    cross, tol = {}, {}
    for lT in grid:
        ok = [c["lambda_W"] for c in cells if c["lambda_T"] == lT and c["matched"]["regret_wave"] <= c["matched"]["regret_thermal"]]
        cross[str(lT)] = min((w for w in ok if w in grid), default=None)  # pre-specified grid only
        tol[str(lT)] = wave_tolerance([(c["lambda_W"], c["matched"]["regret_wave"] <= c["matched"]["regret_thermal"])
                                       for c in cells if c["lambda_T"] == lT])
    link = []
    for lL in grid:
        o = V.observe(cfg, df, 1.0, 1.0, lL)
        X, y = o.iloc[fit_rows], o["L1"].to_numpy()[fit_rows]
        m = ML("hgb", V.FEATURES["state_obs"], hp["ML_hgb_state_obs"]).fit(X, y)
        link.append({"lambda_link": lL, "ML_hgb_state_obs": float(regret(m.predict(o.iloc[test_rows]), ora, V.P(cfg, "ber.regret_floor")).mean())})
    # At lambda_T > 1 Tj_Rth_hat carries (lambda_T - 1) * u_ntc_offset without noise.
    # Ablation: thermal-family HGB with and without Tj_Rth_hat; positive = the feature lowers regret.
    no_rth = [c for c in V.FEATURES["thermal_family"] if c != "Tj_Rth_hat"]
    ablation = []
    for lT in (g for g in grid if g > 1):
        o = V.observe(cfg, df, lT, 1.0, 1.0)
        X, y = o.iloc[fit_rows], o["L1"].to_numpy()[fit_rows]
        r = {k: regret(ML("hgb", cols, hp["ML_hgb_state_obs"]).fit(X, y).predict(o.iloc[test_rows]), ora, V.P(cfg, "ber.regret_floor"))
             for k, cols in (("with", V.FEATURES["thermal_family"]), ("without", no_rth))}
        boot, _ = cluster_boot(o["unit_id"].to_numpy()[test_rows], r["without"], r["with"], V.P(cfg, "evaluation.bootstrap_B"))
        ablation.append({"lambda_T": lT, "regret_with": float(r["with"].mean()), "regret_without": float(r["without"].mean()),
                         "without_minus_with": boot})
    write(run_dir, "sensing", {"cells": cells, "crossover_lambda_W_by_lambda_T": cross, "wave_tolerance_lambda_W_by_lambda_T": tol, "probe_capped_share": {"fit": float(capped[fit_rows].mean()), "test_id": float(capped_test.mean())}, "lambda_W_stress": list(LAMBDA_W_STRESS), "link_sweep": link,
                               "tj_rth_ablation": ablation, "holdout_units": hold_units.tolist()})


def task_learning(run_dir, workers):
    main = json.loads((Path(run_dir) / "results_main.json").read_text())
    obs, cv, cfg, tr, curves_of = prepare(run_dir)
    test = np.flatnonzero(obs["split"] == "test_id")
    ora = curve_rows(cv, "oracle", test)
    units = np.unique(obs["unit_id"].to_numpy()[tr])
    out = []
    for n_units in V.P(cfg, "evaluation.learning_units"):
        for seed in range(V.P(cfg, "evaluation.model_seeds")):
            pick = np.random.default_rng(seed).choice(units, min(n_units, len(units)), replace=False)
            rows = tr[np.isin(obs["unit_id"].to_numpy()[tr], pick)]
            X, y = obs.iloc[rows], obs["L1"].to_numpy()[rows]
            ms = {k: ML(k.split("_")[1], V.FEATURES["state_obs"], v["chosen"], seed)
                  for k, v in main["hyperparameters"].items() if k.endswith("_state_obs")}
            ms["B6a_lut_state"] = LUT(V.FEATURES["state"][:-1], V.P(cfg, "evaluation.lut_state_bins"))
            for k, m in ms.items():
                r = regret(m.fit(X, y).predict(obs.iloc[test]), ora, V.P(cfg, "ber.regret_floor"))
                out.append({"units": int(n_units), "seed": seed, "method": k, "mean_regret": float(r.mean())})
    write(run_dir, "learning", {"rows": out})


def _b8_row(args):
    cfg, row, iters = args
    st, dev, _ = V.build(cfg, row)
    quad = V.quadrature(cfg)
    phi = (np.sqrt(5) - 1) / 2
    calls = [0]

    def f(x):
        d = V.draws(cfg, V.rng_for(cfg, V.STREAM_B8, row["state_id"], calls[0]), V.P(cfg, "ofdm.R_probe"), st["M"])
        calls[0] += 1
        return V.mc_curve(cfg, st, dev, np.array([x]), d, quad)[0][0]

    lo, hi = V.P(cfg, "led.i_min_frac") * row["I_rated"], row["C_hat"]
    x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
    f1, f2 = f(x1), f(x2)
    out = {}
    for it in range(1, max(iters) + 1):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = f(x2)
        if it in iters:
            out[it] = ((lo + hi) / 2, calls[0])
    return out


def task_b8(run_dir, workers):
    obs, cv, cfg, _, _ = prepare(run_dir)
    rows = np.flatnonzero(obs["split"] == "test_id")[: V.P(cfg, "evaluation.b8_test_states")]
    iters = V.P(cfg, "evaluation.b8_iterations")
    jobs = [(cfg, r, iters) for r in obs.iloc[rows].to_dict("records")]
    with Pool(workers) as pool:
        res = pool.map(_b8_row, jobs, chunksize=4)
    cur = curve_rows(cv, "oracle", rows)
    out = []
    for it in iters:
        bh = np.array([r[it][0] for r in res])
        out.append({"iterations": it, "probe_evaluations": int(res[0][it][1]), "probe_frames": int(res[0][it][1] * V.P(cfg, "ofdm.R_probe")),
                    "mean_regret": float(regret(bh, cur, V.P(cfg, "ber.regret_floor")).mean())})
    write(run_dir, "b8", {"frontier": out, "n_states": len(rows)})


# Elsewhere B7 and B6b are built from the data-generating LED form. Misspecification arm, declared in advance: they get a wrong LED model (controller cfg only; data, oracle curves and the
# ML comparator are unchanged). Keyed by the run's true led.model. Form errors swap the transfer
# function; datasheet errors move the controller's nominal kappa_tilde to the ends of the sourced
# range 0.27-0.36 (config.json unit_spread.kappa_tilde note), against true units in 0.30 +- 15 %.
MISSPEC = {
    "quadratic": {"true_form": {},  # identity control: B6b only, must reproduce results_main
                  "form_tsonev_cubic": {"led.model": "tsonev_cubic"},
                  "datasheet_kappa_0.27": {"led.kappa_tilde": 0.27},
                  "datasheet_kappa_0.36": {"led.kappa_tilde": 0.36}},
    "tsonev_cubic": {"true_form": {}, "form_quadratic": {"led.model": "quadratic"}},
}


def misspec_cfg(cfg, over):
    """Copy of cfg with the controller's LED parameters overridden; cfg itself is not touched."""
    c = copy.deepcopy(cfg)
    for k, v in over.items():
        V.set_param(c, k, v)
    return c


def task_misspec(run_dir, workers):
    """B7 and B6b under each MISSPEC variant vs the best ML of results_main (refitted, seed 0, and
    checked to reproduce results_main), paired unit bootstrap per test split."""
    obs, cv, cfg, tr, _ = prepare(run_dir)
    X, y = obs.iloc[tr], obs["L1"].to_numpy()[tr]
    main = json.loads((Path(run_dir) / "results_main.json").read_text())
    best_ml = main["best_ml_state_obs"]
    _, name, fs = best_ml.split("_", 2)
    test = np.flatnonzero(obs["split"].isin(TEST_SPLITS))
    bhat = {best_ml: np.full(len(obs), np.nan)}
    bhat[best_ml][test] = ML(name, V.FEATURES[fs], main["hyperparameters"][best_ml]["chosen"]).fit(X, y).predict(obs.iloc[test])
    variants = MISSPEC[V.P(cfg, "led.model")]
    for v, over in variants.items():
        c = misspec_cfg(cfg, over)
        # B7w (secondary) is built on the same nominal_device path
        methods = [(f"B6b_model_grid@{v}", ModelGrid(c))]
        if over:
            methods += [(f"B7_bayes@{v}", B7(c, workers)), (f"B7w_bayes_wave@{v}", B7(c, workers, "wave"))]
        for key, m in methods:
            bhat[key] = np.full(len(obs), np.nan)
            bhat[key][test] = m.fit(X).predict(obs.iloc[test])
    test_rep, per_row = score_test(obs, cv, bhat, cfg)
    splits = [s for s in TEST_SPLITS if s in test_rep]
    for s in splits:
        got, want = test_rep[s][best_ml]["mean_regret"], main["test"][s][best_ml]["mean_regret"]
        if not np.isclose(got, want, rtol=1e-9, atol=0):
            raise AssertionError(f"refitted {best_ml} on {s}: {got} != results_main {want}")
        got, want = test_rep[s]["B6b_model_grid@true_form"]["mean_regret"], main["test"][s]["B6b_model_grid"]["mean_regret"]
        if not np.isclose(got, want, rtol=1e-9, atol=0):
            raise AssertionError(f"B6b@true_form on {s}: {got} != results_main {want}")
    B = V.P(cfg, "evaluation.bootstrap_B")
    rep = {"declared": variants, "best_ml_state_obs": best_ml, "results_main_eval_sha": main["_eval_provenance"]["sha"],
           "confirmatory_delta_true_form": main["confirmatory"]["delta"],
           "B7w_delta_true_form": main["secondary"].get(f"delta_B7w_bayes_wave_minus_{best_ml}"),
           "delta_minus_best_ml": {}, "test": test_rep}
    for key in bhat:
        if key != best_ml:  # positive = ML lower regret
            # descriptive: 4 splits x up to 10 keys, no multiplicity adjustment
            rep["delta_minus_best_ml"][key] = {s: cluster_boot(obs.loc[obs["split"] == s, "unit_id"].to_numpy(), per_row[(s, key)],
                                                               per_row[(s, best_ml)], B)[0] for s in splits}
    write(run_dir, "misspec", rep)
    return rep


# Device-shift (mirror) arm: every controller, learned and model-based, is
# built for the nominal population of run_dir and deployed on shifted twins of run_dir's test_id units
# (same units, states and noise draws; only the LED changes), generated by sweeps.py --twins.
# Declared before the run: every led.*-only sweep value (joint run excluded, it also changes noise.rho)
# plus the datasheet kappa_tilde 0.27 / 0.36. Deploying on the sweep runs' test_id units would score
# ML on its own training units, hence the twins.
SHIFT = {"led.model=tsonev_cubic": {"led.model": "tsonev_cubic"},
         "led.kappa_tilde=0": {"led.kappa_tilde": 0}, "led.kappa_tilde=0.15": {"led.kappa_tilde": 0.15},
         "led.kappa_tilde=0.27": {"led.kappa_tilde": 0.27}, "led.kappa_tilde=0.36": {"led.kappa_tilde": 0.36},
         "led.kappa_tilde=0.42": {"led.kappa_tilde": 0.42},
         "led.delta=0": {"led.delta": 0}, "led.delta=0.001": {"led.delta": 0.001}, "led.delta=0.003": {"led.delta": 0.003},
         "led.i_min_frac=0.0": {"led.i_min_frac": 0.0}, "led.i_min_frac=0.05": {"led.i_min_frac": 0.05},
         "led.f3dB_over_B=0.5": {"led.f3dB_over_B": 0.5}, "led.f3dB_over_B=null": {"led.f3dB_over_B": None},
         # an LED form outside every controller's set, with per-unit spread
         "led.model=tsonev_cubic_spread": {"led.model": "tsonev_cubic_spread"}}
TWIN_PREFIX = "twin-"
# Declared before the run. Wave-free comparator: the best ML refitted on the thermal family
# (no waveform tap, so no w_min), every seed, every deploy; B7 with a wide uniform kappa_tilde prior
# (0 to 0.48, the twins' whole range) on the identity and kappa_tilde deploys. Read-out: commanded biases of
# READOUT_KEYS re-evaluated by Monte Carlo at the exact bias, on the oracle's realisation (PCHIP error
# alone) and on an independent one (STREAM_READOUT). Also FEC-feasible means and SNR slopes.
# 0.02 spacing, close to the default B7's 0.018, so prior width is not confounded with grid coarseness
#; 25 x 6 x 6 = 900 analytic curves per state
B7_WIDE_KT = tuple(np.round(np.linspace(0.0, 0.48, 25), 2))
B7_WIDE_ON = ("identity",) + tuple(k for k in SHIFT if k.startswith("led.kappa_tilde="))
SNR_SLOPE_DB = 1.0
# Declared before the run. Primary: B7_bayes - best ML (seed mean); secondary:
# B6b, B7w, B6a, B6b@device (ModelGrid of the true device); every other key is supplementary context.
SHIFT_PRIMARY = "B7_bayes"
SHIFT_SECONDARY = ["B6b_model_grid", "B7w_bayes_wave", "B6a_lut_state", "B6b_model_grid@device"]
# columns a twin shares with run_dir's test_id; the swept LED latent may differ
TWIN_SAME = ["unit_id", "I_rated", "sigma_frac", "T_hs", "SNR_ref_dB", "M", "probe_frac", "T_ntc", "C"]
TWIN_LATENT = {"led.kappa_tilde": "u_kt", "led.delta": "u_delta"}


def twin_check(run_dir, d, cfg, ref, prefix=TWIN_PREFIX):
    """Pre-flight for one twin: clean provenance, config = nominal + its shift only,
    and the same units, states and non-LED latents as run_dir's test_id rows (ref, indexed by state_id)."""
    prov = json.loads((d / "provenance.json").read_text())
    dep = d.name[len(prefix):]
    if prov["git"]["dirty"] or prov["diagnostics"]["splits"] != ["test_id"]:
        raise AssertionError(f"{d.name}: dirty or not a test_id twin")
    if prov["config"] != misspec_cfg(cfg, SHIFT[dep]):
        raise AssertionError(f"{d.name}: config differs from nominal by more than {SHIFT[dep]}")
    tw = pd.read_csv(d / "states.csv", low_memory=False).set_index("state_id")
    if not tw["tj_ok"].all():  # a Tj drop would unpair rows hours later
        raise AssertionError(f"{d.name}: {int((~tw['tj_ok']).sum())} rows dropped by the Tj check")
    if not tw.index.equals(ref.index):
        raise AssertionError(f"{d.name}: states are not the test_id states of {Path(run_dir).name}")
    skip = {TWIN_LATENT.get(k) for k in SHIFT[dep]}
    cols = TWIN_SAME + [c for c in ref if c.startswith("u_") and c not in skip]
    bad = [c for c in cols if not np.array_equal(tw[c].to_numpy(), ref[c].to_numpy())]
    if bad:
        raise AssertionError(f"{d.name}: twin columns differ from {Path(run_dir).name}: {bad}")


def pmap(fn, jobs, workers):
    if workers > 1:
        with Pool(workers) as pool:
            return pool.map(fn, jobs, chunksize=8)
    return [fn(j) for j in jobs]


SLOPE_MIN = 0.01  # dec/dB: below this a state is distortion-limited and no SNR increase recovers the regret


def _slope_row(args):
    cfg, row, b = args
    st, dev, _ = V.build(cfg, row)
    b, quad = np.asarray(b, float), V.quadrature(cfg)
    lb = [V.analytic_curve(cfg, st | {"snr_db": st["snr_db"] + x}, dev, b, quad)["lb"] for x in (0.0, SNR_SLOPE_DB)]
    return np.stack([lb[0], (lb[0] - lb[1]) / SNR_SLOPE_DB])


def snr_slope(cfg, df, bh, workers=1):
    """Decades of BER gained per dB of SNR_ref at each row's biases bh (rows x biases), true device,
    Bussgang analytic model. regret / slope at the commanded bias is the first-order SNR penalty in dB:
    the SNR increase that brings the controller's BER down to the oracle's.
    Returns (analytic log10 BER, slope), each rows x biases; the first is a read-out under a second
    BER model."""
    out = np.array(pmap(_slope_row, [(cfg, r, b) for r, b in zip(df.to_dict("records"), bh)], workers))
    return out[:, 0], out[:, 1]


def penalty_db(reg, slope):
    """Per-row SNR penalty in dB; 0 where regret is 0, NaN (unresolvable) where slope < SLOPE_MIN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(reg <= 0, 0.0, np.where(slope >= SLOPE_MIN, reg / slope, np.nan))


def penalty_summary(p):
    ok = ~np.isnan(p)
    return quart(p[ok]) | {"mean": float(p[ok].mean()) if ok.any() else None, "n_unresolvable": int((~ok).sum())}


def _readout_row(args):
    cfg, row, b, stream = args
    st, dev, _ = V.build(cfg, row)
    d = V.draws(cfg, V.rng_for(cfg, stream, row["state_id"]), V.P(cfg, "ofdm.R_oracle"), st["M"])
    return V.mc_curve(cfg, st, dev, np.asarray(b, float), d, V.quadrature(cfg))[0]


def readout(cfg, df, bh, stream, workers=1):
    """Monte Carlo log10 BER at each row's biases bh (rows x biases), R_oracle frames from `stream`.
    With V.STREAM_ORACLE it reproduces the oracle curve's realisation exactly (nodes included)."""
    return np.array(pmap(_readout_row, [(cfg, r, b, stream) for r, b in zip(df.to_dict("records"), bh)], workers))


def quart(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "median": float(np.median(x)), "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75))} if len(x) else {"n": 0}


def task_slopes(run_dir, workers):
    """SNR-slope scale for the main and misspecification tables (per-row predictions there are not stored, so
    per-controller dB penalties come from task_shift's identity deploy on test_id): SNR slope at O_L1
    per test split, quartiles on FEC-feasible states, per-row values in slopes-rows.npz."""
    df, cv, cfg = load(run_dir)
    rows = np.flatnonzero(df["split"].isin(TEST_SPLITS))
    s = snr_slope(cfg, df.iloc[rows], df["O_L1"].to_numpy()[rows, None], workers)[1][:, 0]
    feas = np.nanmin(cv["oracle_lb"][rows], axis=1) <= np.log10(V.P(cfg, "ber.fec_hd"))
    split = df["split"].to_numpy()[rows]
    np.savez_compressed(Path(run_dir) / "slopes-rows.npz", state_id=df["state_id"].to_numpy()[rows], split=split, fec_ok=feas, snr_slope=s)
    write(run_dir, "slopes", {"snr_step_dB": SNR_SLOPE_DB, "slope_min": SLOPE_MIN,
                              "fec_relevant": {sp: quart(s[(split == sp) & feas]) for sp in TEST_SPLITS},
                              "all": {sp: quart(s[split == sp]) for sp in TEST_SPLITS}})


def prev_shift(run_dir, require):
    """The previous results_shift.json, snapshotted once as results_shift.prev-<sha7>.json so a restart after
    write() has overwritten the live file still checks against the old numbers.
    require=False only for a first run (smoke tests)."""
    snaps = sorted(run_dir.glob("results_shift.prev-*.json"))
    if len(snaps) > 1:
        raise RuntimeError(f"several snapshots {[s.name for s in snaps]}: keep the one to check against")
    if not snaps and (run_dir / "results_shift.json").exists():
        old = json.loads((run_dir / "results_shift.json").read_text())
        snaps = [run_dir / f"results_shift.prev-{old['_eval_provenance']['sha'][:7]}.json"]
        shutil.copy2(run_dir / "results_shift.json", snaps[0])
    if not snaps:
        if require:
            raise RuntimeError("no previous results_shift.json to reproduce; pass --first-run for a first run")
        return None, {}
    print(f"checking against {snaps[0].name}", flush=True)
    return snaps[0].name, json.loads(snaps[0].read_text())["by_deploy"]


def readout_preflight(cfg_d, df, cv, n=5):
    """The oracle-stream read-out at O_L1 reproduces the oracle minimum on n rows,
    checked for every deploy before hours of fitting."""
    rows = np.flatnonzero(df["split"] == "test_id")[:n]
    got = readout(cfg_d, df.iloc[rows], df["O_L1"].to_numpy()[rows, None], V.STREAM_ORACLE)[:, 0]
    if not np.allclose(got, np.nanmin(cv["oracle_lb"][rows], axis=1), rtol=0, atol=1e-9):
        raise AssertionError("oracle-stream read-out does not reproduce the oracle minimum")


def task_shift(run_dir, workers, prefix=TWIN_PREFIX, require_prev=True):
    """Controllers fitted on run_dir's training rows (nominal cfg) and scored on each SHIFT twin's
    test_id units against the twin's oracle. ML: every *_state_obs model at its chosen hp and every
    model seed. Per deploy, paired unit bootstraps (positive = ML lower regret):
    Delta = method - best ML (seed mean); the cost of shift, method(twin) - method(identity); and the
    difference in differences, Delta(twin) - Delta(identity). The identity
    deployment (run_dir's own test_id) must reproduce results_main (seed 0)."""
    run_dir, root = Path(run_dir), Path(run_dir).parent
    obs, cv, cfg, tr, curves_of = prepare(run_dir)
    X, y = obs.iloc[tr], obs["L1"].to_numpy()[tr]
    main = json.loads((run_dir / "results_main.json").read_text())
    best_ml, B = main["best_ml_state_obs"], V.P(cfg, "evaluation.bootstrap_B")
    df = load(run_dir)[0]
    ref = df[df["split"] == "test_id"].set_index("state_id")
    if set(obs["unit_id"].iloc[tr]) & set(ref["unit_id"]):
        raise AssertionError("deploy units overlap the training units")
    readout_preflight(cfg, df, load(run_dir)[1])
    for dep in SHIFT:  # pre-flight every twin before any fitting
        twin_check(run_dir, root / f"{prefix}{dep}", cfg, ref, prefix)
        df_t, cv_t, cfg_t = load(root / f"{prefix}{dep}")
        readout_preflight(cfg_t, df_t, cv_t)
    # a re-run must reproduce every number of the previous results_shift.json (keys present in both)
    prev_name, prev = prev_shift(run_dir, require_prev)
    checked = set()
    rows_dir = run_dir / "shift-rows"
    rows_dir.mkdir(exist_ok=True)

    fitted = {k: make().fit(X, y) for k, (make, _) in nonml_methods(cfg, curves_of).items()}
    fitted |= {"B6b_model_grid": ModelGrid(cfg).fit(), "B7_bayes": B7(cfg, workers).fit(X),
               "B7w_bayes_wave": B7(cfg, workers, "wave").fit(X)}
    b7_wide = B7(cfg, workers, kt_axis=B7_WIDE_KT).fit(X)
    seeds = range(V.P(cfg, "evaluation.model_seeds"))
    mls = [k for k in main["hyperparameters"] if k.endswith("_state_obs")]
    for k in mls:
        _, name, fs = k.split("_", 2)
        for s in seeds:
            fitted[f"{k}@seed{s}"] = ML(name, V.FEATURES[fs], main["hyperparameters"][k]["chosen"], s).fit(X, y)
    _, best_name, _ = best_ml.split("_", 2)
    th_ml = f"ML_{best_name}_thermal"  # same model and hyperparameters, no waveform features
    for s in seeds:
        fitted[f"{th_ml}@seed{s}"] = ML(best_name, V.FEATURES["thermal_family"], main["hyperparameters"][best_ml]["chosen"], s).fit(X, y)
    rep = {"declared": list(SHIFT), "primary": f"{SHIFT_PRIMARY} - {best_ml}@seedmean",
           "secondary": SHIFT_SECONDARY, "train_run": run_dir.name, "best_ml_state_obs": best_ml,
           "added_analyses": {"status": "post hoc, descriptive, no multiplicity adjustment",
                           "thermal_ml": th_ml, "b7_wide_kt_axis": list(B7_WIDE_KT), "b7_wide_on": list(B7_WIDE_ON),
                           "snr_step_dB": SNR_SLOPE_DB, "readout_streams": {"same_realisation": V.STREAM_ORACLE, "independent": V.STREAM_READOUT}},
           "results_main_eval_sha": main["_eval_provenance"]["sha"], "by_deploy": {}}
    ml_key, th_key = f"{best_ml}@seedmean", f"{th_ml}@seedmean"
    interp = [SHIFT_PRIMARY] + SHIFT_SECONDARY + ["B7_bayes_wide", th_key]
    # Contrasts (a, b): a - b per row, read out three ways
    contrasts = [(SHIFT_PRIMARY, ml_key), (SHIFT_PRIMARY, f"{best_ml}@seed0"), (SHIFT_PRIMARY, th_key), ("B7_bayes_wide", ml_key)]
    fl = np.log10(V.P(cfg, "ber.regret_floor"))
    ident = None
    for dep in ["identity"] + list(SHIFT):
        if dep == "identity":
            d, obs_d, cv_d, cfg_d = run_dir, obs, cv, cfg
        else:
            d = root / f"{prefix}{dep}"
            obs_d, cv_d, cfg_d, _, _ = prepare(d, belief=cfg)  # readings from the true device, nominal-model terms from cfg
        rows = np.flatnonzero(obs_d["split"] == "test_id")
        # score test_id rows only; below_i_min_share against the device's i_min
        sub = obs_d.iloc[rows].reset_index(drop=True)
        sub = sub.assign(i_min=V.P(cfg_d, "led.i_min_frac") * sub["I_rated"])
        cvs = {k: v[rows] for k, v in cv_d.items()}
        use = fitted | {"B6b_model_grid@device": ModelGrid(cfg_d).fit()}
        if dep in B7_WIDE_ON:
            use["B7_bayes_wide"] = b7_wide
        bhat = {k: m.predict(obs_d.iloc[rows]) for k, m in use.items()}
        t, pr = score_test(sub, cvs, bhat, cfg_d)
        r = {k: pr[("test_id", k)] for k in bhat}
        for k in mls + [th_ml]:
            r[f"{k}@seedmean"] = np.mean([r[f"{k}@seed{s}"] for s in seeds], axis=0)
        units, sid = sub["unit_id"].to_numpy(), sub["state_id"].to_numpy()
        fec_ok = np.nanmin(cvs["oracle_lb"], axis=1) <= np.log10(V.P(cfg_d, "ber.fec_hd"))
        if dep == "identity":
            for k in use:
                base = k.split("@")[0]
                if base not in main["test"]["test_id"]:  # keys added for the wide-prior B7 and the thermal ML
                    continue
                if k.endswith("@seed0") or "@" not in k:
                    got, want = t["test_id"][k]["mean_regret"], main["test"]["test_id"][base]["mean_regret"]
                    tol = 1e-6 if base.startswith("B7") else 1e-9  # grid argmin; covers a BLAS-order tie flip
                    if not np.isclose(got, want, rtol=tol, atol=0):
                        raise AssertionError(f"identity deployment {k}: {got} != results_main {want}")
            ident = (sid, r, fec_ok)
        elif not np.array_equal(sid, ident[0]):
            raise AssertionError(f"{d.name}: test_id rows not aligned with identity")  # pre-flight makes this unreachable
        rid, fec_id = ident[1], ident[2]

        # direct Monte Carlo at each commanded bias (clamped to the curve span, as Curve does), plus
        # O_L1 as a last column: on the oracle's realisation it must reproduce the oracle minimum exactly
        ro_keys = [k for k in bhat if k in (SHIFT_PRIMARY, "B7_bayes_wide") or k.startswith((f"{best_ml}@seed", f"{th_ml}@seed"))]
        lo_b, hi_b = np.nanmin(cvs["oracle_b"], axis=1), np.nanmax(cvs["oracle_b"], axis=1)
        bh = np.column_stack([np.clip(bhat[k], lo_b, hi_b) for k in ro_keys] + [sub["O_L1"].to_numpy()])
        direct = {name: readout(cfg_d, sub, bh, stream, workers)
                  for name, stream in (("same_realisation", V.STREAM_ORACLE), ("independent", V.STREAM_READOUT))}
        if not np.allclose(direct["same_realisation"][:, -1], np.nanmin(cvs["oracle_lb"], axis=1), rtol=0, atol=1e-9):
            raise AssertionError(f"{dep}: same-realisation read-out does not reproduce the oracle minimum")
        dl = {}  # floored direct log10 BER per key, seed means added
        for name, D in direct.items():
            dl[name] = {k: np.maximum(D[:, j], fl) for j, k in enumerate(ro_keys)}
            for k in (best_ml, th_ml):
                dl[name][f"{k}@seedmean"] = np.mean([dl[name][f"{k}@seed{s}"] for s in seeds], axis=0)
        # SNR slope at every controller's commanded bias (clamped as above) -> per-row dB penalty
        sl_keys = list(bhat)
        lb_an, slope = snr_slope(cfg_d, sub, np.column_stack([np.clip(bhat[k], lo_b, hi_b) for k in sl_keys]), workers)
        dl["analytic_model"] = {k: np.maximum(lb_an[:, j], fl) for j, k in enumerate(sl_keys)}
        for k in (best_ml, th_ml):
            dl["analytic_model"][f"{k}@seedmean"] = np.mean([dl["analytic_model"][f"{k}@seed{s}"] for s in seeds], axis=0)
        pdb = {k: penalty_db(r[k], slope[:, j]) for j, k in enumerate(sl_keys)}
        for k in mls + [th_ml]:  # NaN (unresolvable) in any seed propagates to the seed mean
            pdb[f"{k}@seedmean"] = np.mean([pdb[f"{k}@seed{s}"] for s in seeds], axis=0)
        np.savez_compressed(rows_dir / f"{dep}.npz", state_id=sid, unit_id=units, fec_ok=fec_ok, ro_keys=np.array(ro_keys), sl_keys=np.array(sl_keys), snr_slope=slope, analytic_lb=lb_an,
                            **{f"bhat:{k}": v for k, v in bhat.items()}, **{f"regret:{k}": v for k, v in r.items()},
                            **{f"direct:{n}": D for n, D in direct.items()})
        entry = {
            "n_units": int(len(np.unique(units))), "n_rows": int(len(rows)),
            "mean_regret": {k: float(v.mean()) for k, v in r.items()},
            "below_i_min_share": {k: t["test_id"][k]["below_i_min_share"] for k in bhat},
            "ml_seed_range": {k: [float(min(r[f"{k}@seed{s}"].mean() for s in seeds)),
                                  float(max(r[f"{k}@seed{s}"].mean() for s in seeds))] for k in mls},
            # descriptive, no multiplicity adjustment; only the primary and secondary keys are interpreted
            "delta_minus_best_ml": {k: cluster_boot(units, r[k], r[ml_key], B)[0] for k in r if k != ml_key},
            # the primary against each seed, so the seed-mean CI is not the only view
            "primary_by_seed": {s: cluster_boot(units, r[SHIFT_PRIMARY], r[f"{best_ml}@seed{s}"], B)[0] for s in seeds},
            # the primary against the wave-free ML
            "delta_minus_thermal_ml": {k: cluster_boot(units, r[k], r[th_key], B)[0] for k in interp if k in r and k != th_key},
        }
        # paired contrast a - b from the PCHIP oracle (as reported), and read out directly
        entry["readout"] = {}
        for a, b in contrasts:
            if a not in r:
                continue
            dp = r[a] - r[b]
            ds, di, da = (dl[n][a] - dl[n][b] for n in ("same_realisation", "independent", "analytic_model"))
            entry["readout"][f"{a} - {b}"] = {
                "pchip": cluster_boot(units, dp, B=B)[0], "same_realisation": cluster_boot(units, ds, B=B)[0],
                "independent": cluster_boot(units, di, B=B)[0],
                "interp_error": cluster_boot(units, ds - dp, B=B)[0], "readout_error": cluster_boot(units, di - dp, B=B)[0],
                "readout_error_row_rms": float(np.sqrt(np.mean((di - dp) ** 2))),
                # the same commanded biases scored by the Bussgang analytic BER instead of Monte Carlo EVM
                "analytic_model": cluster_boot(units, da, B=B)[0], "model_difference": cluster_boot(units, da - dp, B=B)[0]}
        # FEC-feasible states (oracle minimum meets HD-FEC); SNR slope quartiles there; dB scale
        mr_f = {k: float(v[fec_ok].mean()) for k, v in r.items()} if fec_ok.any() else {}
        entry["fec_relevant"] = {
            "n_rows": int(fec_ok.sum()), "n_units": int(len(np.unique(units[fec_ok]))), "mean_regret": mr_f,
            "snr_slope_dec_per_dB_at_bias": {k: quart(slope[fec_ok, j]) for j, k in enumerate(sl_keys)},
            "penalty_dB": {k: penalty_summary(v[fec_ok]) for k, v in pdb.items()},
            "delta_minus_best_ml": {k: cluster_boot(units[fec_ok], r[k][fec_ok], r[ml_key][fec_ok], B)[0]
                                    for k in interp if k in r and k != ml_key} if fec_ok.any() else None}
        if dep != "identity":
            entry["cost_of_shift"] = {k: cluster_boot(units, r[k], rid[k], B)[0] for k in interp + [ml_key] if k in rid and k in r}
            entry["did_delta"] = {k: cluster_boot(units, r[k] - r[ml_key], rid[k] - rid[ml_key], B)[0] for k in interp if k in rid and k in r}
            both = fec_ok & fec_id  # paired: rows FEC-feasible on the nominal and on the shifted device
            entry["fec_relevant"]["cost_of_shift_both_feasible"] = {
                k: cluster_boot(units[both], r[k][both], rid[k][both], B)[0] | {"n_rows": int(both.sum())}
                for k in interp + [ml_key] if k in rid and k in r} if both.any() else None
            ref_run = root / f"sweep-{dep}" / "results_main.json"
            entry["matched_ref_sweep"] = None  # unpaired context only (units 40-49, 40 training units)
            if ref_run.exists():
                dm = json.loads(ref_run.read_text())
                entry["matched_ref_sweep"] = {"best_ml": dm["best_ml_state_obs"],
                                                "mean_regret": {k: dm["test"]["test_id"][k]["mean_regret"] for k in bhat if k in dm["test"]["test_id"]}}
        for k, v in prev.get(dep, {}).get("mean_regret", {}).items():
            checked.add((dep, k))
            tol = 1e-6 if k.startswith("B7") else 1e-9
            if not np.isclose(entry["mean_regret"][k], v, rtol=tol, atol=0):
                raise AssertionError(f"{dep} {k}: {entry['mean_regret'][k]} != previous results_shift {v}")
        rep["by_deploy"][dep] = entry
        write(run_dir, "shift", rep)  # after every deploy: a crash keeps the finished ones
    missing = {(d, k) for d, e in prev.items() for k in e["mean_regret"]} - checked
    if missing:
        raise AssertionError(f"{len(missing)} previous (deploy, key) pairs were not reproduced, e.g. {sorted(missing)[:3]}")
    rep["reproduced_previous"] = {"file": prev_name, "pairs": len(checked)}
    write(run_dir, "shift", rep)
    return rep


# Post hoc, descriptive, no multiplicity adjustment.
# B7 calibrated on n characterised training units (0 = sensor spec only; 160 = results_main B7).
# the wide-prior B7 (B7_WIDE_KT, kappa_tilde only) on the two i_min twins it was not run on.
B7CAL_UNITS = (10, 20, 40)
B7_WIDE_IMIN = ("led.i_min_frac=0.0", "led.i_min_frac=0.05")


B7CAL_ZMAX = 1.5  # spec_z pre-flight bound for EVM, P_rx, Tj_Vf (Tj_Rth carries unmodelled model error)


def task_b7cal(run_dir, workers):
    """B7 whose likelihood mean residual and SD come from n characterised units, picked exactly as
    task_learning picks them (same seed, same units), and the best ML refitted on the same units, so
    every (n, seed) cell pairs with the learning-curve figure. Paired unit bootstraps on test_id, seed means per row:
    B7(n) - ML(n) (positive = ML better) and B7(n) - B7(160), plus per-seed means (the bootstrap holds
    the unit picks fixed). n = 0: B7(calib="spec"). B7.fit uses at most 1000 rows, so at n >= 10 the
    n axis varies unit diversity, not row count (calib_rows recorded). Each cell is cached in
    b7cal-cells/ with the code SHA; a re-run on the same SHA resumes, a crash loses one cell at most."""
    run_dir = Path(run_dir)
    prov = G.git_info()
    if prov["dirty"]:
        raise RuntimeError("b7cal: code is dirty; commit before the run")
    main = json.loads((run_dir / "results_main.json").read_text())
    learn = {(r["units"], r["seed"], r["method"]): r["mean_regret"]
             for r in json.loads((run_dir / "results_learning.json").read_text())["rows"]}
    obs, cv, cfg, tr, _ = prepare(run_dir)
    test = np.flatnonzero(obs["split"] == "test_id")
    ora, fl, B = curve_rows(cv, "oracle", test), V.P(cfg, "ber.regret_floor"), V.P(cfg, "evaluation.bootstrap_B")
    ut, units = obs["unit_id"].to_numpy()[test], np.unique(obs["unit_id"].to_numpy()[tr])
    seeds = range(V.P(cfg, "evaluation.model_seeds"))
    best_ml = main["best_ml_state_obs"]
    name, hp = best_ml.split("_")[1], main["hyperparameters"][best_ml]["chosen"]
    # pre-flights, all before any B7 prediction: stored B7(160) rows and calibration reproduce
    # results_main; every ML cell reproduces results_learning; sensor-spec SDs describe the residuals
    ident = np.load(run_dir / "shift-rows" / "identity.npz")
    if not np.array_equal(ident["state_id"], obs["state_id"].to_numpy()[test]):
        raise AssertionError("shift-rows/identity.npz rows not aligned with test_id")
    r160 = ident["regret:B7_bayes"]
    if not np.isclose(r160.mean(), main["test"]["test_id"]["B7_bayes"]["mean_regret"], rtol=1e-9, atol=0):
        raise AssertionError("stored B7 regrets do not reproduce results_main")
    full = B7(cfg).fit(obs.iloc[tr])
    for k in B7_OBS:
        if not np.isclose(full.sig[k], main["budget"]["B7_obs_sd"][k], rtol=1e-12, atol=0):
            raise AssertionError(f"B7 calibration SD {k} does not reproduce results_main")
    picks, ml = {}, {}
    for n in B7CAL_UNITS:
        for seed in seeds:
            pick = np.random.default_rng(seed).choice(units, min(n, len(units)), replace=False)
            rows = tr[np.isin(obs["unit_id"].to_numpy()[tr], pick)]
            m = regret(ML(name, V.FEATURES["state_obs"], hp, seed).fit(obs.iloc[rows], obs["L1"].to_numpy()[rows]).predict(obs.iloc[test]), ora, fl)
            if not np.isclose(m.mean(), learn[(n, seed, best_ml)], rtol=1e-9, atol=0):
                raise AssertionError(f"{best_ml} n={n} seed={seed} does not reproduce results_learning")
            picks[(n, seed)], ml[(n, seed)] = (pick, rows), m
    z, z_mean = spec_z(cfg, obs.iloc[tr])
    for k in ("EVM_p", "Prx_hat", "Tj_Vf_hat"):
        if not z[k] <= B7CAL_ZMAX:
            raise AssertionError(f"spec_sig {k}: z SD {z[k]:.2f} > {B7CAL_ZMAX} at the true latents")
    cells_dir = run_dir / "b7cal-cells"
    cells_dir.mkdir(exist_ok=True)

    def cell(key, make):
        f = cells_dir / f"{key}.npz"
        if f.exists():
            c = np.load(f)
            if str(c["sha"]) == prov["sha"]:
                return c["bhat"], json.loads(str(c["meta"]))
        m = make()
        bh = m.predict(obs.iloc[test])
        meta = {"B7_obs_sd": m.sig, "B7_obs_mean_residual": m.bias}
        tmp = cells_dir / f"{key}.tmp.npz"  # atomic: a kill mid-write leaves no truncated cell
        np.savez_compressed(tmp, sha=prov["sha"], bhat=bh, meta=json.dumps(meta))
        os.replace(tmp, f)
        return bh, meta

    rep = {"status": "post hoc, descriptive", "run_provenance": prov,
           "units": list(B7CAL_UNITS), "best_ml_state_obs": best_ml, "spec_z_at_true_latents": z, "spec_z_mean_at_true_latents": z_mean, "complete": False,
           "spec_evm_probe_sd_dB": EVM_PROBE_SD_DB, "cells": [], "by_units": {}}
    bh0, _ = cell("0", lambda: B7(cfg, workers, calib="spec").fit(None))
    r0 = regret(bh0, ora, fl)
    rep["by_units"]["0"] = {"mean_regret": float(r0.mean()), "minus_B7_160": cluster_boot(ut, r0, r160, B)[0]}
    rows_out = {"sha": prov["sha"], "state_id": obs["state_id"].to_numpy()[test], "unit_id": ut,
                "bhat:B7@0": bh0, "regret:B7@0": r0, "regret:B7@160": r160}
    write(run_dir, "b7cal", rep)
    for n in B7CAL_UNITS:
        rb, rm = [], []
        for seed in seeds:
            pick, rows = picks[(n, seed)]
            bh, meta = cell(f"{n}_{seed}", lambda: B7(cfg, workers).fit(obs.iloc[rows]))
            b, m = regret(bh, ora, fl), ml[(n, seed)]
            rb.append(b), rm.append(m)
            rows_out |= {f"bhat:B7@{n}@seed{seed}": bh, f"regret:B7@{n}@seed{seed}": b, f"regret:ML@{n}@seed{seed}": m}
            rep["cells"].append({"units": n, "seed": seed, "B7_mean_regret": float(b.mean()), "ml_mean_regret": float(m.mean()),
                                 "B7_minus_ml": float(b.mean() - m.mean()), "calib_rows": int(min(len(rows), 1000)),
                                 "calib_rows_available": int(len(rows)), **meta, "picked_units": pick.tolist()})
        d_seed = [float(b.mean() - m.mean()) for b, m in zip(rb, rm)]
        rb, rm = np.mean(rb, axis=0), np.mean(rm, axis=0)
        rep["by_units"][str(n)] = {"B7_mean_regret": float(rb.mean()), "ml_mean_regret": float(rm.mean()),
                                   "B7_minus_ml": cluster_boot(ut, rb, rm, B)[0], "minus_B7_160": cluster_boot(ut, rb, r160, B)[0],
                                   "B7_minus_ml_by_seed": d_seed, "seeds_ml_better": int(sum(d > 0 for d in d_seed)), "seeds_b7_better": int(sum(d < 0 for d in d_seed)),
                                   "B7_mean_regret_seed_range": [float(min(c["B7_mean_regret"] for c in rep["cells"] if c["units"] == n)),
                                                                 float(max(c["B7_mean_regret"] for c in rep["cells"] if c["units"] == n))]}
        write(run_dir, "b7cal", rep)  # after every n: a crash keeps the finished ones
    np.savez_compressed(run_dir / "b7cal-rows.npz", **rows_out)
    rep["complete"] = True
    write(run_dir, "b7cal", rep)
    return rep


def task_b7wide_imin(run_dir, workers):
    """The wide-prior B7 of task_shift, refitted identically, deployed on the two i_min twins and
    scored as task_shift scores (twin oracle, nominal-model belief). Paired against the rows stored
    in shift-rows/<deploy>.npz. Its prior widens kappa_tilde only; i_min stays nominal."""
    run_dir, root = Path(run_dir), Path(run_dir).parent
    main = json.loads((run_dir / "results_main.json").read_text())
    best_ml = main["best_ml_state_obs"]
    obs, cv, cfg, tr, _ = prepare(run_dir)
    B, fl = V.P(cfg, "evaluation.bootstrap_B"), V.P(cfg, "ber.regret_floor")
    df = load(run_dir)[0]
    ref = df[df["split"] == "test_id"].set_index("state_id")
    b7w = B7(cfg, workers, kt_axis=B7_WIDE_KT).fit(obs.iloc[tr])
    rows_dir = run_dir / "shift-rows"
    ident = np.load(rows_dir / "identity.npz")
    test = np.flatnonzero(obs["split"] == "test_id")
    # pre-flight: the refitted wide prior reproduces the stored identity biases (first 20 rows)
    if not np.allclose(b7w.predict(obs.iloc[test[:20]]), ident["bhat:B7_bayes_wide"][:20], rtol=0, atol=1e-9):
        raise AssertionError("refitted B7_bayes_wide does not reproduce shift-rows/identity.npz")
    rep = {"status": "post hoc, descriptive", "b7_wide_kt_axis": list(B7_WIDE_KT),
           "best_ml_state_obs": best_ml, "by_deploy": {}}
    for dep in B7_WIDE_IMIN:
        d = root / f"{TWIN_PREFIX}{dep}"
        twin_check(run_dir, d, cfg, ref)
        obs_d, cv_d, cfg_d, _, _ = prepare(d, belief=cfg)
        rows = np.flatnonzero(obs_d["split"] == "test_id")
        z = np.load(rows_dir / f"{dep}.npz")
        if not np.array_equal(z["state_id"], obs_d["state_id"].to_numpy()[rows]):
            raise AssertionError(f"{dep}: shift-rows not aligned with the twin's test_id")
        cur = curve_rows(cv_d, "oracle", rows)
        # scoring parity with task_shift: the stored B7 biases give the stored B7 regrets
        if not np.allclose(regret(z["bhat:B7_bayes"], cur, fl), z["regret:B7_bayes"], rtol=0, atol=1e-12):
            raise AssertionError(f"{dep}: regret() does not reproduce task_shift's stored B7 regrets")
        bh = b7w.predict(obs_d.iloc[rows])
        r, units, fec = regret(bh, cur, fl), z["unit_id"], z["fec_ok"]
        i_min = V.P(cfg_d, "led.i_min_frac") * obs_d["I_rated"].to_numpy()[rows]
        rep["by_deploy"][dep] = {
            "mean_regret": float(r.mean()), "below_i_min_share": float(np.mean(bh < i_min)),
            "delta_minus_best_ml": cluster_boot(units, r, z[f"regret:{best_ml}@seedmean"], B)[0],
            "minus_B7_bayes": cluster_boot(units, r, z["regret:B7_bayes"], B)[0],
            "cost_of_shift": cluster_boot(units, r, ident["regret:B7_bayes_wide"], B)[0],
            "fec_relevant": {"n_rows": int(fec.sum()), "mean_regret": float(r[fec].mean()) if fec.any() else None}}
        np.savez_compressed(rows_dir / f"b7wide-{dep}.npz", state_id=z["state_id"], bhat=bh, regret=r)
    write(run_dir, "b7wide_imin", rep)
    return rep


def write(run_dir, task, obj):
    """results_<task>.json, stamped with the evaluating code's SHA and dirty flag."""
    obj = {"_eval_provenance": G.git_info()} | obj
    path = Path(run_dir) / f"results_{task}.json"
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("task", choices=["main", "sensing", "learning", "b8", "misspec", "shift", "slopes", "b7cal", "b7wide_imin"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--first-run", action="store_true", help="shift: no previous results_shift.json to reproduce")
    a = ap.parse_args()
    if a.workers > 1:
        for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = "1"
    {"main": task_main, "sensing": task_sensing, "learning": task_learning, "b8": task_b8, "misspec": task_misspec, "shift": task_shift, "slopes": task_slopes,
     "b7cal": task_b7cal, "b7wide_imin": task_b7wide_imin}[a.task](a.run_dir, a.workers, *((a.task == "shift") * [TWIN_PREFIX, not a.first_run]))


if __name__ == "__main__":
    sys.exit(main())
