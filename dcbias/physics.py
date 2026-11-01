"""Electrothermal DCO-OFDM link physics: the one module every other module imports.

Parameters come only from config.json.

Signal chain per state and candidate DC bias current b (amps):
  drive   i = clip(b + sigma_I * s, i_min, I_max(T_ntc))     s: unit-variance DCO-OFDM symbol
  heat    Tj = T_hs + Rth * P_diss(b)                         closed form; P_diss is affine in Tj
  light   p = eta0 (1 - delta (Tj - T0)) q(i)                 q: quadratic i - kappa_tilde i^2 / I_rated
                                                              (nominal) or the Tsonev 2013 cubic
  channel first-order LED low-pass H_k, absolute receiver noise floor + signal shot noise
  receive FFT, zero-forcing, one-tap data-aided gain, per-subcarrier EVM -> SINR_k -> BER_M

Two evaluators of the same chain:
  mc_curve        Monte Carlo with common random numbers (labels, oracle)
  analytic_curve  Bussgang decomposition by quadrature (cross-check, B6b, B7)
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import kurtosis, skew

CONFIG_PATH = Path(__file__).with_name("config.json")
T0 = 298.0
LN10 = np.log(10.0)

# Seed-stream purposes: SeedSequence(root, spawn_key=(purpose, index)). Disjoint by construction.
STREAM_UNIT, STREAM_STATE, STREAM_LABEL, STREAM_ORACLE, STREAM_PROBE = 0, 1, 2, 3, 4
STREAM_WAVE, STREAM_SENSE, STREAM_COUNTED, STREAM_B8, STREAM_GATE = 5, 6, 7, 8, 9
STREAM_READOUT = 10  # independent re-evaluation of commanded biases


class ThermalRunaway(RuntimeError):
    """Self-heating has no stable operating point (1 - Rth * dP_diss/dTj <= 0)."""


# --------------------------------------------------------------------------- config

def load_config(path=CONFIG_PATH, overrides=None):
    cfg = json.loads(Path(path).read_text())
    for key, val in (overrides or {}).items():
        set_param(cfg, key, val)
    return cfg


def _leaf(cfg, key):
    node = cfg
    for part in key.split("."):
        node = node[part]
    return node


def P(cfg, key):
    """Value of a config leaf, e.g. P(cfg, 'led.delta')."""
    leaf = _leaf(cfg, key)
    if not isinstance(leaf, dict) or "value" not in leaf:
        raise KeyError(f"{key} is not a parameter leaf")
    return leaf["value"]


def set_param(cfg, key, value):
    leaf = _leaf(cfg, key)
    if not isinstance(leaf, dict) or "value" not in leaf:
        raise KeyError(f"{key} is not a parameter leaf")
    leaf["value"] = value


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def rng_for(cfg, purpose, *index):
    key = (purpose, *(int(i) for i in index))
    return np.random.default_rng(np.random.SeedSequence(P(cfg, "seed"), spawn_key=key))


# --------------------------------------------------------------------------- device and driver

LED_MODELS = ("quadratic", "tsonev_cubic", "tsonev_cubic_spread")


def cubic_coeffs(cfg, kt):
    """Tsonev cubic in i / I_rated, scaled to the quadratic's small-signal slope 1 - kappa_tilde at I_rated / 2.
    tsonev_cubic: one curve for every unit (kt ignored; nominal kappa_tilde). tsonev_cubic_spread: the raw x^3 and x^2 coefficients are multiplied by kt / kappa_tilde before
    the whole cubic is renormalised to slope 1 - kt at I_rated / 2, so a unit's curvature relative to its
    gain spreads along one axis, as the quadratic's kappa_tilde does; at kt = kappa_tilde it is
    tsonev_cubic exactly. Monotone on [0, I_rated] for kt < 0.467."""
    c, k0 = np.asarray(P(cfg, "led.tsonev_cubic"), float), P(cfg, "led.kappa_tilde")
    if P(cfg, "led.model") == "tsonev_cubic_spread":
        if not k0 > 0:
            raise ValueError("tsonev_cubic_spread needs led.kappa_tilde > 0")
        c, k0 = c * np.array([kt / k0, kt / k0, 1.0, 1.0]), kt
    return c * (1.0 - k0) / np.polyval(np.polyder(c), 0.5)


def nominal_device(cfg, I_rated, kt=None):
    """Nominal device; also carries the LED model and the turn-on clip i_min, which no unit varies.
    kt: a unit's kappa_tilde (default nominal); it shapes the cubic only under tsonev_cubic_spread."""
    model = P(cfg, "led.model")
    if model not in LED_MODELS:
        raise ValueError(f"unknown led.model {model!r}")
    kt = P(cfg, "led.kappa_tilde") if kt is None else kt
    cubic = cubic_coeffs(cfg, kt)
    return {
        "model": model, "cubic": cubic, "i_min": P(cfg, "led.i_min_frac") * float(I_rated),
        "I_rated": float(I_rated),
        "eta0": P(cfg, "led.eta0_nom"),
        "kt": kt,
        "delta": P(cfg, "led.delta"),
        "Rth": P(cfg, "thermal.Rth"),
        "V0": P(cfg, "thermal.V0"),
        "Rs": P(cfg, "thermal.Rs"),
        "KV": P(cfg, "thermal.K_V"),
    }


def draw_unit(cfg, unit_id):
    """Latent per-unit parameters (relative to nominal). I_rated is a per-state setting, added later."""
    rng = rng_for(cfg, STREAM_UNIT, unit_id)
    on = P(cfg, "design.unit_spread_on")

    def rel(name, nominal):
        w = P(cfg, f"unit_spread.{name}") if on else 0.0
        return nominal * rng.uniform(1 - w, 1 + w)

    lo, hi = P(cfg, "unit_spread.K_V_range")
    unit = {
        "eta0": rel("eta0", P(cfg, "led.eta0_nom")),
        "kt": rel("kappa_tilde", P(cfg, "led.kappa_tilde")),
        "delta": rel("delta", P(cfg, "led.delta")),
        "Rth": rel("Rth", P(cfg, "thermal.Rth")),
        "V0": rel("V0", P(cfg, "thermal.V0")),
        "Rs": rel("Rs", P(cfg, "thermal.Rs")),
        "KV": rng.uniform(lo, hi) if on else P(cfg, "thermal.K_V"),
        "ntc_offset": rng.normal(0, P(cfg, "unit_spread.ntc_offset_sd")) if on else 0.0,
        "epsK": rng.normal(0, P(cfg, "unit_spread.epsK_sd")) if on else 0.0,
    }
    check_monotone(nominal_device(cfg, 1.0, unit["kt"]))
    return unit


DEVICE_KEYS = ("eta0", "kt", "delta", "Rth", "V0", "Rs", "KV")


def build(cfg, row):
    """True (state, device, unit) for a design row: unit latents, driver ceiling from the unit's own NTC."""
    unit = draw_unit(cfg, row["unit_id"])
    Ir = float(row["I_rated"])
    dev = nominal_device(cfg, Ir, unit["kt"]) | {k: unit[k] for k in DEVICE_KEYS}
    T_ntc = row["T_hs"] + unit["ntc_offset"]
    st = {"I_rated": Ir, "sigma": row["sigma_frac"] * Ir, "T_hs": row["T_hs"], "snr_db": row["SNR_ref_dB"],
          "M": int(row["M"]), "C": float(foldback_ceiling(cfg, Ir, T_ntc)), "T_ntc": T_ntc}
    return st, dev, unit


def check_monotone(dev):
    """q(i) must increase on [0, I_rated]: quadratic iff 2 kt < 1; cubic checked on a grid."""
    if dev["model"] == "quadratic":
        if not 2.0 * dev["kt"] < 1.0:
            raise ValueError(f"LED transfer not monotone on [0, I_rated]: 2*kappa_tilde = {2 * dev['kt']:.3f} >= 1")
    elif not np.all(np.polyval(np.polyder(dev["cubic"]), np.linspace(0, 1, 201)) > 0):
        raise ValueError("Tsonev cubic LED transfer not monotone on [0, I_rated]")


def foldback_ceiling(cfg, I_rated, T_ntc):
    """Driver thermal foldback: I_rated below the knee, linear derating above, floor f_min."""
    frac = 1.0 - P(cfg, "driver.slope_per_K") * np.maximum(np.asarray(T_ntc) - P(cfg, "driver.knee_K"), 0.0)
    return I_rated * np.maximum(P(cfg, "driver.f_min"), frac)


def bias_grid(cfg, I_rated, C):
    """[i_min, C] in steps of grid_step_frac * I_rated, ending exactly at C (the floor is the LED
    turn-on clip i_min). Spacings stay in [step / 2, 1.5 step)."""
    step, lo = P(cfg, "ofdm.grid_step_frac") * I_rated, P(cfg, "led.i_min_frac") * I_rated
    if not C > lo + step:
        raise ValueError(f"ceiling {C:.4g} A is within one grid step of i_min {lo:.4g} A")
    b = np.arange(lo, C + 1e-12, step)
    if C - b[-1] >= 0.5 * step:
        b = np.append(b, C)
    else:
        b[-1] = C
    return b


# --------------------------------------------------------------------------- quadrature and heat

def quadrature(cfg):
    s = np.linspace(-P(cfg, "ofdm.quad_span"), P(cfg, "ofdm.quad_span"), P(cfg, "ofdm.quad_nodes"))
    w = np.exp(-0.5 * s**2)
    return s, w / w.sum()


def _E(x, w):
    """Quadrature expectation x @ w. macOS Accelerate BLAS sets spurious FP flags on finite input,
    so flags are ignored here and finiteness is checked explicitly instead."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        r = x @ w
    if not np.all(np.isfinite(r)):
        raise FloatingPointError("non-finite quadrature expectation")
    return r


def led_q(i, dev):
    """LED transfer q(i) (the one LED function: thermal, receive and the analytic curve all use it)."""
    if dev["model"] == "quadratic":
        return i - dev["kt"] * i**2 / dev["I_rated"]
    return dev["I_rated"] * np.polyval(dev["cubic"], i / dev["I_rated"])


def thermal(b, sigma, C, T_hs, dev, quad):
    """Deterministic electrothermal operating point for each bias in b.

    P_diss(Tj) = E[V_f i] - eta(Tj) E[q(i)] with V_f = V0 + Rs i + K_V (Tj - T0) is affine in Tj,
    P_diss = a + c (Tj - T0), so Tj = T_hs + Rth P_diss has the exact solution below. A stable
    solution needs 1 - Rth c > 0; otherwise the self-heating loop has no fixed point and we raise.
    """
    s, w = quad
    b = np.atleast_1d(np.asarray(b, dtype=float))
    i = np.clip(b[:, None] + sigma * s[None, :], dev["i_min"], C)
    Ei, Ei2, Eq = _E(i, w), _E(i**2, w), _E(led_q(i, dev), w)
    a = dev["V0"] * Ei + dev["Rs"] * Ei2 - dev["eta0"] * Eq
    c = dev["KV"] * Ei + dev["eta0"] * dev["delta"] * Eq
    denom = 1.0 - dev["Rth"] * c
    if np.any(denom <= 0):
        raise ThermalRunaway(f"1 - Rth*dP/dTj = {denom.min():.3g} <= 0")
    Tj = (T_hs + dev["Rth"] * (a - c * T0)) / denom
    eta = dev["eta0"] * (1.0 - dev["delta"] * (Tj - T0))
    if np.any(eta <= 0):
        raise ValueError(f"optical efficiency <= 0 at Tj = {Tj.max():.1f} K")
    P_elec = dev["V0"] * Ei + dev["Rs"] * Ei2 + dev["KV"] * (Tj - T0) * Ei
    return {"i": i, "Tj": Tj, "eta": eta, "Ep": eta * Eq, "P_elec": P_elec, "P_diss": P_elec - eta * Eq}


# --------------------------------------------------------------------------- noise, channel, BER

def noise_var(cfg, state, Ep):
    """Absolute receiver noise: fixed thermal/ambient floor plus signal shot noise proportional to E[p].

    SNR_ref is the linear-LED reference SNR: receiver noise = (eta0 sigma_I)^2 / SNR_ref,
    so the LED's small-signal slope and distortion lower the realised SINR below it.
    The floor is set by the nominal eta0, so a unit's latent eta0 moves its effective SNR. P_ref, the
    shot-noise reference, is E[p] of the nominal LED at b = I_rated/2, T = T0, unclipped.
    """
    eta0, Ir, sig = P(cfg, "led.eta0_nom"), state["I_rated"], state["sigma"]
    var_ref = (eta0 * sig) ** 2 / 10 ** (state["snr_db"] / 10)
    s, w = quadrature(cfg)
    P_ref = eta0 * _E(led_q(Ir / 2 + sig * s, nominal_device(cfg, Ir)), w)
    rho = P(cfg, "noise.rho")
    return var_ref * ((1 - rho) + rho * np.asarray(Ep) / P_ref)


def channel(cfg):
    """LED first-order low-pass at data subcarriers k = 1..Nu; f_k / B = 2k/N with f_s = 2B."""
    N, Nu, r = P(cfg, "ofdm.N"), P(cfg, "ofdm.Nu"), P(cfg, "led.f3dB_over_B")
    k = np.arange(1, Nu + 1)
    if r is None:
        return np.ones(Nu, dtype=complex)
    return 1.0 / (1.0 + 1j * (2.0 * k / N) / r)


_BER_TERMS = {}


def _ber_terms(M):
    """Exact Gray square-QAM BER as weighted decision-region tails, derived from the bit map itself.

    Square M-QAM with gray_bits() is two independent Gray m-PAMs (m = sqrt M) with kb = log2 m bits
    each. For transmitted level j and decided level r != j, P(r | j) = Q(n1 x) - Q(n2 x) with
    n1 = 2|r - j| - 1, n2 = 2|r - j| + 1 (n2 = inf at an outer level), x = a / sigma_axis = sqrt(3 g/(M-1)).
    BER = sum_{j, r} Hamming(j, r) P(r | j) / (m kb). Returns (log weight, n1, n2) per distinct term.
    """
    if M not in _BER_TERMS:
        m, kb = int(round(np.sqrt(M))), int(round(np.log2(np.sqrt(M))))
        axis = gray_bits(M)[::m, :kb]  # bits of the in-phase level j
        acc = {}
        for j in range(m):
            for r in range(m):
                h = int(np.sum(axis[j] != axis[r]))
                if h:
                    dist = abs(r - j)
                    key = (2 * dist - 1, np.inf if r in (0, m - 1) else 2 * dist + 1)
                    acc[key] = acc.get(key, 0.0) + h / (m * kb)
        _BER_TERMS[M] = [(np.log(w), n1, n2) for (n1, n2), w in acc.items()]
    return _BER_TERMS[M]


def log10_ber(sinr, M):
    """log10 of the subcarrier-averaged exact Gray square-QAM BER (see _ber_terms).

    Log domain throughout (log_ndtr), so the objective has no floor ties.
    """
    x = np.sqrt(3.0 * np.asarray(sinr, float) / (M - 1))
    parts = []
    for lw, n1, n2 in _ber_terms(M):
        lq1 = log_ndtr(-n1 * x)
        tail = 0.0 if np.isinf(n2) else np.log1p(-np.exp(log_ndtr(-n2 * x) - lq1))
        parts.append(lw + lq1 + tail)
    lb = logsumexp(np.stack(parts), axis=0)
    return (logsumexp(lb, axis=-1) - np.log(lb.shape[-1])) / LN10


# --------------------------------------------------------------------------- analytic curve

def analytic_curve(cfg, state, dev, b, quad, H=None):
    """Bussgang SINR per subcarrier by quadrature: log10 BER and thermal state per bias."""
    s, w = quad
    sig, N, Nu = state["sigma"], P(cfg, "ofdm.N"), P(cfg, "ofdm.Nu")
    H = channel(cfg) if H is None else H
    th = thermal(b, sig, state["C"], state["T_hs"], dev, quad)
    p = th["eta"][:, None] * led_q(th["i"], dev)
    Ep = _E(p, w)
    K = _E((p - Ep[:, None]) * s[None, :], w) / sig
    var = _E((p - Ep[:, None]) ** 2, w)
    d2 = var - K**2 * sig**2
    if np.any(d2 < -1e-9 * var):
        raise FloatingPointError("negative Bussgang distortion variance beyond round-off")
    d2 = np.maximum(d2, 0.0)
    n2 = noise_var(cfg, state, Ep)
    sinr = (N / (2 * Nu)) * (K**2 * sig**2)[:, None] / (d2[:, None] + n2[:, None] / np.abs(H[None, :]) ** 2)
    return {"lb": log10_ber(sinr, state["M"]), "sinr": sinr, **{k: th[k] for k in ("Tj", "Ep", "P_elec", "P_diss")}}


# --------------------------------------------------------------------------- Monte Carlo curve

def constellation(M):
    m = int(round(np.sqrt(M)))
    if m * m != M:
        raise ValueError(f"M = {M} is not a square QAM order")
    lv = np.arange(-m + 1, m, 2, dtype=float)
    pts = (lv[:, None] + 1j * lv[None, :]).ravel()  # index = ix * m + iq
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def ofdm_symbols(rng, R, M, N, Nu):
    """R Hermitian DCO-OFDM symbols: data on k = 1..Nu, X[0] = X[N/2] = 0.

    s is normalised by the analytic constant sqrt(2 Nu / N), never per symbol, so common random
    numbers hold across biases. Returns symbol indices, symbols S (R, Nu) and s (R, N), E[s^2] = 1.
    """
    idx = rng.integers(0, M, size=(R, Nu))
    S = constellation(M)[idx]
    X = np.zeros((R, N), dtype=complex)
    X[:, 1 : Nu + 1] = S
    X[:, N - Nu :] = np.conj(S[:, ::-1])
    x = np.fft.ifft(X, axis=1, norm="ortho")
    if np.max(np.abs(x.imag)) > 1e-12:
        raise FloatingPointError("OFDM symbol is not real: Hermitian symmetry broken")
    return idx, S, x.real / np.sqrt(2.0 * Nu / N)


def draws(cfg, rng, R, M):
    """Common random numbers for one state: symbols and receiver noise, drawn once, reused at every bias."""
    N, Nu = P(cfg, "ofdm.N"), P(cfg, "ofdm.Nu")
    idx, S, s = ofdm_symbols(rng, R, M, N, Nu)
    Zf = np.fft.rfft(rng.standard_normal((R, N)), axis=1, norm="ortho")[:, 1 : Nu + 1]
    return {"idx": idx, "S": S, "s": s, "Zf": Zf, "M": M}


def mc_receive(cfg, state, dev, b, d, quad, H=None):
    """Received, zero-forced, gain-normalised subcarrier values for each bias in b: (len(b), R, Nu)."""
    H = channel(cfg) if H is None else H
    Nu = P(cfg, "ofdm.Nu")
    th = thermal(b, state["sigma"], state["C"], state["T_hs"], dev, quad)
    n = np.sqrt(noise_var(cfg, state, th["Ep"]))
    out = np.empty((len(b), *d["S"].shape), dtype=complex)
    for j, bj in enumerate(np.atleast_1d(b)):
        i = np.clip(bj + state["sigma"] * d["s"], dev["i_min"], state["C"])
        p = th["eta"][j] * led_q(i, dev)
        Y = np.fft.rfft(p, axis=1, norm="ortho")[:, 1 : Nu + 1] + n[j] * d["Zf"] / H
        a = np.vdot(d["S"], Y) / np.vdot(d["S"], d["S"])
        out[j] = Y / a
    return out, th


def mc_curve(cfg, state, dev, b, d, quad, H=None):
    """Semi-analytic BER from per-subcarrier EVM under CRN: log10 BER per bias."""
    lb = np.empty(len(b))
    chunk = P(cfg, "ofdm.bias_chunk")
    th_all = []
    for j0 in range(0, len(b), chunk):
        Yn, th = mc_receive(cfg, state, dev, b[j0 : j0 + chunk], d, quad, H)
        e2 = np.mean(np.abs(Yn - d["S"][None]) ** 2, axis=1)
        lb[j0 : j0 + chunk] = log10_ber(1.0 / e2, state["M"])
        th_all.append(th)
    th = {k: np.concatenate([t[k] for t in th_all]) for k in ("Tj", "Ep", "P_elec", "P_diss")}
    return lb, th


def evm_db(cfg, state, dev, b, d, quad):
    Yn, _ = mc_receive(cfg, state, dev, np.atleast_1d(b), d, quad)
    return 10 * np.log10(np.mean(np.abs(Yn - d["S"][None]) ** 2, axis=(1, 2)))


# --------------------------------------------------------------------------- counted bits

def gray_bits(M):
    """Bit labels (M, log2 M) for constellation() indices: Gray code per axis."""
    m = int(round(np.sqrt(M)))
    kb = int(np.log2(m))
    g = np.arange(m) ^ (np.arange(m) >> 1)
    axis = ((g[:, None] >> np.arange(kb)[::-1]) & 1).astype(np.int8)
    return np.concatenate([np.repeat(axis, m, axis=0), np.tile(axis, (m, 1))], axis=1)


def hard_decide(Yn, M):
    m = int(round(np.sqrt(M)))
    scale = 1.0 / np.sqrt(2 * (M - 1) / 3)  # amplitude of level 1 after unit-energy normalisation

    def level(v):
        return np.clip(np.rint((v / scale + m - 1) / 2), 0, m - 1).astype(int)

    return level(Yn.real) * m + level(Yn.imag)


def counted_ber(Yn, idx, M):
    bits = gray_bits(M)
    return np.mean(bits[hard_decide(Yn, M)] != bits[idx])


# --------------------------------------------------------------------------- labels

def labels_from_curve(b, lb, cfg):
    """L1 = argmin BER; L2 = minimum b with BER <= target (feasible flag), else argmin (defined branch)."""
    if not np.all(np.isfinite(lb)):
        raise FloatingPointError("non-finite log10 BER on the label curve")
    j = int(np.argmin(lb))
    out = {"L1": b[j], "L1_idx": j, "lb_min": lb[j], "boundary": j in (0, len(b) - 1)}
    for name, key in (("L2", "ber.fec_hd"), ("L2_38", "ber.fec_alt")):
        ok = np.flatnonzero(lb <= np.log10(P(cfg, key)))
        out[name], out[f"{name}_feasible"] = (b[ok[0]], True) if len(ok) else (b[j], False)
    return out


def n_local_min(lb):
    """Local minima of a curve, endpoints included."""
    x = np.concatenate([[np.inf], lb, [np.inf]])
    return int(np.sum((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])))


def refine_grid(cfg, b, j, I_rated, C):
    """Coarse grid plus fine points around b[j]. Fine points closer than half a fine step to an existing
    node are dropped."""
    h, step = P(cfg, "ofdm.refine_half_points"), P(cfg, "ofdm.refine_step_frac") * I_rated
    fine = b[j] + step * np.arange(-h, h + 1)
    fine = fine[(fine >= b[0]) & (fine <= C)]
    fine = fine[np.min(np.abs(fine[:, None] - b[None, :]), axis=1) >= 0.5 * step]
    return np.sort(np.concatenate([b, fine]))


# --------------------------------------------------------------------------- sensing

FEATURES = {
    "config": ["I_rated", "sigma_I", "M", "b_p"],
    "link": ["SNR_hat", "Prx_hat", "EVM_p"],
    "thermal": ["T_ntc_hat", "Tj_Vf_hat", "Tj_Rth_hat"],
    "wave": ["w_mean", "w_std", "w_min", "w_max", "w_skew", "w_kurt"],
}
FEATURES["state"] = ["I_rated", "sigma_I", "T_ntc_hat", "SNR_hat", "M"]
FEATURES["state_obs"] = FEATURES["config"] + FEATURES["link"] + FEATURES["thermal"] + FEATURES["wave"]
FEATURES["thermal_family"] = FEATURES["config"] + FEATURES["link"] + FEATURES["thermal"]
FEATURES["wave_family"] = FEATURES["config"] + FEATURES["link"] + FEATURES["wave"]


def waveform_stats(cfg, row, lam_W, z):
    """Drive-current tap: W symbols at f_s = 2B through a b-bit ADC over 1.2 I_rated, plus N(0, sigma_a).

    The clean waveform is seeded per state (STREAM_WAVE); z are the standard normals for the ADC
    noise. lam_W scales both the noise and the LSB (lam_W = 0 is an ideal ADC).
    """
    N, Nu, W = P(cfg, "ofdm.N"), P(cfg, "ofdm.Nu"), P(cfg, "sensing.window_symbols")
    _, _, s = ofdm_symbols(rng_for(cfg, STREAM_WAVE, row["state_id"]), W, int(row["M"]), N, Nu)
    i = np.clip(row["b_p"] + row["sigma_I"] * s.ravel(), P(cfg, "led.i_min_frac") * row["I_rated"], row["C"])
    fs = P(cfg, "sensing.adc_fs_frac") * row["I_rated"]
    lsb = fs / 2 ** P(cfg, "sensing.adc_bits")
    x = i + lam_W * P(cfg, "sensing.sigma_a_lsb") * lsb * z[: i.size]
    if lam_W > 0:
        x = np.round(x / (lam_W * lsb)) * (lam_W * lsb)
    x = np.clip(x, 0.0, fs)
    return [x.mean(), x.std(ddof=1), x.min(), x.max(), skew(x), kurtosis(x)]


def p_opt_nominal(cfg, I_rated, sigma, b, C, T):
    """Optical power a controller would predict at bias b from the nominal LED at temperature T."""
    dev = nominal_device(cfg, I_rated)
    s, w = quadrature(cfg)
    i = np.clip(b + sigma * s, dev["i_min"], C)
    return dev["eta0"] * (1.0 - dev["delta"] * (T - T0)) * float(_E(led_q(i, dev), w))


def observe(cfg, df, lam_T=1.0, lam_W=1.0, lam_link=1.0, rep=0, belief=None):
    """Noisy observables for every row of a states table. Noise draws depend on (state, rep) only,
    so cells of the lambda map share common random numbers. Configuration values stay exact.

    lambda_T scales every thermal-family error, so lambda_T = 0 returns the true T_ntc and Tj:
      T_ntc_hat   NTC reading (unit offset is physical: the driver's foldback reads it too)
      Tj_Vf_hat   forward-voltage method; per-unit calibrated K-factor with residual epsK, plus V noise
      Tj_Rth_hat  T_ntc_hat + Rth_nom * P_diss_est, with P_diss_est = measured P_elec (relative error
                  sigma_P) minus the nominal-LED optical power at T_ntc_hat (never the true
                  P_diss). The NTC does not see the LED's self-heating (stated scope).
    lambda_T scales the systematic (per-unit) errors deterministically, so Tj_Rth_hat is latent-free only
    at lambda_T in {0, 1}; elsewhere it carries (lambda_T - 1) * u_ntc_offset without noise. task_sensing reports a with/without-Tj_Rth_hat ablation at lambda_T > 1.
    belief: the controller's device model (default cfg), used only where the controller computes from its
    nominal LED: P_opt_nom_est (hence Tj_Rth_hat) and i_min. Physical readings always use cfg (device-shift
    arm).
    """
    belief = cfg if belief is None else belief
    sT, sV = P(cfg, "sensing.sigma_T_ntc"), P(cfg, "sensing.sigma_V") / abs(P(cfg, "sensing.K_V_nom"))
    sP, Rth_nom = P(cfg, "sensing.sigma_P_rel"), P(cfg, "thermal.Rth")
    nw = P(cfg, "sensing.window_symbols") * P(cfg, "ofdm.N")
    out = {k: np.empty(len(df)) for k in FEATURES["link"] + FEATURES["thermal"] + FEATURES["wave"] + ["P_opt_nom_est"]}
    for n, row in enumerate(df.to_dict("records")):
        z = rng_for(cfg, STREAM_SENSE, row["state_id"], rep).standard_normal(5 + nw)
        out["T_ntc_hat"][n] = row["T_ntc"] + lam_T * sT * z[0]
        out["Tj_Vf_hat"][n] = T0 + (1 + lam_T * row["u_epsK"]) * (row["Tj_p"] - T0) + lam_T * sV * z[1]
        C_hat = float(foldback_ceiling(cfg, row["I_rated"], out["T_ntc_hat"][n]))
        p_nom = p_opt_nominal(belief, row["I_rated"], row["sigma_I"], row["b_p"], C_hat, out["T_ntc_hat"][n])
        p_true = row["P_elec_p"] - row["P_diss_p"]
        P_diss_est = row["P_elec_p"] * (1 + lam_T * sP * z[4]) - (p_true + lam_T * (p_nom - p_true))
        R_used = row["u_Rth"] + lam_T * (Rth_nom - row["u_Rth"])
        out["Tj_Rth_hat"][n] = row["T_hs"] + lam_T * (row["u_ntc_offset"] + sT * z[0]) + R_used * P_diss_est
        out["P_opt_nom_est"][n] = p_nom
        out["SNR_hat"][n] = row["SNR_ref_dB"] + lam_link * P(cfg, "sensing.snr_sd_dB") * z[2]
        out["Prx_hat"][n] = row["Prx_dB"] + lam_link * P(cfg, "sensing.prx_sd_dB") * z[3]
        out["EVM_p"][n] = row["EVM_p_dB"]
        for k, v in zip(FEATURES["wave"], waveform_stats(cfg, row, lam_W, z[5:])):
            out[k][n] = v
    obs = df.copy()
    for k, v in out.items():
        obs[k] = v
    obs["C_hat"] = foldback_ceiling(cfg, obs["I_rated"].to_numpy(), obs["T_ntc_hat"].to_numpy())
    obs["i_min"] = P(belief, "led.i_min_frac") * obs["I_rated"]
    return obs
