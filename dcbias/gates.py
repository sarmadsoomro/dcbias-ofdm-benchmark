"""Pre-run gates. Each prints a JSON verdict and exits 1 on failure.

  crn        n ID states x 3 independent label seeds at R_label: within-state label SD (ddof = 1)
             <= 0.003 I_rated (RMS over states) and mean label regret vs an R_oracle curve
             <= 0.005 decades; also MC-vs-analytic label difference. On failure: R_label -> 512.
  counted    n states x b in {b_o, 0.95 b_o, 1.05 b_o}: counted-bit BER (R_oracle symbols) vs the
             semi-analytic EVM BER, within +/-0.1 decade wherever BER >= 1e-5 in >= 95 % of cases.
             A Poisson z-check is reported alongside, so counting noise is visible.
  interp     on a generated run: oracle-curve interpolation error at 100 random biases < 0.002 decades;
             half the biases within +/-3 coarse steps of O_L1, where competitive predictions land.
             Judged on log10 BER floored at ber.regret_floor, the quantity regret uses; raw error reported
  precision  from results_main.json: Delta 95 % CI half-width scaled to 40 test units <= 0.03 decades

Usage: python -m dcbias.gates crn --n 200 --workers 8
       python -m dcbias.gates interp --run runs/<run>
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dcbias import evaluate as E
from dcbias import generate as G
from dcbias import physics as V


def id_rows(cfg, n):
    rows = [r for r in G.design(cfg) if r["split"] in ("train", "test_id")]
    pick = np.random.default_rng(1).choice(len(rows), n, replace=False)
    return [rows[i] for i in sorted(pick)]


def _crn_state(args):
    cfg, row = args
    quad = V.quadrature(cfg)
    st, dev, _ = V.build(cfg, row)
    b = V.bias_grid(cfg, st["I_rated"], st["C"])
    do = V.draws(cfg, V.rng_for(cfg, V.STREAM_ORACLE, row["state_id"]), V.P(cfg, "ofdm.R_oracle"), st["M"])
    lo, _ = V.mc_curve(cfg, st, dev, b, do, quad)
    labs = []
    for rep in range(3):
        d = V.draws(cfg, V.rng_for(cfg, V.STREAM_GATE, row["state_id"], rep), V.P(cfg, "ofdm.R_label"), st["M"])
        labs.append(int(np.argmin(V.mc_curve(cfg, st, dev, b, d, quad)[0])))
    an = int(np.argmin(V.analytic_curve(cfg, st, dev, b, quad)["lb"]))
    fl = np.log10(V.P(cfg, "ber.regret_floor"))
    reg = [max(lo[j], fl) - max(lo.min(), fl) for j in labs]
    return {"sd_over_Ir": float(np.std(b[labs], ddof=1) / st["I_rated"]), "regret": reg,
            "mc_minus_analytic_steps": float(np.mean(labs) - an)}


def _counted_state(args):
    cfg, row = args
    quad = V.quadrature(cfg)
    st, dev, _ = V.build(cfg, row)
    b = V.bias_grid(cfg, st["I_rated"], st["C"])
    d = V.draws(cfg, V.rng_for(cfg, V.STREAM_COUNTED, row["state_id"]), V.P(cfg, "ofdm.R_oracle"), st["M"])
    bo = b[int(np.argmin(V.mc_curve(cfg, st, dev, b, d, quad)[0]))]
    out = []
    for x in (bo, max(0.95 * bo, dev["i_min"]), min(1.05 * bo, st["C"])):
        Yn, _ = V.mc_receive(cfg, st, dev, np.array([x]), d, quad)
        semi = 10 ** V.log10_ber(1.0 / np.mean(np.abs(Yn[0] - d["S"]) ** 2, axis=0), st["M"])
        nbits = d["idx"].size * int(np.log2(st["M"]))
        cnt = V.counted_ber(Yn[0], d["idx"], st["M"])
        out.append({"semi": float(semi), "counted": float(cnt), "nbits": nbits})
    return out


def pool_map(fn, jobs, workers):
    if workers > 1:
        for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = "1"
        with Pool(workers) as p:
            return p.map(fn, jobs, chunksize=2)
    return [fn(j) for j in jobs]


def gate_crn(cfg, n, workers):
    res = pool_map(_crn_state, [(cfg, r) for r in id_rows(cfg, n)], workers)
    sd = float(np.sqrt(np.mean([r["sd_over_Ir"] ** 2 for r in res])))
    reg = np.concatenate([r["regret"] for r in res])
    diff = np.array([r["mc_minus_analytic_steps"] for r in res])
    return {"R_label": V.P(cfg, "ofdm.R_label"), "n_states": n, "rms_within_sd_over_Ir": sd, "mean_label_regret": float(reg.mean()),
            "p95_label_regret": float(np.quantile(reg, 0.95)), "mc_minus_analytic_grid_steps": {"mean": float(diff.mean()), "rms": float(np.sqrt((diff**2).mean()))},
            "pass": bool(sd <= 0.003 and reg.mean() <= 0.005)}


def gate_counted(cfg, n, workers):
    res = [x for r in pool_map(_counted_state, [(cfg, r) for r in id_rows(cfg, n)], workers) for x in r]
    use = [r for r in res if r["semi"] >= 1e-5]
    within = [abs(np.log10(max(r["counted"], 1e-12)) - np.log10(r["semi"])) <= 0.1 for r in use]
    z = [abs(r["counted"] - r["semi"]) * r["nbits"] / np.sqrt(r["semi"] * r["nbits"]) for r in use]
    return {"n_cases": len(use), "share_within_0.1_decade": float(np.mean(within)), "share_poisson_z_le_2.5": float(np.mean(np.array(z) <= 2.5)),
            "median_expected_errors": float(np.median([r["semi"] * r["nbits"] for r in use])),
            "pass": bool(np.mean(within) >= 0.95)}


def gate_interp(run):
    df = G.pd.read_csv(Path(run) / "states.csv")
    cfg = json.loads((Path(run) / "provenance.json").read_text())["config"]
    cv = np.load(Path(run) / "curves.npz")
    rows = np.flatnonzero(df["split"] != "train")
    rng = np.random.default_rng(2)
    quad, err, raw, err12, lt = V.quadrature(cfg), [], [], [], []
    fl = np.log10(V.P(cfg, "ber.regret_floor"))
    for n, r in enumerate(rng.choice(rows, min(100, len(rows)), replace=False)):
        row = df.iloc[r].to_dict()
        st, dev, _ = V.build(cfg, row)
        b, lb = cv["oracle_b"][r], cv["oracle_lb"][r]
        b, lb = b[~np.isnan(b)], lb[~np.isnan(b)]
        near = 3 * V.P(cfg, "ofdm.grid_step_frac") * st["I_rated"]
        x = rng.uniform(dev["i_min"], st["C"]) if n % 2 else rng.uniform(max(row["O_L1"] - near, b[0]), min(row["O_L1"] + near, b[-1]))
        d = V.draws(cfg, V.rng_for(cfg, V.STREAM_ORACLE, row["state_id"]), V.P(cfg, "ofdm.R_oracle"), st["M"])
        true, interp = V.mc_curve(cfg, st, dev, np.array([x]), d, quad)[0][0], float(E.Curve(b, lb)(x))
        raw.append(abs(true - interp))
        err.append(abs(max(true, fl) - max(interp, fl)))
        err12.append(abs(max(true, -12.0) - max(interp, -12.0)))
        lt.append(true)
    err, lt = np.array(err), np.array(lt)
    # per-band counts and maxima: the (-12, -3] band is where the floor and margin matter
    bands = {"le_-12": lt <= -12, "(-12,-3]": (lt > -12) & (lt <= -3), "gt_-3": lt > -3}
    return {"n": len(err), "near_O_L1_share": 0.5, "max_abs_err_decades": float(err.max()), "mean_abs_err": float(err.mean()),
            "max_abs_err_floor_1e-12": float(max(err12)), "raw_max_abs_err_unfloored": float(max(raw)),
            "by_true_log_ber_band": {k: {"n": int(m.sum()), "max_abs_err": float(err[m].max()) if m.any() else None} for k, m in bands.items()},
            "pass": bool(err.max() < 0.002)}


def gate_precision(run):
    r = json.loads((Path(run) / "results_main.json").read_text())
    df = G.pd.read_csv(Path(run) / "states.csv")
    n_units = df.loc[df["split"] == "test_id", "unit_id"].nunique()
    lo, hi = r["confirmatory"]["delta"]["test_id"]["ci"]
    hw40 = (hi - lo) / 2 * np.sqrt(n_units / 40)
    return {"n_test_units_pilot": int(n_units), "halfwidth_pilot": (hi - lo) / 2, "halfwidth_at_40_units": float(hw40),
            "pass": bool(hw40 <= 0.03), "on_fail": "increase test units to 80 before the full run"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", choices=["crn", "counted", "interp", "precision"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cfg = V.load_config(overrides=G.parse_sets(a.set))
    out = {"crn": lambda: gate_crn(cfg, a.n or 200, a.workers), "counted": lambda: gate_counted(cfg, a.n or 300, a.workers),
           "interp": lambda: gate_interp(a.run), "precision": lambda: gate_precision(a.run)}[a.gate]()
    out["config_hash"] = V.config_hash(cfg)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
