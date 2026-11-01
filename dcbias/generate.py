"""Generate a dataset: states, labels, probe observables, oracle curves.

Writes runs/<name>/ with
  states.csv       one row per state (exogenous state, latent unit parameters prefixed u_, labels)
  curves.npz       label curves (R_label, all states) and oracle curves (R_oracle + refinement,
                   test states only); NaN-padded rows aligned with states.csv
  provenance.json  git SHA, config + hash, versions, seeds, runtime, label diagnostics
Never overwrites an existing directory.

Usage:
  python -m dcbias.generate --name smoke --set design.n_units_id=4 design.n_train_units=2 \
      design.n_units_ood=1 design.states_per_unit=5 --workers 4
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from dcbias import physics as V

REPO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_DIR / "runs"
SPLITS_OOD = ("ood_T", "ood_SNR", "ood_sigma")


def design(cfg):
    """Unit and state table. ID units: train then test_id; then n_units_ood units per OOD set.
    Each OOD set leaves the ID region along exactly one axis.

    T_hs is a two-stratum mixture: odd-indexed states of
    every unit outside OOD-T come from the foldback stratum [s_lo, T_id], the rest from the base stratum
    [T_lo, s_lo). OOD-T states come from (T_id, T_hi]."""
    def P_(key):
        return V.P(cfg, key)

    nid, ntr, nood = P_("design.n_units_id"), P_("design.n_train_units"), P_("design.n_units_ood")
    if not 0 < ntr < nid:
        raise ValueError("need 0 < n_train_units < n_units_id")
    T_lo, T_hi = P_("states.T_hs_K")
    S_lo, S_hi = P_("states.SNR_ref_dB")
    f_lo, f_hi = P_("states.sigma_frac")
    T_id, S_id, f_id = P_("id_region.T_hs_max_K"), P_("id_region.SNR_max_dB"), P_("id_region.sigma_frac_max")
    s_lo, s_hi = P_("id_region.stratum_T_hs_K")
    if not (T_lo < s_lo < s_hi == T_id < T_hi and P_("id_region.stratum_share") == 0.5):
        raise ValueError("T strata must tile [T_lo, T_hi] as base < stratum <= T_id < OOD-T, with share 0.5")
    units = [(u, "train" if u < ntr else "test_id") for u in range(nid)]
    units += [(nid + j * nood + k, split) for j, split in enumerate(SPLITS_OOD) for k in range(nood)]
    rows, spu = [], P_("design.states_per_unit")
    for u, split in units:
        rng = V.rng_for(cfg, V.STREAM_STATE, u)
        for k in range(spu):
            stratum = "ood_T" if split == "ood_T" else ("foldback" if k % 2 else "base")
            T = rng.uniform(*{"ood_T": (T_id, T_hi), "foldback": (s_lo, s_hi), "base": (T_lo, s_lo)}[stratum])
            S = rng.uniform(S_id, S_hi) if split == "ood_SNR" else rng.uniform(S_lo, S_id)
            f = rng.uniform(f_id, f_hi) if split == "ood_sigma" else rng.uniform(f_lo, f_id)
            rows.append({
                "state_id": u * spu + k, "unit_id": u, "split": split, "stratum": stratum,
                "I_rated": rng.uniform(*P_("states.I_rated_A")), "sigma_frac": f, "T_hs": T,
                "SNR_ref_dB": S, "M": int(rng.choice(P_("states.M"))), "probe_frac": rng.uniform(*P_("states.probe_frac")),
            })
    return rows


def binding(cfg, L1, L1_free, Ir):
    """The ceiling binds when it moves the analytic L1 by more than one grid step."""
    return bool(abs(L1 - L1_free) > V.P(cfg, "ofdm.grid_step_frac") * Ir + 1e-12)


def probe_bias(cfg, row, Ir, C, T_ntc, i_min):
    """Probe bias: probe_frac * I_rated, capped at 0.9 x the ceiling the controller reads from its NTC at
    probe time.
    The reading is T_ntc + sigma_T_ntc * z0 with z0 the first STREAM_SENSE normal of rep 0, the same
    draw V.observe uses for T_ntc_hat at lambda_T = 1, so b_p and C_hat agree at nominal sensing. The
    driver clips any command above the true ceiling at C (essentially never: needs C_hat > C / 0.9)."""
    nw = V.P(cfg, "sensing.window_symbols") * V.P(cfg, "ofdm.N")
    z0 = V.rng_for(cfg, V.STREAM_SENSE, row["state_id"], 0).standard_normal(5 + nw)[0]
    C_hat = float(V.foldback_ceiling(cfg, Ir, T_ntc + V.P(cfg, "sensing.sigma_T_ntc") * z0))
    return max(min(row["probe_frac"] * Ir, 0.9 * C_hat, C), i_min)


def simulate(args):
    cfg, row = args
    quad = V.quadrature(cfg)
    st, dev, unit = V.build(cfg, row)
    Ir, C, T_ntc = st["I_rated"], st["C"], st["T_ntc"]
    g = row["state_id"]

    b = V.bias_grid(cfg, Ir, C)
    d = V.draws(cfg, V.rng_for(cfg, V.STREAM_LABEL, g), V.P(cfg, "ofdm.R_label"), row["M"])
    lb, _ = V.mc_curve(cfg, st, dev, b, d, quad)
    lab = V.labels_from_curve(b, lb, cfg)
    b_p = probe_bias(cfg, row, Ir, C, T_ntc, dev["i_min"])
    an = V.analytic_curve(cfg, st, dev, np.append(b, b_p), quad)
    lb_an, sinr_db = an["lb"][:-1], 10 * np.log10(np.mean(an["sinr"], axis=1))
    j_an = int(np.argmin(lb_an))
    # foldback-binding diagnostic: analytic L1 under the ceiling vs with the ceiling at I_rated
    b_free = V.bias_grid(cfg, Ir, Ir)
    L1_free = b_free[int(np.argmin(V.analytic_curve(cfg, st | {"C": Ir}, dev, b_free, quad)["lb"]))]
    Tj_L1 = float(V.thermal([lab["L1"]], st["sigma"], C, row["T_hs"], dev, quad)["Tj"][0])

    th_p = V.thermal([b_p], st["sigma"], C, row["T_hs"], dev, quad)
    dp = V.draws(cfg, V.rng_for(cfg, V.STREAM_PROBE, g), V.P(cfg, "ofdm.R_probe"), row["M"])

    out = dict(row)
    out.update({f"u_{k}": v for k, v in unit.items()})
    out.update({
        "sigma_I": st["sigma"], "T_ntc": T_ntc, "C": C, "n_grid": len(b),
        "L1": lab["L1"], "L1_idx": lab["L1_idx"], "lb_min": lab["lb_min"], "boundary": lab["boundary"],
        "L2": lab["L2"], "L2_feasible": lab["L2_feasible"], "L2_38": lab["L2_38"], "L2_38_feasible": lab["L2_38_feasible"],
        "L1_analytic": b[j_an], "n_local_min": V.n_local_min(lb_an),
        "binding": binding(cfg, b[j_an], L1_free, Ir),
        "Tj_L1": Tj_L1, "tj_ok": bool(Tj_L1 <= V.P(cfg, "led.Tj_max_K")),
        "sinr_p_dB": sinr_db[-1], "sinr_L1_dB": sinr_db[lab["L1_idx"]],
        "b_p": b_p, "Tj_p": th_p["Tj"][0], "P_diss_p": th_p["P_diss"][0], "P_elec_p": th_p["P_elec"][0],
        "Prx_dB": 10 * np.log10(th_p["Ep"][0]), "EVM_p_dB": V.evm_db(cfg, st, dev, b_p, dp, quad)[0],
    })
    curves = {"label_b": b, "label_lb": lb}

    if row["split"] != "train":
        do = V.draws(cfg, V.rng_for(cfg, V.STREAM_ORACLE, g), V.P(cfg, "ofdm.R_oracle"), row["M"])
        lo, tho = V.mc_curve(cfg, st, dev, b, do, quad)
        bo = V.refine_grid(cfg, b, int(np.argmin(lo)), Ir, C)
        new = ~np.isin(bo, b)
        lo_new, tho_new = V.mc_curve(cfg, st, dev, bo[new], do, quad)
        lbo, pel = np.empty(len(bo)), np.empty(len(bo))
        lbo[~new], lbo[new] = lo, lo_new
        pel[~new], pel[new] = tho["P_elec"], tho_new["P_elec"]
        olab = V.labels_from_curve(bo, lbo, cfg)
        out.update({"O_L1": olab["L1"], "O_lb_min": olab["lb_min"], "O_L2": olab["L2"], "O_L2_feasible": olab["L2_feasible"],
                    "O_L2_38": olab["L2_38"], "O_L2_38_feasible": olab["L2_38_feasible"], "O_boundary": olab["boundary"]})
        curves.update({"oracle_b": bo, "oracle_lb": lbo, "oracle_pelec": pel})
    return out, curves


def pad(rows, key, width):
    a = np.full((len(rows), width), np.nan)
    for n, c in enumerate(rows):
        if key in c:
            a[n, : len(c[key])] = c[key]
    return a


def git_info():
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()
    sha = run("rev-parse", "HEAD")
    if not sha:
        raise RuntimeError("git SHA unavailable: provenance must name the code version")
    return {"sha": sha, "dirty": bool(run("status", "--porcelain", "--", "dcbias"))}


def parse_sets(items):
    return {k: json.loads(v) for k, v in (s.split("=", 1) for s in items or [])}


def generate(cfg, out_dir, workers=1, limit=None, splits=None):
    """splits: keep only these design splits (shifted twins of the test_id units);
    every kept row is identical to the full design's, since each unit draws from its own stream."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; a dataset is never overwritten")
    rows = [r for r in design(cfg) if splits is None or r["split"] in splits][:limit]
    t0 = time.time()
    if workers > 1:
        for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = "1"  # spawned workers: one BLAS thread each, no oversubscription
        with Pool(workers) as pool:
            res = pool.map(simulate, [(cfg, r) for r in rows], chunksize=4)
    else:
        res = [simulate((cfg, r)) for r in rows]
    df = pd.DataFrame([r for r, _ in res])
    cv = [c for _, c in res]
    out_dir.mkdir(parents=True)
    df.to_csv(out_dir / "states.csv", index=False)
    np.savez_compressed(out_dir / "curves.npz", state_id=df["state_id"].to_numpy(),
                        **{k: pad(cv, k, max((len(c.get(k, [])) for c in cv), default=0))
                           for k in ("label_b", "label_lb", "oracle_b", "oracle_lb", "oracle_pelec")})
    diag = {
        "n_states": len(df), "runtime_s": time.time() - t0, "workers": workers, "splits": splits,
        "boundary_share_L1": float(df["boundary"].mean()),
        "boundary_rows": df.loc[df["boundary"], "state_id"].tolist(),
        "L2_feasible_share": float(df["L2_feasible"].mean()),
        "mean_abs_L1_mc_minus_analytic_over_Irated": float(((df["L1"] - df["L1_analytic"]).abs() / df["I_rated"]).mean()),
        "binding_share_by_split_stratum": {f"{a}/{b}": float(g.mean()) for (a, b), g in df.groupby(["split", "stratum"])["binding"]},
        "tj_drop_share_by_split": {k: float(1 - g.mean()) for k, g in df.groupby("split")["tj_ok"]},
        # the Tj check is inert by design: report the headroom, not only the drop rate
        "tj_margin_min_K_by_split": {k: float(V.P(cfg, "led.Tj_max_K") - g.max()) for k, g in df.groupby("split")["Tj_L1"]},
        "multimodal_share_by_split": {k: float((g > 1).mean()) for k, g in df.groupby("split")["n_local_min"]},
        "realised_sinr_L1_dB_range_by_split": {k: [float(g.min()), float(g.max())] for k, g in df.groupby("split")["sinr_L1_dB"]},
    }
    prov = {"git": git_info(), "config_hash": V.config_hash(cfg), "config": cfg, "argv": sys.argv,
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "python": sys.version},
            "diagnostics": diag}
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    return df, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output folder under runs/")
    ap.add_argument("--set", nargs="*", help="config overrides key=json, e.g. led.delta=0.003")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first n states only (smoke tests)")
    ap.add_argument("--splits", nargs="*", default=None, help="keep only these splits, e.g. test_id")
    a = ap.parse_args()
    cfg = V.load_config(overrides=parse_sets(a.set))
    _, diag = generate(cfg, RUNS_DIR / a.name, a.workers, a.limit, a.splits)
    print(json.dumps({k: v for k, v in diag.items() if k != "boundary_rows"}, indent=1))
    if diag["boundary_share_L1"] > 0.005:
        print(f"WARNING: boundary label share {diag['boundary_share_L1']:.3%} > 0.5 % (investigate)", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
