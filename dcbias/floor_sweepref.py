"""Sweep-reference contrast, regret noise floor and oracle-bias displacement. Reprocess only: reads stored results, predicts nothing.

1. Sweep contrast: each sweep's Delta minus the 50-unit reference Delta (sweep-nominal), so the sweep direction is
read against a reference trained on the same 40 units. Per-row regrets are not stored for the sweeps,
so the interval treats the two runs as independent (SE from each 95 % CI half-width / 1.96). The
sweeps share units and noise streams with the reference, which correlates them positively, so this
interval is wider than a paired one would be.

2. Noise floor of the regret metric on the ID test states.
  (i)  realised PCHIP error: regret from the stored direct Monte Carlo read-out at each commanded bias
       (same realisation as the oracle, and an independent one) minus the stored PCHIP regret, per
       controller in shift-rows/identity.npz;
  (ii) the training label's own regret: the R_label minimiser L1 scored on the R_oracle curve, as
       score_test scores every controller.
The script asserts that its PCHIP regrets reproduce results_main.json before writing anything.

3. Displacement: the oracle-bias displacement on each device-shift twin, in units of sigma_I, from the stored labels.

    python -m dcbias.floor_sweepref runs sweep- runs/baseline   (twins: runs/twin-*)  ->  runs/baseline/results_floor_sweepref.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dcbias import evaluate as E
from dcbias import physics as V


def sweep_ref(root, prefix, ref="nominal"):
    s = json.loads((Path(root) / f"{prefix}{ref}" / "results_main.json").read_text())["confirmatory"]["delta"]["test_id"]
    se = lambda d: (d["ci"][1] - d["ci"][0]) / (2 * 1.96)  # noqa: E731
    by = {}
    for d in sorted(Path(root).glob(f"{prefix}*")):
        name = d.name.removeprefix(prefix)
        if name == ref:
            continue
        x = json.loads((d / "results_main.json").read_text())["confirmatory"]["delta"]["test_id"]
        p, h = x["point"] - s["point"], 1.96 * np.hypot(se(x), se(s))
        by[name] = {"delta": x["point"], "delta_minus_ref": {"point": p, "ci": [p - h, p + h]}}
    pts = [r["delta_minus_ref"]["point"] for r in by.values()]
    return {"reference": ref, "reference_delta": s, "ci_method": "independent normal approximation (conservative)",
            "by_sweep": by,
            "range": {"n": len(by), "point_min": min(pts), "point_max": max(pts),
                      "n_point_above_0": sum(p > 0 for p in pts), "n_point_below_0": sum(p < 0 for p in pts),
                      "n_ci_above_0": sum(r["delta_minus_ref"]["ci"][0] > 0 for r in by.values()),
                      "n_ci_below_0": sum(r["delta_minus_ref"]["ci"][1] < 0 for r in by.values())}}


def floor(run_dir):
    run = Path(run_dir)
    main_ = json.loads((run / "results_main.json").read_text())
    df, cv, cfg = E.load(run)
    B = cfg["evaluation"]["bootstrap_B"]["value"]
    fl = np.log10(V.P(cfg, "ber.regret_floor"))
    z = np.load(run / "shift-rows" / "identity.npz")
    ro = [str(k) for k in z["ro_keys"]]
    units = z["unit_id"]

    # (ii) label regret on test_id, scored exactly as score_test scores a controller
    rows = np.flatnonzero(df["split"].to_numpy() == "test_id")
    if not np.array_equal(df["state_id"].to_numpy()[rows], z["state_id"]):
        raise ValueError("identity rows are not the test_id rows")
    cur = E.curve_rows(cv, "oracle", rows)
    lab = E.regret(df["L1"].to_numpy()[rows], cur, V.P(cfg, "ber.regret_floor"))
    for k in ("B7_bayes", main_["best_ml_state_obs"]):
        got = E.regret(z[f"bhat:{k}" if k == "B7_bayes" else f"bhat:{k}@seed0"], cur, V.P(cfg, "ber.regret_floor"))
        want = z[f"regret:{k}" if k == "B7_bayes" else f"regret:{k}@seed0"]
        if not np.allclose(got, want, rtol=0, atol=1e-12):
            raise ValueError(f"{k}: recomputed PCHIP regret does not reproduce identity.npz")
    if not np.isclose(z["regret:B7_bayes"].mean(), main_["test"]["test_id"]["B7_bayes"]["mean_regret"], rtol=0, atol=1e-12):
        raise ValueError("identity.npz B7 regret does not reproduce results_main")
    fec = z["fec_ok"]
    q = lambda r, u: {"mean": E.cluster_boot(u, r, B=B)[0], "median": float(np.median(r)),  # noqa: E731
                      "p95": float(np.quantile(r, 0.95)), "share_zero": float(np.mean(r <= 1e-12))}
    label = {"all": q(lab, units), "fec_relevant": q(lab[fec], units[fec]) | {"n_rows": int(fec.sum())}}
    # the label's floor against the controllers on the same rows (positive: controller above the label)
    vs = {k: E.cluster_boot(units, z[f"regret:{k}"], lab, B)[0]
          for k in ("B7_bayes", "B6b_model_grid", f"{main_['best_ml_state_obs']}@seedmean", "ML_ridge_state_obs@seedmean")}

    # (i) realised read-out error per controller; last column of each direct block is O_L1
    same, ind = z["direct:same_realisation"], z["direct:independent"]
    if not np.allclose(same[:, -1], [c.min for c in cur], rtol=0, atol=1e-9):
        raise ValueError("same-realisation O_L1 column does not reproduce the oracle minimum")
    best = np.maximum(same[:, -1], fl)
    interp = {}
    for j, k in enumerate(ro):
        r = z[f"regret:{k}"]
        es = np.maximum(same[:, j], fl) - best - r
        ei = np.maximum(ind[:, j], fl) - best - r
        interp[k] = {"pchip_regret": float(r.mean()),
                     "same_realisation_minus_pchip": E.cluster_boot(units, es, B=B)[0], "same_row_rms": float(np.sqrt(np.mean(es**2))),
                     "same_row_max_abs": float(np.abs(es).max()),
                     "independent_minus_pchip": E.cluster_boot(units, ei, B=B)[0], "independent_row_rms": float(np.sqrt(np.mean(ei**2)))}
    e_all = [interp[k]["same_realisation_minus_pchip"]["point"] for k in ro]
    return {"label_regret_test_id": label, "controller_minus_label": vs, "readout_error": interp,
            "readout_error_range": {"same_mean_min": min(e_all), "same_mean_max": max(e_all),
                                    "same_row_rms_max": max(v["same_row_rms"] for v in interp.values())},
            "R_label": V.P(cfg, "ofdm.R_label"), "R_oracle": V.P(cfg, "ofdm.R_oracle")}


def displacement(root, run_dir, prefix=E.TWIN_PREFIX):
    """Oracle-bias displacement per twin, (O_L1 on the twin - O_L1 on the unshifted unit) / sigma_I,
    on the test_id states (quartiles over states)."""
    ref = E.load(run_dir)[0]
    ref = ref[ref["split"] == "test_id"].set_index("state_id")
    out = {}
    for d in sorted(Path(root).glob(f"{prefix}*")):
        t = pd.read_csv(d / "states.csv", low_memory=False).set_index("state_id").loc[ref.index]
        if not np.allclose(t["sigma_I"], ref["sigma_I"], rtol=0, atol=1e-12):
            raise ValueError(f"{d.name}: states not aligned with {run_dir} test_id")
        x = ((t["O_L1"] - ref["O_L1"]) / ref["sigma_I"]).to_numpy()
        out[d.name.removeprefix(prefix)] = {"n": int(len(x)), "median": float(np.median(x)),
                                            "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75))}
    return out


def main(root, prefix, run_dir):
    out = {"source": "sweep results_main.json (sweep contrast); baseline curves + shift-rows/identity.npz (noise floor); twin-*/states.csv (displacement); "
                     "reprocess",
           "positive": "sweep contrast: sweep Delta above the reference (learning relatively better); noise floor: controller regret above the label's; "
                       "displacement: twin oracle bias above the nominal one",
           "sweep_minus_reference": sweep_ref(root, prefix), "noise_floor": floor(run_dir),
           "oracle_displacement_sigma": displacement(root, run_dir)}
    E.write(Path(run_dir), "floor_sweepref", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
