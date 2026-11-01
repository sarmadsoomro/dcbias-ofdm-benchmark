"""Wide-prior B7 on the i_min twins, extra contrasts: per deploy, the
FEC-relevant Delta against the best ML seed mean, the per-seed range of that Delta (full set and
FEC-relevant), and the difference in differences against the identity deployment, as task_shift
reports them for its interpreted keys. Reprocess only: reads the saved rows, predicts nothing.

Rows: shift-rows/b7wide-<dep>.npz (task_b7wide_imin) paired with shift-rows/<dep>.npz and
shift-rows/identity.npz (task_shift). The script asserts that its full-set Delta reproduces
results_b7wide_imin.json before writing anything.

    python -m dcbias.b7_wide_extras runs/baseline  ->  runs/baseline/results_b7wide_extras.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from dcbias import evaluate as E


def seed_range(r, z, best_ml, mask):
    d = [float((r[mask] - z[f"regret:{best_ml}@seed{s}"][mask]).mean()) for s in range(5)]
    return [min(d), max(d)]


def main(run_dir):
    run = Path(run_dir)
    main_ = json.loads((run / "results_main.json").read_text())
    wide = json.loads((run / "results_b7wide_imin.json").read_text())
    B = json.loads((run / "provenance.json").read_text())["config"]["evaluation"]["bootstrap_B"]["value"]
    best_ml = main_["best_ml_state_obs"]
    ml = f"regret:{best_ml}@seedmean"
    ident = np.load(run / "shift-rows" / "identity.npz")
    rid = ident["regret:B7_bayes_wide"]
    out = {"source": "shift-rows/b7wide-<dep>.npz + shift-rows/<dep>.npz + identity.npz (reprocess)",
           "positive": "ML better (wide-prior B7 higher regret)", "best_ml_state_obs": best_ml, "by_deploy": {}}
    for dep in E.B7_WIDE_IMIN:
        w = np.load(run / "shift-rows" / f"b7wide-{dep}.npz")
        z = np.load(run / "shift-rows" / f"{dep}.npz")
        if not (np.array_equal(w["state_id"], z["state_id"]) and np.array_equal(z["state_id"], ident["state_id"])):
            raise ValueError(f"{dep}: rows not aligned")
        r, units, fec = w["regret"], z["unit_id"], z["fec_ok"]
        full = E.cluster_boot(units, r, z[ml], B)[0]
        want = wide["by_deploy"][dep]["delta_minus_best_ml"]["point"]
        if not np.isclose(full["point"], want, rtol=0, atol=1e-12):
            raise ValueError(f"{dep}: full-set Delta {full['point']} != results_b7wide_imin {want}")
        allr = np.ones(len(r), bool)
        out["by_deploy"][dep] = {
            "delta_minus_best_ml": full,
            "delta_minus_best_ml_seed_range": seed_range(r, z, best_ml, allr),
            "did_delta": E.cluster_boot(units, r - z[ml], rid - ident[ml], B)[0],
            "fec_relevant": {
                "n_rows": int(fec.sum()), "n_units": int(len(np.unique(units[fec]))),
                "delta_minus_best_ml": E.cluster_boot(units[fec], r[fec], z[ml][fec], B)[0],
                "delta_minus_best_ml_seed_range": seed_range(r, z, best_ml, fec),
                # context: the nominal-prior B7 on the same rows
                "B7_bayes_delta_minus_best_ml": E.cluster_boot(units[fec], z["regret:B7_bayes"][fec], z[ml][fec], B)[0]}}
    E.write(run, "b7wide_extras", out)


if __name__ == "__main__":
    main(sys.argv[1])
