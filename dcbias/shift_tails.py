"""Shift-arm reporting summary: per deploy, the MLP seed-mean regret
median / p90 / p99 and the share of regret carried by the top 5 % of rows, and the range of the
primary contrast over MLP seeds.

Per-row predictions come from an independent refit of the MLP (`<run>/refit-shift/*.npz`; seeds 0-4 for all 14 deploys). The script asserts that its seed-mean
regret reproduces results_shift.json before writing anything, so the tails describe the same numbers.

    python -m dcbias.shift_tails runs/baseline   ->  runs/baseline/results_shifttails.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from dcbias import evaluate as E


def tails(r):
    s = np.sort(r)[::-1]
    return {"mean": float(r.mean()), "median": float(np.median(r)), "p90": float(np.quantile(r, 0.90)),
            "p99": float(np.quantile(r, 0.99)), "top5_share": float(s[: int(np.ceil(0.05 * len(s)))].sum() / s.sum())}


def main(run_dir):
    run = Path(run_dir)
    shift = json.loads((run / "results_shift.json").read_text())["by_deploy"]
    floor = json.loads((run / "provenance.json").read_text())["config"]["ber"]["regret_floor"]["value"]
    out = {"source": "refit-shift/*.npz (independent refit), checked against results_shift.json",
           "by_deploy": {}}
    for dep, v in shift.items():
        z = np.load(run / "refit-shift" / f"{dep}.npz")
        cur = [E.Curve(b[~np.isnan(b)], lb[~np.isnan(b)]) for b, lb in zip(z["oracle_b"], z["oracle_lb"])]
        r = np.mean([E.regret(z[f"mlp{s}"], cur, floor) for s in range(5)], axis=0)
        want = v["mean_regret"]["ML_mlp_state_obs@seedmean"]
        if not np.isclose(r.mean(), want, rtol=0, atol=1e-9):
            raise ValueError(f"{dep}: seed-mean regret {r.mean()} != results_shift {want}")
        ps = [p["point"] for p in v["primary_by_seed"].values()]
        idm = shift["identity"]["mean_regret"]
        cost = {m: x - idm[m] for m, x in v["mean_regret"].items() if m in idm and "@seed" not in m}
        cost.update({m: x - idm[m] for m, x in v["mean_regret"].items() if m.endswith("@seedmean")})
        out["by_deploy"][dep] = {"mlp_seedmean": tails(r), "primary_seed_min": min(ps), "primary_seed_max": max(ps),
                                 "cost_point_all": cost}  # point cost for every controller; CIs only in results_shift
        print(f"{dep:26s} mean {r.mean():.4f} median {np.median(r):.4f} p90 {np.quantile(r, .9):.4f} "
              f"seed range {min(ps):+.4f}..{max(ps):+.4f}")
    (run / "results_shifttails.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
