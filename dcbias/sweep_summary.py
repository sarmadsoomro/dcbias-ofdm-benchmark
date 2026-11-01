"""Collect the per-sweep test_id comparison into one small JSON beside the main run, so macros.py turns
it into \\vnum keys. Rendering every
sweep's full results as macros would be ~70 MB of TeX; the paper only cites this summary.

    \\vnum{sweeps/by_sweep/led.model=tsonev_cubic/delta/point}
    \\vnum{sweeps/range/delta_point_min}

Usage: python -m dcbias.sweep_summary <runs> <prefix> <out.json>   e.g. runs sweep- runs/baseline/results_sweeps.json
"""
import json
import sys

from pathlib import Path


def summarise(root, prefix):
    by = {}
    for d in sorted(Path(root).glob(f"{prefix}*")):
        m = json.loads((d / "results_main.json").read_text())
        non, ml = m["best_non_ml"], m["best_ml_state_obs"]
        c = m["confirmatory"]
        row = {"best_non_ml": non, "best_ml": ml, "framing": c["framing"],
               "regret_non_ml": m["test"]["test_id"][non]["mean_regret"],
               "regret_ml": m["test"]["test_id"][ml]["mean_regret"],
               "delta": c["delta"]["test_id"], "eval_sha": m["_eval_provenance"]["sha"]}
        if (d / "results_misspec.json").exists():
            row["misspec"] = json.loads((d / "results_misspec.json").read_text())["delta_minus_best_ml"]
        by[d.name.removeprefix(prefix)] = row
    if not by:
        raise SystemExit(f"no {prefix}* runs under {root}")
    pts = [r["delta"]["point"] for r in by.values()]
    rng = {"n": len(by), "delta_point_min": min(pts), "delta_point_max": max(pts),
           "ci_abs_max": max(abs(x) for r in by.values() for x in r["delta"]["ci"]),
           "n_framing_b": sum(r["framing"].startswith("(b)") for r in by.values()),
           "n_best_non_ml_B7": sum(r["best_non_ml"] == "B7_bayes" for r in by.values()),
           "n_ci_above_0": sum(r["delta"]["ci"][0] > 0 for r in by.values()),
           "n_ci_below_0": sum(r["delta"]["ci"][1] < 0 for r in by.values())}
    return {"by_sweep": by, "range": rng}


if __name__ == "__main__":
    root, prefix, out = sys.argv[1:4]
    Path(out).write_text(json.dumps(summarise(root, prefix), indent=1))
    print(f"wrote {out}")
