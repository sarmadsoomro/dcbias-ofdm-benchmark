"""Band-resolved read-out of the counted-bit run. No new counting.

Reads counted-rows/*.npz (written by counted.py) and the saved shift-rows/*.npz. For each stress twin and
controller and each band of twin oracle log10 BER on S, plus the off-S rows (PCHIP only):
n and share of the full-set PCHIP cost; mean counted, semi and PCHIP cost with unit-bootstrap CIs; counted - semi
in dec and in % of the semi cost; the S-only full-set floor (cost 0 off S) with CI; B7 - MLP twin regret by band.
Asserts that the on-S counted cost reproduces results_counted.json before writing.

    python -m dcbias.counted_bands runs/baseline   ->  results_counted_bands.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from dcbias import counted as C

MLP = f"{C.MLP}@seedmean"


def load_regrets(path):
    """counted-rows npz -> (state_id, unit_id, oracle_lb, {read-out: {key: per-row regret}}) for counted and semi."""
    z = np.load(path, allow_pickle=False)
    assert tuple(z["keys"]) == ("O_L1",) + C.KEYS, "column order changed since the run"
    r = C.regrets(z["errors"], z["nbits"], z["semi_lb"])
    return z["state_id"], z["unit_id"], z["oracle_lb"], {m: C.with_seedmean(v) for m, v in r.items()}


def pchip_regrets(path):
    z = np.load(path, allow_pickle=False)
    return z["state_id"], C.with_seedmean(np.column_stack([z[f"regret:{k}"] for k in C.KEYS]))


def bands(tw, idn, full, inS, lbt, units, units_full, keys=C.REPORT, contrast=("B7_bayes", MLP)):
    """One twin. tw / idn: read-out -> key -> per-row regret on S (same states); full: key -> full-set PCHIP cost
    per row; inS: mask of S over the full rows; lbt: twin oracle log10 BER on S. Returns key -> band -> stats."""
    out = {}
    for k in keys:
        c = {m: tw[m][k] - idn[m][k] for m in ("counted", "semi", "pchip")}
        tot = full[k].sum()
        floor = np.zeros_like(full[k])
        floor[inS] = c["counted"]
        e = {"S_only_floor": C.boot(units_full, floor)}
        for lo, hi in C.BANDS:
            m = (lbt >= lo) & (lbt < hi)
            if not m.any():
                continue
            cs = C.boot(units[m], c["counted"][m], c["semi"][m])
            semi = float(c["semi"][m].mean())
            e[f"[{lo},{hi})"] = {
                "n": int(m.sum()), "share_of_full_set_pchip_cost": float(full[k][inS][m].sum() / tot),
                **{m_: C.boot(units[m], c[m_][m]) for m_ in ("counted", "semi", "pchip")},
                "counted_minus_semi": cs, "counted_minus_semi_pct_of_semi": 100 * cs["point"] / semi if semi else None,
                "contrast_twin": {m_: C.boot(units[m], tw[m_][contrast[0]][m], tw[m_][contrast[1]][m]) for m_ in ("counted", "semi", "pchip")}}
        off = ~inS
        e["off_S"] = {"n": int(off.sum()), "share_of_full_set_pchip_cost": float(full[k][off].sum() / tot),
                      "pchip": C.boot(units_full[off], full[k][off])} if off.any() else {"n": 0}
        out[k] = e
    return out


def main(run_dir):
    run_dir = Path(run_dir)
    ref = json.loads((run_dir / "results_counted.json").read_text())
    sid_i, _, _, ident = load_regrets(run_dir / "counted-rows" / "identity.npz")
    sid_full, pi = pchip_regrets(run_dir / "shift-rows" / "identity.npz")
    pos_i, pos_full = {s: j for j, s in enumerate(sid_i)}, {s: j for j, s in enumerate(sid_full)}
    df = C.deploy(run_dir, "identity")[0]  # unit_id for every test_id row; off-S rows are absent from counted-rows
    assert np.array_equal(df["state_id"].to_numpy(), sid_full)
    units_full = df["unit_id"].to_numpy()
    res = {"source": "results_counted.json + counted-rows/ + shift-rows/ (no recount)",
           "results_counted_provenance": ref["_eval_provenance"], "bands": [list(b) for b in C.BANDS], "by_deploy": {}}
    for name in C.DEPLOYS:
        sid, units, lbt, tw = load_regrets(run_dir / "counted-rows" / f"{name}.npz")
        sid_t, pt = pchip_regrets(run_dir / "shift-rows" / f"{name}.npz")
        assert np.array_equal(sid_t, sid_full), f"{name}: shift-rows states differ from identity"
        ii = np.array([pos_i[s] for s in sid])
        inS = np.zeros(len(sid_full), bool)
        inS[[pos_full[s] for s in sid]] = True
        assert np.array_equal(sid_full[inS], sid), f"{name}: counted-rows order differs from shift-rows order on S"
        full = {k: pt[k] - pi[k] for k in C.REPORT}
        tw["pchip"] = {k: v[inS] for k, v in pt.items()}
        idn = {m: {k: v[ii] for k, v in ident[m].items()} for m in ("counted", "semi")} | {"pchip": {k: v[inS] for k, v in pi.items()}}
        e = bands(tw, idn, full, inS, lbt, units, units_full)
        for k in C.REPORT:  # the on-S counted cost must reproduce the registered run
            got = float((tw["counted"][k] - idn["counted"][k]).mean())
            want = ref["by_deploy"][name]["cost"][k]["counted"]["point"]
            assert abs(got - want) < 1e-12, f"{name} {k}: {got} != {want}"
        res["by_deploy"][name] = {"n_rows": int(len(sid_full)), "n_S": int(inS.sum()), "by_key": e}
    out = run_dir / "results_counted_bands.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
