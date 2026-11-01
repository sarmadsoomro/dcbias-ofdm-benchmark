"""Counted-bit check of the stress-shift costs.

For the twins kappa_tilde 0 and i_min 0 and the identity deploy, every test_id state whose oracle minimum BER
is >= 1e-7 on both deploys is re-evaluated at the oracle bias and at the commanded biases of B7, B6b and the
five MLP seeds (read from the saved shift-rows/*.npz, no refit). Bit errors are counted on a fresh stream;
on the same received samples the semi-analytic EVM BER is computed, so counted - semi isolates the
Gaussian-error assumption (both use the genie-aided LS gain of V.mc_receive). The number of 1024-symbol
batches per row is fixed before counting from the oracle curve.

Rules reported per deploy (rule names are the keys in results_counted.json):
  R1  agreement: the counted-minus-semi-analytic cost interval lies within +/-TOL_R1 decades (PASS / FAIL).
  R2  position of the corrected full-set cost interval against the 0.10-decade reference
      (above / below / straddles).
  R3  at i_min = 0 only: whether the counted B7 - MLP twin contrast is positive (resolved / unresolved).

    python -m dcbias.counted runs/baseline --workers 9 2>&1 | tee runs/baseline/counted.log
        ->  results_counted.json, counted-rows/*.npz
"""
import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from dcbias import evaluate as E
from dcbias import generate as G
from dcbias import physics as V

DEPLOYS = ("led.kappa_tilde=0", "led.i_min_frac=0.0")
MLP = "ML_mlp_state_obs"
KEYS = ("B7_bayes", "B6b_model_grid") + tuple(f"{MLP}@seed{s}" for s in range(5))
REPORT = ("B7_bayes", "B6b_model_grid", f"{MLP}@seedmean")
STREAM_COUNTED_SHIFT = 12  # unseen stream: not label 2, oracle 3, gate 7, read-out 10, or 11
LB_MIN = -7.0  # stratum S: oracle minimum log10 BER >= this on identity and twin
TARGET_ERRORS = 250
MAX_BATCHES = 4096
TOL_R1 = 0.03
REF = 0.10
BANDS = ((-3.0, np.inf), (-4.5, -3.0), (-6.0, -4.5), (-7.0, -6.0))  # twin oracle log10 BER bands for the R1 read-out


def n_batches(lb_min, M, cfg):
    """Batches of R_oracle symbols giving TARGET_ERRORS expected errors at the oracle's semi-analytic BER."""
    bits = V.P(cfg, "ofdm.R_oracle") * V.P(cfg, "ofdm.Nu") * int(np.log2(M))
    return int(np.clip(np.ceil(TARGET_ERRORS / (10.0 ** lb_min * bits)), 1, MAX_BATCHES))


def count_row(args):
    """Counted bit errors and same-sample semi-analytic log10 BER at each bias in b, over nb batches.
    stream == STREAM_ORACLE with nb == 1 draws exactly the oracle's realisation (preflight)."""
    cfg, row, b, nb, stream = args
    assert stream != V.STREAM_ORACLE or nb == 1, "the oracle stream has one realisation per state"
    st, dev, _ = V.build(cfg, row)
    M, R = st["M"], V.P(cfg, "ofdm.R_oracle")
    quad, bits = V.quadrature(cfg), V.gray_bits(M)
    b = np.asarray(b, float)
    err, e2, nsym, nbits = np.zeros(len(b)), np.zeros((len(b), V.P(cfg, "ofdm.Nu"))), 0, 0
    for k in range(nb):
        rng = V.rng_for(cfg, stream, row["state_id"]) if stream == V.STREAM_ORACLE else V.rng_for(cfg, stream, row["state_id"], k)
        d = V.draws(cfg, rng, R, M)
        Yn, _ = V.mc_receive(cfg, st, dev, b, d, quad)
        ref = bits[d["idx"]]
        for j in range(len(b)):
            err[j] += np.count_nonzero(bits[V.hard_decide(Yn[j], M)] != ref)
        e2 += np.sum(np.abs(Yn - d["S"][None]) ** 2, axis=1)
        nsym, nbits = nsym + R, nbits + d["idx"].size * int(np.log2(M))
    return err, nbits, V.log10_ber(nsym / e2, M)


def pool_map(fn, jobs, workers):
    if workers > 1:
        for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = "1"
        with Pool(workers) as p:
            return p.map(fn, jobs, chunksize=1)  # rows differ up to 4000x in cost
    return [fn(j) for j in jobs]


def deploy(run_dir, name):
    """(test_id rows, curves, true-device cfg, shift-rows npz) for one deploy, aligned by state_id."""
    d = run_dir if name == "identity" else run_dir.parent / f"{E.TWIN_PREFIX}{name}"
    df, cv, cfg = E.load(d)
    rows = np.flatnonzero(df["split"] == "test_id")
    z = np.load(run_dir / "shift-rows" / f"{name}.npz", allow_pickle=False)
    if not np.array_equal(z["state_id"], df["state_id"].to_numpy()[rows]):
        raise AssertionError(f"{name}: shift-rows not aligned with test_id rows")
    return df.iloc[rows].reset_index(drop=True), {k: v[rows] for k, v in cv.items()}, cfg, z


def biases(df, cv, z):
    """Columns: O_L1, then KEYS, clamped to the oracle-curve span as Curve does."""
    lo, hi = np.nanmin(cv["oracle_b"], axis=1), np.nanmax(cv["oracle_b"], axis=1)
    return np.column_stack([df["O_L1"].to_numpy()] + [np.clip(z[f"bhat:{k}"], lo, hi) for k in KEYS])


def preflight(cfg, df, cv, n=5):
    rows = np.linspace(0, len(df) - 1, n).astype(int)
    for r in rows:
        _, _, semi = count_row((cfg, df.iloc[r].to_dict(), [df["O_L1"].iloc[r]], 1, V.STREAM_ORACLE))
        if not np.isclose(semi[0], np.nanmin(cv["oracle_lb"][r]), rtol=0, atol=1e-9):
            raise AssertionError(f"state {df['state_id'].iloc[r]}: same-sample semi BER does not reproduce the oracle minimum")


def run_counts(cfg, df, bh, lb, rows, workers):
    nb = np.array([n_batches(lb[r], int(df["M"].iloc[r]), cfg) for r in rows])
    jobs = [(cfg, df.iloc[r].to_dict(), bh[r], k, STREAM_COUNTED_SHIFT) for r, k in zip(rows, nb)]
    out = pool_map(count_row, jobs, workers)
    err = np.array([o[0] for o in out])
    nbits = np.array([o[1] for o in out], float)
    semi = np.array([o[2] for o in out])
    return err, nbits, semi, nb


def regrets(err, nbits, semi):
    """Per-row signed regret vs column 0 (the oracle bias), counted and semi; zero counts scored at 0.5."""
    lc = np.log10(np.maximum(err, 0.5) / nbits[:, None])
    return {"counted": lc[:, 1:] - lc[:, :1], "semi": semi[:, 1:] - semi[:, :1]}


def with_seedmean(r):
    """KEYS columns -> dict per key, plus the MLP seed mean."""
    out = {k: r[:, j] for j, k in enumerate(KEYS)}
    out[f"{MLP}@seedmean"] = np.mean([out[f"{MLP}@seed{s}"] for s in range(5)], axis=0)
    return out


def boot(units, a, b=None):
    return E.cluster_boot(units, a, b, B=10000, seed=0)[0]


def r2(ci):
    return "above" if ci[0] > REF else "below" if ci[1] < REF else "straddles"


def corrected_cost(units_full, full, inS, d, tail, B=10000, seed=0):
    """R2: full-set PCHIP mean cost corrected by d = cost_counted - cost_pchip on S, with the
    CI from one joint unit bootstrap of both terms. Sensitivities: `no_shift_off_S`
    adds d only for S rows; `tail_shift_off_S` gives the off-S rows the mean d of the tail band (mask `tail`
    over S rows). Returns {name: {"point", "ci"}}."""
    u, inv = np.unique(units_full, return_inverse=True)
    inv_s, tail_s = inv[inS], inv[inS][tail]
    sf, nf = np.bincount(inv, full, len(u)), np.bincount(inv, minlength=len(u))
    sd, ns = np.bincount(inv_s, d, len(u)), np.bincount(inv_s, minlength=len(u))
    st, nt = np.bincount(tail_s, d[tail], len(u)), np.bincount(tail_s, minlength=len(u))
    pick = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    idx = np.arange(len(u))[None]  # the unresampled point uses every unit once

    def est(p):
        F, N, Sd, Ns = sf[p].sum(-1), nf[p].sum(-1), sd[p].sum(-1), ns[p].sum(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            Dt = st[p].sum(-1) / nt[p].sum(-1)
        return {"primary": F / N + Sd / Ns, "no_shift_off_S": (F + Sd) / N, "tail_shift_off_S": (F + Sd) / N + (1 - Ns / N) * Dt}

    pt, dr = est(idx), est(pick)
    return {k: {"point": float(pt[k][0]), "ci": [float(np.nanquantile(dr[k], 0.025)), float(np.nanquantile(dr[k], 0.975))],
                "n_nan_draws": int(np.isnan(dr[k]).sum())} for k in pt}


def summarise(units, tw, idn, full, inS, lbt, units_full, keys=REPORT, contrast=("B7_bayes", f"{MLP}@seedmean")):
    """One twin. tw / idn: read-out -> key -> per-row regret on S (twin / identity, same states); full: key ->
    full-set PCHIP cost per row (all test_id rows, units units_full); inS: bool mask of S over those rows; lbt:
    twin oracle log10 BER on S. Returns cost, R1 (with bands), R2 (selection-neutral) and R2 on S, contrasts."""
    if not np.array_equal(np.unique(units), np.unique(units_full)):
        raise AssertionError("every unit must have rows in S for the joint bootstrap")
    e = {"cost": {}, "R1": {}, "R1_bands": {}, "R2": {}, "R2_on_S": {}, "contrast_twin": {}, "contrast_identity": {}}
    for k in keys:
        c = {m: tw[m][k] - idn[m][k] for m in ("counted", "semi", "pchip")}
        fm = float(full[k].mean())
        d_cp = boot(units, c["counted"], c["pchip"])
        corrs = corrected_cost(units_full, full[k], inS, c["counted"] - c["pchip"], lbt < BANDS[-1][1])
        corr = corrs["primary"]
        e["cost"][k] = {m: boot(units, v) for m, v in c.items()} | {
            "counted_minus_semi": boot(units, c["counted"], c["semi"]), "counted_minus_pchip": d_cp,
            "pchip_full_set_mean": fm, "corrected_full_set": corr,
            "corrected_sensitivity": {n: v | {"R2": r2(v["ci"])} for n, v in corrs.items() if n != "primary"},
            "share_of_full_set_cost_in_S": float(full[k][inS].sum() / full[k].sum())}
        ci = e["cost"][k]["counted_minus_semi"]["ci"]
        e["R1"][k] = "PASS" if -TOL_R1 <= ci[0] and ci[1] <= TOL_R1 else "FAIL"
        e["R2"][k] = r2(corr["ci"])
        e["R2_on_S"][k] = {"counted": r2(e["cost"][k]["counted"]["ci"]), "pchip": r2(e["cost"][k]["pchip"]["ci"])}
        e["R1_bands"][k] = {}
        for lo, hi in BANDS:
            m = (lbt >= lo) & (lbt < hi)
            e["R1_bands"][k][f"[{lo},{hi})"] = {
                "n": int(m.sum()), "share_of_full_set_cost": float(full[k][inS][m].sum() / full[k].sum()),
                **({"counted_minus_semi": boot(units[m], c["counted"][m], c["semi"][m])} if m.any() else {})}
    a, b = contrast
    for m in ("counted", "semi", "pchip"):
        e["contrast_twin"][m] = boot(units, tw[m][a], tw[m][b])
        e["contrast_identity"][m] = boot(units, idn[m][a], idn[m][b])
    ci = e["contrast_twin"]["counted"]["ci"]
    e["R3"] = "resolved" if ci[0] > 0 else "unresolved"  # applied at i_min 0 only (reported sign: positive)
    return e


def sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def main(run_dir, workers):
    run_dir = Path(run_dir)
    prov = G.git_info()
    if prov["dirty"]:
        raise RuntimeError("code tree is dirty: commit before a counted run")
    out_dir = run_dir / "counted-rows"
    out_dir.mkdir(exist_ok=True)
    names = ("identity",) + DEPLOYS
    inputs = {"results_shift_sha": json.loads((run_dir / "results_shift.json").read_text())["_eval_provenance"]["sha"],
              "shift_rows_sha1": {n: sha1(run_dir / "shift-rows" / f"{n}.npz") for n in names}}
    dep = {name: deploy(run_dir, name) for name in names}
    for name, (df, cv, cfg, _) in dep.items():
        preflight(cfg, df, cv)
        print(f"preflight {name}: ok", flush=True)
    sid = dep["identity"][0]["state_id"].to_numpy()
    for name in DEPLOYS:
        if not np.array_equal(dep[name][0]["state_id"].to_numpy(), sid):
            raise AssertionError(f"{name}: test_id states differ from identity")
    lb = {name: np.nanmin(cv["oracle_lb"], axis=1) for name, (_, cv, _, _) in dep.items()}
    S = {name: (lb["identity"] >= LB_MIN) & (lb[name] >= LB_MIN) for name in DEPLOYS}
    need = {"identity": np.flatnonzero(np.logical_or.reduce(list(S.values())))} | {n: np.flatnonzero(S[n]) for n in DEPLOYS}

    reg = {}
    for name, (df, cv, cfg, z) in dep.items():
        rows = need[name]
        bh = biases(df, cv, z)
        err, nbits, semi, nb = run_counts(cfg, df, bh, lb[name], rows, workers)
        rr = regrets(err, nbits, semi)
        pchip = np.column_stack([z[f"regret:{k}"][rows] for k in KEYS])
        np.savez_compressed(out_dir / f"{name}.npz", state_id=sid[rows], unit_id=df["unit_id"].to_numpy()[rows], M=df["M"].to_numpy()[rows],
                            keys=np.array(("O_L1",) + KEYS), bias=bh[rows], errors=err, nbits=nbits, semi_lb=semi, batches=nb, oracle_lb=lb[name][rows])
        reg[name] = {"rows": rows, "err": err, "batches": nb,
                     **{m: with_seedmean(v) for m, v in (("counted", rr["counted"]), ("semi", rr["semi"]), ("pchip", pchip))}}
        print(f"{name}: {len(rows)} rows, {int(nb.sum())} batches, zero-count cells {int((err == 0).sum())}", flush=True)

    units_all = dep["identity"][0]["unit_id"].to_numpy()
    res = {"protocol": "rules R1-R3 as defined in the dcbias.counted module docstring",
           "status": "post hoc, descriptive", "inputs": inputs,
           "stream": STREAM_COUNTED_SHIFT, "lb_min": LB_MIN, "target_errors": TARGET_ERRORS, "max_batches": MAX_BATCHES,
           "tol_r1": TOL_R1, "ref": REF, "by_deploy": {}}
    ident = reg["identity"]
    pos = {r: i for i, r in enumerate(ident["rows"])}
    for name in DEPLOYS:
        tw, rows = reg[name], need[name]
        ii = np.array([pos[r] for r in rows])  # identity entries for the same states
        z_t, z_i = dep[name][3], dep["identity"][3]
        full = {k: z_t[f"regret:{k}"] - z_i[f"regret:{k}"] for k in REPORT}
        sub = lambda R, idx: {m: {k: v[idx] for k, v in R[m].items()} for m in ("counted", "semi", "pchip")}  # noqa: E731
        e = {"n_S": int(len(rows)), "n_rows": int(len(sid)), "n_units": int(len(np.unique(units_all[rows])))}
        e |= summarise(units_all[rows], sub(tw, slice(None)), sub(ident, ii), full, S[name], lb[name][rows], units_all)
        e["counting"] = {side: {"zero_count_cells": int((r["err"][ix] == 0).sum()), "min_oracle_errors": int(r["err"][ix, 0].min()),
                                "batches_median": float(np.median(r["batches"][ix])), "batches_max": int(r["batches"][ix].max()),
                                "n_capped": int((r["batches"][ix] == MAX_BATCHES).sum())}
                         for side, r, ix in (("twin", tw, slice(None)), ("identity", ident, ii))}
        if name != "led.i_min_frac=0.0":
            e.pop("R3")
        res["by_deploy"][name] = e
        print(name, json.dumps({k: e[k] for k in ("R1", "R2")}), e.get("R3", ""), flush=True)
    res["R1_all_pass"] = all(v == "PASS" for e in res["by_deploy"].values() for v in e["R1"].values())

    if G.git_info() != prov:
        raise RuntimeError("code changed during the run")
    res = {"_eval_provenance": prov} | res
    (run_dir / "results_counted.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"wrote {run_dir / 'results_counted.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    main(a.run_dir, a.workers)
