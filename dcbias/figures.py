"""Manuscript figures: plotting only. Reads the stored results JSON of the baseline run
(runs/baseline/results_*.json, which includes the sweep summary and the device-shift arm) and computes no statistic:
every plotted number, interval and sign is a stored value. Writes PDFs to figures/.

Usage: python -m dcbias.figures
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "baseline"
OUT = ROOT / "figures"

SPLITS = ("test_id", "ood_T", "ood_SNR", "ood_sigma")
SPLIT_LABEL = {"test_id": "ID (unseen units)", "ood_T": "OOD-T", "ood_SNR": "OOD-SNR", "ood_sigma": "OOD-$\\sigma$"}
COL = {"test_id": "#1b6ca8", "ood_T": "#d1495b", "ood_SNR": "#edae49", "ood_sigma": "#00798c"}
MARK = {"test_id": "o", "ood_T": "s", "ood_SNR": "^", "ood_sigma": "D"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "text.usetex": False, "mathtext.fontset": "stix",
})


def load(name):
    return json.loads((RUN / f"results_{name}.json").read_text())


def regret_dotplot(main):
    rows = [
        ("B0_static", "B0 static bias"), ("B3_centre", "B3 window centre (= B2)"),
        ("B1c_kfit", "B1c $k\\sigma$, $k$ fitted"), ("B4_lut1d_C", "B4 1-D LUT on $\\hat C$"),
        ("B5_lut2d_Ir_T", "B5 2-D LUT ($I_{\\rm rated}, \\hat T$)"), ("B6a_lut_state", "B6a state LUT (data)"),
        ("B6b_model_grid", "B6b model-grid LUT"), ("B7_bayes", "B7 Bayesian calibrate-then-optimise"),
        ("B7w_bayes_wave", "B7w (exploratory)"),
        ("ML_ridge_state_obs", "Ridge (cubic), state + obs."), ("ML_rf_state_obs", "Random forest, state + obs."),
        ("ML_hgb_state_obs", "HistGB, state + obs."), ("ML_mlp_state_obs", "MLP, state + obs."),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 3.7))
    y = np.arange(len(rows))[::-1]
    for s in SPLITS:
        x = [main["test"][s][k]["mean_regret"] for k, _ in rows]
        ax.scatter(x, y, s=16, marker=MARK[s], color=COL[s], label=SPLIT_LABEL[s], zorder=3, linewidths=0)
    ax.axvline(0.10, color="k", ls="--", lw=0.8)
    ax.text(0.085, y[0] + 0.1, "pre-specified\nmargin", ha="right", va="top", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_xlabel("BER regret (decades)")
    ax.grid(axis="x", color="0.88", lw=0.5, zorder=0)
    ax.axhline(y[9] + 0.5, color="0.6", lw=0.5)
    ax.axhline(y[6] + 0.5, color="0.6", lw=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.16), ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_regret.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def forest(main, mis, sw):
    conf = main["confirmatory"]["delta"]
    d = mis["delta_minus_best_ml"]
    rows = [("hdr", "Confirmatory, true LED form (B7 $-$ MLP)")]
    rows += [("conf", SPLIT_LABEL[s], conf[s], COL[s]) for s in SPLITS]
    rows += [("hdr", "Misspecified controller, ID units")]
    lab = {"form_tsonev_cubic": "wrong LED form", "datasheet_kappa_0.27": "$\\tilde\\kappa$ = 0.27", "datasheet_kappa_0.36": "$\\tilde\\kappa$ = 0.36"}
    for who, wl in (("B7_bayes", "B7"), ("B6b_model_grid", "B6b")):
        for v, vl in lab.items():
            rows.append(("mis", f"{wl}, {vl}", d[f"{who}@{v}"]["test_id"], "#6a4c93" if who == "B7_bayes" else "#8a817c"))
    rows += [("hdr", "Misspecified controller, OOD-SNR")]
    for who, wl in (("B7_bayes", "B7"), ("B6b_model_grid", "B6b")):
        rows.append(("mis", f"{wl}, wrong LED form", d[f"{who}@form_tsonev_cubic"]["ood_SNR"], "#6a4c93" if who == "B7_bayes" else "#8a817c"))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 3.5), gridspec_kw={"width_ratios": [1.2, 1]})
    y = len(rows)
    yt, yl, hdr = [], [], []
    for r in rows:
        y -= 1
        yt.append(y)
        if r[0] == "hdr":
            yl.append(r[1])
            hdr.append(True)
            continue
        _, name, st, c = r
        ax.errorbar(st["point"], y, xerr=[[st["point"] - st["ci"][0]], [st["ci"][1] - st["point"]]], fmt="o", ms=3.2, color=c, elinewidth=1.0, capsize=1.5)
        yl.append("   " + name)
        hdr.append(False)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xscale("symlog", linthresh=1e-3, linscale=0.6)
    ax.set_xlim(-0.03, 0.12)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    for t, h in zip(ax.get_yticklabels(), hdr):
        if h:
            t.set_fontweight("bold")
    ax.set_xlabel("$\\Delta$ = method $-$ ML (decades; > 0: ML better)")
    ax.grid(axis="x", color="0.9", lw=0.5)
    ax.set_title("(a) Confirmatory and misspecification", loc="left", fontsize=7.5)

    # paper notation, not config keys
    sym = {"driver.f_min": "$f_{\\min}$", "driver.knee_K": "$T_k$ (K)", "driver.slope_per_K": "$s_d$ (1/K)",
           "led.delta": "$\\delta$ (1/K)", "led.f3dB_over_B": "$f_{3\\rm dB}/B$", "led.i_min_frac": "$i_{\\min}/I_{\\rm rated}$",
           "led.kappa_tilde": "$\\tilde\\kappa$", "noise.rho": "$\\rho$", "unit_spread.Rth": "$R_{th}$ spread"}

    def sweep_label(k):
        if k.startswith("joint_"):
            return "joint $\\tilde\\kappa$ = 0, $\\rho$ = 0"
        if k == "led.model=tsonev_cubic":
            return "cubic LED"
        p, v = k.split("=")
        return f"{sym[p]} = {'none' if v == 'null' else v}"

    # each sweep against the 50-unit reference trained on the same 40 units
    items = sorted(sw["by_sweep"].items(), key=lambda kv: kv[1]["delta_minus_ref"]["point"])
    for j, (k, v) in enumerate(items):
        d = v["delta_minus_ref"]
        lo, hi = d["ci"]
        c, m = ("#d1495b", "^") if lo > 0 else (("#00798c", "v") if hi < 0 else ("0.45", "o"))  # shape backs colour
        bx.errorbar(d["point"], j, xerr=[[d["point"] - lo], [hi - d["point"]]], fmt=m, ms=2.8, color=c, elinewidth=0.9, capsize=1.2)
    bx.axvline(0, color="k", lw=0.7)
    bx.set_yticks(range(len(items)))
    bx.set_yticklabels([sweep_label(k) for k, _ in items], fontsize=5.5)
    bx.set_xlabel("$\\Delta - \\Delta_{\\mathrm{ref}}$ (decades; > 0: ML relatively better)", fontsize=7)
    bx.grid(axis="x", color="0.9", lw=0.5)
    bx.set_title("(b) 26 sweeps minus the 50-unit reference, ID units", loc="left", fontsize=7.5)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "fig_forest.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# Cost of each device shift per controller family, and the oracle-bias
# displacement per twin. Twins ordered by basis: in the modelled LED family (D/E), assumed (A), outside (S).
SHIFT_GROUPS = [
    ("in modelled family", [("led.model=tsonev_cubic", "cubic"), ("led.model=tsonev_cubic_spread", "cubic + spread"),
                            ("led.kappa_tilde=0.27", "$\\tilde\\kappa$ 0.27"), ("led.kappa_tilde=0.36", "$\\tilde\\kappa$ 0.36")]),
    ("assumed (A)", [("led.delta=0", "$\\delta$ 0"), ("led.delta=0.001", "$\\delta$ 0.001"), ("led.delta=0.003", "$\\delta$ 0.003"),
                     ("led.f3dB_over_B=0.5", "$f_{3\\rm dB}$ half"), ("led.f3dB_over_B=null", "no low-pass")]),
    ("outside the family (S)", [("led.kappa_tilde=0.15", "$\\tilde\\kappa$ 0.15"), ("led.kappa_tilde=0.42", "$\\tilde\\kappa$ 0.42"),
                                ("led.i_min_frac=0.05", "$i_{\\min}$ 0.05"), ("led.kappa_tilde=0", "$\\tilde\\kappa$ 0"),
                                ("led.i_min_frac=0.0", "$i_{\\min}$ 0")]),
]
SHIFT_CTRL = [("B6a_lut_state", "B6a data table", "#8a817c", "s"), ("B6b_model_grid", "B6b model grid", "#edae49", "^"),
              ("B7_bayes", "B7", "#6a4c93", "o"), ("B7_bayes_wide", "B7, wide prior (post hoc)", "#6a4c93", "D"),
              ("ML_mlp_state_obs@seedmean", "MLP (5-seed mean)", "#1b6ca8", "P")]


def shift_cost(shift, wide, floor):
    deps = [d for _, g in SHIFT_GROUPS for d in g]
    x = np.arange(len(deps))
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.1, 3.6), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    off = np.linspace(-0.28, 0.28, len(SHIFT_CTRL))
    for (key, lab, c, m), o in zip(SHIFT_CTRL, off):
        xs, ys, lo, hi = [], [], [], []
        for i, (d, _) in enumerate(deps):
            v = shift["by_deploy"][d]["cost_of_shift"].get(key)
            if v is None and key == "B7_bayes_wide" and d in wide["by_deploy"]:
                v = wide["by_deploy"][d]["cost_of_shift"]
            if v is None:
                continue
            xs.append(i + o); ys.append(v["point"]); lo.append(v["point"] - v["ci"][0]); hi.append(v["ci"][1] - v["point"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=m, ms=3.4, color=c, mfc="white" if key == "B7_bayes_wide" else c,
                    elinewidth=0.8, capsize=1.0, label=lab, zorder=3)
    n0 = len(SHIFT_GROUPS[0][1])
    for a in (ax, bx):
        a.axvspan(-0.5, n0 - 0.5, color="#e8f1e4", zorder=0)
        a.axvline(n0 + len(SHIFT_GROUPS[1][1]) - 0.5, color="0.6", lw=0.5)
        a.axvline(n0 - 0.5, color="0.6", lw=0.5)
    ax.axhline(0.10, color="k", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.text(n0 + 1.5, 0.085, "0.10-decade reference", ha="center", va="top", fontsize=6.5)
    ax.set_yscale("symlog", linthresh=1e-3, linscale=0.6)
    ax.set_ylim(-0.02, 1.0)
    ax.set_ylabel("cost of shift (decades)")
    ax.set_xlim(-0.6, len(deps) - 0.4)
    ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    for k, (g, ds) in enumerate(SHIFT_GROUPS):
        start = sum(len(h[1]) for h in SHIFT_GROUPS[:k])
        ax.text(start + (len(ds) - 1) / 2, 0.9, g, ha="center", va="top", fontsize=6.5, style="italic")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.0)
    ax.text(-0.45, 0.45, "(a)", fontsize=7.5, fontweight="bold", va="center")
    disp = floor["oracle_displacement_sigma"]
    med = [disp[d]["median"] for d, _ in deps]
    bx.errorbar(x, med, yerr=[[m_ - disp[d]["q25"] for m_, (d, _) in zip(med, deps)], [disp[d]["q75"] - m_ for m_, (d, _) in zip(med, deps)]],
                fmt="o", ms=3, color="0.25", elinewidth=0.8, capsize=1.0)
    bx.axhline(0, color="k", lw=0.5)
    bx.set_ylabel("$\\Delta b^{\\circ}$ ($\\sigma_I$)")
    bx.text(-0.45, 0.9, "(b)", fontsize=7.5, fontweight="bold", va="center")
    bx.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    bx.set_xticks(x)
    bx.set_xticklabels([l for _, l in deps], rotation=35, ha="right", fontsize=6.5)
    fig.tight_layout(pad=0.3, h_pad=0.4)
    fig.savefig(OUT / "fig_shift.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def sensing_grid(sens):
    cells = sens["cells"]
    lT = sorted({c["lambda_T"] for c in cells})
    lW = sorted({c["lambda_W"] for c in cells})
    Z = np.full((len(lT), len(lW)), np.nan)
    sig = np.zeros_like(Z, dtype=bool)
    for c in cells:
        i, j = lT.index(c["lambda_T"]), lW.index(c["lambda_W"])
        st = c["matched"]["wave_minus_thermal"]
        Z[i, j] = st["point"]
        sig[i, j] = st["ci"][0] > 0 or st["ci"][1] < 0
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    lim = np.nanmax(np.abs(Z))
    im = ax.imshow(Z, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim * 0.25, lim), origin="lower", aspect="auto")
    for i in range(len(lT)):
        for j in range(len(lW)):
            ax.text(j, i, f"{Z[i, j] * 1e3:.1f}" + ("*" if not sig[i, j] else ""), ha="center", va="center", fontsize=5.2,
                    color="white" if abs(Z[i, j]) > 0.55 * lim else "black")
    ax.set_xticks(range(len(lW)))
    ax.set_xticklabels([f"{w:g}" for w in lW])
    ax.set_yticks(range(len(lT)))
    ax.set_yticklabels([f"{t:g}" for t in lT])
    ax.set_xlabel("Waveform-sensor noise scale $\\lambda_W$")
    ax.set_ylabel("Thermal-sensor noise\nscale $\\lambda_T$")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("wave $-$ thermal regret ($10^{-3}$ dec)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_sensing.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def learning_curve(lc, main):
    """Every stored (units, seed) value is plotted; the line joins seed 0 (no averaging)."""
    rows = lc["rows"]
    units = sorted({r["units"] for r in rows})
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    names = {"ML_ridge_state_obs": "Ridge", "ML_rf_state_obs": "RF", "ML_hgb_state_obs": "HistGB", "ML_mlp_state_obs": "MLP", "B6a_lut_state": "B6a LUT"}
    for k, lab in names.items():
        line = ax.plot(units, [next(r["mean_regret"] for r in rows if r["method"] == k and r["units"] == u and r["seed"] == 0) for u in units],
                       marker="o", ms=3, lw=1.0, label=lab, ls="--" if k.startswith("B") else "-")[0]
        ax.scatter([r["units"] for r in rows if r["method"] == k], [r["mean_regret"] for r in rows if r["method"] == k],
                   s=5, color=line.get_color(), alpha=0.35, linewidths=0)
    ref = main["test"]["test_id"]["B7_bayes"]["mean_regret"]
    ax.axhline(ref, color="k", lw=0.8, ls=":")
    ax.text(units[0], ref * 1.08, "B7 (no bias labels)", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(units)
    ax.set_xticklabels([str(u) for u in units])
    ax.set_xlabel("Training units")
    ax.set_ylabel("ID regret (decades)")
    ax.legend(ncol=2, frameon=False)
    ax.grid(color="0.9", lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_learning.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def pipeline():
    """Block diagram (no data)."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.1, 2.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    def box(x, y, w, h, t, fc="#f1f1f1"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1", fc=fc, ec="k", lw=0.6))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.3)
        return (x, y, w, h)
    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7))
    box(1, 18, 20, 9, "State $s$\n$I_{rated},\\sigma_I,T_{hs},$\nSNR$_{ref}$, $M$")
    box(1, 3, 20, 9, "Latent unit $\\theta$\n$\\tilde\\kappa,\\delta,\\eta_0,R_{th},$\n$V_0,R_s,K_V$, offsets")
    box(28, 9, 24, 12, "Electrothermal\nDCO-OFDM link\nclip $\\to$ LED $\\to T_j$ loop\n$\\to$ noise $\\to$ ZF $\\to$ BER$(b)$")
    box(59, 20, 17, 8, "Oracle $b^\\circ$\n$R$=1024, CRN", "#fbe3c8")
    box(59, 2, 17, 8, "Training label\n$R$=512, 160 units")
    box(81, 20, 18, 8, "Regret $r(\\hat b)$\nunit-cluster\nbootstrap")
    box(79, 2, 20, 11, "Controllers\nB0-B8, ML\nnoisy obs.\nB0: none", "#dbe7f5")
    arr(21, 22, 28, 18); arr(21, 8, 28, 12); arr(52, 17, 59, 23); arr(52, 13, 59, 6)
    arr(76, 24, 81, 24); arr(76, 6, 79, 6); arr(89, 13, 89, 20)
    fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main, mis, sens, lc = load("main"), load("misspec"), load("sensing"), load("learning")
    sw = load("floor_sweepref")["sweep_minus_reference"]
    regret_dotplot(main)
    forest(main, mis, sw)
    sensing_grid(sens)
    shift_cost(load("shift"), load("b7wide_imin"), load("floor_sweepref"))
    learning_curve(lc, main)
    pipeline()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
