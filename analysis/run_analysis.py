"""
Full analysis: measurement system, capability, life testing. Writes results/summary.json and
the figures in media/.

Run from the repo root:  python analysis/run_analysis.py
"""
import csv
import json
import os

import numpy as np
from scipy import stats

from quality import SPECS, capability, gage_rr_crossed, repeatability, verdict
from reliability import compare, fit, median_ranks, weibull_life

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#1baf7a"
NAMES = {"thickness_mm": "Thickness", "length_mm": "Length", "width_mm": "Width"}


def load(name):
    with open(os.path.join(ROOT, "data", name)) as f:
        return list(csv.DictReader(f))


def col(rows, c, **where):
    return np.array([float(r[c]) for r in rows if all(r[k] == str(v) for k, v in where.items())])


def part_means(rows, c):
    return np.array([np.mean([float(r[c]) for r in rows if int(r["part"]) == p]) for p in range(1, 11)])


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": INK, "axes.titlecolor": INK})

    old, new, life = load("gage_rr_original_hinges.csv"), load("repeatability_new_hinges.csv"), load("life_test.csv")
    R = dict(measurement_system={}, capability={}, life={}, dimensions_vs_life={})

    # ---------- 1. measurement system ----------
    for c, (_, lsl, usl) in SPECS.items():
        tol = usl - lsl
        g = gage_rr_crossed([r["part"] for r in old], [r["operator"] for r in old], col(old, c), tol)
        g["operator_bias_mm"] = float(col(old, c, operator=2).mean() - col(old, c, operator=1).mean())
        g["verdict_tolerance"] = verdict(g["pct_tolerance_grr"])
        rp = repeatability([r["part"] for r in new], col(new, c), tol)
        rp["verdict_tolerance"] = verdict(rp["pct_tolerance_grr"])
        R["measurement_system"][c] = dict(original_two_operators=g, redesigned_one_operator=rp)

    # ---------- 2. capability ----------
    for c, (_, lsl, usl) in SPECS.items():
        R["capability"][c] = dict(
            original_both_operators=capability(col(old, c), lsl, usl),
            original_operator1_only=capability(col(old, c, operator=1), lsl, usl),
            redesigned=capability(col(new, c), lsl, usl))

    # ---------- 3. life testing ----------
    data = {}
    for des in ("original", "redesigned"):
        t = np.array([float(r["cycles"]) for r in life if r["design"] == des])
        f = np.array([r["status"] == "failed" for r in life if r["design"] == des])
        data[des] = (t, f)
        w = weibull_life(t, f)
        w["aic"] = {d: fit(t, f, d)["aic"] for d in ("weibull", "lognormal", "normal")}
        R["life"][des] = w
    R["life"]["comparison"] = compare(*data["original"], *data["redesigned"])

    # ---------- 4. do dimensions predict life? ----------
    for des, rows in (("original", old), ("redesigned", new)):
        cyc = data[des][0]
        R["dimensions_vs_life"][des] = {c: dict(zip(("spearman_rho", "p"), map(float, stats.spearmanr(part_means(rows, c), cyc))))
                                        for c in SPECS}

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as fjs:
        json.dump(R, fjs, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # ---------- figures ----------
    # capability
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), facecolor=SURFACE)
    for ax, (c, (nom, lsl, usl)) in zip(axs, SPECS.items()):
        style(ax)
        a, b = col(old, c, operator=1), col(new, c)
        lo, hi = min(lsl, a.min(), b.min()), max(usl, a.max(), b.max())
        span = hi - lo
        bins = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 40)
        ax.hist(a, bins, color=ORANGE, alpha=0.85, label="Original", edgecolor=SURFACE)
        ax.hist(b, bins, color=BLUE, alpha=0.85, label="Redesigned", edgecolor=SURFACE)
        for x, lab in ((lsl, "LSL"), (usl, "USL")):
            ax.axvline(x, color=INK, lw=1.2, ls="--")
            ax.text(x, ax.get_ylim()[1] * 0.97, lab, ha="center", va="top", fontsize=8, color=INK,
                    bbox=dict(fc=SURFACE, ec="none", pad=1))
        co, cn = R["capability"][c]["original_operator1_only"], R["capability"][c]["redesigned"]
        ax.set_title(f"{NAMES[c]}: Cpk {co['cpk']:.2f} → {cn['cpk']:.2f}", loc="left", fontsize=10)
        ax.set_xlabel("mm")
    axs[0].set_ylabel("Measurements")
    axs[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.88))
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "capability.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # measurement system
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]
    style(ax)
    x = np.arange(3)
    vo = [R["measurement_system"][c]["original_two_operators"]["pct_tolerance_grr"] for c in SPECS]
    vn = [R["measurement_system"][c]["redesigned_one_operator"]["pct_tolerance_grr"] for c in SPECS]
    ax.axhspan(0, 10, color=BAND, alpha=0.12, lw=0)
    ax.axhline(10, color=MUTED, lw=1, ls="--")
    ax.axhline(30, color=MUTED, lw=1, ls="--")
    ax.text(2.55, 10.5, "10% good", fontsize=8, color=MUTED, ha="right")
    ax.text(2.55, 30.5, "30% limit", fontsize=8, color=MUTED, ha="right")
    ax.bar(x - 0.2, vo, 0.38, color=ORANGE, label="Original (2 operators)", edgecolor=SURFACE, lw=2)
    ax.bar(x + 0.2, vn, 0.38, color=BLUE, label="Redesigned (1 operator)", edgecolor=SURFACE, lw=2)
    for xi, v in zip(x, vo):
        ax.text(xi - 0.2, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
    for xi, v in zip(x, vn):
        ax.text(xi + 0.2, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[c] for c in SPECS])
    ax.set_ylabel("Gage R&R, % of tolerance")
    ax.set_ylim(0, 48)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Measurement error vs tolerance", loc="left", fontsize=10)
    ax = axs[1]
    style(ax)
    for op, colr, off in (("1", ORANGE, -0.12), ("2", INK, 0.12)):
        for p in range(1, 11):
            ys = col(old, "length_mm", part=p, operator=op)
            ax.plot([p + off] * len(ys), ys, "o", ms=4.5, color=colr, alpha=0.8, markeredgecolor=SURFACE,
                    label=f"Operator {op}" if p == 1 else None)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Original hinge (part)")
    ax.set_ylabel("Length (mm)")
    bias = R["measurement_system"]["length_mm"]["original_two_operators"]["operator_bias_mm"]
    ax.set_title(f"Length: operator 2 reads {bias:.2f} mm longer", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "measurement_system.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # reliability
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    ax = axs[0]
    style(ax)
    qs = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99]
    yq = lambda F: np.log(-np.log(1 - F))
    for des, colr in (("original", ORANGE), ("redesigned", BLUE)):
        t, f = data[des]
        pts = median_ranks(t, f)
        w = R["life"][des]
        ax.plot(pts[:, 0], yq(pts[:, 1]), "o", color=colr, ms=6, markeredgecolor=SURFACE,
                label=f"{des.capitalize()}: β {w['beta']:.1f}, η {w['eta']:.0f}")
        tt = np.logspace(np.log10(5), np.log10(3000), 200)
        ax.plot(tt, w["beta"] * np.log(tt / w["eta"]), color=colr, lw=2)
    ax.set_xscale("log")
    ax.set_yticks(yq(np.array(qs)))
    ax.set_yticklabels([f"{int(q * 100)}%" for q in qs])
    ax.set_ylim(yq(0.01), yq(0.99))
    ax.set_xlim(8, 3000)
    ax.set_xlabel("Cycles to failure")
    ax.set_ylabel("Share failed (Weibull scale)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Weibull probability plot", loc="left", fontsize=10)
    ax = axs[1]
    style(ax)
    tt = np.linspace(0, 1300, 400)
    for des, colr in (("original", ORANGE), ("redesigned", BLUE)):
        w = R["life"][des]
        ax.plot(tt, 100 * np.exp(-(tt / w["eta"]) ** w["beta"]), color=colr, lw=2.2, label=des.capitalize())
        ax.plot([w["b10"]], [90], "o", color=colr, ms=7, markeredgecolor=SURFACE)
        ax.annotate(f"B10 = {w['b10']:.0f}", (w["b10"], 90), xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xlabel("Cycles")
    ax.set_ylabel("Hinges still working (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower left")
    ratio = R["life"]["redesigned"]["median"] / R["life"]["original"]["median"]
    ax.set_title(f"Reliability: median life {ratio:.1f}× longer", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "media", "reliability.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("saved results/summary.json and figures")
    return R


if __name__ == "__main__":
    main()
