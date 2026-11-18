from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
TIMINGS = HERE / "timings.csv"
OUT = HERE.parent / "man" / "figures" / "speedup.png"

TITLE = "FastconText on conText's help-page examples"
XLABEL = "Run time (seconds)"
TASK_LABELS = { 
    "reg_jackknife_permute": "conText(): full jackknife + 100 permutations",
    "get_nns_ratio": "get_nns_ratio(): 100 bootstraps + 100 permutations",
    "contrast_nns": "contrast_nns(): 100 bootstraps + 100 permutations",
    "get_ncs": "get_ncs(): 100 bootstraps",
    "get_nns": "get_nns(): 100 bootstraps",
    "get_cos_sim": "get_cos_sim(): 100 bootstraps",
}
LEGEND = {
    "orig": "conText 3.0.1 (one core)",
    "par": "conText 3.0.1 (18 workers)",
    "cpu": "FastconText, CPU engine",
    "gpu": "FastconText, Apple GPU engine",
}
C_ORIG, C_CPU, C_GPU = "#7F7F7F", "#1F5FA8", "#E07B00"
FONT = "Helvetica Neue"
# ------------------------------------------------------------------------------------

mpl.rcParams.update({"font.family": FONT, "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 8, "savefig.bbox": "tight"})

t = pd.read_csv(TIMINGS)
w = t.pivot_table(index="task", columns=["pkg", "engine"], values="median_s")
w.columns = ["_".join(c) for c in w.columns]
w["speedup_cpu"] = w["conText_original"] / w["FastconText_cpu"]
w["speedup_vs_par"] = w["conText_parallel"] / w["FastconText_cpu"]

order = list(TASK_LABELS)
y = np.arange(len(order))[::-1]
XMAX = 300  # right edge of the axis; speedup labels sit just outside it

fig, ax = plt.subplots(figsize=(7.4, 3.5))
for yi, task in zip(y, order):
    r = w.loc[task]
    has_par = not np.isnan(r.conText_parallel)
    # rows with a parallel conText option are compared against parallel conText only
    ref = r.conText_parallel if has_par else r.conText_original
    xs = [r.FastconText_cpu, r.FastconText_metal, ref]
    ax.plot([min(xs), max(xs)], [yi, yi], color="#D0D0D0", lw=1.2, zorder=1)
    if has_par:
        ax.scatter(ref, yi, s=34, facecolor="white", edgecolor=C_ORIG, lw=1.2, zorder=3)
    else:
        ax.scatter(ref, yi, s=34, color=C_ORIG, zorder=3)
    ax.scatter(r.FastconText_cpu, yi, s=34, color=C_CPU, zorder=4)
    ax.scatter(r.FastconText_metal, yi, s=34, marker="D", facecolor="none", edgecolor=C_GPU, lw=1.2, zorder=5)
    sp = ref / r.FastconText_cpu
    txt = f"{sp:,.0f}× faster" if sp >= 10 else f"{sp:.1f}× faster"
    ax.text(XMAX, yi + (0.17 if has_par else 0), txt, va="center", ha="left", fontsize=8,
            color=C_CPU if sp >= 2 else "#555555")
    if has_par:
        ax.text(XMAX, yi - 0.25, "(vs. 18 workers)", va="center", ha="left", fontsize=7.5, color="#555555")

ax.set_yticks(y)
ax.set_yticklabels([TASK_LABELS[k] for k in order])
ax.set_xscale("log")
ax.set_xlim(0.012, XMAX)
ax.set_xticks([0.1, 1, 10, 100])
ax.set_xticklabels(["0.1 s", "1 s", "10 s", "100 s"])
ax.set_xlabel(XLABEL, labelpad=6)
ax.set_ylim(-0.6, len(order) - 0.4)
ax.grid(axis="x", color="#EEEEEE", lw=0.6)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
handles = [
    Line2D([], [], ls="", marker="o", ms=6, color=C_ORIG, label=LEGEND["orig"]),
    Line2D([], [], ls="", marker="o", ms=6, mfc="white", mec=C_ORIG, mew=1.2, label=LEGEND["par"]),
    Line2D([], [], ls="", marker="o", ms=6, color=C_CPU, label=LEGEND["cpu"]),
    Line2D([], [], ls="", marker="D", ms=5.5, mfc="none", mec=C_GPU, mew=1.2, label=LEGEND["gpu"]),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, -0.26), ncol=2, frameon=False,
          fontsize=7.5, handletextpad=0.3, columnspacing=1.2)
ax.set_title(TITLE, loc="left", fontsize=9, pad=26)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300)
print(f"saved {OUT}")
