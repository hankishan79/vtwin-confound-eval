# -*- coding: utf-8 -*-
"""make_figures.py — V-TWIN confound-aware framework: Figures 1-4 (300 dpi PNG)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 300})
C = dict(full="#2E5496", phys="#4CAF82", device="#E07B39", qc="#B0413E",
         age="#7E57C2", net="#5B8DB8", chance="#999999", v="#2E5496", q="#B0413E")
OUT = "/mnt/user-data/outputs/figures"
import os; os.makedirs(OUT, exist_ok=True)

def label_bars(ax, bars, fmt="%.2f", dy=0.01, fs=8):
    for b in bars:
        h = b.get_height()
        if np.isnan(h): continue
        ax.text(b.get_x()+b.get_width()/2, h+dy, fmt % h, ha="center", va="bottom", fontsize=fs)

def chance(ax):
    ax.axhline(0.5, ls="--", lw=1, color=C["chance"])
    ax.set_ylim(0, 1.0); ax.set_ylabel("AUROC")

# =========================================================================
# FIG 1 — Confound map: mean age by group/center
# =========================================================================
fig, ax = plt.subplots(figsize=(7, 3.6))
groups = ["IBNI\nLC", "IBNI\nCOPD", "IBNI\ncontrol", "Acıbadem", "Ankara Üni.", "AÜ Kalp"]
ages   = [61.6, 67.6, 44.1, 29.1, 25.0, 65.9]
ns     = [57, 36, 71, 88, 24, 17]
cols   = ["#B0413E", "#E07B39", "#4CAF82", "#7E57C2", "#9C7BC2", "#5B8DB8"]
bars = ax.bar(groups, ages, color=cols, edgecolor="white")
for b, n in zip(bars, ns):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.0f}y\n(n={n})",
            ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Mean age (years)"); ax.set_ylim(0, 80)
ax.set_title("Figure 1. Age differs systematically by group and center\n"
             "(cancer cases are older; control sources are younger and label-free)", fontsize=10)
ax.axhline(np.average(ages, weights=ns), ls=":", color="#555", lw=1)
plt.tight_layout(); plt.savefig(f"{OUT}/Figure1_confound_map.png", bbox_inches="tight"); plt.close()

# =========================================================================
# FIG 2 — Within-IBNI negative controls
# =========================================================================
contrasts = ["LC vs non-LC", "LC vs COPD", "LC vs all controls"]
data = {  # full, phys, device, QC-only, AGE-only, NET-bio
    "LC vs non-LC":      [0.795, 0.627, 0.867, 0.712, 0.799, 0.725],
    "LC vs COPD":        [0.849, 0.718, 0.930, 0.951, 0.700, 0.644],
    "LC vs all controls":[0.797, 0.652, 0.868, 0.774, 0.619, 0.732],
}
series = ["Full Voice45", "Physiological", "Spectral/acq.", "QC-only", "Age-only", "Confound-adj."]
colors = [C["full"], C["phys"], C["device"], C["qc"], C["age"], C["net"]]
fig, ax = plt.subplots(figsize=(8.2, 4.2))
x = np.arange(len(contrasts)); w = 0.13
for i, (s, col) in enumerate(zip(series, colors)):
    vals = [data[c][i] for c in contrasts]
    bars = ax.bar(x + (i-2.5)*w, vals, w, label=s, color=col, edgecolor="white", linewidth=0.4)
    label_bars(ax, bars, fs=6.5)
chance(ax); ax.set_xticks(x); ax.set_xticklabels(contrasts)
ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)
ax.set_title("Figure 2. Confound-only models match or exceed disease models (within single center)\n"
             "Spectral features beat physiological ones; age-matched physiological AUROC ≈ 0.55 (chance)",
             fontsize=9.5)
ax.annotate("age-matched\nphysiological\n≈ 0.55", xy=(0, 0.55), xytext=(0.0, 0.30),
            fontsize=7.5, ha="center", color="#4CAF82",
            arrowprops=dict(arrowstyle="->", color="#4CAF82", lw=1))
plt.tight_layout(); plt.savefig(f"{OUT}/Figure2_negative_controls.png", bbox_inches="tight"); plt.close()

# =========================================================================
# FIG 3 — Cross-cohort transfer collapse + leakage (headline)
# =========================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
subsets = ["Full", "Physiological", "Spectral/acq."]
cvA = [0.797, 0.652, 0.868]; cvB = [0.925, 0.781, 0.918]
ab  = [0.477, 0.295, 0.621]; ba  = [0.568, 0.385, 0.608]
x = np.arange(len(subsets)); w = 0.2
for i, (vals, lab, col) in enumerate([(cvA, "Internal CV (center)", "#9DB8D6"),
                                      (cvB, "Internal CV (web)", "#C9D9EC"),
                                      (ab, "Transfer A→B", "#2E5496"),
                                      (ba, "Transfer B→A", "#1B3A66")]):
    bars = axL.bar(x + (i-1.5)*w, vals, w, label=lab, color=col, edgecolor="white", linewidth=0.4)
    label_bars(axL, bars, fs=6.5)
chance(axL); axL.set_xticks(x); axL.set_xticklabels(subsets)
axL.legend(fontsize=7.5, frameon=False, loc="upper right")
axL.set_title("High internal CV, but transfer collapses;\nphysiological transfer ≤ chance", fontsize=9)

# right: residualized transfer up but leakage stays high
labels = ["raw\nA→B", "resid\nA→B", "raw\nB→A", "resid\nB→A", "source\nleakage", "leakage\nafter resid"]
vals   = [0.477, 0.718, 0.568, 0.650, 0.924, 0.918]
cols   = ["#2E5496", "#5B8DB8", "#1B3A66", "#3E6FA8", "#B0413E", "#D08B89"]
bars = axR.bar(labels, vals, color=cols, edgecolor="white")
label_bars(axR, bars, fs=7); axR.axhline(0.5, ls="--", lw=1, color=C["chance"])
axR.set_ylim(0, 1.0); axR.set_ylabel("AUROC")
axR.set_title("Residualised transfer rises, but\nleakage stays ≈0.91 → artefact", fontsize=9)
fig.suptitle("Figure 3. Cross-cohort transfer is confound-driven, not disease-driven", fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(f"{OUT}/Figure3_transfer_collapse.png", bbox_inches="tight"); plt.close()

# =========================================================================
# FIG 4 — Multi-center placebo (headline)
# =========================================================================
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.8))
# (a) site-leakage
sites = ["IBNI", "Acıbadem", "Kırıkkale", "Ankara"]
vv = [0.890, 0.894, 0.999, 0.986]; qq = [0.953, 0.917, 0.922, 0.961]
xx = np.arange(len(sites)); w = 0.38
b1 = a1.bar(xx-w/2, vv, w, label="from voice", color=C["v"]); b2 = a1.bar(xx+w/2, qq, w, label="from QC", color=C["q"])
label_bars(a1, b1, fs=7); label_bars(a1, b2, fs=7)
a1.set_xticks(xx); a1.set_xticklabels(sites, fontsize=8); a1.axhline(0.5, ls="--", lw=1, color=C["chance"])
a1.set_ylim(0,1.05); a1.set_ylabel("AUROC"); a1.legend(fontsize=7.5, frameon=False)
a1.set_title("(a) Site-leakage: center is\npredicted from voice (0.89–1.00)", fontsize=9)
# (b) sham cancer-free
sl = ["Full", "Physio", "Spectral", "QC", "Age"]; sv = [0.904, 0.905, 0.853, 0.969, 0.763]
sc = [C["full"], C["phys"], C["device"], C["qc"], C["age"]]
bb = a2.bar(sl, sv, color=sc); label_bars(a2, bb, fs=7); a2.axhline(0.5, ls="--", lw=1, color=C["chance"])
a2.set_ylim(0,1.05); a2.set_ylabel("AUROC")
a2.set_title("(b) SHAM: two cancer-FREE groups\nseparated at 0.90 (no disease!)", fontsize=9)
# (c) honest vs naive
gl = ["Full", "QC-only", "Age-only"]; hon = [0.795, 0.712, 0.799]; nai = [0.948, 0.981, 0.972]
xx = np.arange(len(gl)); w = 0.38
h1 = a3.bar(xx-w/2, hon, w, label="within-center control", color="#4CAF82")
h2 = a3.bar(xx+w/2, nai, w, label="cross-center control", color="#B0413E")
label_bars(a3, h1, fs=7); label_bars(a3, h2, fs=7)
a3.set_xticks(xx); a3.set_xticklabels(gl, fontsize=8); a3.axhline(0.5, ls="--", lw=1, color=C["chance"])
a3.set_ylim(0,1.05); a3.set_ylabel("AUROC"); a3.legend(fontsize=7.5, frameon=False)
a3.set_title("(c) Naive controls inflate LC\ndetection 0.80→0.95 (= age/QC)", fontsize=9)
fig.suptitle("Figure 4. Multi-center placebo: apparent LC detection is center/age, not disease",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig(f"{OUT}/Figure4_placebo.png", bbox_inches="tight"); plt.close()

print("Yazıldı:", os.listdir(OUT))
