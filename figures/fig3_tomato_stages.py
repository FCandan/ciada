"""
Fig. 3 — CIADA Convergence by Tomato Growth Stage + Optimal N Bar Chart
========================================================================
Çalıştırma:
    pip install numpy matplotlib
    python fig3_tomato_stages.py

Çıktı:
    fig3_tomato_stages.png
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── CIADA Optimizer ───────────────────────────────────────────────────
def ciada_run(n=20, G=50, alpha=2.0, ideal=85, sigma=25, L=0, U=250, seed=42):
    np.random.seed(seed)
    pebbles  = np.random.uniform(L, U, n)
    best_fit = -np.inf
    best_p   = pebbles[0]
    history  = []
    for t in range(G):
        fits = 100.0 * np.exp(-((pebbles - ideal)**2) / (2 * sigma**2))
        idx  = np.argmax(fits)
        if fits[idx] > best_fit:
            best_fit = fits[idx]
            best_p   = pebbles[idx]
        history.append(best_fit)
        dv = (U - L) * np.exp(-alpha * t / G) * np.random.rand()
        new = []
        for i in range(n):
            if fits[i] < best_fit:
                p = pebbles[i] + dv * (best_p - pebbles[i]) * np.random.rand()
            else:
                p = pebbles[i] + np.random.uniform(-1, 1) * dv
            new.append(np.clip(p, L, U))
        pebbles = np.array(new)
    return history

# ── Domates büyüme evreleri (FAO 2022) ────────────────────────────────
STAGES = {
    "Seedling\n(0–30 days)"    : {"ideal": 45,  "sigma": 12},
    "Vegetative\n(30–60 days)" : {"ideal": 75,  "sigma": 18},
    "Flowering\n(60–90 days)"  : {"ideal": 95,  "sigma": 20},
    "Fruiting\n(90–120 days)"  : {"ideal": 120, "sigma": 25},
    "Harvest\n(120–150 days)"  : {"ideal": 85,  "sigma": 20},
}

COLORS = ["#8E44AD", "#2980B9", "#27AE60", "#E67E22", "#C0392B"]

# ── 10 çalıştırma ortalaması ──────────────────────────────────────────
stage_histories = {}
for name, cfg in STAGES.items():
    runs = [
        ciada_run(ideal=cfg["ideal"], sigma=cfg["sigma"],
                  L=0, U=250, seed=s)
        for s in range(10)
    ]
    stage_histories[name] = np.array(runs).mean(axis=0)

# ── Grafik ────────────────────────────────────────────────────────────
iters = np.arange(1, 51)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# Sol panel — yakınsama eğrileri
ax = axes[0]
for i, (name, hist) in enumerate(stage_histories.items()):
    ax.plot(iters, hist, color=COLORS[i], linewidth=2.2, label=name)

ax.axhline(99, color="gray", linestyle="--", linewidth=1.2,
           alpha=0.6, label="99% Threshold")
ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
ax.set_ylabel("Fitness (%)", fontsize=11, fontweight="bold")
ax.set_title("CIADA Convergence by Tomato Growth Stage",
             fontsize=11, fontweight="bold", color="#1F3864")
ax.legend(fontsize=8.5, framealpha=0.9)
ax.set_xlim(1, 50)
ax.set_ylim(40, 104)
ax.grid(True, alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

# Sağ panel — optimum azot bar grafiği
ax2 = axes[1]
stage_labels = [s.split("\n")[0] for s in STAGES.keys()]
n_ideals     = [cfg["ideal"] for cfg in STAGES.values()]

bars = ax2.bar(stage_labels, n_ideals, color=COLORS,
               edgecolor="white", linewidth=1.5, width=0.6, zorder=3)

for bar, val in zip(bars, n_ideals):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{val} mg/kg",
        ha="center", va="bottom",
        fontsize=9, fontweight="bold"
    )

ax2.set_xlabel("Growth Stage", fontsize=11, fontweight="bold")
ax2.set_ylabel("Optimum Nitrogen (mg/kg)", fontsize=11, fontweight="bold")
ax2.set_title("Optimum Nitrogen Requirement\nper Growth Stage",
              fontsize=11, fontweight="bold", color="#1F3864")
ax2.set_ylim(0, 145)
ax2.grid(True, alpha=0.3, axis="y")
ax2.spines[["top", "right"]].set_visible(False)

plt.suptitle(
    "Fig. 3 — Tomato Plant Nutrition: Seasonal Variability Analysis (FAO, 2022)",
    fontsize=12, fontweight="bold", color="#1F3864", y=1.02
)
plt.tight_layout(pad=2.0)
plt.savefig("fig3_tomato_stages.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Kaydedildi: fig3_tomato_stages.png")
