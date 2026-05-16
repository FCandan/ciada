"""
Fig. 6 — Parameter Calibration: n-G Heatmap + Marginal Fitness Analysis
========================================================================
Left : Fitness quality heatmap for n-G combinations
Right: Marginal fitness improvement vs. population size (diminishing returns)

Çalıştırma:
    pip install numpy matplotlib
    python fig6_calibration.py

Çıktı:
    fig6_calibration.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

np.random.seed(42)

# ── CIADA Optimizer ───────────────────────────────────────────────────
def ciada_run(n, G, alpha=2.0, ideal=85.0, sigma=25.0, L=0.0, U=150.0, seed=42):
    np.random.seed(seed)
    pebbles  = np.random.uniform(L, U, n)
    best_fit = -np.inf
    best_p   = pebbles[0]
    for t in range(G):
        fits = 100.0 * np.exp(-((pebbles - ideal)**2) / (2 * sigma**2))
        idx  = np.argmax(fits)
        if fits[idx] > best_fit:
            best_fit = fits[idx]
            best_p   = pebbles[idx]
        dv  = (U - L) * np.exp(-alpha * t / G) * np.random.rand()
        new = []
        for i in range(n):
            if fits[i] < best_fit:
                p = pebbles[i] + dv * (best_p - pebbles[i]) * np.random.rand()
            else:
                p = pebbles[i] + np.random.uniform(-1, 1) * dv
            new.append(np.clip(p, L, U))
        pebbles = np.array(new)
    return best_fit

def mean_fitness(n, G, n_runs=10):
    return np.mean([ciada_run(n, G, seed=s) for s in range(n_runs)])

# ── Sol panel: n-G ısı haritası ────────────────────────────────────────
N_VALS = [5, 10, 15, 20, 30, 50]
G_VALS = [10, 20, 30, 50, 75, 100]

print("n-G ısı haritası hesaplanıyor...")
heatmap = np.zeros((len(N_VALS), len(G_VALS)))
for i, n in enumerate(N_VALS):
    for j, G in enumerate(G_VALS):
        heatmap[i, j] = mean_fitness(n, G, n_runs=10)
        print(f"  n={n:3d}, G={G:4d} → {heatmap[i,j]:.2f}%")

# ── Sağ panel: Marjinal fitness iyileşmesi ─────────────────────────────
N_MARG  = [5, 10, 15, 20, 30, 50, 75, 100]
G_FIXED = 50
print("\nMarjinal fitness hesaplanıyor (G=50)...")
marg_fits = []
for n in N_MARG:
    f = mean_fitness(n, G_FIXED, n_runs=15)
    marg_fits.append(f)
    print(f"  n={n:4d} → {f:.4f}%")

marg_fits = np.array(marg_fits)

# Marjinal kazanç (bir önceki n'ye göre artış)
marginal_gain = np.diff(marg_fits)

# ── Grafik ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# Sol — Isı haritası
ax = axes[0]
im  = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto",
                vmin=heatmap.min(), vmax=100)
ax.set_xticks(range(len(G_VALS)))
ax.set_yticks(range(len(N_VALS)))
ax.set_xticklabels([str(g) for g in G_VALS], fontsize=10)
ax.set_yticklabels([str(n) for n in N_VALS], fontsize=10)
ax.set_xlabel("Maximum Iterations (G)", fontsize=12, fontweight="bold")
ax.set_ylabel("Population Size (n)", fontsize=12, fontweight="bold")
ax.set_title("Fitness Quality Heatmap\n(n-G Combinations, 10 runs avg.)",
             fontsize=11, fontweight="bold", color="#1F3864")

# Hücre değerleri
for i in range(len(N_VALS)):
    for j in range(len(G_VALS)):
        val = heatmap[i, j]
        color = "white" if val < 97 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)

# Önerilen bölge — yeşil çerçeve (n=20, G=50 → i=3, j=3)
from matplotlib.patches import Rectangle
rec = Rectangle((2.5, 2.5), 2, 1,
                linewidth=3, edgecolor="#27AE60",
                facecolor="none", zorder=5,
                label="Recommended zone")
ax.add_patch(rec)
ax.legend(fontsize=9, loc="upper left")
fig.colorbar(im, ax=ax, label="Avg. Fitness (%)")

# Sağ — Marjinal kazanç
ax2 = axes[1]
ax2_twin = ax2.twinx()

# Fitness çizgisi
ax2.plot(N_MARG, marg_fits, "o-", color="#1F3864",
         linewidth=2.5, markersize=7, zorder=5, label="Avg. Fitness (%)")
ax2.fill_between(N_MARG, marg_fits, marg_fits.min(),
                 alpha=0.08, color="#1F3864")

# Marjinal kazanç çubuğu
ax2_twin.bar(N_MARG[1:], marginal_gain,
             width=[n*0.25 for n in N_MARG[1:]],
             color="#E67E22", alpha=0.5, zorder=3,
             label="Marginal Gain")
ax2_twin.axhline(0, color="gray", lw=0.8, ls="--")

# Önerilen n=20 işareti
ax2.axvline(20, color="#27AE60", linestyle="--",
            linewidth=1.8, alpha=0.8, label="Recommended n=20")

ax2.set_xlabel("Population Size (n)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Average Fitness (%)", fontsize=12,
               fontweight="bold", color="#1F3864")
ax2_twin.set_ylabel("Marginal Fitness Gain (%)", fontsize=11,
                    fontweight="bold", color="#E67E22")
ax2.set_title("Marginal Fitness Improvement\nvs. Population Size (G=50 fixed)",
              fontsize=11, fontweight="bold", color="#1F3864")

# Azalan getiri alanı
ax2.axvspan(30, 105, alpha=0.05, color="#C0392B",
            label="Diminishing returns zone")

lines1, labs1 = ax2.get_legend_handles_labels()
lines2, labs2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8.5,
           loc="lower right")
ax2.grid(True, alpha=0.3)
ax2.spines[["top"]].set_visible(False)
ax2_twin.spines[["top"]].set_visible(False)

plt.suptitle(
    "Fig. 6 — CIADA Parameter Calibration Analysis\n"
    "Green zone = recommended operating range (n=20, G=50)",
    fontsize=13, fontweight="bold", color="#1F3864", y=1.02
)
plt.tight_layout(pad=2.0)
plt.savefig("fig6_calibration.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
print("\nKaydedildi: fig6_calibration.png")
