"""
Fig. 4 — Effect of Damping Coefficient α on CIADA Final Fitness
================================================================
Çalıştırma:
    pip install numpy matplotlib
    python fig4_alpha_sensitivity.py

Çıktı:
    fig4_alpha_sensitivity.png
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── CIADA Optimizer ───────────────────────────────────────────────────
def ciada_run(n=20, G=50, alpha=2.0, ideal=85, sigma=25, L=0, U=150, seed=42):
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
    return history[-1]   # final fitness

# ── α değerleri ve 30 çalıştırma ortalaması ───────────────────────────
ALPHAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
N_RUNS = 30

mean_fits = []
for a in ALPHAS:
    fits = [ciada_run(alpha=a, seed=s) for s in range(N_RUNS)]
    mean_fits.append(np.mean(fits))
    print(f"  α={a:.1f}  →  Ort. Fitness: {np.mean(fits):.4f}%")

# ── Grafik ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")

# Renk: güvenli aralık mavi, dışarısı kırmızı, optimal koyu mavi
bar_colors = [
    "#C0392B" if (a < 1.5 or a > 3.0) else
    "#1F3864" if a == 2.0 else
    "#5B8DB8"
    for a in ALPHAS
]

bars = ax.bar(
    [str(a) for a in ALPHAS],
    mean_fits,
    color=bar_colors,
    edgecolor="white",
    linewidth=1.5,
    zorder=3
)

# Güvenli bölge — yeşil şeffaf alan (index 2–5 arası: 1.5, 2.0, 2.5, 3.0)
ax.axvspan(1.5, 4.5, alpha=0.08, color="green",
           label="Safe Operating Zone α ∈ [1.5, 3.0]")
ax.axhline(99, color="gray", linestyle="--", linewidth=1,
           alpha=0.6, label="99% Fitness Threshold")

# Değer etiketleri
for bar, val in zip(bars, mean_fits):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{val:.2f}",
        ha="center", va="bottom",
        fontsize=9, fontweight="bold", color="#333333"
    )

# Optimal işareti
opt_idx = ALPHAS.index(2.0)
ax.annotate(
    "Optimal\n(α = 2.0)",
    xy=(opt_idx, mean_fits[opt_idx]),
    xytext=(opt_idx + 1.3, mean_fits[opt_idx] - 1.8),
    arrowprops=dict(arrowstyle="->", color="#1F3864", lw=1.5),
    fontsize=9, color="#1F3864", fontweight="bold"
)

ax.set_xlabel("α (Damping Coefficient)", fontsize=12, fontweight="bold")
ax.set_ylabel("Average Final Fitness (%)", fontsize=12, fontweight="bold")
ax.set_title(
    "Effect of Damping Coefficient α on CIADA Final Fitness\n"
    f"(n=20, G=50, {N_RUNS} independent runs per α)",
    fontsize=12, fontweight="bold", color="#1F3864"
)
ax.set_ylim(88, 102)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, axis="y")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("fig4_alpha_sensitivity.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Kaydedildi: fig4_alpha_sensitivity.png")
