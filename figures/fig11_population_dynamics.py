"""
Fig. 11 — CIADA Population Dynamics (6 Snapshots)
==================================================
Çalıştırma:
    pip install numpy matplotlib
    python fig11_population_dynamics.py

Çıktı:
    fig11_population_dynamics.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(99)

# ── Parametreler ──────────────────────────────────────────────────────
N_PEBBLES = 30
G         = 50
IDEAL     = 85.0
SIGMA     = 25.0
L, U      = 0.0, 150.0
ALPHA     = 2.0
SNAPSHOTS = [0, 4, 9, 19, 29, 49]   # 0-tabanlı → iter 1,5,10,20,30,50

TITLES = [
    "Iteration 1\n(Random Initialization)",
    "Iteration 5\n(First Clustering)",
    "Iteration 10\n(Leader Convergence)",
    "Iteration 20\n(Cluster Formation)",
    "Iteration 30\n(Fine-Tuning Phase)",
    "Iteration 50\n(Convergence Complete)",
]

# ── CIADA — snapshot kayıtlı çalıştırma ──────────────────────────────
def run_with_snapshots():
    pebbles  = np.random.uniform(L, U, N_PEBBLES)
    best_fit = -np.inf
    best_p   = pebbles[0]
    snaps    = {}
    history  = []

    for t in range(G):
        fits = 100.0 * np.exp(-((pebbles - IDEAL)**2) / (2 * SIGMA**2))
        idx  = np.argmax(fits)
        if fits[idx] > best_fit:
            best_fit = fits[idx]
            best_p   = pebbles[idx]
        history.append(best_fit)

        if t in SNAPSHOTS:
            snaps[t] = {
                "pebbles"  : pebbles.copy(),
                "fits"     : fits.copy(),
                "best_p"   : best_p,
                "best_fit" : best_fit,
                "dv"       : (U - L) * np.exp(-ALPHA * t / G),
            }

        dv  = (U - L) * np.exp(-ALPHA * t / G) * np.random.rand()
        new = []
        for i in range(N_PEBBLES):
            if fits[i] < best_fit:
                p = pebbles[i] + dv * (best_p - pebbles[i]) * np.random.rand()
            else:
                p = pebbles[i] + np.random.uniform(-1, 1) * dv
            new.append(np.clip(p, L, U))
        pebbles = np.array(new)

    return snaps, history

snaps, history = run_with_snapshots()

# ── Fitness eğrisi (görselleştirme için) ─────────────────────────────
x_vals = np.linspace(L, U, 500)
y_vals = 100.0 * np.exp(-((x_vals - IDEAL)**2) / (2 * SIGMA**2))
iters_all = np.arange(1, G + 1)
dv_vals   = (U - L) * np.exp(-ALPHA * iters_all / G)

# ── Grafik ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("white")
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

for plot_idx, (t, title) in enumerate(zip(SNAPSHOTS, TITLES)):
    row, col = divmod(plot_idx, 3)
    ax = fig.add_subplot(gs[row, col])

    # Fitness eğrisi (arka plan)
    ax.plot(x_vals, y_vals, color="#1F3864", linewidth=2, alpha=0.35, zorder=1)
    ax.fill_between(x_vals, y_vals, alpha=0.05, color="#1F3864")

    if t in snaps:
        s       = snaps[t]
        pebbles = s["pebbles"]
        fits    = s["fits"]
        best_p  = s["best_p"]
        best_fit= s["best_fit"]
        std     = np.std(pebbles)
        mean_p  = np.mean(pebbles)

        # Pebble scatter — renk fitness'a göre
        sc = ax.scatter(pebbles, fits,
                        c=fits, cmap="RdYlGn", vmin=0, vmax=100,
                        s=75, zorder=4, edgecolors="white",
                        linewidths=0.5, alpha=0.9)

        # Leader pebble (altın yıldız)
        leader_fit = 100.0 * np.exp(-((best_p - IDEAL)**2) / (2 * SIGMA**2))
        ax.scatter([best_p], [leader_fit], color="gold", s=220, zorder=5,
                   marker="*", edgecolors="#1F3864", linewidths=1.5,
                   label=f"Leader: {best_p:.1f}")

        # Standart sapma bandı (turuncu)
        ax.axvspan(mean_p - std, mean_p + std,
                   alpha=0.10, color="#E67E22", zorder=0)

        # Metin kutusu
        spread = pebbles.max() - pebbles.min()
        ax.text(0.03, 0.97,
                f"Spread: {spread:.1f}\nBest: {best_fit:.1f}%\nStd: {std:.1f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="#CCCCCC"))
        ax.legend(fontsize=7.5, loc="lower right")

    # Hedef çizgisi
    ax.axvline(IDEAL, color="red", linestyle="--",
               linewidth=1.2, alpha=0.6, label=f"Target ({IDEAL})")

    ax.set_xlim(0, 150)
    ax.set_ylim(-5, 108)
    ax.set_xlabel("Nitrogen (mg/kg)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Fitness (%)", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=9.5, fontweight="bold", color="#1F3864")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

# ── Sağ alt panel: Yakınsama + ΔV daralması ──────────────────────────
ax_conv = fig.add_subplot(gs[2, 2])
ax_twin = ax_conv.twinx()

ax_conv.plot(iters_all, history, color="#1F3864",
             linewidth=2.5, label="Fitness (%)")
ax_twin.plot(iters_all, dv_vals, color="#E67E22",
             linewidth=1.8, linestyle="--", label="ΔV (Step Size)")

ax_conv.axhline(99, color="gray", linestyle=":", linewidth=1, alpha=0.7)

# Snapshot dikey çizgileri
for snap_t in SNAPSHOTS:
    ax_conv.axvline(snap_t + 1, color="#27AE60",
                    linewidth=0.8, alpha=0.5, linestyle=":")

ax_conv.set_xlabel("Iteration", fontsize=9, fontweight="bold")
ax_conv.set_ylabel("Fitness (%)", fontsize=9,
                   fontweight="bold", color="#1F3864")
ax_twin.set_ylabel("ΔV (Step Size)", fontsize=9,
                   fontweight="bold", color="#E67E22")
ax_conv.set_title("Convergence + ΔV Decay\n(Green lines = snapshot iterations)",
                  fontsize=9.5, fontweight="bold", color="#1F3864")
ax_conv.set_xlim(1, G)

lines1, labs1 = ax_conv.get_legend_handles_labels()
lines2, labs2 = ax_twin.get_legend_handles_labels()
ax_conv.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="lower right")
ax_conv.grid(True, alpha=0.25)
ax_conv.spines[["top"]].set_visible(False)

plt.suptitle(
    "Fig. 11 — CIADA Population Dynamics\n"
    f"(n={N_PEBBLES} pebbles, target={IDEAL} mg/kg, "
    "gold star = leader pebble, orange band = ±std range)",
    fontsize=13, fontweight="bold", color="#1F3864", y=1.01
)

plt.savefig("fig11_population_dynamics.png", dpi=160,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Kaydedildi: fig11_population_dynamics.png")
