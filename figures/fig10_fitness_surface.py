"""
Fig. 10 — CIADA Fitness Surface in N-W-pH Space (Tomato)
=========================================================
Çalıştırma:
    pip install numpy matplotlib
    python fig10_fitness_surface.py

Çıktı:
    fig10_fitness_surface.png

Domates için:
    N_ideal  = 120 mg/kg
    W_ideal  = 4.5 L/day
    pH_ideal = 6.2
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

# ── Parametreler ──────────────────────────────────────────────────────
N_IDEAL  = 120.0;  SIGMA_N  = 25.0
W_IDEAL  = 4.5;    SIGMA_W  = 1.5
PH_IDEAL = 6.2;    SIGMA_PH = 0.5

# ── Fitness fonksiyonları ─────────────────────────────────────────────
def score(x, ideal, sigma):
    return 100.0 * np.exp(-((x - ideal)**2) / (2 * sigma**2))

# Eksen vektörleri
N   = np.linspace(0,   250,  120)
W   = np.linspace(0,   15,   120)
pH  = np.linspace(4.0, 9.0,  120)

# N-W düzlemi (pH sabit = ideal)
NN, WW   = np.meshgrid(N, W)
Z_NW     = score(NN, N_IDEAL, SIGMA_N) * score(WW, W_IDEAL, SIGMA_W) \
            * score(PH_IDEAL, PH_IDEAL, SIGMA_PH) / 100

# N-pH düzlemi (W sabit = ideal)
NN2, PH2 = np.meshgrid(N, pH)
Z_NpH    = score(NN2, N_IDEAL, SIGMA_N) * score(W_IDEAL, W_IDEAL, SIGMA_W) \
            * score(PH2, PH_IDEAL, SIGMA_PH) / 100

# W-pH düzlemi (N sabit = ideal)
WW3, PH3 = np.meshgrid(W, pH)
Z_WpH    = score(N_IDEAL, N_IDEAL, SIGMA_N) * score(WW3, W_IDEAL, SIGMA_W) \
            * score(PH3, PH_IDEAL, SIGMA_PH) / 100

# ── Grafik ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("white")

# ── Üst satır: 3D yüzeyler ───────────────────────────────────────────
for col, (X, Y, Z, xlabel, ylabel, cmap, opt_x, opt_y) in enumerate([
    (NN,  WW,  Z_NW,  "N (mg/kg)", "W (L/day)", "viridis", N_IDEAL,  W_IDEAL),
    (NN2, PH2, Z_NpH, "N (mg/kg)", "pH",         "plasma",  N_IDEAL,  PH_IDEAL),
    (WW3, PH3, Z_WpH, "W (L/day)", "pH",         "cool",    W_IDEAL,  PH_IDEAL),
]):
    ax3d = fig.add_subplot(2, 3, col + 1, projection="3d")
    surf = ax3d.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, linewidth=0)
    ax3d.scatter([opt_x], [opt_y], [Z.max()],
                 color="red", s=80, zorder=5, marker="*")
    ax3d.set_xlabel(xlabel, fontsize=9, labelpad=5)
    ax3d.set_ylabel(ylabel, fontsize=9, labelpad=5)
    ax3d.set_zlabel("Fitness (%)", fontsize=9, labelpad=5)
    titles = ["N-W Fitness Surface\n(pH = 6.2 fixed)",
              "N-pH Fitness Surface\n(W = 4.5 fixed)",
              "W-pH Fitness Surface\n(N = 120 fixed)"]
    ax3d.set_title(titles[col], fontsize=10, fontweight="bold", color="#1F3864")
    fig.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1)

# ── Alt satır: Kontur haritaları + risk haritası ──────────────────────

# N-W kontur
ax4 = fig.add_subplot(2, 3, 4)
ct1 = ax4.contourf(NN, WW, Z_NW, levels=25, cmap="viridis")
ax4.contour(NN, WW, Z_NW, levels=10, colors="white", alpha=0.3, linewidths=0.5)
ax4.scatter([N_IDEAL], [W_IDEAL], color="red", s=120, zorder=5,
            marker="*", label=f"Optimum ({N_IDEAL}, {W_IDEAL})")
ax4.set_xlabel("N (mg/kg)", fontsize=10, fontweight="bold")
ax4.set_ylabel("W (L/day)", fontsize=10, fontweight="bold")
ax4.set_title("N-W Contour Map\n(pH = 6.2 fixed)", fontsize=10,
              fontweight="bold", color="#1F3864")
ax4.legend(fontsize=9)
fig.colorbar(ct1, ax=ax4)

# N-pH kontur
ax5 = fig.add_subplot(2, 3, 5)
ct2 = ax5.contourf(NN2, PH2, Z_NpH, levels=25, cmap="plasma")
ax5.contour(NN2, PH2, Z_NpH, levels=10, colors="white", alpha=0.3, linewidths=0.5)
ax5.scatter([N_IDEAL], [PH_IDEAL], color="yellow", s=120, zorder=5,
            marker="*", label=f"Optimum ({N_IDEAL}, {PH_IDEAL})")
ax5.set_xlabel("N (mg/kg)", fontsize=10, fontweight="bold")
ax5.set_ylabel("pH", fontsize=10, fontweight="bold")
ax5.set_title("N-pH Contour Map\n(W = 4.5 fixed)", fontsize=10,
              fontweight="bold", color="#1F3864")
ax5.legend(fontsize=9)
fig.colorbar(ct2, ax=ax5)

# Toksisite Risk Haritası
ax6 = fig.add_subplot(2, 3, 6)
risk = np.zeros_like(Z_NW)
risk[Z_NW < 50]  = 1   # Yüksek risk
risk[(Z_NW >= 50) & (Z_NW < 80)] = 2  # Orta risk
risk[Z_NW >= 80] = 3   # Güvenli bölge

cmap_risk = matplotlib.colors.ListedColormap(["#C0392B", "#E67E22", "#27AE60"])
ax6.pcolormesh(NN, WW, risk, cmap=cmap_risk, shading="auto")
ax6.scatter([N_IDEAL], [W_IDEAL], color="white", s=150, zorder=5,
            marker="*", edgecolors="black", linewidths=1)
ax6.set_xlabel("N (mg/kg)", fontsize=10, fontweight="bold")
ax6.set_ylabel("W (L/day)", fontsize=10, fontweight="bold")
ax6.set_title("Toxicity Risk Zone Map\n(Red=High Risk, Orange=Caution, Green=Safe)",
              fontsize=10, fontweight="bold", color="#1F3864")
legend_elements = [
    mpatches.Patch(facecolor="#C0392B", label="High Risk  (f < 50%)"),
    mpatches.Patch(facecolor="#E67E22", label="Caution    (50–80%)"),
    mpatches.Patch(facecolor="#27AE60", label="Safe Zone  (f > 80%)"),
]
ax6.legend(handles=legend_elements, fontsize=8, loc="upper right")

plt.suptitle(
    "Fig. 10 — N-W-pH Fitness Landscape Visualization\n"
    f"Tomato (N_ideal={N_IDEAL} mg/kg, W_ideal={W_IDEAL} L/day, pH_ideal={PH_IDEAL})",
    fontsize=13, fontweight="bold", color="#1F3864", y=1.01
)
plt.tight_layout(pad=2.0)
plt.savefig("fig10_fitness_surface.png", dpi=160,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Kaydedildi: fig10_fitness_surface.png")
