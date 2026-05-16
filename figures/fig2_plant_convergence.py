"""
Fig. 2 — CIADA Convergence Curves for 10 Plant Species
=======================================================
Çalıştırma:
    pip install numpy matplotlib
    python fig2_plant_convergence.py

Çıktı:
    fig2_plant_convergence.png
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── CIADA Optimizer ───────────────────────────────────────────────────
def ciada_run(n=20, G=50, alpha=2.0, ideal=85, sigma=25, L=0, U=250, seed=42):
    np.random.seed(seed)
    pebbles = np.random.uniform(L, U, n)
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

# ── Bitki profilleri (FAO 2022, USDA-NRCS 2023) ──────────────────────
PLANTS = {
    "Tomato"    : {"ideal": 120, "sigma": 20, "L": 0, "U": 250},
    "Corn"      : {"ideal": 180, "sigma": 25, "L": 0, "U": 300},
    "Lettuce"   : {"ideal":  60, "sigma": 15, "L": 0, "U": 150},
    "Wheat"     : {"ideal":  95, "sigma": 22, "L": 0, "U": 200},
    "Pepper"    : {"ideal":  75, "sigma": 18, "L": 0, "U": 180},
    "Potato"    : {"ideal": 110, "sigma": 28, "L": 0, "U": 250},
    "Soybean"   : {"ideal":  65, "sigma": 20, "L": 0, "U": 180},
    "Cotton"    : {"ideal": 130, "sigma": 30, "L": 0, "U": 300},
    "Sunflower" : {"ideal":  88, "sigma": 22, "L": 0, "U": 200},
    "Strawberry": {"ideal":  55, "sigma": 16, "L": 0, "U": 150},
}

# ── 10 çalıştırma ortalaması ──────────────────────────────────────────
plant_histories = {}
for name, cfg in PLANTS.items():
    runs = [
        ciada_run(ideal=cfg["ideal"], sigma=cfg["sigma"],
                  L=cfg["L"], U=cfg["U"], seed=s)
        for s in range(10)
    ]
    plant_histories[name] = np.array(runs).mean(axis=0)

# ── Grafik ────────────────────────────────────────────────────────────
iters  = np.arange(1, 51)
colors = plt.cm.tab10(np.linspace(0, 1, 10))

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")

for i, (name, hist) in enumerate(plant_histories.items()):
    lw = 2.8 if name in ("Tomato", "Corn", "Lettuce") else 1.8
    ls = "-" if i < 5 else "--"
    ax.plot(iters, hist, color=colors[i], linewidth=lw,
            linestyle=ls, label=name)

ax.axhline(99, color="gray", linestyle=":", linewidth=1.2,
           alpha=0.7, label="99% Threshold")

ax.set_xlabel("Iteration", fontsize=12, fontweight="bold")
ax.set_ylabel("Fitness (%)", fontsize=12, fontweight="bold")
ax.set_title(
    "CIADA — Convergence Curves Across 10 Plant Species\n"
    "(n = 10 run average per species)",
    fontsize=12, fontweight="bold", color="#1F3864"
)
ax.legend(fontsize=9, ncol=2, framealpha=0.9, loc="lower right")
ax.set_xlim(1, 50)
ax.set_ylim(40, 104)
ax.grid(True, alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("fig2_plant_convergence.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Kaydedildi: fig2_plant_convergence.png")
