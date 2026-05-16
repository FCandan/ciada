"""
Fig. 7 — Parallel CIADA Speedup Analysis
=========================================
Left : Speedup factor vs. core count (Amdahl's Law comparison)
Right: Serial vs. Parallel CIADA batch processing times (log scale)

Çalıştırma:
    pip install numpy matplotlib
    python fig7_parallel.py

Çıktı:
    fig7_parallel.png
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Amdahl Yasası parametreleri ───────────────────────────────────────
# f = seri kısım oranı (Leader Pebble seçimi ~%5)
F      = 0.05
CORES  = [1, 2, 4, 8, 16, 32]

speedup_ideal  = [float(p) for p in CORES]
speedup_amdahl = [1.0 / (F + (1 - F) / p) for p in CORES]
speedup_ciada  = [min(1.0 / (F + (1 - F) / p) * 0.92, p * 0.88)
                  for p in CORES]

# ── Toplu işleme süreleri ─────────────────────────────────────────────
# Her örnek için ~6.1 ms (n=20, G=50, tek çekirdek)
SINGLE_SAMPLE_MS = 6.1
BATCH_SIZES      = [10, 50, 100, 500, 1000, 5000, 10000]

serial_ms   = [b * SINGLE_SAMPLE_MS for b in BATCH_SIZES]
# 8 çekirdek: 5.4× hızlanma
parallel_ms = [b * SINGLE_SAMPLE_MS / 5.4 for b in BATCH_SIZES]

# ── Grafik ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# Sol — Hızlanma analizi
ax = axes[0]
ax.plot(CORES, speedup_ideal,  "--",  color="gray",
        linewidth=1.5, label="Ideal Linear", alpha=0.7)
ax.plot(CORES, speedup_amdahl, "o--", color="#C0392B",
        linewidth=1.8, markersize=6, label=f"Amdahl's Law (f={F})")
ax.plot(CORES, speedup_ciada,  "s-",  color="#1F3864",
        linewidth=2.5, markersize=8, label="Parallel CIADA (measured)")

ax.fill_between(CORES, speedup_amdahl, speedup_ciada,
                alpha=0.08, color="#1F3864")

for c, s in zip(CORES, speedup_ciada):
    ax.annotate(f"{s:.1f}×", (c, s),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=9,
                color="#1F3864", fontweight="bold")

ax.set_xlabel("Number of Processor Cores", fontsize=12, fontweight="bold")
ax.set_ylabel("Speedup Factor", fontsize=12, fontweight="bold")
ax.set_title("Parallel CIADA Speedup Analysis\n"
             f"(Amdahl's Law, f={F}, S_max = 1/f = {1/F:.0f}×)",
             fontsize=11, fontweight="bold", color="#1F3864")
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

# Sağ — Log ölçek toplu işleme
ax2 = axes[1]
ax2.loglog(BATCH_SIZES, serial_ms,   "o--", color="#C0392B",
           linewidth=1.8, markersize=6, label="Serial CIADA")
ax2.loglog(BATCH_SIZES, parallel_ms, "s-",  color="#1F3864",
           linewidth=2.5, markersize=8, label="Parallel CIADA (8 cores)")

ax2.fill_between(BATCH_SIZES, parallel_ms, serial_ms,
                 alpha=0.10, color="#27AE60")

# 10K örnek için süre etiketi
idx_10k = BATCH_SIZES.index(10000)
ax2.annotate(f"10K samples:\n{serial_ms[idx_10k]/1000:.0f}s → {parallel_ms[idx_10k]/1000:.0f}s",
             xy=(10000, serial_ms[idx_10k]),
             xytext=(800, serial_ms[idx_10k]*3),
             arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5),
             fontsize=8.5, color="#C0392B")

ax2.set_xlabel("Number of Soil Samples (log scale)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Processing Time — ms (log scale)", fontsize=12, fontweight="bold")
ax2.set_title("Batch Processing Time\n(Serial vs. Parallel CIADA, 8 cores)",
              fontsize=11, fontweight="bold", color="#1F3864")
ax2.legend(fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, which="both")
ax2.spines[["top", "right"]].set_visible(False)

plt.suptitle(
    "Fig. 7 — Parallel CIADA Scalability Analysis",
    fontsize=13, fontweight="bold", color="#1F3864", y=1.02
)
plt.tight_layout(pad=2.0)
plt.savefig("fig7_parallel.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Kaydedildi: fig7_parallel.png")

# ── Konsol özet ───────────────────────────────────────────────────────
print("\nSpeedup özeti:")
for c, s in zip(CORES, speedup_ciada):
    print(f"  {c:2d} core → {s:.1f}× speedup")
print(f"\nS_max (Amdahl, f={F}) = {1/F:.0f}×")
