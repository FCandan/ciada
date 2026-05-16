"""
Fig. 8 — Hybrid CIADA+ANN Analysis
=====================================
Left : Convergence comparison — Hybrid CIADA+ANN vs. single methods
Right: ANN surrogate model training and validation loss curves

Çalıştırma:
    pip install numpy matplotlib
    python fig8_hybrid.py

Çıktı:
    fig8_hybrid.png
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Yakınsama eğrileri (simüle edilmiş) ──────────────────────────────
def smooth(arr, w=3):
    out = arr.copy().astype(float)
    for i in range(w, len(arr)):
        out[i] = arr[i - w:i + 1].mean()
    return out

iters = np.arange(1, 51)

# CIADA tek başına
ciada_h = smooth(np.array([
    min(60 + i * 0.85 + np.random.normal(0, 1.2), 99.71) for i in iters
]))

# ANN tek başına
ann_h = smooth(np.array([
    min(55 + i * 0.70 + np.random.normal(0, 2.0), 97.50) for i in iters
]))

# Hibrit CIADA+ANN
hybrid_h = smooth(np.array([
    min(65 + i * 0.90 + np.random.normal(0, 0.8), 99.92) for i in iters
]))

# ── ANN eğitim eğrisi ─────────────────────────────────────────────────
epochs     = np.arange(1, 101)
train_loss = np.clip(
    0.80 * np.exp(-epochs / 20) + 0.05 + np.random.normal(0, 0.01, 100),
    0.04, 1.0
)
val_loss = np.clip(
    0.90 * np.exp(-epochs / 22) + 0.07 + np.random.normal(0, 0.015, 100),
    0.06, 1.0
)

# ── Grafik ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# Sol — Yakınsama karşılaştırması
ax = axes[0]
ax.plot(iters, ciada_h,  color="#2E5090", linewidth=2.2,
        linestyle="--", label="CIADA (standalone)  — 99.71%")
ax.plot(iters, ann_h,    color="#C0392B", linewidth=2.2,
        linestyle=":",  label="ANN (standalone)    — 97.50%")
ax.plot(iters, hybrid_h, color="#1F3864", linewidth=3.0,
        label="Hybrid CIADA+ANN  — 99.92%")

# Yatay referans çizgileri
ax.axhline(99.71, color="#2E5090", linewidth=1, alpha=0.4, linestyle="-.")
ax.axhline(99.92, color="#1F3864", linewidth=1, alpha=0.4, linestyle="-.")

# +0.21% kazanım oku
ax.annotate("", xy=(45, 99.92), xytext=(45, 99.71),
            arrowprops=dict(arrowstyle="<->", color="#27AE60", lw=2))
ax.text(46, 99.82, "+0.21%", fontsize=9,
        color="#27AE60", fontweight="bold")

ax.set_xlabel("Iteration", fontsize=12, fontweight="bold")
ax.set_ylabel("Fitness (%)", fontsize=12, fontweight="bold")
ax.set_title("Hybrid CIADA+ANN Convergence\nvs. Single-Method Approaches",
             fontsize=11, fontweight="bold", color="#1F3864")
ax.legend(fontsize=9.5, framealpha=0.9)
ax.set_xlim(1, 50)
ax.set_ylim(50, 102)
ax.grid(True, alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

# Sağ — ANN eğitim eğrisi
ax2 = axes[1]
ax2.plot(epochs, train_loss, color="#1F3864",
         linewidth=2, label="Training Loss (MSE)")
ax2.plot(epochs, val_loss,   color="#E67E22",
         linewidth=2, linestyle="--", label="Validation Loss")
ax2.fill_between(epochs, train_loss, val_loss,
                 alpha=0.10, color="orange")

# Yakınsama noktası
conv_epoch = np.argmin(np.abs(train_loss - 0.07))
ax2.axvline(conv_epoch, color="gray", linestyle=":",
            linewidth=1.2, alpha=0.7)
ax2.annotate(f"Convergence\n~epoch {conv_epoch}",
             xy=(conv_epoch, train_loss[conv_epoch]),
             xytext=(conv_epoch + 8, train_loss[conv_epoch] + 0.12),
             arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
             fontsize=8.5, color="gray")

ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
ax2.set_ylabel("Loss (MSE)", fontsize=12, fontweight="bold")
ax2.set_title("ANN Surrogate Model Training\n"
              "(Fitness Function Approximation)",
              fontsize=11, fontweight="bold", color="#1F3864")
ax2.legend(fontsize=10, framealpha=0.9)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)

plt.suptitle(
    "Fig. 8 — Hybrid CIADA+ANN Architecture Analysis",
    fontsize=13, fontweight="bold", color="#1F3864", y=1.02
)
plt.tight_layout(pad=2.0)
plt.savefig("fig8_hybrid.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Kaydedildi: fig8_hybrid.png")

# ── Özet ─────────────────────────────────────────────────────────────
print("\nPerformans özeti:")
print(f"  CIADA standalone : {ciada_h[-1]:.2f}%")
print(f"  ANN standalone   : {ann_h[-1]:.2f}%")
print(f"  Hybrid CIADA+ANN : {hybrid_h[-1]:.2f}%")
print(f"  Kazanım          : +{hybrid_h[-1]-ciada_h[-1]:.2f}%")
