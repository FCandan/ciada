"""
Fig. 9 — CIADA Algorithm Complete Flowchart
============================================
Çalıştırma:
    pip install numpy matplotlib
    python fig9_flowchart.py

Çıktı:
    fig9_flowchart.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 20))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 10)
ax.set_ylim(0, 22)
ax.axis("off")

# ── Renk paleti ───────────────────────────────────────────────────────
C_END  = "#6C3483"   # Mor   — Başlangıç / Bitiş
C_MAIN = "#1F3864"   # Koyu mavi — Girdi / Çıktı
C_CALC = "#1A5276"   # Orta mavi — Hesaplama
C_COND = "#B03A2E"   # Kırmızı — Koşul
C_ACT  = "#1E8449"   # Yeşil — Stratejik Atış
C_EXP  = "#8B4513"   # Kahve — Keşif Modu
BG     = "white"

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────
def box(ax, x, y, w, h, text, color, fs=10.5, sub=None):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          linewidth=2.2, edgecolor=color, facecolor=BG, zorder=3)
    ax.add_patch(rect)
    if sub:
        ax.text(x, y + 0.13, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=color, zorder=4)
        ax.text(x, y - 0.22, sub, ha="center", va="center",
                fontsize=8.5, color="#777777", zorder=4, style="italic")
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=color, zorder=4)

def diamond(ax, x, y, w, h, text, color, fs=9.5):
    pts = np.array([[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]])
    poly = plt.Polygon(pts, closed=True, facecolor=BG,
                       edgecolor=color, linewidth=2.2, zorder=3)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color, zorder=4)

def arrow(ax, x1, y1, x2, y2, color="#2C3E50", lw=2.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))

# ── Legend ────────────────────────────────────────────────────────────
ax.text(0.3, 21.7, "Color Codes:", fontsize=9, fontweight="bold", color=C_MAIN)
legend_items = [
    ("Start / End",      C_END),
    ("Input / Output",   C_MAIN),
    ("Computation",      C_CALC),
    ("Decision",         C_COND),
    ("Strategic Throw",  C_ACT),
    ("Exploration Mode", C_EXP),
]
for i, (label, color) in enumerate(legend_items):
    rect = FancyBboxPatch((0.3 + i*1.6, 21.1), 1.4, 0.42,
                          boxstyle="round,pad=0.05",
                          facecolor=BG, edgecolor=color, lw=2, zorder=3)
    ax.add_patch(rect)
    ax.text(1.0 + i*1.6, 21.31, label, ha="center", va="center",
            fontsize=7.5, color=color, fontweight="bold", zorder=4)

# ── Başlık ────────────────────────────────────────────────────────────
ax.text(5, 20.7, "CIADA Algorithm — Complete Flowchart",
        ha="center", va="center", fontsize=15, fontweight="bold", color=C_MAIN)
ax.text(5, 20.3, "Crow-Inspired Adaptive Displacement Algorithm",
        ha="center", va="center", fontsize=11, color="#555555", style="italic")

# ── Kutular ───────────────────────────────────────────────────────────
# START
box(ax, 5, 19.9, 3.5, 0.7, "START", C_END, fs=11)
arrow(ax, 5, 19.55, 5, 19.1)

# Inputs
box(ax, 5, 18.7, 5.8, 0.75,
    "Inputs: n, L, U, G, T, α", C_MAIN,
    sub="Population | Bounds | Iterations | Target | Damping")
arrow(ax, 5, 18.32, 5, 17.65)

# Initial Population
box(ax, 5, 17.25, 5.8, 0.72,
    "P(t=0) = {x₁, x₂, ..., xₙ} ~ Uniform(L, U)", C_CALC,
    sub="n solution candidates (pebbles) randomly initialized")
arrow(ax, 5, 16.89, 5, 16.25)

# Compute Fitness
box(ax, 5, 15.85, 5.5, 0.72,
    "Compute Fitness: f(Xᵢ) for i = 1..n", C_CALC,
    sub="f(x) = 100 × exp(−(x−x_ideal)² / 2σ²)")
arrow(ax, 5, 15.49, 5, 14.85)

# Decision 1
diamond(ax, 5, 14.35, 4.0, 0.88, "t < G  and\nMax(f) < T ?", C_COND)
arrow(ax, 5, 13.91, 5, 13.25, color="#2C3E50")
ax.text(5.2, 13.56, "Yes", fontsize=9, color="#2C3E50", fontweight="bold")

# No branch → right
ax.annotate("", xy=(8.5, 14.35), xytext=(7.0, 14.35),
            arrowprops=dict(arrowstyle="->", color=C_COND, lw=2.0))
ax.text(8.65, 14.35, "No", fontsize=9, color=C_COND,
        fontweight="bold", va="center")

# Select Leader
box(ax, 5, 12.85, 5.5, 0.72,
    "Select Leader: X_best = argmax f(Xᵢ)", C_CALC,
    sub="Pebble with highest fitness becomes leader")
arrow(ax, 5, 12.49, 5, 11.85)

# Compute ΔV
box(ax, 5, 11.45, 5.8, 0.72,
    "ΔV = (U − L) × exp(−2t/G) × Rand(0,1)   [α=2.0]", C_CALC,
    sub="Volumetric Displacement Operator — exponential adaptive step")
arrow(ax, 5, 11.09, 5, 10.45)

# FOR loop
box(ax, 5, 10.05, 4.0, 0.72,
    "FOR each Xᵢ  (i = 1..n)", C_MAIN,
    sub="All pebbles updated sequentially")
arrow(ax, 5, 9.69, 5, 9.05)

# Decision 2
diamond(ax, 5, 8.55, 4.5, 0.88, "f(Xᵢ) < f(X_best) ?", C_COND)

# Yes → left (Strategic Throw)
ax.annotate("", xy=(2.2, 8.55), xytext=(3.0, 8.55),
            arrowprops=dict(arrowstyle="->", color=C_ACT, lw=2.0))
ax.text(1.55, 8.55, "Yes", fontsize=8.5, color=C_ACT,
        fontweight="bold", va="center")

# No → right (Exploration)
ax.annotate("", xy=(7.8, 8.55), xytext=(7.0, 8.55),
            arrowprops=dict(arrowstyle="->", color=C_EXP, lw=2.0))
ax.text(7.95, 8.55, "No", fontsize=8.5, color=C_EXP,
        fontweight="bold", va="center")

# Strategic Throw box
box(ax, 1.8, 7.55, 3.2, 1.4,
    "STRATEGIC THROW\n(Exploitation Mode)", C_ACT, fs=9.5)
ax.text(1.8, 7.2, "Xᵢ_new = Xᵢ + ΔV × (X_best − Xᵢ)",
        ha="center", fontsize=8, color=C_ACT, zorder=5)

# Exploration Mode box
box(ax, 8.2, 7.55, 3.2, 1.4,
    "EXPLORATION MODE\n(Random Direction)", C_EXP, fs=9.5)
ax.text(8.2, 7.2, "Xᵢ_new = Xᵢ + Rand(−1,1) × ΔV",
        ha="center", fontsize=8, color=C_EXP, zorder=5)

# Merge arrows down
arrow(ax, 1.8, 6.85, 1.8, 6.35)
arrow(ax, 8.2, 6.85, 8.2, 6.35)
ax.plot([1.8, 5], [6.35, 6.35], color="#2C3E50", lw=2)
ax.plot([8.2, 5], [6.35, 6.35], color="#2C3E50", lw=2)
arrow(ax, 5, 6.35, 5, 5.85)

# Boundary Control
box(ax, 5, 5.45, 5.5, 0.72,
    "Boundary Control: Xᵢ_new = Clamp(Xᵢ_new, L, U)", C_CALC,
    sub="Container wall — toxicity boundary violation prevented")
arrow(ax, 5, 5.09, 5, 4.45)

# Update Population
box(ax, 5, 4.05, 5.5, 0.72,
    "Update Population: P(t+1) ← P_new", C_MAIN,
    sub="t = t + 1  |  New generation ready")

# Feedback loop — orange dashed
ax.plot([2.25, 0.8, 0.8], [4.05, 4.05, 14.35],
        color="#E67E22", lw=2.0, linestyle="--")
ax.annotate("", xy=(3.0, 14.35), xytext=(0.8, 14.35),
            arrowprops=dict(arrowstyle="->", color="#E67E22", lw=2.0))
ax.text(0.4, 9.2, "Next\nIteration", fontsize=8.5, color="#E67E22",
        fontweight="bold", va="center", ha="center", rotation=90)

# Best_X Found (from No branch)
box(ax, 8.5, 12.85, 2.5, 0.72, "Best_X\nFound", C_ACT, fs=10)
arrow(ax, 8.5, 12.49, 8.5, 11.75)

# END
box(ax, 8.5, 11.35, 2.5, 0.72, "END", C_END, fs=11)

# Output box
arrow(ax, 5, 3.69, 5, 3.05)
box(ax, 5, 2.65, 5.0, 0.72,
    "OUTPUT: Best_X = Optimal Nutrition Value", C_ACT,
    sub="Within toxicity bounds, meeting target yield threshold")

plt.tight_layout(pad=0.5)
plt.savefig("fig9_flowchart.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Kaydedildi: fig9_flowchart.png")
