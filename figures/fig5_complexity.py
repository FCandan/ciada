"""
Fig. 5 — CIADA Time Complexity Analysis
========================================
Left : Computation time vs. population size n (G=50 fixed)
Right: Computation time vs. maximum iterations G  (n=20 fixed)

Çalıştırma:
    pip install numpy matplotlib
    python fig5_complexity.py

Çıktı:
    fig5_complexity.png
"""

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

# ── CIADA tek çalıştırma ─────────────────────────────────────────────
def ciada_run(n, G, alpha=2.0, ideal=85.0, sigma=25.0, L=0.0, U=150.0):
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

# ── Ölçüm fonksiyonu (5 tekrar ortalaması) ────────────────────────────
def measure_time(n, G, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ciada_run(n, G)
        times.append((time.perf_counter() - t0) * 1000)  # ms
    return float(np.mean(times))

# ── n vs Süre (G=50 sabit) ────────────────────────────────────────────
N_VALS  = [5, 10, 20, 30, 50, 100, 200, 500, 1000]
G_FIXED = 50
print("n vs Süre (G=50):")
times_n = []
for n in N_VALS:
    t = measure_time(n, G_FIXED)
    times_n.append(t)
    print(f"  n={n:5d}: {t:.3f} ms")

# ── G vs Süre (n=20 sabit) ────────────────────────────────────────────
G_VALS  = [10, 20, 30, 50, 75, 100]
N_FIXED = 20
print("\nG vs Süre (n=20):")
times_G = []
for G in G_VALS:
    t = measure_time(N_FIXED, G)
    times_G.append(t)
    print(f"  G={G:5d}: {t:.3f} ms")

# ── Grafik ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

for ax, x_vals, times, xlabel, title, rec_span in [
    (axes[0], N_VALS, times_n,
     "Population Size (n)",
     f"Computation Time vs. n  (G={G_FIXED} fixed)",
     (10, 30)),
    (axes[1], G_VALS, times_G,
     "Maximum Iterations (G)",
     f"Computation Time vs. G  (n={N_FIXED} fixed)",
     (40, 75)),
]:
    # Ölçülen değerler
    ax.plot(x_vals, times, "o-", color="#1F3864",
            linewidth=2.5, markersize=7, zorder=5, label="Measured")

    # Teorik O(n) / O(G) eğrisi
    theory = [times[0] / x_vals[0] * x for x in x_vals]
    ax.plot(x_vals, theory, "--", color="#C0392B",
            linewidth=1.5, alpha=0.7, label="Theoretical O(n)")

    # Önerilen aralık — yeşil şeffaf alan
    ax.axvspan(rec_span[0], rec_span[1], alpha=0.08,
               color="green", label=f"Recommended [{rec_span[0]}–{rec_span[1]}]")

    # Değer etiketleri
    for x, t in zip(x_vals, times):
        ax.annotate(f"{t:.1f} ms", (x, t),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#333333")

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Computation Time (ms)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1F3864")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.fill_between(x_vals, times, alpha=0.08, color="#1F3864")
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle(
    "Fig. 5 — CIADA Time Complexity Analysis\n"
    "O(n × G × d) — Green zone = recommended operating range",
    fontsize=13, fontweight="bold", color="#1F3864", y=1.02
)
plt.tight_layout(pad=2.0)
plt.savefig("fig5_complexity.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
print("\nKaydedildi: fig5_complexity.png")
