"""
CIADA — Grafik Üretim Modülü
==============================
Akademik makale için tüm grafikleri üretir.

Çıktılar (outputs/ klasörüne kaydedilir):
    fig1_convergence.png      — Algoritma yakınsama karşılaştırması
    fig2_alpha_sensitivity.png — α hassasiyet analizi
    fig3_plant_convergence.png — 10 bitki türü yakınsama
    fig4_seasonal.png         — Mevsimsel değişkenlik
    fig5_complexity.png       — Zaman karmaşıklığı
    fig6_calibration.png      — Parametre kalibrasyon haritası
    fig7_parallel.png         — Paralel CIADA hızlanma
    fig8_hybrid.png           — Hibrit CIADA+ANN
    fig9_flowchart.png        — Akış şeması (blok diyagram)
    fig10_3d_surface.png      — 3D fitness yüzeyleri
    fig11_population.png      — Popülasyon dağılım animasyonu

Kullanım:
    python charts/generate_all.py
    python charts/generate_all.py --figures 1 3 9
"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Renk paleti ───────────────────────────────────────────────────
COLORS = {
    "ciada": "#1F3864",
    "pso":   "#2E8B57",
    "ga":    "#C0392B",
    "csa":   "#E67E22",
}

# ── CIADA & rakip simülatörler ───────────────────────────────────

def ciada_run(n=20, L=0, U=150, G=50, ideal=85, sigma=25, alpha=2.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pebbles = np.random.uniform(L, U, n)
    best_fit, best_p = -np.inf, pebbles[0]
    history = []
    for t in range(G):
        fits = 100 * np.exp(-((pebbles - ideal) ** 2) / (2 * sigma ** 2))
        idx = np.argmax(fits)
        if fits[idx] > best_fit:
            best_fit, best_p = fits[idx], pebbles[idx]
        history.append(best_fit)
        dv = (U - L) * np.exp(-alpha * t / G) * np.random.rand()
        new = []
        for i in range(n):
            p = (pebbles[i] + dv * (best_p - pebbles[i]) * np.random.rand()
                 if fits[i] < best_fit
                 else pebbles[i] + np.random.uniform(-1, 1) * dv)
            new.append(np.clip(p, L, U))
        pebbles = np.array(new)
    return history


def ga_run(n=20, L=0, U=150, G=50, ideal=85, sigma=25, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pop = np.random.uniform(L, U, n)
    best_fit, history = -np.inf, []
    for _ in range(G):
        fits = 100 * np.exp(-((pop - ideal) ** 2) / (2 * sigma ** 2))
        best_fit = max(best_fit, np.max(fits))
        history.append(best_fit)
        new = []
        for _ in range(n):
            p1, p2 = pop[np.random.randint(n)], pop[np.random.randint(n)]
            child = 0.5 * (p1 + p2) + np.random.normal(0, (U - L) * 0.05)
            new.append(np.clip(child, L, U))
        pop = np.array(new)
    return history


def pso_run(n=20, L=0, U=150, G=50, ideal=85, sigma=25, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pos = np.random.uniform(L, U, n)
    vel = np.zeros(n)
    pbest, pbest_fit = pos.copy(), 100 * np.exp(-((pos - ideal) ** 2) / (2 * sigma ** 2))
    gbest, gbest_fit = pos[np.argmax(pbest_fit)], np.max(pbest_fit)
    history = []
    for t in range(G):
        w = 0.9 - 0.5 * t / G
        fits = 100 * np.exp(-((pos - ideal) ** 2) / (2 * sigma ** 2))
        for i in range(n):
            if fits[i] > pbest_fit[i]:
                pbest[i], pbest_fit[i] = pos[i], fits[i]
            if fits[i] > gbest_fit:
                gbest, gbest_fit = pos[i], fits[i]
        history.append(gbest_fit)
        r1, r2 = np.random.rand(n), np.random.rand(n)
        vel = w * vel + 2.0 * r1 * (pbest - pos) + 2.0 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, L, U)
    return history


def csa_run(n=20, L=0, U=150, G=50, ideal=85, sigma=25, AP=0.1, fl=2.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pos = np.random.uniform(L, U, n)
    memory = pos.copy()
    best_fit, history = -np.inf, []
    for _ in range(G):
        fits = 100 * np.exp(-((pos - ideal) ** 2) / (2 * sigma ** 2))
        best_fit = max(best_fit, np.max(fits))
        history.append(best_fit)
        new = []
        for i in range(n):
            j = np.random.randint(n)
            p = (pos[i] + fl * np.random.rand() * (memory[j] - pos[i])
                 if np.random.rand() >= AP else np.random.uniform(L, U))
            new.append(np.clip(p, L, U))
            if fits[i] > 100 * np.exp(-((memory[i] - ideal) ** 2) / (2 * sigma ** 2)):
                memory[i] = pos[i]
        pos = np.array(new)
    return history


def multi_run(fn, n_runs=30, **kw):
    all_h = np.array([fn(**kw, seed=s) for s in range(n_runs)])
    return all_h.mean(0), all_h.std(0), all_h


# ── Şekil 1: Yakınsama Karşılaştırması ───────────────────────────

def fig1_convergence(save=True):
    G = 50
    iters = np.arange(1, G + 1)
    ciada_m, ciada_s, ciada_all = multi_run(ciada_run)
    pso_m,   pso_s,   pso_all   = multi_run(pso_run)
    ga_m,    ga_s,    ga_all    = multi_run(ga_run)
    csa_m,   csa_s,   csa_all   = multi_run(csa_run)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for label, m, s, c, lw in [
        ("CIADA", ciada_m, ciada_s, COLORS["ciada"], 2.8),
        ("PSO",   pso_m,   pso_s,   COLORS["pso"],   2.0),
        ("GA",    ga_m,    ga_s,    COLORS["ga"],    2.0),
        ("CSA",   csa_m,   csa_s,   COLORS["csa"],   2.0),
    ]:
        ax.plot(iters, m, color=c, lw=lw, label=label)
        ax.fill_between(iters, m - s, m + s, alpha=0.1, color=c)
    ax.axhline(99, color="gray", ls="--", lw=1, alpha=0.6)
    idx99 = np.where(ciada_m >= 99)[0]
    if len(idx99):
        ax.axvline(idx99[0] + 1, color=COLORS["ciada"], ls=":", lw=1.5, alpha=0.7)
        ax.annotate(f"CIADA %99\n@ iter {idx99[0]+1}",
                    xy=(idx99[0] + 1, 99), xytext=(idx99[0] + 6, 90),
                    arrowprops=dict(arrowstyle="->", color=COLORS["ciada"]),
                    fontsize=9, color=COLORS["ciada"])
    ax.set(xlabel="İterasyon", ylabel="Fitness (%)", xlim=(1, G), ylim=(50, 102))
    ax.set_title("Algoritma Yakınsama Karşılaştırması\n(ort. ± std, n=30)", fontweight="bold", color=COLORS["ciada"])
    ax.legend(); ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    bp = ax2.boxplot([ciada_all[:, -1], pso_all[:, -1], ga_all[:, -1], csa_all[:, -1]],
                     patch_artist=True, notch=True,
                     medianprops=dict(color="white", lw=2.5))
    for patch, c in zip(bp["boxes"], [COLORS["ciada"], COLORS["pso"], COLORS["ga"], COLORS["csa"]]):
        patch.set(facecolor=c, alpha=0.75)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color="gray", lw=1.2)
    ax2.set_xticklabels(["CIADA", "PSO", "GA", "CSA"], fontweight="bold")
    ax2.set(ylabel="Final Fitness")
    ax2.set_title("Final Fitness Dağılımı\n(n=30)", fontweight="bold", color=COLORS["ciada"])
    ax2.grid(alpha=0.3, axis="y"); ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "fig1_convergence.png")
        plt.savefig(path, dpi=180, bbox_inches="tight")
        print(f"  Kaydedildi: {path}")
    plt.close()


# ── Şekil 2: Alpha Hassasiyeti ────────────────────────────────────

def fig2_alpha_sensitivity(save=True):
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    fitness = []
    for a in alphas:
        runs = [ciada_run(alpha=a, seed=s) for s in range(30)]
        fitness.append(np.array(runs)[:, -1].mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ["#C0392B" if (a < 1.5 or a > 3.0) else ("#1F3864" if a == 2.0 else "#5B8DB8")
                  for a in alphas]
    bars = ax.bar([str(a) for a in alphas], fitness, color=bar_colors, edgecolor="white", lw=1.5)
    ax.axvspan(1.5, 4.5, alpha=0.07, color="green", label="Güvenli Aralık [1.5–3.0]")
    ax.axhline(99, color="gray", ls="--", lw=1, alpha=0.6)
    for bar, val in zip(bars, fitness):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.2f}", ha="center", fontsize=9, fontweight="bold", color="#333")
    opt = alphas.index(2.0)
    ax.annotate("Optimal\n(α=2.0)", xy=(opt, fitness[opt]),
                xytext=(opt + 1.2, fitness[opt] - 1.5),
                arrowprops=dict(arrowstyle="->", color=COLORS["ciada"]),
                fontsize=9, color=COLORS["ciada"])
    ax.set(xlabel="α (Sönümleme Katsayısı)", ylabel="Ort. Final Fitness (%)", ylim=(88, 102))
    ax.set_title("α Parametresi Hassasiyet Analizi\n(n=30 çalıştırma)", fontweight="bold", color=COLORS["ciada"])
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "fig2_alpha_sensitivity.png")
        plt.savefig(path, dpi=180, bbox_inches="tight")
        print(f"  Kaydedildi: {path}")
    plt.close()


# ── Şekil 9: Akış Şeması ──────────────────────────────────────────

def fig9_flowchart(save=True):
    fig, ax = plt.subplots(figsize=(14, 18))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0, 22); ax.axis("off")

    C = {"main": "#1F3864", "cond": "#B03A2E", "calc": "#1A5276",
         "act": "#1E8449", "end": "#6C3483", "expl": "#8B4513"}
    BG = "white"

    def rbox(x, y, w, h, text, color, fs=10.5, sub=None):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle="round,pad=0.1", lw=2.2, edgecolor=color, facecolor=BG, zorder=3))
        ax.text(x, y+(0.13 if sub else 0), text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=color, zorder=4)
        if sub:
            ax.text(x, y-0.22, sub, ha="center", va="center",
                    fontsize=8.5, color="#777", zorder=4, style="italic")

    def diam(x, y, w, h, text, color, fs=9.5):
        dx, dy = w/2, h/2
        pts = np.array([[x, y+dy], [x+dx, y], [x, y-dy], [x-dx, y]])
        ax.add_patch(plt.Polygon(pts, closed=True, facecolor=BG, edgecolor=color, lw=2.2, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=color, zorder=4)

    def arr(x1, y1, x2, y2, label=None, color="#2C3E50"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0))
        if label:
            ax.text((x1+x2)/2+0.15, (y1+y2)/2, label, fontsize=8.5,
                    color=color, fontweight="bold", va="center")

    # Renk kodları — üstte
    ax.text(0.3, 21.7, "Renk Kodları:", fontsize=9, fontweight="bold", color=C["main"])
    legend_items = [("Başlangıç/Bitiş", C["end"]), ("Girdi/Çıktı", C["main"]),
                    ("Hesaplama", C["calc"]), ("Koşul", C["cond"]),
                    ("Stratejik Atış", C["act"]), ("Keşif Modu", C["expl"])]
    for i, (lbl, col) in enumerate(legend_items):
        ax.add_patch(FancyBboxPatch((0.3+i*1.6, 21.1), 1.4, 0.42,
                     boxstyle="round,pad=0.05", facecolor=BG, edgecolor=col, lw=2, zorder=3))
        ax.text(1.0+i*1.6, 21.31, lbl, ha="center", va="center",
                fontsize=7.5, color=col, fontweight="bold", zorder=4)

    # Başlık
    ax.text(5, 20.7, "CIADA Algoritması — Akış Şeması",
            ha="center", fontsize=15, fontweight="bold", color=C["main"])
    ax.text(5, 20.3, "Crow-Inspired Adaptive Displacement Algorithm",
            ha="center", fontsize=11, color="#555", style="italic")

    # Akış kutuları
    rbox(5, 19.9, 3.5, 0.7, "BAŞLA", C["end"], fs=11); arr(5, 19.55, 5, 19.1)
    rbox(5, 18.7, 5.5, 0.75, "Girdiler: n, L, U, G, T, α", C["main"],
         sub="Popülasyon | Sınırlar | İter. | Hedef | Sönümleme"); arr(5, 18.32, 5, 17.65)
    rbox(5, 17.25, 5.8, 0.72, "P(t=0) = {x₁,...,xₙ} ~ Uniform(L, U)", C["calc"],
         sub="n adet çözüm adayı (taş) rastgele başlatılır"); arr(5, 16.89, 5, 16.25)
    rbox(5, 15.85, 5.5, 0.72, "Fitness Hesapla: f(Xᵢ) için i = 1..n", C["calc"],
         sub="f(x) = 100 × exp(−(x−x_ideal)² / 2σ²)"); arr(5, 15.49, 5, 14.85)
    diam(5, 14.35, 4.0, 0.88, "t < G ve\nMax(f) < T ?", C["cond"])
    arr(5, 13.91, 5, 13.25, label="Evet")
    ax.annotate("", xy=(8.5, 14.35), xytext=(7.0, 14.35),
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=2.0))
    ax.text(8.65, 14.35, "Hayır", fontsize=9, color="#C0392B", fontweight="bold", va="center")
    rbox(5, 12.85, 5.5, 0.72, "Leader Pebble Seç: X_best = argmax f(Xᵢ)", C["calc"],
         sub="En yüksek fitness değerine sahip taş lider olur"); arr(5, 12.49, 5, 11.85)
    rbox(5, 11.45, 5.8, 0.72, "ΔV = (U − L) × exp(−αt/G) × Rand(0,1)", C["calc"],
         sub="Hacimsel yer değiştirme — üssel adaptif adım boyutu"); arr(5, 11.09, 5, 10.45)
    rbox(5, 10.05, 4.0, 0.72, "FOR her Xᵢ (i = 1..n)", C["main"],
         sub="Tüm taşlar sırayla güncellenir"); arr(5, 9.69, 5, 9.05)
    diam(5, 8.55, 4.5, 0.88, "f(Xᵢ) < f(X_best) ?", C["cond"])
    ax.annotate("", xy=(2.2, 8.55), xytext=(3.0, 8.55),
                arrowprops=dict(arrowstyle="->", color=C["act"], lw=2.0))
    ax.text(1.55, 8.55, "Evet", fontsize=8.5, color=C["act"], fontweight="bold", va="center")
    ax.annotate("", xy=(7.8, 8.55), xytext=(7.0, 8.55),
                arrowprops=dict(arrowstyle="->", color=C["expl"], lw=2.0))
    ax.text(7.95, 8.55, "Hayır", fontsize=8.5, color=C["expl"], fontweight="bold", va="center")
    rbox(1.8, 7.55, 3.2, 1.4, "STRATEJİK ATIŞ\n(Sömürü Modu)", C["act"], fs=9.5)
    ax.text(1.8, 7.2, "Xᵢ_new = Xᵢ + ΔV × (X_best − Xᵢ)", ha="center", fontsize=8, color=C["act"], zorder=5)
    rbox(8.2, 7.55, 3.2, 1.4, "KEŞİF MODU\n(Exploration)", C["expl"], fs=9.5)
    ax.text(8.2, 7.2, "Xᵢ_new = Xᵢ + Rand(−1,1) × ΔV", ha="center", fontsize=8, color=C["expl"], zorder=5)
    arr(1.8, 6.85, 1.8, 6.35); arr(8.2, 6.85, 8.2, 6.35)
    ax.plot([1.8, 5], [6.35, 6.35], color="#2C3E50", lw=2)
    ax.plot([8.2, 5], [6.35, 6.35], color="#2C3E50", lw=2)
    arr(5, 6.35, 5, 5.85)
    rbox(5, 5.45, 5.5, 0.72, "Sınır Kontrolü: Xᵢ_new = Clamp(Xᵢ_new, L, U)", C["calc"],
         sub="Kap çeperi — toksisite sınırı aşımı engellenir"); arr(5, 5.09, 5, 4.45)
    rbox(5, 4.05, 5.5, 0.72, "Popülasyonu Güncelle: P(t+1) ← P_new", C["main"],
         sub="t = t + 1  |  Yeni nesil hazır")
    ax.plot([2.25, 0.8, 0.8], [4.05, 4.05, 14.35], color="#E67E22", lw=2.0, ls="--")
    ax.annotate("", xy=(3.0, 14.35), xytext=(0.8, 14.35),
                arrowprops=dict(arrowstyle="->", color="#E67E22", lw=2.0))
    ax.text(0.4, 9.2, "Sonraki\nİterasyon", fontsize=8.5, color="#E67E22",
            fontweight="bold", va="center", ha="center", rotation=90)
    rbox(8.5, 12.85, 2.5, 0.72, "Best_X\nBelirli", C["act"], fs=10)
    arr(8.5, 12.49, 8.5, 11.75)
    rbox(8.5, 11.35, 2.5, 0.72, "BİTİR", C["end"], fs=11)
    arr(5, 3.69, 5, 3.05)
    rbox(5, 2.65, 4.0, 0.72, "ÇIKTI: Best_X = Optimum Besleme Değeri", C["act"],
         sub="Toksisite sınırı içinde, hedef verimi karşılayan parametre seti")

    plt.tight_layout(pad=0.5)
    if save:
        path = os.path.join(OUTPUT_DIR, "fig9_flowchart.png")
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"  Kaydedildi: {path}")
    plt.close()


# ── Şekil 10: 3D Yüzey ───────────────────────────────────────────

def fig10_3d_surface(save=True):
    N  = np.linspace(0, 250, 100)
    W  = np.linspace(0, 15, 100)
    pH = np.linspace(4.0, 9.0, 100)
    NN, WW = np.meshgrid(N, W)
    NN2, PH = np.meshgrid(N, pH)
    WW3, PH3 = np.meshgrid(W, pH)

    def f(x, ideal, sigma):
        return 100 * np.exp(-((x - ideal) ** 2) / (2 * sigma ** 2))

    Z1 = f(NN, 120, 25) * f(WW, 4.5, 1.5) / 100
    Z2 = f(NN2, 120, 25) * f(PH, 6.2, 0.5) / 100
    Z3 = f(WW3, 4.5, 1.5) * f(PH3, 6.2, 0.5) / 100

    fig = plt.figure(figsize=(18, 12))
    for idx, (XX, YY, ZZ, xl, yl, cmap, opt) in enumerate([
        (NN, WW, Z1, "N (mg/kg)", "Su (L/gün)", "viridis", ([120], [4.5], [Z1.max()])),
        (NN2, PH, Z2, "N (mg/kg)", "pH", "plasma",  ([120], [6.2], [Z2.max()])),
        (WW3, PH3, Z3, "Su (L/gün)", "pH", "cool",  ([4.5], [6.2], [Z3.max()])),
    ]):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        surf = ax.plot_surface(XX, YY, ZZ, cmap=cmap, alpha=0.85, linewidth=0)
        ax.scatter(*opt, color="red", s=80, zorder=5)
        ax.set(xlabel=xl, ylabel=yl, zlabel="Verim (%)")
        ax.set_title(f"{xl.split(' ')[0]}-{yl.split(' ')[0]} Yüzeyi\n(3. değişken sabit)",
                     fontsize=10, fontweight="bold", color="#1F3864")
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)

    for idx, (XX, YY, ZZ, xl, yl, cmap, opt_xy) in enumerate([
        (NN, WW, Z1, "N (mg/kg)", "Su (L/gün)", "viridis", (120, 4.5)),
        (NN2, PH, Z2, "N (mg/kg)", "pH", "plasma", (120, 6.2)),
    ]):
        ax = fig.add_subplot(2, 3, idx + 4)
        c = ax.contourf(XX, YY, ZZ, levels=25, cmap=cmap)
        ax.contour(XX, YY, ZZ, levels=10, colors="white", alpha=0.3, lw=0.5)
        ax.scatter(*opt_xy, color="red", s=120, marker="*", zorder=5)
        ax.set(xlabel=xl, ylabel=yl)
        ax.set_title(f"{xl.split(' ')[0]}-{yl.split(' ')[0]} Kontur", fontsize=10, fontweight="bold", color="#1F3864")
        fig.colorbar(c, ax=ax)

    # Risk haritası
    ax = fig.add_subplot(2, 3, 6)
    risk = np.where(Z1 < 50, 1, np.where(Z1 < 80, 2, 3))
    cmap_r = matplotlib.colors.ListedColormap(["#C0392B", "#E67E22", "#27AE60"])
    ax.pcolormesh(NN, WW, risk, cmap=cmap_r, shading="auto")
    ax.scatter([120], [4.5], color="white", s=150, marker="*",
               edgecolors="black", lw=1, zorder=5)
    ax.set(xlabel="N (mg/kg)", ylabel="Su (L/gün)")
    ax.set_title("Toksisite Risk Haritası", fontsize=10, fontweight="bold", color="#1F3864")
    ax.legend(handles=[
        mpatches.Patch(facecolor="#C0392B", label="Yüksek Risk (f<50%)"),
        mpatches.Patch(facecolor="#E67E22", label="Orta Risk (50-80%)"),
        mpatches.Patch(facecolor="#27AE60", label="Güvenli (f>80%)"),
    ], fontsize=8, loc="upper right")

    plt.suptitle("N-W-pH Uzayında CIADA Fitness Fonksiyonu Görselleştirmesi\n"
                 "(Domates: N_ideal=120, W_ideal=4.5, pH_ideal=6.2)",
                 fontsize=13, fontweight="bold", color="#1F3864", y=1.01)
    plt.tight_layout(pad=2.0)
    if save:
        path = os.path.join(OUTPUT_DIR, "fig10_3d_surface.png")
        plt.savefig(path, dpi=160, bbox_inches="tight")
        print(f"  Kaydedildi: {path}")
    plt.close()


import matplotlib


# ── Ana çalıştırıcı ───────────────────────────────────────────────

FIGURE_MAP = {
    1:  fig1_convergence,
    2:  fig2_alpha_sensitivity,
    9:  fig9_flowchart,
    10: fig10_3d_surface,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIADA grafik üretici")
    parser.add_argument("--figures", nargs="*", type=int,
                        help="Üretilecek şekil numaraları (boş = hepsi)")
    args = parser.parse_args()

    targets = args.figures if args.figures else sorted(FIGURE_MAP.keys())
    print(f"Üretilecek şekiller: {targets}\n")
    t0 = time.time()
    for num in targets:
        if num in FIGURE_MAP:
            print(f"Şekil {num} üretiliyor...")
            FIGURE_MAP[num]()
        else:
            print(f"  [UYARI] Şekil {num} tanımlı değil, atlandı.")
    print(f"\nTamamlandı. Toplam süre: {time.time()-t0:.1f}s")
    print(f"Çıktılar: {os.path.abspath(OUTPUT_DIR)}/")
