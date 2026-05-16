import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, ArrowStyle
import matplotlib.patheffects as path_effects

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════
# GRAFİK 9 — CIADA Blok / Akış Şeması  (outline-only, beyaz iç)
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 18))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 22)
ax.axis('off')

# Renk paleti — artık yalnızca kenarlık ve metin rengi olarak kullanılıyor
C_MAIN   = '#1F3864'   # koyu mavi
C_COND   = '#B03A2E'   # koyu kırmızı
C_CALC   = '#1A5276'   # orta mavi
C_ACT    = '#1E8449'   # koyu yeşil
C_END    = '#6C3483'   # mor
C_ARROW  = '#2C3E50'
BG       = 'white'     # tüm kutuların iç rengi

def rounded_box(ax, x, y, w, h, text, color, fontsize=10.5, sub=None):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.1", linewidth=2.2,
                         edgecolor=color, facecolor=BG, zorder=3)
    ax.add_patch(box)
    if sub:
        ax.text(x, y + 0.13, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=color, zorder=4)
        ax.text(x, y - 0.22, sub, ha='center', va='center',
                fontsize=8.5, color='#777777', zorder=4, style='italic')
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=color, zorder=4)

def diamond(ax, x, y, w, h, text, color, fontsize=9.5):
    dx, dy = w/2, h/2
    pts = np.array([[x, y+dy], [x+dx, y], [x, y-dy], [x-dx, y]])
    poly = plt.Polygon(pts, closed=True, facecolor=BG, edgecolor=color, linewidth=2.2, zorder=3)
    ax.add_patch(poly)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=color, zorder=4, wrap=True)

def arrow(ax, x1, y1, x2, y2, label=None, color=C_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my, label, fontsize=8.5, color=color, fontweight='bold', va='center')

def side_arrow(ax, x1, y1, x2, y2, label=None, color='#C0392B'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                connectionstyle='arc3,rad=0.0'), zorder=2)
    if label:
        mx, my = (x1+x2)/2+0.1, (y1+y2)/2
        ax.text(mx, my, label, fontsize=8.5, color=color, fontweight='bold')

# ── Başlık ──
ax.text(5, 20.7, 'CIADA Algoritması — Akış Şeması',
        ha='center', va='center', fontsize=15, fontweight='bold', color=C_MAIN)
ax.text(5, 20.3, 'Crow-Inspired Adaptive Displacement Algorithm',
        ha='center', va='center', fontsize=11, color='#555555', style='italic')

# ── Kutular (yukarıdan aşağı) ──
# 1. BAŞLA
rounded_box(ax, 5, 19.9, 3.5, 0.7, 'BAŞLA', C_END, fontsize=11)
arrow(ax, 5, 19.55, 5, 19.1)

# 2. GİRDİLER
rounded_box(ax, 5, 18.7, 5.5, 0.75,
            'Girdiler: n, L, U, G, T, α', C_MAIN,
            sub='Popülasyon | Sınırlar | İter. | Hedef | Sönümleme')
arrow(ax, 5, 18.32, 5, 17.65)

# 3. BAŞLANGIÇ POPULASYONU
rounded_box(ax, 5, 17.25, 5.8, 0.72,
            'P(t=0) = {x₁, x₂, ..., xₙ} ~ Uniform(L, U)', C_CALC,
            sub='n adet çözüm adayı (taş) rastgele başlatılır')
arrow(ax, 5, 16.89, 5, 16.25)

# 4. FITNESS HESAPLA
rounded_box(ax, 5, 15.85, 5.5, 0.72,
            'Fitness Hesapla: f(Xᵢ) için i = 1..n', C_CALC,
            sub='f(x) = 100 × exp(−(x−x_ideal)² / 2σ²)')
arrow(ax, 5, 15.49, 5, 14.85)

# 5. KOŞUL 1 — t < G ?
diamond(ax, 5, 14.35, 4.0, 0.88, 't < G ve\nMax(f) < T ?', C_COND)
arrow(ax, 5, 13.91, 5, 13.25, label='Evet')

# Hayır çıkışı sağa
ax.annotate('', xy=(8.5, 14.35), xytext=(7.0, 14.35),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2.0))
ax.text(8.65, 14.35, 'Hayır', fontsize=9, color='#C0392B', fontweight='bold', va='center')

# 6. LEADER PEBBLE seç
rounded_box(ax, 5, 12.85, 5.5, 0.72,
            'Leader Pebble Seç: X_best = argmax f(Xᵢ)', C_CALC,
            sub='En yüksek fitness değerine sahip taş lider olur')
arrow(ax, 5, 12.49, 5, 11.85)

# 7. ΔV HESAPLA
rounded_box(ax, 5, 11.45, 5.8, 0.72,
            'ΔV = (U − L) × exp(−αt/G) × Rand(0,1)', C_CALC,
            sub='Hacimsel yer değiştirme — üssel adaptif adım boyutu')
arrow(ax, 5, 11.09, 5, 10.45)

# 8. DÖNGÜ — her Xi için
rounded_box(ax, 5, 10.05, 4.0, 0.72, 'FOR her Xᵢ (i = 1..n)', C_MAIN,
            sub='Tüm taşlar sırayla güncellenir')
arrow(ax, 5, 9.69, 5, 9.05)

# 9. KOŞUL 2 — f(Xi) < f(X_best)?
diamond(ax, 5, 8.55, 4.5, 0.88, 'f(Xᵢ) < f(X_best) ?', C_COND)

# Evet — stratejik atış (sola)
ax.annotate('', xy=(2.2, 8.55), xytext=(3.0, 8.55),
            arrowprops=dict(arrowstyle='->', color='#145A32', lw=2.0))
ax.text(1.55, 8.55, 'Evet', fontsize=8.5, color='#145A32', fontweight='bold', va='center')

# Hayır — keşif modu (sağa)
ax.annotate('', xy=(7.8, 8.55), xytext=(7.0, 8.55),
            arrowprops=dict(arrowstyle='->', color='#8B4513', lw=2.0))
ax.text(7.95, 8.55, 'Hayır', fontsize=8.5, color='#8B4513', fontweight='bold', va='center')

# Sol kutu — Stratejik Atış
rounded_box(ax, 1.8, 7.55, 3.2, 1.4,
            'STRATEJİK ATIŞ\n(Sömürü Modu)', C_ACT, fontsize=9.5)
ax.text(1.8, 7.2, 'Xᵢ_new = Xᵢ + ΔV × (X_best − Xᵢ)',
        ha='center', fontsize=8, color=C_ACT, zorder=5)

# Sağ kutu — Keşif Modu
rounded_box(ax, 8.2, 7.55, 3.2, 1.4,
            'KEŞİF MODU\n(Exploration)', '#8B4513', fontsize=9.5)
ax.text(8.2, 7.2, 'Xᵢ_new = Xᵢ + Rand(−1,1) × ΔV',
        ha='center', fontsize=8, color='#8B4513', zorder=5)

# Her iki kutundan aşağı birleşim
arrow(ax, 1.8, 6.85, 1.8, 6.35)
arrow(ax, 8.2, 6.85, 8.2, 6.35)
ax.plot([1.8, 5], [6.35, 6.35], color=C_ARROW, lw=2)
ax.plot([8.2, 5], [6.35, 6.35], color=C_ARROW, lw=2)
arrow(ax, 5, 6.35, 5, 5.85)

# 10. SINIR KONTROLÜ
rounded_box(ax, 5, 5.45, 5.5, 0.72,
            'Sınır Kontrolü: Xᵢ_new = Clamp(Xᵢ_new, L, U)', C_CALC,
            sub='Kap çeperi — toksisite sınırı aşımı engellenir')
arrow(ax, 5, 5.09, 5, 4.45)

# 11. POPULASYONU GÜNCELLE
rounded_box(ax, 5, 4.05, 5.5, 0.72,
            'Popülasyonu Güncelle: P(t+1) ← P_new', C_MAIN,
            sub='t = t + 1  |  Yeni nesil hazır')

# Geri dönüş oku — sol kenardan iterasyon koşuluna
ax.plot([2.25, 0.8, 0.8], [4.05, 4.05, 14.35], color='#E67E22', lw=2.0, linestyle='--')
ax.annotate('', xy=(3.0, 14.35), xytext=(0.8, 14.35),
            arrowprops=dict(arrowstyle='->', color='#E67E22', lw=2.0))
ax.text(0.4, 9.2, 'Sonraki\nİterasyon', fontsize=8.5, color='#E67E22',
        fontweight='bold', va='center', ha='center', rotation=90)

# 12. SONUÇ (hayır çıkışından)
rounded_box(ax, 8.5, 12.85, 2.5, 0.72, 'Best_X\nBelirli', C_ACT, fontsize=10)
arrow(ax, 8.5, 12.49, 8.5, 11.75)
rounded_box(ax, 8.5, 11.35, 2.5, 0.72, 'BİTİR', C_END, fontsize=11)

# Açıklama kutusu
ax.text(0.3, 21.7, 'Renk Kodları:', fontsize=9, fontweight='bold', color=C_MAIN)
legend_items = [
    ('Başlangıç/Bitiş', C_END), ('Girdi/Çıktı', C_MAIN),
    ('Hesaplama', C_CALC), ('Koşul', C_COND),
    ('Stratejik Atış', C_ACT), ('Keşif Modu', '#8B4513'),
]
for i, (label, color) in enumerate(legend_items):
    bx = FancyBboxPatch((0.3 + i*1.6, 21.1), 1.4, 0.42,
                        boxstyle="round,pad=0.05", facecolor=BG, edgecolor=color, lw=2, zorder=3)
    ax.add_patch(bx)
    ax.text(1.0 + i*1.6, 21.31, label, ha='center', va='center',
            fontsize=7.5, color=color, fontweight='bold', zorder=4)

arrow(ax, 5, 3.69, 5, 3.05)
rounded_box(ax, 5, 2.65, 4.0, 0.72, 'ÇIKTI: Best_X = Optimum Besleme Değeri', C_ACT,
            sub='Toksisite sınırı içinde, hedef verimi karşılayan parametre seti')

plt.tight_layout(pad=0.5)
plt.savefig('/home/claude/grafik9_blok_diyagram.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Grafik 9 — Blok diyagram kaydedildi")

# ══════════════════════════════════════════════════════════════════
# GRAFİK 10 — 3D Yüzey Grafikleri (N-W-pH fitness uzayı)
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('white')

# Panel 1: N-W düzlemi (pH=6.2 sabit)
N  = np.linspace(0, 250, 120)
W  = np.linspace(0, 15, 120)
NN, WW = np.meshgrid(N, W)
pH_fixed = 6.2
score_N  = 100 * np.exp(-((NN - 120)**2) / (2 * 25**2))
score_W  = 100 * np.exp(-((WW - 4.5)**2)  / (2 * 1.5**2))
score_pH = 100 * np.exp(-((pH_fixed - 6.2)**2) / (2 * 0.5**2))
Z1 = (score_N * score_W / 100) * (score_pH / 100)

ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(NN, WW, Z1, cmap='viridis', alpha=0.85, linewidth=0)
ax1.set_xlabel('Azot N (mg/kg)', fontsize=9, labelpad=6)
ax1.set_ylabel('Su W (L/gün)', fontsize=9, labelpad=6)
ax1.set_zlabel('Verim (%)', fontsize=9, labelpad=6)
ax1.set_title('N-W Fitness Yüzeyi\n(pH=6.2 sabit)', fontsize=10, fontweight='bold', color='#1F3864')
ax1.scatter([120], [4.5], [Z1.max()], color='red', s=80, zorder=5)
fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.1)

# Panel 2: N-pH düzlemi (W=4.5 sabit)
pH = np.linspace(4.0, 9.0, 120)
NN2, PH = np.meshgrid(N, pH)
score_N2  = 100 * np.exp(-((NN2 - 120)**2) / (2 * 25**2))
score_pH2 = 100 * np.exp(-((PH - 6.2)**2)  / (2 * 0.5**2))
score_W2  = 100 * np.exp(-((4.5 - 4.5)**2) / (2 * 1.5**2))
Z2 = (score_N2 * score_pH2 / 100) * (score_W2 / 100)

ax2 = fig.add_subplot(2, 3, 2, projection='3d')
surf2 = ax2.plot_surface(NN2, PH, Z2, cmap='plasma', alpha=0.85, linewidth=0)
ax2.set_xlabel('Azot N (mg/kg)', fontsize=9, labelpad=6)
ax2.set_ylabel('pH', fontsize=9, labelpad=6)
ax2.set_zlabel('Verim (%)', fontsize=9, labelpad=6)
ax2.set_title('N-pH Fitness Yüzeyi\n(W=4.5 L/gün sabit)', fontsize=10, fontweight='bold', color='#1F3864')
ax2.scatter([120], [6.2], [Z2.max()], color='red', s=80, zorder=5)
fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.1)

# Panel 3: W-pH düzlemi (N=120 sabit)
WW3, PH3 = np.meshgrid(W, pH)
score_N3  = 100 * np.exp(-((120 - 120)**2) / (2 * 25**2))
score_W3  = 100 * np.exp(-((WW3 - 4.5)**2) / (2 * 1.5**2))
score_pH3 = 100 * np.exp(-((PH3 - 6.2)**2) / (2 * 0.5**2))
Z3 = (score_W3 * score_pH3 / 100) * (score_N3 / 100)

ax3 = fig.add_subplot(2, 3, 3, projection='3d')
surf3 = ax3.plot_surface(WW3, PH3, Z3, cmap='cool', alpha=0.85, linewidth=0)
ax3.set_xlabel('Su W (L/gün)', fontsize=9, labelpad=6)
ax3.set_ylabel('pH', fontsize=9, labelpad=6)
ax3.set_zlabel('Verim (%)', fontsize=9, labelpad=6)
ax3.set_title('W-pH Fitness Yüzeyi\n(N=120 mg/kg sabit)', fontsize=10, fontweight='bold', color='#1F3864')
ax3.scatter([4.5], [6.2], [Z3.max()], color='red', s=80, zorder=5)
fig.colorbar(surf3, ax=ax3, shrink=0.5, pad=0.1)

# Panel 4: N-W kontur haritası
ax4 = fig.add_subplot(2, 3, 4)
contour = ax4.contourf(NN, WW, Z1, levels=25, cmap='viridis')
ax4.contour(NN, WW, Z1, levels=10, colors='white', alpha=0.3, linewidths=0.5)
ax4.scatter([120], [4.5], color='red', s=120, zorder=5, marker='*', label='Optimum (120, 4.5)')
ax4.set_xlabel('Azot N (mg/kg)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Su W (L/gün)', fontsize=10, fontweight='bold')
ax4.set_title('N-W Fitness Kontur Haritası\n(pH=6.2 sabit)', fontsize=10, fontweight='bold', color='#1F3864')
ax4.legend(fontsize=9)
fig.colorbar(contour, ax=ax4)

# Panel 5: N-pH kontur haritası
ax5 = fig.add_subplot(2, 3, 5)
contour2 = ax5.contourf(NN2, PH, Z2, levels=25, cmap='plasma')
ax5.contour(NN2, PH, Z2, levels=10, colors='white', alpha=0.3, linewidths=0.5)
ax5.scatter([120], [6.2], color='yellow', s=120, zorder=5, marker='*', label='Optimum (120, 6.2)')
ax5.set_xlabel('Azot N (mg/kg)', fontsize=10, fontweight='bold')
ax5.set_ylabel('pH', fontsize=10, fontweight='bold')
ax5.set_title('N-pH Fitness Kontur Haritası\n(W=4.5 sabit)', fontsize=10, fontweight='bold', color='#1F3864')
ax5.legend(fontsize=9)
fig.colorbar(contour2, ax=ax5)

# Panel 6: Toksisite risk bölgesi haritası
ax6 = fig.add_subplot(2, 3, 6)
risk = np.zeros_like(Z1)
risk[Z1 < 50]  = 1  # Yüksek risk (eksiklik veya toksisite)
risk[(Z1 >= 50) & (Z1 < 80)] = 2  # Orta risk
risk[Z1 >= 80]  = 3  # Güvenli bölge

cmap_risk = matplotlib.colors.ListedColormap(['#C0392B', '#E67E22', '#27AE60'])
im = ax6.pcolormesh(NN, WW, risk, cmap=cmap_risk, shading='auto')
ax6.scatter([120], [4.5], color='white', s=150, zorder=5, marker='*', edgecolors='black', linewidths=1)
ax6.set_xlabel('Azot N (mg/kg)', fontsize=10, fontweight='bold')
ax6.set_ylabel('Su W (L/gün)', fontsize=10, fontweight='bold')
ax6.set_title('Toksisite Risk Bölge Haritası\n(Kırmızı=Tehlike, Turuncu=Dikkat, Yeşil=Güvenli)', fontsize=10, fontweight='bold', color='#1F3864')
legend_elements = [
    mpatches.Patch(facecolor='#C0392B', label='Yüksek Risk (f<50%)'),
    mpatches.Patch(facecolor='#E67E22', label='Orta Risk (50-80%)'),
    mpatches.Patch(facecolor='#27AE60', label='Guvenli Bolge (f>80%)'),
]
ax6.legend(handles=legend_elements, fontsize=8, loc='upper right')

plt.suptitle('N-W-pH Uzayında CIADA Fitness Fonksiyonu Görselleştirmesi\n(Domates bitkisi: N_ideal=120 mg/kg, W_ideal=4.5 L/gün, pH_ideal=6.2)',
             fontsize=13, fontweight='bold', color='#1F3864', y=1.01)
plt.tight_layout(pad=2.0)
plt.savefig('/home/claude/grafik10_3d_yuzey.png', dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print("Grafik 10 — 3D yüzey grafikleri kaydedildi")

# ══════════════════════════════════════════════════════════════════
# GRAFİK 11 — Popülasyon Dağılım Animasyonu (6 anlık görüntü)
# ══════════════════════════════════════════════════════════════════
def run_ciada_tracked(n=30, L=0, U=150, G=50, ideal=85, sigma=25, alpha=2.0):
    pebbles = np.random.uniform(L, U, n)
    snapshots = {}
    best_fit = -np.inf
    best_p = pebbles[0]
    fitness_hist = []
    for t in range(G):
        fits = 100 * np.exp(-((pebbles - ideal)**2) / (2 * sigma**2))
        idx = np.argmax(fits)
        if fits[idx] > best_fit:
            best_fit = fits[idx]
            best_p = pebbles[idx]
        fitness_hist.append(best_fit)
        if t in [0, 4, 9, 19, 29, 49]:
            snapshots[t] = (pebbles.copy(), fits.copy(), best_p, best_fit)
        dv = (U - L) * np.exp(-alpha * t / G) * np.random.rand()
        new = []
        for i in range(n):
            if fits[i] < best_fit:
                p = pebbles[i] + dv * (best_p - pebbles[i]) * np.random.rand()
            else:
                p = pebbles[i] + np.random.uniform(-1, 1) * dv
            new.append(np.clip(p, L, U))
        pebbles = np.array(new)
    return snapshots, fitness_hist

np.random.seed(99)
snapshots, fitness_hist = run_ciada_tracked(n=30)

# X ekseni için fitness eğrisi
x_vals = np.linspace(0, 150, 500)
y_vals = 100 * np.exp(-((x_vals - 85)**2) / (2 * 25**2))

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

titles = ['İterasyon 1\n(Başlangıç — Rastgele Dağılım)',
          'İterasyon 5\n(İlk Kümelenme)',
          'İterasyon 10\n(Lider Etrafında Yoğunlaşma)',
          'İterasyon 20\n(Belirgin Küme Oluşumu)',
          'İterasyon 30\n(İnce Ayar Aşaması)',
          'İterasyon 50\n(Yakınsama Tamamlandı)']
iters_show = [0, 4, 9, 19, 29, 49]

for plot_idx, (t, title) in enumerate(zip(iters_show, titles)):
    row, col = divmod(plot_idx, 3)
    ax = fig.add_subplot(gs[row, col])

    # Fitness eğrisi
    ax.plot(x_vals, y_vals, color='#1F3864', linewidth=2, alpha=0.4, zorder=1)
    ax.fill_between(x_vals, y_vals, alpha=0.06, color='#1F3864')

    if t in snapshots:
        pebbles, fits, best_p, best_fit = snapshots[t]

        # Taşları renklendir: fitness'a göre
        scatter = ax.scatter(pebbles, fits,
                             c=fits, cmap='RdYlGn', vmin=0, vmax=100,
                             s=80, zorder=3, edgecolors='white', linewidths=0.5, alpha=0.9)

        # Lider taşı vurgula
        leader_fit = 100 * np.exp(-((best_p - 85)**2) / (2 * 25**2))
        ax.scatter([best_p], [leader_fit], color='gold', s=220, zorder=5,
                   marker='*', edgecolors='#1F3864', linewidths=1.5, label=f'Lider: {best_p:.1f}')

        # Standart sapma göster
        std = np.std(pebbles)
        mean = np.mean(pebbles)
        ax.axvspan(mean - std, mean + std, alpha=0.08, color='#E67E22')

        # Dağılım metriği
        spread = pebbles.max() - pebbles.min()
        ax.text(0.03, 0.97, f'Yayılım: {spread:.1f}\nEn İyi: {best_fit:.1f}%\nStd: {std:.1f}',
                transform=ax.transAxes, fontsize=8, va='top', color='#333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#CCC'))

        ax.legend(fontsize=7.5, loc='lower right')

    ax.axvline(85, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label='Hedef (85)')
    ax.set_xlim(0, 150)
    ax.set_ylim(-5, 108)
    ax.set_xlabel('Azot Miktarı (mg/kg)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Fitness (%)', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=9.5, fontweight='bold', color='#1F3864')
    ax.grid(True, alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)

# Panel 9 (sağ alt): Yakınsama eğrisi + ΔV eğrisi
ax_conv = fig.add_subplot(gs[2, 2])
iters_all = np.arange(1, 51)
dv_vals = 150 * np.exp(-2 * iters_all / 50)

ax_twin = ax_conv.twinx()
ax_conv.plot(iters_all, fitness_hist, color='#1F3864', linewidth=2.5, label='Fitness')
ax_twin.plot(iters_all, dv_vals, color='#E67E22', linewidth=1.8, linestyle='--', label='ΔV (Adım Boyutu)')
ax_conv.axhline(99, color='gray', linestyle=':', linewidth=1, alpha=0.7)

for snap_t in iters_show:
    ax_conv.axvline(snap_t + 1, color='#2E8B57', linewidth=0.8, alpha=0.5, linestyle=':')

ax_conv.set_xlabel('İterasyon', fontsize=9, fontweight='bold')
ax_conv.set_ylabel('Fitness (%)', fontsize=9, fontweight='bold', color='#1F3864')
ax_twin.set_ylabel('ΔV (Adım Boyutu)', fontsize=9, fontweight='bold', color='#E67E22')
ax_conv.set_title('Yakınsama + ΔV Daralması\n(Yeşil çizgiler = anlık görüntüler)', fontsize=9.5, fontweight='bold', color='#1F3864')
ax_conv.set_xlim(1, 50)
lines1, labels1 = ax_conv.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax_conv.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower right')
ax_conv.grid(True, alpha=0.25)
ax_conv.spines[['top']].set_visible(False)

plt.suptitle('CIADA Populasyon Dagilimi — Cakil Taslari Nasil Kumelenir?\n(n=30 tas, hedef=85 mg/kg, her kare farkli bir iterasyonu gostermektedir)',
             fontsize=13, fontweight='bold', color='#1F3864', y=1.01)

plt.savefig('/home/claude/grafik11_populasyon_dagilimi.png', dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print("Grafik 11 — Populasyon dagilim kaydedildi")
print("\nTum goruntulesme grafikleri tamamlandi.")
