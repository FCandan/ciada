# CIADA — Algoritma Detayları

## Matematiksel Temel

### Hacimsel Yer Değiştirme Operatörü

```
ΔV = (U − L) × exp(−αt/G) × Rand(0, 1)
```

| Sembol | Anlam |
|--------|-------|
| `U − L` | Arama uzayının genişliği (kap hacmi) |
| `exp(−αt/G)` | Üstel sönümleme — zaman içinde adım küçülür |
| `Rand(0,1)` | Stokastik bileşen |
| `α` | Sönümleme katsayısı (önerilen: 2.0) |
| `t` | Mevcut iterasyon |
| `G` | Maksimum iterasyon |

### Sönümleme Profili (α=2.0)

| Aşama | t/G | exp(−αt/G) | Davranış |
|-------|-----|------------|----------|
| Erken | 0.0 | 1.000 | Büyük adımlar — global arama |
| Orta  | 0.5 | 0.368 | Orta adımlar — bölgesel tarama |
| Son   | 1.0 | 0.135 | Küçük adımlar — hassas ayar |

### Güncelleme Kuralları

**Stratejik Atış Modu** (f(Xᵢ) < f(X_best)):
```
Xᵢ_new = Xᵢ + ΔV × (X_best − Xᵢ) × Rand(0,1)
```

**Keşif Modu** (f(Xᵢ) ≥ f(X_best)):
```
Xᵢ_new = Xᵢ + Rand(−1, 1) × ΔV
```

**Kap Çeperi Kontrolü** (her güncellemeden sonra):
```
Xᵢ_new = Clamp(Xᵢ_new, L, U)
```

### Fitness Fonksiyonları

**Tek değişkenli Gaussian:**
```
f(x) = 100 × exp(−(x − x_ideal)² / (2σ²))
```

**Çok değişkenli sinerjik (Liebig [8]):**
```
scoreᵢ = 100 × exp(−(xᵢ − tᵢ)² / (2σᵢ²))
f(x)   = mean(scoreᵢ) × (min(scoreᵢ) / 100)
```

### Zaman Karmaşıklığı

```
T(n, G, d) = O(n × G × d)
```

- `n`: popülasyon büyüklüğü
- `G`: iterasyon sayısı  
- `d`: değişken boyutu (N-W-pH için d=3)

### Paralel Hızlanma (Amdahl [17])

```
S(p) = 1 / (f + (1−f)/p)    f = 0.05 (seri kısım oranı)
S_max = 1/f = 20x           (teorik üst sınır)
```

## Bitki Besleme Bağlamı

### Liebig'in Minimum Yasası [8]

Bitkinin büyümesi en kısıtlayıcı besin tarafından sınırlandırılır.
CIADA bunu matematiksel olarak şu şekilde modeller:

- `L` (alt sınır): kritik eksiklik eşiği
- `U` (üst sınır): toksisite başlangıç noktası
- `Clamp(x, L, U)`: kap çeperi mekanizması

### Önerilen Parametre Aralıkları

| Bitki | N (mg/kg) | Su (L/gün) | pH |
|-------|-----------|------------|-----|
| Domates | 120 | 4.5 | 6.2 |
| Mısır | 180 | 8.0 | 6.8 |
| Marul | 60 | 2.5 | 5.8 |
| Buğday | 95 | 4.0 | 6.5 |

Tam liste: `src/plant_profiles.py`

## İstatistiksel Doğrulama

- **Wilcoxon İşaret-Sıra Testi [11]**: ikili karşılaştırma
- **Friedman Testi [12]**: çoklu algoritma sıralama
- **Etki büyüklüğü (r)**: pratik anlamlılık ölçümü

CIADA tüm karşılaştırmalarda p < 0.001, r > 0.74 (Büyük etki) elde etmiştir.
