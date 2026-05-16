"""
CIADA — Fitness Fonksiyonları
==============================
Bitki besleme ve standart benchmark fonksiyonları.

Kullanım:
    from fitness_functions import gaussian_1d, plant_nutrition_fitness
    from fitness_functions import sphere, rastrigin, ackley, rosenbrock, schwefel

Tüm fonksiyonlar MAKSİMİZASYON için tanımlanmıştır.
(Minimizasyon problemleri negatif alınarak dönüştürülür.)
"""

import numpy as np
from typing import Callable, Union


# ══════════════════════════════════════════════════════════════════════
# BİTKİ BESLEME FITNESS FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════

def gaussian_1d(ideal: float, sigma: float) -> Callable:
    """
    Tek boyutlu Gaussian fitness fonksiyonu.

    f(x) = 100 × exp(−(x − ideal)² / (2σ²))

    Parametreler
    ------------
    ideal : float — Hedef parametre değeri (örn. 85 mg/kg)
    sigma : float — Hassasiyet katsayısı (düşük σ = dar tepe)

    Döndürür
    --------
    f: float → float  (0–100 arası)

    Örnek
    -----
    fn = gaussian_1d(ideal=85.0, sigma=25.0)
    print(fn(85.0))   # → 100.0
    print(fn(60.0))   # → ~60.7
    """
    def f(x):
        x = np.atleast_1d(np.array(x, dtype=float))[0]
        return float(100.0 * np.exp(-((x - ideal) ** 2) / (2 * sigma ** 2)))
    return f


def plant_nutrition_fitness(
    ideal: Union[list, np.ndarray],
    sigma: Union[list, np.ndarray]
) -> Callable:
    """
    Çok boyutlu bağımsız ortalama fitness fonksiyonu (N, W, pH).

    f(x) = mean(scoreᵢ)
    scoreᵢ = 100 × exp(−(xᵢ − idealᵢ)² / (2σᵢ²))

    Parametreler
    ------------
    ideal : list/array — Her boyut için ideal değer
                         Örn. [120.0, 4.5, 6.2]  → N, W, pH
    sigma : list/array — Her boyut için hassasiyet
                         Örn. [25.0, 1.5, 0.5]

    Döndürür
    --------
    f: array → float  (0–100 arası)

    Örnek
    -----
    fn = plant_nutrition_fitness([120, 4.5, 6.2], [25, 1.5, 0.5])
    print(fn([120, 4.5, 6.2]))   # → 100.0
    print(fn([40, 1.0, 7.5]))    # → düşük değer
    """
    ideal_arr = np.array(ideal, dtype=float)
    sigma_arr = np.array(sigma, dtype=float)

    def f(x):
        x_arr = np.array(x, dtype=float)
        scores = 100.0 * np.exp(-((x_arr - ideal_arr) ** 2) / (2 * sigma_arr ** 2))
        return float(np.mean(scores))

    return f


# ── Hazır bitki profilleri (FAO [13], USDA-NRCS [14]) ─────────────────
PLANT_PROFILES = {
    "domates"   : {"ideal": [120.0, 4.5, 6.2], "sigma": [25.0, 1.5, 0.5],
                   "lower": [0, 0, 4], "upper": [250, 15, 9]},
    "misir"     : {"ideal": [180.0, 8.0, 6.8], "sigma": [25.0, 2.0, 0.6],
                   "lower": [0, 0, 4], "upper": [300, 20, 9]},
    "marul"     : {"ideal": [60.0,  2.5, 5.8], "sigma": [15.0, 0.8, 0.4],
                   "lower": [0, 0, 4], "upper": [150, 10, 9]},
    "bugday"    : {"ideal": [95.0,  4.0, 6.5], "sigma": [22.0, 1.2, 0.5],
                   "lower": [0, 0, 4], "upper": [200, 12, 9]},
    "biber"     : {"ideal": [75.0,  3.2, 6.0], "sigma": [18.0, 1.0, 0.4],
                   "lower": [0, 0, 4], "upper": [180, 12, 9]},
    "patates"   : {"ideal": [110.0, 5.5, 6.1], "sigma": [28.0, 1.5, 0.5],
                   "lower": [0, 0, 4], "upper": [250, 15, 9]},
    "soya"      : {"ideal": [65.0,  3.8, 6.3], "sigma": [20.0, 1.1, 0.5],
                   "lower": [0, 0, 4], "upper": [180, 12, 9]},
    "pamuk"     : {"ideal": [130.0, 6.2, 6.3], "sigma": [30.0, 1.8, 0.6],
                   "lower": [0, 0, 4], "upper": [300, 20, 9]},
    "aycicegi"  : {"ideal": [88.0,  4.1, 6.4], "sigma": [22.0, 1.2, 0.5],
                   "lower": [0, 0, 4], "upper": [200, 12, 9]},
    "cilek"     : {"ideal": [55.0,  2.8, 5.9], "sigma": [16.0, 0.9, 0.4],
                   "lower": [0, 0, 4], "upper": [150, 10, 9]},
}

# ── Domates büyüme evreleri (FAO [13]) ────────────────────────────────
TOMATO_GROWTH_STAGES = {
    "fide"        : {"gun": "0-30",   "ideal": [45.0,  1.5, 6.0], "sigma": [12.0, 0.5, 0.3]},
    "vejetatif"   : {"gun": "30-60",  "ideal": [75.0,  2.8, 6.1], "sigma": [18.0, 0.8, 0.4]},
    "ciceklenme"  : {"gun": "60-90",  "ideal": [95.0,  3.5, 6.2], "sigma": [20.0, 1.0, 0.4]},
    "meyve"       : {"gun": "90-120", "ideal": [120.0, 4.5, 6.2], "sigma": [25.0, 1.5, 0.5]},
    "hasat"       : {"gun": "120-150","ideal": [85.0,  3.0, 6.3], "sigma": [20.0, 1.0, 0.4]},
}


# ══════════════════════════════════════════════════════════════════════
# STANDART BENCHMARK FONKSİYONLARI (IEEE/CEC)
# Tümü MAKSİMİZASYON formatında — minimum = 0 → maksimum = 0
# ══════════════════════════════════════════════════════════════════════

def sphere(x) -> float:
    """
    Sphere — Unimodal referans fonksiyonu [Yao et al., 1999]
    f(x) = -sum(xᵢ²)
    Global max @ x* = (0,...,0), f(x*) = 0
    Önerilen aralık: xᵢ ∈ [-5.12, 5.12]
    """
    return -float(np.sum(np.array(x, dtype=float) ** 2))


def rastrigin(x) -> float:
    """
    Rastrigin — Multimodal benchmark [Mühlenbein et al., 1991]
    f(x) = -(10d + sum(xᵢ² - 10cos(2πxᵢ)))
    Global max @ x* = (0,...,0), f(x*) = 0
    Önerilen aralık: xᵢ ∈ [-5.12, 5.12]
    """
    x_arr = np.array(x, dtype=float)
    d = len(x_arr)
    val = 10 * d + np.sum(x_arr ** 2 - 10 * np.cos(2 * np.pi * x_arr))
    return -float(val)


def ackley(x) -> float:
    """
    Ackley — Multimodal, dar global minimum [Ackley, 1987]
    Global max @ x* = (0,...,0), f(x*) = 0
    Önerilen aralık: xᵢ ∈ [-32, 32]
    """
    x_arr = np.array(x, dtype=float)
    d = len(x_arr)
    a, b, c = 20.0, 0.2, 2 * np.pi
    s1 = np.sqrt(np.mean(x_arr ** 2))
    s2 = np.mean(np.cos(c * x_arr))
    val = -a * np.exp(-b * s1) - np.exp(s2) + a + np.e
    return -float(val)


def rosenbrock(x) -> float:
    """
    Rosenbrock — 'Banana valley', dar kıvrımlı vadi [Rosenbrock, 1960]
    Global max @ x* = (1,...,1), f(x*) = 0
    Önerilen aralık: xᵢ ∈ [-2, 2]
    """
    x_arr = np.array(x, dtype=float)
    val = np.sum(
        100 * (x_arr[1:] - x_arr[:-1] ** 2) ** 2 + (1 - x_arr[:-1]) ** 2
    )
    return -float(val)


def schwefel(x) -> float:
    """
    Schwefel — Aldatıcı, global minimum sınırda [Yao et al., 1999]
    Global max @ x* ≈ (420.968,...,420.968), f(x*) ≈ 0
    Önerilen aralık: xᵢ ∈ [-500, 500]
    """
    x_arr = np.array(x, dtype=float)
    d = len(x_arr)
    val = 418.9829 * d - np.sum(x_arr * np.sin(np.sqrt(np.abs(x_arr))))
    return -float(val)


def griewank(x) -> float:
    """
    Griewank — Geniş periyodik yapı [Griewank, 1981]
    Global max @ x* = (0,...,0), f(x*) = 0
    Önerilen aralık: xᵢ ∈ [-600, 600]
    """
    x_arr = np.array(x, dtype=float)
    s = np.sum(x_arr ** 2) / 4000
    p = np.prod(np.cos(x_arr / np.sqrt(np.arange(1, len(x_arr) + 1))))
    return -(s - p + 1)


def sine_1d(x) -> float:
    """
    Sine — Basit unimodal
    Global max @ x* = π/2 ≈ 1.5708, f(x*) = 1.0
    Önerilen aralık: x ∈ [0, 2π]
    """
    return float(np.sin(np.atleast_1d(np.array(x, dtype=float))[0]))


def multimodal_1d(x) -> float:
    """
    Çok tepeli Sine — Birden fazla yerel maksimum
    Global max @ x* ≈ 0.23
    Önerilen aralık: x ∈ [-3, 3]
    """
    xv = float(np.atleast_1d(np.array(x, dtype=float))[0])
    return float(np.sin(5 * xv) * np.exp(-0.5 * xv ** 2))


# ── Benchmark konfigurasyon sözlüğü ────────────────────────────────────
BENCHMARKS = {
    "sphere"    : {"fn": sphere,      "lower": -5.12,  "upper": 5.12,
                   "x_opt": 0.0,     "note": "Unimodal"},
    "rastrigin" : {"fn": rastrigin,   "lower": -5.12,  "upper": 5.12,
                   "x_opt": 0.0,     "note": "Multimodal (50+ local minima)"},
    "ackley"    : {"fn": ackley,      "lower": -32.0,  "upper": 32.0,
                   "x_opt": 0.0,     "note": "Multimodal, deep global min"},
    "rosenbrock": {"fn": rosenbrock,  "lower": -2.0,   "upper": 2.0,
                   "x_opt": 1.0,     "note": "Banana valley"},
    "schwefel"  : {"fn": schwefel,    "lower": -500.0, "upper": 500.0,
                   "x_opt": 420.968, "note": "Deceptive, boundary global min"},
    "griewank"  : {"fn": griewank,    "lower": -600.0, "upper": 600.0,
                   "x_opt": 0.0,     "note": "Wide periodic structure"},
    "sine"      : {"fn": sine_1d,     "lower": 0.0,    "upper": 6.2832,
                   "x_opt": 1.5708,  "note": "Simple unimodal"},
    "multimodal": {"fn": multimodal_1d,"lower": -3.0,  "upper": 3.0,
                   "x_opt": 0.23,    "note": "Multi-peak"},
}


# ══════════════════════════════════════════════════════════════════════
# DEMO — Doğrudan çalıştırma
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Fitness Fonksiyonları — Demo")
    print("=" * 55)

    # Bitki besleme
    print("\n[Bitki Besleme]")
    fn = gaussian_1d(ideal=85.0, sigma=25.0)
    for x in [85.0, 60.0, 120.0, 0.0]:
        print(f"  gaussian_1d(x={x:6.1f}) = {fn(x):.2f}%")

    fn3d = plant_nutrition_fitness([120, 4.5, 6.2], [25, 1.5, 0.5])
    print(f"\n  plant_nutrition([120, 4.5, 6.2]) = {fn3d([120, 4.5, 6.2]):.2f}%  (optimum)")
    print(f"  plant_nutrition([40, 1.0, 7.5])  = {fn3d([40, 1.0, 7.5]):.2f}%  (mevcut durum)")

    # Benchmark
    print("\n[Benchmark Fonksiyonları — Optimum Noktada Değer]")
    test_points = {
        "sphere":     [0.0, 0.0],
        "rastrigin":  [0.0, 0.0],
        "ackley":     [0.0, 0.0],
        "rosenbrock": [1.0, 1.0],
        "schwefel":   [420.968, 420.968],
        "griewank":   [0.0, 0.0],
        "sine":       [1.5708],
        "multimodal": [0.23],
    }
    for name, pt in test_points.items():
        fn_b = BENCHMARKS[name]["fn"]
        val  = fn_b(pt)
        note = BENCHMARKS[name]["note"]
        print(f"  {name:12s}: f(x*) = {val:8.4f}   [{note}]")

    print("\nBitki profilleri:", list(PLANT_PROFILES.keys()))
    print("Domates evreleri:", list(TOMATO_GROWTH_STAGES.keys()))
