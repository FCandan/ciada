"""
CIADA — İstatistiksel Analiz Modülü
=====================================
Wilcoxon [11] ve Friedman [12] testleri ile
algoritma karşılaştırma fonksiyonları.
"""

import numpy as np
from scipy import stats
from typing import List, Dict


def run_multiple(optimizer_fn, n_runs: int = 30, **kwargs) -> np.ndarray:
    """
    Bir algoritmayı n_runs kez çalıştırıp final fitness değerlerini döner.

    Args:
        optimizer_fn : Çağrılabilir; history listesi dönen fonksiyon
        n_runs       : Bağımsız çalıştırma sayısı
        **kwargs     : optimizer_fn'e iletilecek parametreler

    Returns:
        shape (n_runs,) array — her çalıştırmanın final fitness değeri
    """
    results = []
    for seed in range(n_runs):
        np.random.seed(seed)
        result = optimizer_fn(**kwargs)
        final = result.history[-1] if hasattr(result, 'history') else result[-1]
        results.append(final)
    return np.array(results)


def descriptive_stats(data: np.ndarray) -> Dict:
    """Betimsel istatistikler."""
    return {
        "mean":   float(np.mean(data)),
        "std":    float(np.std(data)),
        "min":    float(np.min(data)),
        "max":    float(np.max(data)),
        "median": float(np.median(data)),
    }


def wilcoxon_test(data_a: np.ndarray,
                  data_b: np.ndarray,
                  alpha: float = 0.05) -> Dict:
    """
    İki algoritma arasında Wilcoxon İşaret-Sıra Testi [11].

    Args:
        data_a : Algoritma A'nın n_runs final fitness değerleri
        data_b : Algoritma B'nin n_runs final fitness değerleri
        alpha  : Anlamlılık düzeyi (varsayılan 0.05)

    Returns:
        {"statistic", "p_value", "significant", "effect_size_r"}
    """
    stat, p = stats.wilcoxon(data_a, data_b)
    n = len(data_a)
    # Etki büyüklüğü r = Z / sqrt(N)
    z = stats.norm.ppf(1 - p / 2)
    r = abs(z) / np.sqrt(n)

    magnitude = "Kucuk" if r < 0.3 else ("Orta" if r < 0.5 else "Buyuk")

    return {
        "statistic":    float(stat),
        "p_value":      float(p),
        "significant":  bool(p < alpha),
        "effect_size_r": float(r),
        "magnitude":    magnitude,
    }


def friedman_test(datasets: List[np.ndarray],
                  labels: List[str]) -> Dict:
    """
    Çoklu algoritma karşılaştırması için Friedman Testi [12].

    Args:
        datasets : Her algoritma için n_runs değerleri listesi
        labels   : Algoritma isimleri

    Returns:
        {"statistic", "p_value", "mean_ranks", "ranking"}
    """
    stat, p = stats.friedmanchisquare(*datasets)

    # Ortalama sıralamalar
    all_data = np.column_stack(datasets)
    ranks = np.array([stats.rankdata(row) for row in all_data])
    mean_ranks = ranks.mean(axis=0)

    ranking = sorted(zip(labels, mean_ranks), key=lambda x: x[1], reverse=True)

    return {
        "statistic":  float(stat),
        "p_value":    float(p),
        "significant": bool(p < 0.05),
        "mean_ranks": {l: float(r) for l, r in zip(labels, mean_ranks)},
        "ranking":    [(l, float(r)) for l, r in ranking],
    }


def convergence_speed(histories: List[List[float]],
                      threshold: float = 99.0) -> Dict:
    """
    Belirli bir fitness eşiğine kaç iterasyonda ulaşıldığını analiz eder.

    Args:
        histories  : Her çalıştırmanın iterasyon bazlı fitness listesi
        threshold  : Hedef fitness eşiği (%)

    Returns:
        {"mean_iter", "std_iter", "success_rate", "failed_runs"}
    """
    conv_iters = []
    failed = 0
    for hist in histories:
        reached = [i + 1 for i, f in enumerate(hist) if f >= threshold]
        if reached:
            conv_iters.append(reached[0])
        else:
            failed += 1

    if conv_iters:
        return {
            "mean_iter":    float(np.mean(conv_iters)),
            "std_iter":     float(np.std(conv_iters)),
            "success_rate": len(conv_iters) / len(histories) * 100,
            "failed_runs":  failed,
        }
    return {
        "mean_iter": None, "std_iter": None,
        "success_rate": 0.0, "failed_runs": failed,
    }
