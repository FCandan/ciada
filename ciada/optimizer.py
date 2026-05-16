"""
CIADA — Crow-Inspired Adaptive Displacement Algorithm
======================================================
Karga İlhamlı Adaptif Yer Değiştirme Algoritması

Referans:
    Bu modül, CIADA akademik makalesinde tanımlanan algoritmayı
    Python ile uygular. IEEE atıf formatı için makaleye bakınız.

Kullanım:
    from ciada import CIADAOptimizer
    opt = CIADAOptimizer(n=20, G=50, alpha=2.0)
    result = opt.run(lower=0, upper=150, target=85)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class CIADAResult:
    """Optimizasyon sonucu veri sınıfı."""
    best_x: np.ndarray          # Optimum parametre vektörü
    best_fitness: float         # Elde edilen maksimum fitness değeri
    convergence_iter: int       # %99 eşiğine ulaşılan iterasyon (-1: ulaşılamadı)
    history: list               # Tüm iterasyonlardaki best_fitness geçmişi
    population_snapshots: dict  # Seçili iterasyonlarda popülasyon anlık görüntüleri
    elapsed_ms: float           # Toplam çalışma süresi (milisaniye)


class CIADAOptimizer:
    """
    CIADA — Crow-Inspired Adaptive Displacement Algorithm

    Parametreler
    ------------
    n : int
        Popülasyon büyüklüğü (taş sayısı). Önerilen: 20.
    G : int
        Maksimum iterasyon sayısı. Önerilen: 50.
    alpha : float
        Üstel sönümleme katsayısı. Güvenli aralık: [1.5, 3.0]. Önerilen: 2.0.
    seed : int, optional
        Tekrarlanabilirlik için rastgele tohum değeri.
    track_snapshots : list, optional
        Anlık görüntü alınacak iterasyon indisleri (0 tabanlı).
    """

    def __init__(
        self,
        n: int = 20,
        G: int = 50,
        alpha: float = 2.0,
        seed: Optional[int] = None,
        track_snapshots: Optional[list] = None,
    ):
        self.n = n
        self.G = G
        self.alpha = alpha
        self.seed = seed
        self.track_snapshots = track_snapshots or [0, 4, 9, 19, 29, 49]

        # seed run() başında uygulanır

    # ------------------------------------------------------------------
    # Ana optimizasyon metodu
    # ------------------------------------------------------------------
    def run(
        self,
        lower,
        upper,
        target: float = 100.0,
        fitness_fn: Optional[Callable] = None,
        ideal=None,
        sigma=None,
    ) -> CIADAResult:
        """
        CIADA optimizasyonunu çalıştırır.

        Parametreler
        ------------
        lower : float veya array-like
            Arama uzayı alt sınırı (L). Skalar veya d-boyutlu vektör.
        upper : float veya array-like
            Arama uzayı üst sınırı (U). Skalar veya d-boyutlu vektör.
        target : float
            Durdurma koşulu — bu fitness değerine ulaşınca algoritma durur.
        fitness_fn : callable, optional
            Özel fitness fonksiyonu: f(x) -> float. Verilmezse Gaussian modeli kullanılır.
        ideal : float veya array-like, optional
            Gaussian fitness modeli için ideal değer(ler).
        sigma : float veya array-like, optional
            Gaussian fitness modeli için standart sapma.

        Döndürür
        --------
        CIADAResult
        """
        import time

        lower = np.atleast_1d(np.array(lower, dtype=float))
        upper = np.atleast_1d(np.array(upper, dtype=float))
        d = len(lower)

        if fitness_fn is None:
            _ideal = np.atleast_1d(np.array(ideal if ideal is not None else (lower + upper) / 2))
            _sigma = np.atleast_1d(np.array(sigma if sigma is not None else (upper - lower) / 6))
            fitness_fn = self._gaussian_fitness(_ideal, _sigma)

        # Başlangıç popülasyonu — P(t=0) ~ Uniform(L, U)
        population = np.random.uniform(lower, upper, size=(self.n, d))

        best_fit = -np.inf
        best_x = population[0].copy()
        history = []
        snapshots = {}
        convergence_iter = -1

        if self.seed is not None:
            np.random.seed(self.seed)
        t_start = time.perf_counter()

        for t in range(self.G):
            # Fitness hesapla
            fits = np.array([fitness_fn(population[i]) for i in range(self.n)])

            # Leader Pebble seç
            leader_idx = np.argmax(fits)
            if fits[leader_idx] > best_fit:
                best_fit = fits[leader_idx]
                best_x = population[leader_idx].copy()

            history.append(best_fit)

            # Anlık görüntü al
            if t in self.track_snapshots:
                snapshots[t] = {
                    "population": population.copy(),
                    "fits": fits.copy(),
                    "best_x": best_x.copy(),
                    "best_fit": best_fit,
                    "delta_v": self._delta_v(t, lower, upper),
                }

            # Yakınsama kontrolü
            if best_fit >= target:
                if convergence_iter == -1:
                    convergence_iter = t
                break

            # Hacimsel Yer Değiştirme Operatörü: ΔV = (U-L) × exp(-αt/G) × Rand
            dv = self._delta_v(t, lower, upper)

            # Popülasyonu güncelle
            new_population = np.empty_like(population)
            for i in range(self.n):
                if fits[i] < best_fit:
                    # Stratejik Atış (Sömürü Modu)
                    step = dv * (best_x - population[i]) * np.random.rand(d)
                    new_population[i] = population[i] + step
                else:
                    # Keşif Modu
                    step = np.random.uniform(-1, 1, d) * dv
                    new_population[i] = population[i] + step

                # Sınır Kontrolü — Kap Çeperi
                new_population[i] = np.clip(new_population[i], lower, upper)

            population = new_population

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return CIADAResult(
            best_x=best_x,
            best_fitness=best_fit,
            convergence_iter=convergence_iter,
            history=history,
            population_snapshots=snapshots,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Toplu işleme (batch) — birden fazla toprak örneği için
    # ------------------------------------------------------------------
    def run_batch(self, samples: list, **kwargs) -> list:
        """
        Birden fazla toprak örneğini sıralı olarak optimize eder.

        Parametreler
        ------------
        samples : list of dict
            Her eleman {'lower': ..., 'upper': ..., 'ideal': ..., 'sigma': ...} içermeli.

        Döndürür
        --------
        list of CIADAResult
        """
        return [self.run(**s, **kwargs) for s in samples]

    # ------------------------------------------------------------------
    # Paralel toplu işleme
    # ------------------------------------------------------------------
    def run_parallel(self, samples: list, n_cores: int = 4, **kwargs) -> list:
        """
        Toprak örneklerini çok işlemcili mimaride paralel optimize eder.

        Parametreler
        ------------
        samples : list of dict
        n_cores : int
            Kullanılacak işlemci çekirdeği sayısı.
        """
        from multiprocessing import Pool

        def _worker(sample):
            opt = CIADAOptimizer(n=self.n, G=self.G, alpha=self.alpha)
            return opt.run(**sample, **kwargs)

        with Pool(processes=n_cores) as pool:
            results = pool.map(_worker, samples)
        return results

    # ------------------------------------------------------------------
    # Yardımcı metodlar
    # ------------------------------------------------------------------
    def _delta_v(self, t: int, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Hacimsel Yer Değiştirme Operatörü: ΔV = (U-L) × exp(-αt/G) × Rand(0,1)"""
        return (upper - lower) * np.exp(-self.alpha * t / self.G) * np.random.rand(len(lower))

    @staticmethod
    def _gaussian_fitness(ideal: np.ndarray, sigma: np.ndarray) -> Callable:
        """Çok boyutlu sinerjik Gaussian fitness fonksiyonu."""
        def fitness(x: np.ndarray) -> float:
            scores = 100.0 * np.exp(-((x - ideal) ** 2) / (2 * sigma ** 2))
            # Bağımsız boyut ortalaması — koordinat bazlı optimizasyon modeli
            return float(np.mean(scores))
        return fitness
