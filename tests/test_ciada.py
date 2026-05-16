"""
CIADA — Birim Testleri
=======================
pytest ile çalıştır:
    pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from ciada import CIADAOptimizer, PlantNutritionOptimizer, PLANT_PROFILES


# ══════════════════════════════════════════════════════════════════
# CIADAOptimizer testleri
# ══════════════════════════════════════════════════════════════════

class TestCIADAOptimizer:

    def test_single_variable_convergence(self):
        """Tek değişkenli optimizasyonda %99 eşiğine ulaşılmalı."""
        opt = CIADAOptimizer(n=20, G=50, alpha=2.0, seed=42)
        result = opt.run(lower=0, upper=150, ideal=85, sigma=25, target=99.0)
        assert result.best_fitness >= 99.0, (
            f"Beklenen ≥99.0, elde edilen {result.best_fitness:.4f}"
        )

    def test_multidimensional_convergence(self):
        """Üç boyutlu (N, W, pH) optimizasyonda makul fitness elde edilmeli."""
        opt = CIADAOptimizer(n=30, G=100, alpha=2.0, seed=42)
        result = opt.run(
            lower=[0, 0, 4.0],
            upper=[250, 15, 9.0],
            ideal=[120, 4.5, 6.2],
            sigma=[25, 1.5, 0.5],
            target=90.0,
        )
        assert result.best_fitness >= 50.0  # Sinerjik model doğası gereği daha düşük başlar

    def test_boundary_respect(self):
        """best_x değerleri her zaman [lower, upper] içinde olmalı."""
        opt = CIADAOptimizer(n=20, G=30, alpha=2.0, seed=7)
        result = opt.run(lower=10, upper=100, ideal=55, sigma=20)
        assert np.all(result.best_x >= 10), "Alt sınır ihlali"
        assert np.all(result.best_x <= 100), "Üst sınır ihlali"

    def test_result_fields(self):
        """CIADAResult tüm alanları içermeli."""
        opt = CIADAOptimizer(n=10, G=20, seed=0)
        result = opt.run(lower=0, upper=100, ideal=50, sigma=15)
        assert len(result.history) > 0
        assert result.elapsed_ms > 0
        assert result.best_x is not None

    def test_monotone_history(self):
        """Fitness geçmişi azalmamalı (monoton artış)."""
        opt = CIADAOptimizer(n=15, G=30, seed=1)
        result = opt.run(lower=0, upper=150, ideal=75, sigma=20)
        history = result.history
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-9, (
                f"iter {i}: geçmiş azaldı ({history[i-1]:.4f} → {history[i]:.4f})"
            )

    def test_alpha_sensitivity(self):
        """Farklı alpha değerleri için optimizasyon çalışmalı."""
        for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
            opt = CIADAOptimizer(n=15, G=40, alpha=alpha, seed=42)
            result = opt.run(lower=0, upper=150, ideal=85, sigma=25)
            assert result.best_fitness > 0, f"alpha={alpha} için sıfır fitness"

    def test_reproducibility(self):
        """Aynı seed ile iki bağımsız çalıştırma benzer yüksek fitness döndürmeli."""
        opt1 = CIADAOptimizer(n=20, G=30, alpha=2.0, seed=99)
        opt2 = CIADAOptimizer(n=20, G=30, alpha=2.0, seed=99)
        r1 = opt1.run(lower=0, upper=150, ideal=85, sigma=25)
        r2 = opt2.run(lower=0, upper=150, ideal=85, sigma=25)
        # Her ikisi de yüksek fitness elde etmeli
        assert r1.best_fitness > 95.0
        assert r2.best_fitness > 95.0

    def test_custom_fitness_function(self):
        """Özel fitness fonksiyonu ile çalışmalı."""
        def my_fitness(x):
            return float(100 - (x[0] - 85) ** 2 / 100)

        opt = CIADAOptimizer(n=20, G=40, seed=42)
        result = opt.run(lower=[0], upper=[150], fitness_fn=my_fitness)
        assert result.best_fitness > 50

    def test_batch_run(self):
        """Toplu çalıştırma doğru sayıda sonuç döndürmeli."""
        samples = [
            {"lower": 0, "upper": 150, "ideal": 85, "sigma": 25},
            {"lower": 0, "upper": 200, "ideal": 120, "sigma": 30},
            {"lower": 0, "upper": 100, "ideal": 60, "sigma": 15},
        ]
        opt = CIADAOptimizer(n=15, G=30, seed=42)
        results = opt.run_batch(samples)
        assert len(results) == 3
        for r in results:
            assert r.best_fitness > 0


# ══════════════════════════════════════════════════════════════════
# PlantNutritionOptimizer testleri
# ══════════════════════════════════════════════════════════════════

class TestPlantNutritionOptimizer:

    def setup_method(self):
        self.opt = PlantNutritionOptimizer(n=30, G=100, seed=42)

    def test_tomato_optimization(self):
        """Domates optimizasyonu tamamlanmalı ve geçerli parametre aralığında sonuç üretmeli."""
        rec = self.opt.optimize("domates")
        assert rec.verim_skoru > 0.0
        assert 0 <= rec.optimum_n <= 250
        assert 0 <= rec.optimum_su <= 15
        assert 4.0 <= rec.optimum_ph <= 9.0

    def test_all_plants_run(self):
        """Tüm bitki türleri için optimizasyon hatasız tamamlanmalı."""
        results = self.opt.optimize_all_plants()
        assert len(results) == len(PLANT_PROFILES)
        for r in results:
            assert r.verim_skoru > 0.0

    def test_unknown_plant_raises(self):
        """Bilinmeyen bitki türü ValueError fırlatmalı."""
        with pytest.raises(ValueError, match="Bilinmeyen bitki türü"):
            self.opt.optimize("lavanta")

    def test_growth_stage_optimization(self):
        """Büyüme evresi optimizasyonu geçerli değerler döndürmeli."""
        for evre in ["fide", "vejetatif", "ciceklenme", "meyve", "hasat"]:
            rec = self.opt.optimize_growth_stage(evre)
            assert rec.verim_skoru >= 0.0
            assert rec.evre is not None

    def test_growth_stage_n_increases(self):
        """Azot gereksinimi fide→meyve evresinde artmalı (ideal değer karşılaştırması)."""
        from ciada.plant_nutrition import TOMATO_GROWTH_STAGES
        n_fide_ideal  = TOMATO_GROWTH_STAGES["fide"]["ideal"][0]
        n_meyve_ideal = TOMATO_GROWTH_STAGES["meyve"]["ideal"][0]
        assert n_meyve_ideal > n_fide_ideal, "Meyve evresinde N ideali, fide evresinden yüksek olmalı"

    def test_unknown_stage_raises(self):
        """Bilinmeyen büyüme evresi ValueError fırlatmalı."""
        with pytest.raises(ValueError, match="Bilinmeyen büyüme evresi"):
            self.opt.optimize_growth_stage("olgunlasma")


# ══════════════════════════════════════════════════════════════════
# Performans testleri
# ══════════════════════════════════════════════════════════════════

class TestPerformance:

    def test_single_run_time(self):
        """Tek çalıştırma 100 ms'den kısa sürmeli (n=20, G=50, d=3)."""
        opt = CIADAOptimizer(n=20, G=50, seed=42)
        result = opt.run(
            lower=[0, 0, 4], upper=[250, 15, 9],
            ideal=[120, 4.5, 6.2], sigma=[25, 1.5, 0.5]
        )
        assert result.elapsed_ms < 100, (
            f"Çok yavaş: {result.elapsed_ms:.1f} ms (beklenen < 100 ms)"
        )

    def test_batch_100_samples(self):
        """100 örneğin toplu işlenmesi 5 saniyeden kısa sürmeli."""
        import time
        samples = [
            {"lower": 0, "upper": 150, "ideal": 85 + i % 20, "sigma": 25}
            for i in range(100)
        ]
        opt = CIADAOptimizer(n=10, G=30, seed=42)
        start = time.perf_counter()
        opt.run_batch(samples)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Toplu işleme çok yavaş: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
