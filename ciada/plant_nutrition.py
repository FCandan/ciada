"""
CIADA — Bitki Besleme Modülü
=============================
Tarımsal optimizasyon için hazır yapılandırmalar.

Desteklenen bitki türleri ve büyüme evreleri FAO [13] ve
USDA-NRCS [14] referanslı değerlere dayanmaktadır.
"""

import numpy as np
from .optimizer import CIADAOptimizer, CIADAResult
from dataclasses import dataclass
from typing import Optional


# ------------------------------------------------------------------
# Bitki Profilleri (FAO [13], USDA-NRCS [14])
# ------------------------------------------------------------------
PLANT_PROFILES = {
    "domates": {
        "ideal": np.array([120.0, 4.5, 6.2]),   # [N mg/kg, Su L/gün, pH]
        "sigma":  np.array([25.0,  1.5, 0.5]),
        "lower":  np.array([0.0,   0.0, 4.0]),
        "upper":  np.array([250.0, 15.0, 9.0]),
        "kaynak": "FAO (2022) [13]",
    },
    "misir": {
        "ideal": np.array([180.0, 8.0, 6.8]),
        "sigma":  np.array([25.0,  2.0, 0.6]),
        "lower":  np.array([0.0,   0.0, 4.0]),
        "upper":  np.array([300.0, 20.0, 9.0]),
        "kaynak": "USDA-NRCS (2023) [14]",
    },
    "marul": {
        "ideal": np.array([60.0,  2.5, 5.8]),
        "sigma":  np.array([15.0,  0.8, 0.4]),
        "lower":  np.array([0.0,   0.0, 4.0]),
        "upper":  np.array([150.0, 10.0, 9.0]),
        "kaynak": "FAO (2022) [13]",
    },
    "bugday": {
        "ideal": np.array([95.0,  4.0, 6.5]),
        "sigma":  np.array([22.0,  1.2, 0.5]),
        "lower":  np.array([0.0,   0.0, 4.0]),
        "upper":  np.array([200.0, 12.0, 9.0]),
        "kaynak": "CIMMYT (2021) [15]",
    },
    "biber": {
        "ideal": np.array([75.0,  3.2, 6.0]),
        "sigma":  np.array([18.0,  1.0, 0.4]),
        "lower":  np.array([0.0,   0.0, 4.0]),
        "upper":  np.array([180.0, 12.0, 9.0]),
        "kaynak": "FAO (2022) [13]",
    },
    "patates": {
        "ideal": np.array([110.0, 5.5, 6.1]),
        "sigma":  np.array([28.0,  1.5, 0.5]),
        "lower":  np.array([0.0,   0.0, 4.0]),
        "upper":  np.array([250.0, 15.0, 9.0]),
        "kaynak": "CIP (2020) [16]",
    },
    "soya": {
        "ideal": np.array([65.0,  3.8, 6.3]),
        "sigma":  np.array([20.0,  1.1, 0.5]),
        "lower":  np.array([0.0,   0.0, 4.0]),
        "upper":  np.array([180.0, 12.0, 9.0]),
        "kaynak": "USDA-NRCS (2023) [14]",
    },
    "pamuk": {
        "ideal": np.array([130.0, 6.2, 6.3]),
        "sigma":  np.array([30.0,  1.8, 0.6]),
        "lower":  np.array([0.0,   0.0, 4.0]),
        "upper":  np.array([300.0, 20.0, 9.0]),
        "kaynak": "FAO (2022) [13]",
    },
    "aycicegi": {
        "ideal": np.array([88.0,  4.1, 6.4]),
        "sigma":  np.array([22.0,  1.2, 0.5]),
        "lower":  np.array([0.0,   0.0, 4.0]),
        "upper":  np.array([200.0, 12.0, 9.0]),
        "kaynak": "FAO (2022) [13]",
    },
    "cilek": {
        "ideal": np.array([55.0,  2.8, 5.9]),
        "sigma":  np.array([16.0,  0.9, 0.4]),
        "lower":  np.array([0.0,   0.0, 4.0]),
        "upper":  np.array([150.0, 10.0, 9.0]),
        "kaynak": "USDA-NRCS (2023) [14]",
    },
}

# ------------------------------------------------------------------
# Domates Büyüme Evreleri (FAO [13])
# ------------------------------------------------------------------
TOMATO_GROWTH_STAGES = {
    "fide": {
        "gun_araligi": "0–30",
        "ideal": np.array([45.0,  1.5, 6.0]),
        "sigma":  np.array([12.0,  0.5, 0.3]),
    },
    "vejetatif": {
        "gun_araligi": "30–60",
        "ideal": np.array([75.0,  2.8, 6.1]),
        "sigma":  np.array([18.0,  0.8, 0.4]),
    },
    "ciceklenme": {
        "gun_araligi": "60–90",
        "ideal": np.array([95.0,  3.5, 6.2]),
        "sigma":  np.array([20.0,  1.0, 0.4]),
    },
    "meyve": {
        "gun_araligi": "90–120",
        "ideal": np.array([120.0, 4.5, 6.2]),
        "sigma":  np.array([25.0,  1.5, 0.5]),
    },
    "hasat": {
        "gun_araligi": "120–150",
        "ideal": np.array([85.0,  3.0, 6.3]),
        "sigma":  np.array([20.0,  1.0, 0.4]),
    },
}


@dataclass
class NutritionRecommendation:
    """Bitki besleme optimizasyon sonuç raporu."""
    bitki: str
    evre: Optional[str]
    optimum_n: float        # mg/kg
    optimum_su: float       # L/gün
    optimum_ph: float
    verim_skoru: float      # %
    yakinsama_iter: int
    sure_ms: float
    kaynak: str


class PlantNutritionOptimizer:
    """
    Bitki besleme için yüksek seviye CIADA sarmalayıcısı.

    Kullanım
    --------
    opt = PlantNutritionOptimizer()
    rec = opt.optimize("domates")
    print(rec)

    rec_evre = opt.optimize_growth_stage("meyve")
    """

    def __init__(self, n: int = 20, G: int = 50, alpha: float = 2.0, seed: int = 42):
        self.ciada = CIADAOptimizer(n=n, G=G, alpha=alpha, seed=seed)

    def optimize(self, bitki: str) -> NutritionRecommendation:
        """Belirli bir bitki türü için optimum besleme değerlerini hesaplar."""
        bitki = bitki.lower()
        if bitki not in PLANT_PROFILES:
            raise ValueError(
                f"Bilinmeyen bitki türü: '{bitki}'. "
                f"Desteklenenler: {list(PLANT_PROFILES.keys())}"
            )

        profile = PLANT_PROFILES[bitki]
        result = self.ciada.run(
            lower=profile["lower"],
            upper=profile["upper"],
            ideal=profile["ideal"],
            sigma=profile["sigma"],
            target=99.0,
        )

        return NutritionRecommendation(
            bitki=bitki,
            evre=None,
            optimum_n=float(result.best_x[0]),
            optimum_su=float(result.best_x[1]),
            optimum_ph=float(result.best_x[2]),
            verim_skoru=result.best_fitness,
            yakinsama_iter=result.convergence_iter,
            sure_ms=result.elapsed_ms,
            kaynak=profile["kaynak"],
        )

    def optimize_growth_stage(self, evre: str) -> NutritionRecommendation:
        """Domates bitkisinin belirli bir büyüme evresi için optimizasyon yapar."""
        evre = evre.lower()
        if evre not in TOMATO_GROWTH_STAGES:
            raise ValueError(
                f"Bilinmeyen büyüme evresi: '{evre}'. "
                f"Desteklenenler: {list(TOMATO_GROWTH_STAGES.keys())}"
            )

        stage = TOMATO_GROWTH_STAGES[evre]
        lower = np.array([0.0, 0.0, 4.0])
        upper = np.array([250.0, 15.0, 9.0])

        result = self.ciada.run(
            lower=lower,
            upper=upper,
            ideal=stage["ideal"],
            sigma=stage["sigma"],
            target=99.0,
        )

        return NutritionRecommendation(
            bitki="domates",
            evre=f"{evre} ({stage['gun_araligi']} gün)",
            optimum_n=float(result.best_x[0]),
            optimum_su=float(result.best_x[1]),
            optimum_ph=float(result.best_x[2]),
            verim_skoru=result.best_fitness,
            yakinsama_iter=result.convergence_iter,
            sure_ms=result.elapsed_ms,
            kaynak="FAO (2022) [13]",
        )

    def optimize_all_plants(self) -> list:
        """Tüm desteklenen bitki türleri için toplu optimizasyon çalıştırır."""
        return [self.optimize(bitki) for bitki in PLANT_PROFILES]
