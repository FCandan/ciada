"""
CIADA — Crow-Inspired Adaptive Displacement Algorithm
======================================================

Karga İlhamlı Adaptif Yer Değiştirme Algoritması
Bitki Besleme Optimizasyonu için Meta-Sezgisel Yaklaşım

Hızlı başlangıç:
    >>> from ciada import CIADAOptimizer, PlantNutritionOptimizer
    >>> opt = PlantNutritionOptimizer()
    >>> result = opt.optimize("domates")
    >>> print(f"Optimum N: {result.optimum_n:.2f} mg/kg, Verim: {result.verim_skoru:.2f}%")
"""

__version__ = "1.0.0"
__author__  = "CIADA Research Team"
__license__ = "MIT"

from .optimizer import CIADAOptimizer, CIADAResult
from .plant_nutrition import PlantNutritionOptimizer, NutritionRecommendation, PLANT_PROFILES

__all__ = [
    "CIADAOptimizer",
    "CIADAResult",
    "PlantNutritionOptimizer",
    "NutritionRecommendation",
    "PLANT_PROFILES",
]
