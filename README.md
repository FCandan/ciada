# CIADA: Crow-Inspired Adaptive Displacement Algorithm

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange)](https://www.editorialmanager.com/ASOC)

> **"CIADA: Crow-Inspired Adaptive Displacement Algorithm for Plant Nutrition Optimization"**  
> Assist. Prof. Dr. Fuat CANDAN — Istanbul Beykent University  
> *Applied Soft Computing* (under review)  
> ORCID: [0000-0003-3166-0493](https://orcid.org/0000-0003-3166-0493)

---

## The Algorithm

CIADA is a novel bio-inspired meta-heuristic optimization method grounded in the tool-using behavior of crows (*Corvus spp.*). Its central innovation is the **Volumetric Displacement Operator**:

```
ΔV = (U − L) × e^(−2t/G) × Rand(0,1)   [α = 2.0]
```

This exponential damping mechanism adaptively transitions from global exploration to local exploitation without manual step-size tuning — a structural limitation shared by standard GA and PSO.

---

## Repository Structure

```
ciada/
├── src/
│   ├── ciada.py               # CIADAOptimizer class
│   ├── fitness_functions.py   # Plant nutrition + benchmark functions
│   └── __init__.py
├── figures/
│   ├── fig2_plant_convergence.py    # Fig. 2  — 10 species convergence
│   ├── fig3_tomato_stages.py        # Fig. 3  — Seasonal analysis
│   ├── fig4_alpha_sensitivity.py    # Fig. 4  — Alpha sensitivity
│   ├── fig5_complexity.py           # Fig. 5  — Time complexity
│   ├── fig6_calibration.py          # Fig. 6  — n-G heatmap
│   ├── fig7_parallel.py             # Fig. 7  — Parallel speedup
│   ├── fig8_hybrid.py               # Fig. 8  — Hybrid CIADA+ANN
│   ├── fig9_flowchart.py            # Fig. 9  — Algorithm flowchart
│   ├── fig10_fitness_surface.py     # Fig. 10 — 3D fitness surface
│   ├── fig11_population_dynamics.py # Fig. 11 — Population snapshots
│   └── fig12_benchmark.py           # Fig. 12 — Benchmark analysis
├── notebooks/
│   └── CIADA_Colab.ipynb      # Google Colab notebook (full analysis)
├── data/
│   └── soil_validation.csv    # Turkish soil validation data (8 locations)
├── run_demo.py                # Quick start demo
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/ciada-research/ciada.git
cd ciada
pip install -r requirements.txt
python run_demo.py
```

### Basic Usage

```python
from src.ciada import CIADAOptimizer
from src.fitness_functions import gaussian_1d, plant_nutrition_fitness

# 1D nitrogen optimization
fn  = gaussian_1d(ideal=85.0, sigma=25.0)
opt = CIADAOptimizer(n=20, G=50, alpha=2.0, seed=42)
r   = opt.run(lower=0, upper=150, fitness_fn=fn)
print(r)

# 3D tomato optimization (N, W, pH)
from src.fitness_functions import PLANT_PROFILES
p    = PLANT_PROFILES["domates"]
fn3d = plant_nutrition_fitness(p["ideal"], p["sigma"])
opt3 = CIADAOptimizer(n=20, G=50, alpha=2.0, seed=42)
r3   = opt3.run(lower=p["lower"], upper=p["upper"], fitness_fn=fn3d)
print(r3)

# 30 independent runs
stats = opt.run_multiple(30, lower=0, upper=150, fitness_fn=fn)
print(f"Mean: {stats['mean']:.4f}% +/- {stats['std']:.4f}")
```

---

## Google Colab

Open `notebooks/CIADA_Colab.ipynb` directly in Google Colab — no local installation required:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ciada-research/ciada/blob/main/notebooks/CIADA_Colab.ipynb)

---

## Key Results

| Metric | Value |
|--------|-------|
| Average Fitness (30 runs) | 99.97% ± 0.19 |
| RMSE — Nitrogen (N) | 0.99 mg/kg |
| RMSE — Irrigation (W) | 0.00 L/day |
| RMSE — pH | 0.00 |
| Convergence Threshold (99%) | Iteration 24 |
| Benchmark — Sphere RMSE | 0.000 |
| Benchmark — Rastrigin RMSE | 0.000 |
| Parallel Speedup (8 cores) | 5.4× |
| Hybrid CIADA+ANN Fitness | 99.92% |

---

## Plant Profiles

Ten plant species are pre-configured (FAO 2022, USDA-NRCS 2023):

```python
from src.fitness_functions import PLANT_PROFILES
# Available: domates, misir, marul, bugday, biber,
#            patates, soya, pamuk, aycicegi, cilek
```

---

## Hyperparameter Guide

| Parameter | Minimum | Recommended | Maximum |
|-----------|---------|-------------|---------|
| α (damping) | 1.5 | **2.0** | 3.0 |
| n (population) | 10 | **20** | 30 |
| G (iterations) | 40 | **50** | 75 |

---

## Citation

If you use CIADA in your research, please cite:

```bibtex
@article{candan2025ciada,
  title   = {CIADA: Crow-Inspired Adaptive Displacement Algorithm
             for Plant Nutrition Optimization},
  author  = {Candan, Fuat},
  journal = {Applied Soft Computing},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
