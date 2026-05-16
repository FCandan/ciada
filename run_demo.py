"""
CIADA Quick Demo
================
Run: python run_demo.py
"""
import sys
sys.path.insert(0, ".")
from src.ciada import CIADAOptimizer, plot_convergence
from src.fitness_functions import gaussian_1d, plant_nutrition_fitness, PLANT_PROFILES

print("=" * 55)
print("  CIADA — Quick Demo")
print("=" * 55)

# 1D nitrogen optimization
print("\n[1] 1D — Nitrogen Optimization (target: 85 mg/kg)")
fn1 = gaussian_1d(ideal=85.0, sigma=25.0)
opt = CIADAOptimizer(n=20, G=50, alpha=2.0, seed=42)
r1  = opt.run(lower=0, upper=150, fitness_fn=fn1)
print(r1)

# 3D tomato optimization
print("\n[2] 3D — Tomato (N, W, pH)")
p   = PLANT_PROFILES["domates"]
fn3 = plant_nutrition_fitness(p["ideal"], p["sigma"])
opt3 = CIADAOptimizer(n=20, G=50, alpha=2.0, seed=42)
r3   = opt3.run(lower=p["lower"], upper=p["upper"], fitness_fn=fn3)
print(r3)
labels = ["N (mg/kg)", "W (L/day)", "pH"]
for i, lbl in enumerate(labels):
    err = abs(r3.best_x[i] - p["ideal"][i])
    print(f"  {lbl}: {r3.best_x[i]:.4f}  (error: {err:.4f})")

# 30 runs statistics
print("\n[3] 30 Independent Runs")
stats = opt.run_multiple(30, lower=0, upper=150, fitness_fn=fn1)
print(f"  Mean: {stats['mean']:.4f}% +/- {stats['std']:.4f}")
print(f"  Min / Max: {stats['min']:.4f} / {stats['max']:.4f}")

plot_convergence(r1.history, title="CIADA — Nitrogen Optimization",
                 save_path="demo_convergence.png")
print("\nDone. See demo_convergence.png")
