"""
CIADA — Crow-Inspired Adaptive Displacement Algorithm
======================================================
Author : Assist. Prof. Dr. Fuat CANDAN
Inst.  : Istanbul Beykent University, Dept. of Computer Engineering
ORCID  : 0000-0003-3166-0493
Paper  : CIADA: Crow-Inspired Adaptive Displacement Algorithm
         for Plant Nutrition Optimization
         Applied Soft Computing (under review)

Formula: DeltaV = (U - L) x exp(-2t/G) x Rand(0,1)   [alpha = 2.0]
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional


class CIADAResult:
    def __init__(self, best_x, best_fitness, convergence_iter, history):
        self.best_x           = best_x
        self.best_fitness     = best_fitness
        self.convergence_iter = convergence_iter
        self.history          = history

    def __str__(self):
        ci = self.convergence_iter if self.convergence_iter >= 0 else "not reached"
        return (
            f"  best_x        : {self.best_x}\n"
            f"  best_fitness  : {self.best_fitness:.4f}%\n"
            f"  convergence   : iteration {ci}\n"
            f"  total_iters   : {len(self.history)}"
        )


class CIADAOptimizer:
    """
    CIADA Optimizer.

    Parameters
    ----------
    n     : Population size (pebbles). Recommended: 20
    G     : Maximum iterations.        Recommended: 50
    alpha : Damping coefficient.       Safe range [1.5, 3.0], default 2.0
    seed  : Random seed (None = random each run)

    Quick start
    -----------
    from src.fitness_functions import gaussian_1d
    fn  = gaussian_1d(ideal=85.0, sigma=25.0)
    opt = CIADAOptimizer(n=20, G=50, alpha=2.0, seed=42)
    r   = opt.run(lower=0, upper=150, fitness_fn=fn)
    print(r)
    """

    def __init__(self, n=20, G=50, alpha=2.0, seed=None):
        self.n, self.G, self.alpha, self.seed = n, G, alpha, seed

    def run(self, lower, upper, fitness_fn, target=99.0):
        if self.seed is not None:
            np.random.seed(self.seed)

        L = np.atleast_1d(np.array(lower, dtype=float))
        U = np.atleast_1d(np.array(upper, dtype=float))
        d = len(L)

        population = np.random.uniform(L, U, (self.n, d))
        best_fit   = -np.inf
        best_x     = population[0].copy()
        history    = []
        conv_iter  = -1

        for t in range(self.G):
            fits = np.array([fitness_fn(population[i]) for i in range(self.n)])
            idx  = np.argmax(fits)
            if fits[idx] > best_fit:
                best_fit = fits[idx]
                best_x   = population[idx].copy()
            history.append(best_fit)

            if best_fit >= target and conv_iter == -1:
                conv_iter = t
                break

            # DeltaV = (U-L) x exp(-alpha*t/G) x Rand(0,1)   [alpha=2.0]
            dv = (U - L) * np.exp(-self.alpha * t / self.G) * np.random.rand(d)

            new_pop = np.empty_like(population)
            for i in range(self.n):
                if fits[i] < best_fit:
                    # Strategic Throw (Exploitation)
                    new_pop[i] = population[i] + dv * (best_x - population[i]) * np.random.rand(d)
                else:
                    # Exploration Mode
                    new_pop[i] = population[i] + np.random.uniform(-1, 1, d) * dv
                # Boundary Control (Clamp)
                new_pop[i] = np.clip(new_pop[i], L, U)
            population = new_pop

        return CIADAResult(best_x, best_fit, conv_iter, history)

    def run_multiple(self, n_runs=30, **kwargs):
        results = [CIADAOptimizer(self.n, self.G, self.alpha, seed=s).run(**kwargs)
                   for s in range(n_runs)]
        fits = np.array([r.best_fitness for r in results])
        return {"results": results, "fits": fits,
                "mean": float(fits.mean()), "std": float(fits.std()),
                "min": float(fits.min()), "max": float(fits.max()),
                "median": float(np.median(fits)),
                "best": results[int(np.argmax(fits))]}


def plot_convergence(history, title="CIADA Convergence", save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(history)+1), history, color="#1F3864", lw=2.5)
    ax.axhline(99, color="gray", ls="--", lw=1, alpha=0.6, label="99% threshold")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Fitness (%)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    path = save_path or "convergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")
