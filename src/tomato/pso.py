"""Particle Swarm Optimization for SVM hyperparameters.

Matches the thesis setup: 20 particles, 100 iterations, inertia weight decreasing
linearly from 0.9 to 0.4, c1 = c2 = 2.0, fitness = mean 5-fold CV accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class PSOResult:
    best_position: np.ndarray
    best_fitness: float
    history: list[float] = field(default_factory=list)  # global best per iteration


def pso_maximize(
    fitness: Callable[[np.ndarray], float],
    lower: np.ndarray,
    upper: np.ndarray,
    n_particles: int = 20,
    n_iter: int = 100,
    w_max: float = 0.9,
    w_min: float = 0.4,
    c1: float = 2.0,
    c2: float = 2.0,
    seed: int | None = 42,
    verbose: bool = False,
) -> PSOResult:
    rng = np.random.default_rng(seed)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    dim = lower.size
    span = upper - lower
    v_max = 0.2 * span  # velocity clamp keeps particles from overshooting the box

    pos = lower + rng.random((n_particles, dim)) * span
    vel = (rng.random((n_particles, dim)) - 0.5) * v_max
    pbest = pos.copy()
    pbest_fit = np.array([fitness(p) for p in pos])
    g = int(np.argmax(pbest_fit))
    gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
    history = [gbest_fit]

    for t in range(n_iter):
        w = w_max - (w_max - w_min) * t / max(n_iter - 1, 1)
        r1, r2 = rng.random((n_particles, dim)), rng.random((n_particles, dim))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -v_max, v_max)
        pos = np.clip(pos + vel, lower, upper)

        fit = np.array([fitness(p) for p in pos])
        improved = fit > pbest_fit
        pbest[improved], pbest_fit[improved] = pos[improved], fit[improved]
        g = int(np.argmax(pbest_fit))
        if pbest_fit[g] > gbest_fit:
            gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
        history.append(gbest_fit)
        if verbose and (t % 10 == 0 or t == n_iter - 1):
            print(f"  iter {t + 1:3d}/{n_iter}  best fitness = {gbest_fit:.4f}")

    return PSOResult(gbest, gbest_fit, history)
