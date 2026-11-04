"""(9) Evolutionary loop + stopping criteria"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crossover import OnePointCrossover
from .fairness import FairnessObjective
from .mutation import SwapMutation
from .population import PopulationFactory
from .selection import TournamentSelection


@dataclass
class GAConfig:
    pop_size: int = 100
    generations: int = 1000
    elite_size: int = 5
    tournament_k: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.15
    patience: int = 200        # หยุดถ้า best cost ไม่ดีขึ้นเกินกี่รุ่น
    min_improvement: float = 1e-9
    seed: int | None = 42


@dataclass
class GAResult:
    best_chromosome: np.ndarray
    best_cost: float
    best_history: list[float] = field(default_factory=list)
    mean_history: list[float] = field(default_factory=list)
    generations_run: int = 0
    stopped_reason: str = ""


class GeneticAlgorithm:
    def __init__(self, objective: FairnessObjective, config: GAConfig | None = None):
        self.objective = objective
        self.cfg = config or GAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        n = len(objective.data)
        self.factory = PopulationFactory(n, self.rng)
        self.selection = TournamentSelection(self.rng, self.cfg.tournament_k)
        self.crossover = OnePointCrossover(self.rng, self.cfg.crossover_rate)
        self.mutation = SwapMutation(self.rng, self.cfg.mutation_rate)

    def run(self, verbose: bool = True) -> GAResult:
        cfg = self.cfg
        pop = self.factory.create(cfg.pop_size)
        result = GAResult(best_chromosome=pop[0].copy(), best_cost=np.inf)
        stale = 0

        for gen in range(1, cfg.generations + 1):
            cost = self.objective.cost(pop)
            order = np.argsort(cost)
            pop, cost = pop[order], cost[order]

            # บันทึกสถิติ + ตรวจ early stopping
            if result.best_cost - cost[0] > cfg.min_improvement:
                result.best_cost = float(cost[0])
                result.best_chromosome = pop[0].copy()
                stale = 0
            else:
                stale += 1
            result.best_history.append(float(cost[0]))
            result.mean_history.append(float(cost.mean()))
            result.generations_run = gen

            if verbose and (gen == 1 or gen % 50 == 0):
                print(f"gen {gen:4d} | best cost = {cost[0]:.5f} | mean = {cost.mean():.5f}")

            if stale >= cfg.patience:
                result.stopped_reason = f"converged (no improvement for {cfg.patience} generations)"
                break

            # Elitism: เก็บตัวเก่งสุดไว้ตรงๆ
            next_pop = [pop[i].copy() for i in range(cfg.elite_size)]
            # สร้างลูก
            while len(next_pop) < cfg.pop_size:
                p1 = self.selection.select(pop, cost)
                p2 = self.selection.select(pop, cost)
                child = self.crossover.cross(p1, p2)
                next_pop.append(self.mutation.mutate(child))
            pop = np.array(next_pop)
        else:
            result.stopped_reason = f"reached max generations ({cfg.generations})"

        return result
