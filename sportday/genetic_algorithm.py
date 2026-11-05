"""(3, 4, 9) ประชากร 0/1 (0 = สีที่ 1, 1 = สีที่ 2) + วนรอบวิวัฒนาการ"""
from dataclasses import dataclass, field

import numpy as np

from .operators import OnePointCrossover, SwapMutation, TournamentSelection


@dataclass
class GAConfig:
    pop_size: int = 100
    generations: int = 1000
    elite_size: int = 5
    tournament_k: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.15
    patience: int = 200  # หยุดถ้า best cost ไม่ดีขึ้นกี่รุ่น
    seed: int | None = 42


@dataclass
class GAResult:
    best: np.ndarray = None
    best_cost: float = np.inf
    best_history: list = field(default_factory=list)
    mean_history: list = field(default_factory=list)


class GeneticAlgorithm:
    def __init__(self, objective, config=None):
        self.objective = objective
        self.cfg = cfg = config or GAConfig()
        self.rng = np.random.default_rng(cfg.seed)
        self.selection = TournamentSelection(self.rng, cfg.tournament_k)
        self.crossover = OnePointCrossover(self.rng, cfg.crossover_rate)
        self.mutation = SwapMutation(self.rng, cfg.mutation_rate)

    def _initial_population(self):
        n = len(self.objective.data)
        base = np.zeros(n, dtype=np.int8)
        base[: n // 2] = 1  # 179 ต่อ 179
        return np.array([self.rng.permutation(base) for _ in range(self.cfg.pop_size)])

    def run(self):
        cfg, result = self.cfg, GAResult()
        pop, stale = self._initial_population(), 0

        for gen in range(1, cfg.generations + 1):
            cost = self.objective.cost(pop)
            order = np.argsort(cost)
            pop, cost = pop[order], cost[order]

            if result.best_cost - cost[0] > 1e-9:
                result.best, result.best_cost, stale = pop[0].copy(), float(cost[0]), 0
            else:
                stale += 1
            result.best_history.append(float(cost[0]))
            result.mean_history.append(float(cost.mean()))
            if gen == 1 or gen % 50 == 0:
                print(f"gen {gen:4d} | best cost = {cost[0]:.5f} | mean = {cost.mean():.5f}")
            if stale >= cfg.patience:
                break

            nxt = [pop[i].copy() for i in range(cfg.elite_size)]  # elitism
            while len(nxt) < cfg.pop_size:
                child = self.crossover.cross(self.selection.select(pop, cost), self.selection.select(pop, cost))
                nxt.append(self.mutation.mutate(child))
            pop = np.array(nxt)

        return result
