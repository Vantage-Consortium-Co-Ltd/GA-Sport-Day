"""(6) Selection"""
import numpy as np


class TournamentSelection:
    """สุ่มมา k คน เลือกคนที่ cost ต่ำสุดเป็นพ่อ/แม่"""

    def __init__(self, rng: np.random.Generator, k: int = 3):
        self.rng = rng
        self.k = k

    def select(self, population: np.ndarray, cost: np.ndarray) -> np.ndarray:
        idx = self.rng.integers(0, len(population), size=self.k)
        return population[idx[np.argmin(cost[idx])]]
