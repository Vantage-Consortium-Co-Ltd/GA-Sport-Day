"""(8) Mutation"""
import numpy as np


class SwapMutation:
    """สลับสีของพนักงาน 2 คนที่อยู่คนละสี -> จำนวนคนในทีมยังคงเท่าเดิม"""

    def __init__(self, rng: np.random.Generator, rate: float = 0.15, max_swaps: int = 3):
        self.rng = rng
        self.rate = rate
        self.max_swaps = max_swaps

    def mutate(self, chrom: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.rate:
            return chrom
        for _ in range(self.rng.integers(1, self.max_swaps + 1)):
            i = self.rng.choice(np.flatnonzero(chrom == 0))
            j = self.rng.choice(np.flatnonzero(chrom == 1))
            chrom[i], chrom[j] = 1, 0
        return chrom
