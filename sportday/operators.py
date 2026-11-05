"""(6, 7, 8) Selection / Crossover / Mutation"""
import numpy as np


class TournamentSelection:
    """สุ่มมา k คน เลือกคนที่ cost ต่ำสุดเป็นพ่อ/แม่"""

    def __init__(self, rng, k=3):
        self.rng, self.k = rng, k

    def select(self, population, cost):
        idx = self.rng.integers(0, len(population), size=self.k)
        return population[idx[np.argmin(cost[idx])]]


class OnePointCrossover:
    """ตัดจุดเดียวแล้วสลับท่อน จากนั้น repair ให้ 0 กับ 1 เท่ากัน"""

    def __init__(self, rng, rate=0.9):
        self.rng, self.rate = rng, rate

    def cross(self, p1, p2):
        if self.rng.random() > self.rate:
            return p1.copy()
        point = self.rng.integers(1, len(p1))
        child = np.concatenate([p1[:point], p2[point:]])
        excess = int(child.sum()) - len(child) // 2
        if excess:  # ยีนฝั่งที่เกินถูกสุ่มพลิกกลับให้สมดุล
            value = 1 if excess > 0 else 0
            flip = self.rng.choice(np.flatnonzero(child == value), size=abs(excess), replace=False)
            child[flip] = 1 - value
        return child


class SwapMutation:
    """สลับสีพนักงาน 2 คนที่อยู่คนละสี -> จำนวนคนในทีมเท่าเดิมเสมอ"""

    def __init__(self, rng, rate=0.15, max_swaps=3):
        self.rng, self.rate, self.max_swaps = rng, rate, max_swaps

    def mutate(self, chrom):
        if self.rng.random() > self.rate:
            return chrom
        for _ in range(self.rng.integers(1, self.max_swaps + 1)):
            i = self.rng.choice(np.flatnonzero(chrom == 0))
            j = self.rng.choice(np.flatnonzero(chrom == 1))
            chrom[i], chrom[j] = 1, 0
        return chrom
