"""(7) Crossover"""
import numpy as np


class OnePointCrossover:
    """ตัดที่จุดเดียวแล้วสลับท่อน จากนั้น repair ให้ 0 กับ 1 เท่ากัน"""

    def __init__(self, rng: np.random.Generator, rate: float = 0.9):
        self.rng = rng
        self.rate = rate

    def cross(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.rate:
            return p1.copy()
        point = self.rng.integers(1, len(p1))
        child = np.concatenate([p1[:point], p2[point:]])
        return self._repair(child)

    def _repair(self, child: np.ndarray) -> np.ndarray:
        """ถ้าจำนวน 1 เกิน/ขาดจากครึ่งหนึ่ง สุ่มพลิกยีนฝั่งที่เกินกลับให้สมดุล"""
        target = len(child) // 2
        excess = int(child.sum()) - target
        if excess != 0:
            value = 1 if excess > 0 else 0
            candidates = np.flatnonzero(child == value)
            flip = self.rng.choice(candidates, size=abs(excess), replace=False)
            child[flip] = 1 - value
        return child
