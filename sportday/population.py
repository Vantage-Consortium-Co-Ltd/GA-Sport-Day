"""(3, 4) โครโมโซมและการสร้างประชากรเริ่มต้น"""
import numpy as np


class PopulationFactory:
    """
    โครโมโซม = array 0/1 ยาว n  (0 = สีที่ 1, 1 = สีที่ 2)
    ประชากรทั้งหมด = เมทริกซ์ (pop_size x n)
    """

    def __init__(self, n_genes: int, rng: np.random.Generator):
        self.n = n_genes
        self.rng = rng

    def create(self, size: int) -> np.ndarray:
        base = np.zeros(self.n, dtype=np.int8)
        base[: self.n // 2] = 1  # 179 ต่อ 179
        return np.array([self.rng.permutation(base) for _ in range(size)])
