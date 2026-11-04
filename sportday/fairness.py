"""(2, 5) นิยามความยุติธรรมและการคำนวณ fitness (cost ยิ่งต่ำยิ่งดี)"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import EmployeeDataset


@dataclass
class FairnessWeights:
    """น้ำหนักของแต่ละมิติ ปรับได้ตามที่อยากเน้นความแฟร์"""
    gender: float = 1.0
    weight: float = 1.0
    height: float = 1.0
    age: float = 1.0
    age_group: float = 1.0
    department: float = 1.0
    size_penalty: float = 100.0  # โทษหนักถ้าสองทีมคนไม่เท่ากัน


class FairnessObjective:
    """
    คำนวณ cost ของโครโมโซม (ยิ่งน้อยยิ่งดี = fitness สูง)
    cost = Σ weight_k * |metric_k(สี 1) - metric_k(สี 2)| + penalty * |ขนาดทีมต่าง|
    """

    METRIC_NAMES = ["gender", "weight", "height", "age", "age_group", "department"]

    def __init__(self, data: EmployeeDataset, weights: FairnessWeights | None = None):
        self.data = data
        self.w = weights or FairnessWeights()

    @staticmethod
    def _group_means(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
        """ค่าเฉลี่ยของ values ในกลุ่ม mask  (mask: pop x n, values: n) -> pop"""
        count = np.maximum(mask.sum(axis=1), 1)
        return (mask @ values) / count

    def breakdown(self, population: np.ndarray) -> dict[str, np.ndarray]:
        """ความต่างรายมิติของทุกโครโมโซม (คืน dict ชื่อมิติ -> array ขนาด pop)"""
        d = self.data
        pop = np.atleast_2d(population)
        team2 = pop.astype(float)  # 1 = สีที่ 2
        team1 = 1.0 - team2        # 0 = สีที่ 1

        def diff_mean(values):
            return np.abs(self._group_means(team1, values) - self._group_means(team2, values))

        out = {
            "gender": diff_mean(d.is_male),  # สัดส่วนชายในแต่ละทีมต่างกันเท่าไร
            "weight": diff_mean(d.normalized["weight"]),
            "height": diff_mean(d.normalized["height"]),
            "age": diff_mean(d.normalized["age"]),
            "age_group": diff_mean(d.is_young) + diff_mean(d.is_senior),
        }
        # แผนก: ผลรวมความต่าง (จำนวนคนต่อแผนก) / จำนวนคนทั้งหมด
        dept1 = team1 @ d.dept_matrix
        dept2 = team2 @ d.dept_matrix
        out["department"] = np.abs(dept1 - dept2).sum(axis=1) / d.n
        out["size"] = np.abs(team1.sum(axis=1) - team2.sum(axis=1))
        return out

    def cost(self, population: np.ndarray) -> np.ndarray:
        b = self.breakdown(population)
        total = sum(getattr(self.w, k) * b[k] for k in self.METRIC_NAMES)
        return total + self.w.size_penalty * b["size"]
