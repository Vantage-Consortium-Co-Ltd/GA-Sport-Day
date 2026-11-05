"""(2, 5) นิยามความยุติธรรม: cost = Σ น้ำหนัก * |ความต่างสี 1 vs สี 2| (ยิ่งต่ำยิ่งดี)"""
import numpy as np

DEFAULT_WEIGHTS = {
    "gender": 1, "weight": 1, "height": 1, "age": 1, "age_group": 1, "department": 1,
    "size": 100,  # โทษหนักถ้าสองทีมคนไม่เท่ากัน
}


class FairnessObjective:
    def __init__(self, data, weights=None):
        self.data = data
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}  # ส่งมาเฉพาะมิติที่อยากเปลี่ยน

    def breakdown(self, population):
        """ความต่างรายมิติของทุกโครโมโซม -> dict ชื่อมิติ -> array ขนาด pop"""
        d = self.data
        team2 = np.atleast_2d(population).astype(float)  # 1 = สีที่ 2
        team1 = 1.0 - team2                              # 0 = สีที่ 1

        def diff_mean(v):
            return np.abs(team1 @ v / np.maximum(team1.sum(1), 1) - team2 @ v / np.maximum(team2.sum(1), 1))

        return {
            "gender": diff_mean(d.is_male),
            "weight": diff_mean(d.normalized["weight"]),
            "height": diff_mean(d.normalized["height"]),
            "age": diff_mean(d.normalized["age"]),
            "age_group": diff_mean(d.is_young) + diff_mean(d.is_senior),
            "department": np.abs(team1 @ d.dept_matrix - team2 @ d.dept_matrix).sum(1) / d.n,
            "size": np.abs(team1.sum(1) - team2.sum(1)),
        }

    def cost(self, population):
        b = self.breakdown(population)
        return sum(self.w[k] * b[k] for k in self.w)
