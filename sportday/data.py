"""(1) อ่านข้อมูลพนักงาน + normalize"""
import csv

import numpy as np


class EmployeeDataset:
    def __init__(self, csv_path):
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            # strip: HR.csv มีช่องว่างนำหน้าค่า (" R", " 89")
            self.rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
        self.n = len(self.rows)

        col = lambda c: np.array([r[c] for r in self.rows])
        self.gender, self.department = col("gender"), col("department")
        self.raw = {c: col(c).astype(float) for c in ("weight", "height", "age")}

        # min-max -> 0..1 เพื่อให้ทุกมิติมีน้ำหนักเทียบกันได้
        self.normalized = {c: (v - v.min()) / (v.max() - v.min()) for c, v in self.raw.items()}

        self.is_male = (self.gender == "Male").astype(float)
        self.dept_matrix = (self.department[:, None] == np.unique(self.department)).astype(float)

        # รุ่นใหม่ / อาวุโส = 1/3 ล่าง / 1/3 บนของอายุ
        lo, hi = np.quantile(self.raw["age"], [1 / 3, 2 / 3])
        self.is_young = (self.raw["age"] <= lo).astype(float)
        self.is_senior = (self.raw["age"] >= hi).astype(float)

    def __len__(self):
        return self.n
