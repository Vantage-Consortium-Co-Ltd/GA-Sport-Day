"""(1) Data ingestion: อ่านข้อมูลพนักงาน + normalize (ใช้ csv + numpy ไม่ใช้ pandas)"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import DEFAULT_CSV


class EmployeeDataset:
    """อ่านข้อมูลพนักงาน แยกประเภทข้อมูล และ normalize ตัวเลขให้อยู่ช่วง 0-1"""

    NUMERIC_COLS = ["weight", "height", "age"]
    CATEGORICAL_COLS = ["gender", "department"]

    def __init__(self, csv_path: Path | str = DEFAULT_CSV):
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            # strip: ใน HR.csv มีช่องว่างนำหน้าค่า (" R", " 89")
            self.rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
        self.n = len(self.rows)

        self.gender = np.array([r["gender"] for r in self.rows])
        self.department = np.array([r["department"] for r in self.rows])
        self.raw = {c: np.array([r[c] for r in self.rows], dtype=float) for c in self.NUMERIC_COLS}

        # Min-max normalization -> ค่า 0..1 เพื่อให้ทุกมิติมีน้ำหนักเทียบกันได้
        self.normalized = {c: (v - v.min()) / (v.max() - v.min()) for c, v in self.raw.items()}

        # เตรียมเมทริกซ์ไว้ใช้คำนวณแบบ vectorized
        self.is_male = (self.gender == "Male").astype(float)
        self.departments = sorted(set(self.department))
        self.dept_matrix = (self.department[:, None] == np.array(self.departments)).astype(float)

        # คนรุ่นใหม่ / ผู้อาวุโส = 1/3 ล่าง และ 1/3 บนของอายุ
        age = self.raw["age"]
        lo, hi = np.quantile(age, [1 / 3, 2 / 3])
        self.is_young = (age <= lo).astype(float)
        self.is_senior = (age >= hi).astype(float)

    def __len__(self) -> int:
        return self.n
