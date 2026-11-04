"""(1) Data ingestion: อ่านข้อมูลพนักงาน + normalize"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CSV


class EmployeeDataset:
    """อ่านข้อมูลพนักงาน แยกประเภทข้อมูล และ normalize ตัวเลขให้อยู่ช่วง 0-1"""

    NUMERIC_COLS = ["weight", "height", "age"]
    CATEGORICAL_COLS = ["gender", "department"]

    def __init__(self, csv_path: Path | str = DEFAULT_CSV):
        # skipinitialspace: ใน HR.csv มีช่องว่างนำหน้าค่า (" R", " 89")
        self.df = pd.read_csv(csv_path, skipinitialspace=True)
        for col in self.CATEGORICAL_COLS:
            self.df[col] = self.df[col].astype(str).str.strip()
        self.n = len(self.df)

        # Min-max normalization -> ค่า 0..1 เพื่อให้ทุกมิติมีน้ำหนักเทียบกันได้
        numeric = self.df[self.NUMERIC_COLS].astype(float)
        self.normalized = (numeric - numeric.min()) / (numeric.max() - numeric.min())

        # เตรียมเมทริกซ์ไว้ใช้คำนวณแบบ vectorized
        self.is_male = (self.df["gender"] == "Male").to_numpy(dtype=float)
        self.departments = sorted(self.df["department"].unique())
        self.dept_matrix = pd.get_dummies(self.df["department"])[self.departments].to_numpy(dtype=float)

        # คนรุ่นใหม่ / ผู้อาวุโส = 1/3 ล่าง และ 1/3 บนของอายุ
        age = self.df["age"].to_numpy(dtype=float)
        lo, hi = np.quantile(age, [1 / 3, 2 / 3])
        self.is_young = (age <= lo).astype(float)
        self.is_senior = (age >= hi).astype(float)

    def __len__(self) -> int:
        return self.n
