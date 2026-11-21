import csv

import numpy as np

NUMERIC = ['weight', 'height', 'age']


class Employees:
    """อ่าน HR.csv แล้วเตรียมข้อมูลให้พร้อมคำนวณ  (ทุกอย่างเป็น array เรียงตามแถวในไฟล์)"""

    def __init__(self, path):
        with open(path, newline='', encoding='utf-8-sig') as f:
            # HR.csv มีช่องว่างนำหน้าค่า เช่น " R", " 89" จึงต้อง strip
            self.rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
        self.n = len(self.rows)

        self.gender = np.array([r['gender'] for r in self.rows])
        self.department = np.array([r['department'] for r in self.rows])
        self.values = {c: np.array([float(r[c]) for r in self.rows]) for c in NUMERIC}

        # z-score: (ค่า - ค่าเฉลี่ย) / ส่วนเบี่ยงเบนมาตรฐาน -> ทุกมิติมีหน่วยเดียวกัน (จำนวน SD)
        self.z = {c: (v - v.mean()) / v.std() for c, v in self.values.items()}

        self.male = (self.gender == 'Male').astype(float)

        # one-hot ของแผนก: (n, จำนวนแผนก)
        self.dept_names = np.unique(self.department)
        self.dept_onehot = (self.department[:, None] == self.dept_names).astype(float)

        # ช่วงอายุ 3 กลุ่ม แบ่งที่ 1/3 และ 2/3 ของอายุ -> one-hot (n, 3): รุ่นใหม่ / กลาง / อาวุโส
        cut = np.quantile(self.values['age'], [1 / 3, 2 / 3])
        self.age_group = np.digitize(self.values['age'], cut, right=True)
        self.age_onehot = (self.age_group[:, None] == np.arange(3)).astype(float)

    def __len__(self):
        return self.n
