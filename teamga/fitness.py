import numpy as np

# ความสำคัญของแต่ละมิติ (ค่าเริ่มต้นเท่ากันหมด)
WEIGHTS = {'gender': 1, 'weight': 1, 'height': 1, 'age': 1, 'age_group': 1, 'department': 1}


class Fitness:
    """วัดความไม่ยุติธรรม (unfairness) ของการแบ่งทีม: ยิ่งต่ำยิ่งดี, 0 = สองทีมเหมือนกันทุกมิติ

    โครโมโซม = array 0/1 ยาว n (0 = สี A, 1 = สี B) และมี 1 อยู่ครึ่งหนึ่งพอดีเสมอ
    รับเป็นทั้งประชากร (pop, n) ได้ในครั้งเดียว"""

    def __init__(self, emp, weights=None):
        self.emp = emp
        self.w = {**WEIGHTS, **(weights or {})}

    def parts(self, pop):
        """ความต่างรายมิติ -> dict ชื่อมิติ -> array (pop,)"""
        e = self.emp
        B = np.atleast_2d(pop).astype(float)
        A = 1 - B
        nA, nB = A.sum(1), B.sum(1)

        def mean_gap(v):       # |ค่าเฉลี่ยทีม A - ค่าเฉลี่ยทีม B|
            return np.abs(A @ v / nA - B @ v / nB)

        def share_gap(onehot):  # ผลรวม |สัดส่วนในทีม A - สัดส่วนในทีม B| ของทุกกลุ่ม
            return np.abs((A @ onehot) / nA[:, None] - (B @ onehot) / nB[:, None]).sum(1)

        return {
            'gender': mean_gap(e.male),             # ต่างกันกี่ % ของสัดส่วนผู้ชาย
            'weight': mean_gap(e.z['weight']),      # ต่างกันกี่ SD
            'height': mean_gap(e.z['height']),
            'age': mean_gap(e.z['age']),
            'age_group': share_gap(e.age_onehot),
            'department': share_gap(e.dept_onehot),
        }

    def __call__(self, pop):
        p = self.parts(pop)
        return sum(self.w[k] * p[k] for k in self.w)
