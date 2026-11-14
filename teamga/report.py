import csv
import os

import matplotlib

matplotlib.use('Agg')       # บันทึกรูปลงไฟล์ ไม่เปิดหน้าต่าง
import matplotlib.pyplot as plt
import numpy as np


class Report:
    def __init__(self, emp, fitness, out_dir):
        self.emp, self.fitness, self.out_dir = emp, fitness, out_dir
        os.makedirs(out_dir, exist_ok=True)

    def table(self, team):
        """[(หัวข้อ, ค่าสี A, ค่าสี B)] ไว้เทียบสองทีม"""
        e = self.emp
        A, B = team == 0, team == 1
        rows = [('จำนวนคน', A.sum(), B.sum()),
                ('ชาย', (A & (e.gender == 'Male')).sum(), (B & (e.gender == 'Male')).sum()),
                ('หญิง', (A & (e.gender == 'Female')).sum(), (B & (e.gender == 'Female')).sum())]
        for c, unit in [('weight', 'kg'), ('height', 'cm'), ('age', 'ปี')]:
            v = e.values[c]
            rows.append((f'{c} เฉลี่ย ({unit})', round(v[A].mean(), 2), round(v[B].mean(), 2)))
        for g, name in enumerate(['รุ่นใหม่', 'วัยกลาง', 'อาวุโส']):
            rows.append((f'อายุ: {name}', (A & (e.age_group == g)).sum(), (B & (e.age_group == g)).sum()))
        for d in e.dept_names:
            rows.append((f'แผนก {d}', (A & (e.department == d)).sum(), (B & (e.department == d)).sum()))
        return rows

    def show(self, ga, random_team):
        team = ga.best
        print(f'\nunfairness: สุ่มแบ่ง = {float(self.fitness(random_team)[0]):.4f}  ->  GA = {ga.best_cost:.4f}')
        print('\nความต่างรายมิติ (0 = เท่ากันพอดี)')
        before, after = self.fitness.parts(random_team), self.fitness.parts(team)
        for k in after:
            print(f'  {k:<11} สุ่ม {before[k][0]:.4f}  ->  GA {after[k][0]:.4f}')

        print(f'\n{"":<22}{"สี A":>8}{"สี B":>8}')
        for name, a, b in self.table(team):
            print(f'{name:<22}{a:>8}{b:>8}')

        # กราฟ convergence
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(ga.best_log, label='best')
        ax.plot(ga.mean_log, label='mean', alpha=0.6)
        ax.set(yscale='log', xlabel='generation', ylabel='unfairness (lower = fairer)', title='GA convergence')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'convergence.png'), dpi=130)
        plt.close(fig)

        # รายชื่อพร้อมสีที่ได้
        with open(os.path.join(self.out_dir, 'team_assignment.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=[*self.emp.rows[0], 'team'])
            w.writeheader()
            for r, t in zip(self.emp.rows, team):
                w.writerow({**r, 'team': 'B' if t else 'A'})
        print(f'\nบันทึกไว้ที่ {self.out_dir}/convergence.png และ team_assignment.csv')
