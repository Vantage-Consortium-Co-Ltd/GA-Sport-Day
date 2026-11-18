from dataclasses import dataclass

import numpy as np


@dataclass
class Settings:
    pop_size: int = 120
    generations: int = 1500
    elite: int = 4              # จำนวนตัวที่ดีที่สุดที่ส่งต่อรุ่นถัดไปตรง ๆ
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3
    max_swaps: int = 4          # mutation สลับคนได้สูงสุดกี่คู่
    patience: int = 300         # หยุดถ้า best ไม่ดีขึ้นติดกันกี่รุ่น
    seed: int = 42


class GeneticAlgorithm:
    def __init__(self, fitness, n, settings=None):
        self.fitness = fitness
        self.n = n
        self.s = settings or Settings()
        self.rng = np.random.default_rng(self.s.seed)
        self.best_log, self.mean_log = [], []

    # ---------- ประชากรเริ่มต้น ----------
    def random_team(self):
        """สุ่มให้ครึ่งหนึ่งเป็นสี B (1) ที่เหลือเป็นสี A (0) -> จำนวนคนเท่ากันตั้งแต่แรก"""
        chrom = np.zeros(self.n, dtype=np.int8)
        chrom[self.rng.choice(self.n, self.n // 2, replace=False)] = 1
        return chrom

    # ---------- Selection: Rank selection ----------
    def select(self, pop, cost):
        """เรียงจากดีสุดไปแย่สุด ตัวอันดับ 1 ได้โอกาสถูกเลือกมากสุด (น้ำหนักตามอันดับ ไม่ใช่ตามค่า cost)
        ข้อดี: ถ้า cost ใกล้กันมาก ตัวที่ดีกว่าเล็กน้อยก็ยังได้เปรียบ และตัวเด่นตัวเดียวไม่ครองประชากรเร็วเกินไป"""
        rank_weight = np.arange(len(pop), 0, -1, dtype=float)
        p = rank_weight / rank_weight.sum()
        order = np.argsort(cost)
        return pop[order[self.rng.choice(len(pop), p=p)]]

    # ---------- Crossover: Uniform + repair ----------
    def crossover(self, a, b):
        """แต่ละคนสุ่มว่าจะเอาสีจากพ่อหรือแม่ (50/50) แล้วซ่อมให้สองทีมมีคนเท่ากัน"""
        if self.rng.random() > self.s.crossover_rate:
            return a.copy()
        child = np.where(self.rng.random(self.n) < 0.5, a, b).astype(np.int8)
        extra = int(child.sum()) - self.n // 2          # >0 = สี B เกิน, <0 = สี A เกิน
        if extra:
            # ซ่อมเฉพาะคนที่พ่อแม่ให้สีไม่ตรงกัน จะได้ไม่ทำลายส่วนที่พ่อแม่เห็นตรงกัน
            side = 1 if extra > 0 else 0
            candidates = np.flatnonzero((child == side) & (a != b))
            if len(candidates) < abs(extra):
                candidates = np.flatnonzero(child == side)
            child[self.rng.choice(candidates, abs(extra), replace=False)] = 1 - side
        return child

    # ---------- Mutation: Swap ----------
    def mutate(self, chrom):
        """สลับสีคน 1 ถึง max_swaps คู่ (คนสี A กับคนสี B) -> จำนวนคนในทีมไม่เปลี่ยน"""
        if self.rng.random() < self.s.mutation_rate:
            for _ in range(self.rng.integers(1, self.s.max_swaps + 1)):
                i = self.rng.choice(np.flatnonzero(chrom == 0))
                j = self.rng.choice(np.flatnonzero(chrom == 1))
                chrom[i], chrom[j] = 1, 0
        return chrom

    # ---------- วนรอบวิวัฒนาการ ----------
    def run(self):
        s = self.s
        pop = np.array([self.random_team() for _ in range(s.pop_size)])
        best, best_cost, stale = None, np.inf, 0

        for gen in range(1, s.generations + 1):
            cost = self.fitness(pop)
            i = int(np.argmin(cost))
            if cost[i] < best_cost - 1e-12:
                best, best_cost, stale = pop[i].copy(), float(cost[i]), 0
            else:
                stale += 1
            self.best_log.append(best_cost)
            self.mean_log.append(float(cost.mean()))

            if gen == 1 or gen % 100 == 0:
                print(f'gen {gen:5d}  best = {best_cost:.6f}  mean = {cost.mean():.6f}')
            if stale >= s.patience:
                print(f'gen {gen:5d}  ไม่ดีขึ้นมา {s.patience} รุ่น -> หยุด')
                break

            # elitism: เก็บตัวดีที่สุดไว้ แล้วสร้างลูกจนเต็มประชากร
            children = [pop[j].copy() for j in np.argsort(cost)[:s.elite]]
            while len(children) < s.pop_size:
                child = self.crossover(self.select(pop, cost), self.select(pop, cost))
                children.append(self.mutate(child))
            pop = np.array(children)

        self.best, self.best_cost = best, best_cost
        return best
