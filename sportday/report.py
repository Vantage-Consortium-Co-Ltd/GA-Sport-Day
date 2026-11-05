"""(10) กราฟ ตารางเปรียบเทียบ และไฟล์ผลลัพธ์"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # บันทึกรูปลงไฟล์ ไม่เปิดหน้าต่าง
import matplotlib.pyplot as plt


class ReportGenerator:
    def __init__(self, data, objective, out_dir):
        self.data, self.objective = data, objective
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

    def comparison_table(self, chrom):
        """คืน [(ชื่อมิติ, ค่าสี A, ค่าสี B)]"""
        d = self.data
        teams = (chrom == 0, chrom == 1)
        stat = lambda f: [f(m) for m in teams]
        rows = [
            ("Headcount", *stat(lambda m: m.sum())),
            ("Male", *stat(lambda m: (m & (d.gender == "Male")).sum())),
            ("Female", *stat(lambda m: (m & (d.gender == "Female")).sum())),
            ("Weight_kg", *stat(lambda m: d.raw["weight"][m].mean())),
            ("Height_cm", *stat(lambda m: d.raw["height"][m].mean())),
            ("Age_yrs", *stat(lambda m: d.raw["age"][m].mean())),
        ]
        for dep in sorted(set(d.department)):
            rows.append((f"Dept {dep}", *stat(lambda m: (m & (d.department == dep)).sum())))
        return rows

    def report(self, result):
        chrom = result.best
        print(f"\nStopped after {len(result.best_history)} generations | best cost = {result.best_cost:.6f}")
        print("\nPer-metric discrepancy (0 = perfectly balanced):")
        for k, v in self.objective.breakdown(chrom).items():
            print(f"  {k:<11s}: {v[0]:.5f}")
        print("\nFairness comparison:")
        print(f"{'':<12}{'Color A':>10}{'Color B':>10}")
        for name, a, b in self.comparison_table(chrom):
            print(f"{name:<12}{a:>10.6g}{b:>10.6g}")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(result.best_history, label="Best cost")
        ax.plot(result.mean_history, label="Mean cost", alpha=0.6)
        ax.set(yscale="log", xlabel="Generation", ylabel="Discrepancy (lower = fairer)", title="GA convergence")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.out_dir / "convergence.png", dpi=130)
        plt.close(fig)

        with open(self.out_dir / "team_assignment.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=[*self.data.rows[0], "team"])
            w.writeheader()
            w.writerows({**r, "team": "Color B" if g else "Color A"} for r, g in zip(self.data.rows, chrom))
        print(f"\nSaved to {self.out_dir}")
