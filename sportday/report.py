"""(10) Reporting & Visualization"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # บันทึกรูปลงไฟล์โดยไม่ต้องเปิดหน้าต่าง
import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR
from .data import EmployeeDataset
from .fairness import FairnessObjective
from .genetic_algorithm import GAResult


class ReportGenerator:
    TEAM_NAMES = {0: "Color A", 1: "Color B"}

    def __init__(self, data: EmployeeDataset, objective: FairnessObjective, out_dir: Path = OUTPUT_DIR):
        self.data = data
        self.objective = objective
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def plot_convergence(self, result: GAResult) -> Path:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(result.best_history, label="Best cost")
        ax.plot(result.mean_history, label="Mean cost", alpha=0.6)
        ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Discrepancy (lower = fairer)")
        ax.set_title("GA convergence")
        ax.legend()
        ax.grid(alpha=0.3)
        path = self.out_dir / "convergence.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def comparison_table(self, chrom: np.ndarray) -> list[tuple]:
        """คืนรายการ (ชื่อมิติ, ค่าสี A, ค่าสี B)"""
        d = self.data
        teams = (chrom == 0, chrom == 1)

        def stat(fn):
            return [fn(m) for m in teams]

        rows = [
            ("Headcount", *stat(lambda m: m.sum())),
            ("Male", *stat(lambda m: (m & (d.gender == "Male")).sum())),
            ("Female", *stat(lambda m: (m & (d.gender == "Female")).sum())),
            ("Weight_kg", *stat(lambda m: d.raw["weight"][m].mean())),
            ("Height_cm", *stat(lambda m: d.raw["height"][m].mean())),
            ("Age_yrs", *stat(lambda m: d.raw["age"][m].mean())),
        ]
        for dep in d.departments:
            rows.append((f"Dept {dep}", *stat(lambda m: (m & (d.department == dep)).sum())))
        return rows

    def save_assignment(self, chrom: np.ndarray) -> Path:
        path = self.out_dir / "team_assignment.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[*self.data.rows[0], "team"])
            writer.writeheader()
            for row, gene in zip(self.data.rows, chrom):
                writer.writerow({**row, "team": self.TEAM_NAMES[int(gene)]})
        return path

    def report(self, result: GAResult) -> None:
        print("\n" + "=" * 60)
        print(f"Stopped after {result.generations_run} generations: {result.stopped_reason}")
        print(f"Best cost: {result.best_cost:.6f}")
        print("=" * 60)
        print("\nPer-metric discrepancy (0 = perfectly balanced):")
        for k, v in self.objective.breakdown(result.best_chromosome).items():
            print(f"  {k:<11s}: {v[0]:.5f}")
        print("\nFairness comparison:")
        print(f"{'':<12}{'Color A':>10}{'Color B':>10}")
        for name, a, b in self.comparison_table(result.best_chromosome):
            print(f"{name:<12}{a:>10.6g}{b:>10.6g}")
        print("\nSaved:", self.plot_convergence(result))
        print("Saved:", self.save_assignment(result.best_chromosome))
