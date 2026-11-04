"""(10) Reporting & Visualization"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # บันทึกรูปลงไฟล์โดยไม่ต้องเปิดหน้าต่าง
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    def comparison_table(self, chrom: np.ndarray) -> pd.DataFrame:
        df = self.data.df
        a, b = df[chrom == 0], df[chrom == 1]

        def row(name, fa, fb):
            return {"Metric": name, "Color A": fa, "Color B": fb}

        rows = [
            row("Headcount", len(a), len(b)),
            row("Male : Female", f"{(a.gender == 'Male').sum()} : {(a.gender == 'Female').sum()}",
                f"{(b.gender == 'Male').sum()} : {(b.gender == 'Female').sum()}"),
            row("Mean weight (kg)", round(a.weight.mean(), 2), round(b.weight.mean(), 2)),
            row("Mean height (cm)", round(a.height.mean(), 2), round(b.height.mean(), 2)),
            row("Mean age (yrs)", round(a.age.mean(), 2), round(b.age.mean(), 2)),
        ]
        table = pd.DataFrame(rows)
        dept = pd.DataFrame({
            "Color A": a.department.value_counts(),
            "Color B": b.department.value_counts(),
        }).fillna(0).astype(int)
        dept.index = [f"Dept {d}" for d in dept.index]
        dept = dept.reset_index().rename(columns={"index": "Metric"})
        return pd.concat([table, dept], ignore_index=True)

    def save_assignment(self, chrom: np.ndarray) -> Path:
        out = self.data.df.copy()
        out["team"] = [self.TEAM_NAMES[int(g)] for g in chrom]
        path = self.out_dir / "team_assignment.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
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
        print(self.comparison_table(result.best_chromosome).to_string(index=False))
        print("\nSaved:", self.plot_convergence(result))
        print("Saved:", self.save_assignment(result.best_chromosome))
