"""
จุดรันหลัก:  python main.py

    sportday/data.py               (1)     EmployeeDataset
    sportday/fairness.py           (2,5)   FairnessObjective
    sportday/operators.py          (6,7,8) Selection / Crossover / Mutation
    sportday/genetic_algorithm.py  (3,4,9) GAConfig, GeneticAlgorithm
    sportday/report.py             (10)    ReportGenerator
"""
from pathlib import Path

from sportday.data import EmployeeDataset
from sportday.fairness import FairnessObjective
from sportday.genetic_algorithm import GAConfig, GeneticAlgorithm
from sportday.report import ReportGenerator

BASE = Path(__file__).resolve().parent


def main():
    data = EmployeeDataset(BASE / "data" / "HR.csv")
    objective = FairnessObjective(data)  # เช่น FairnessObjective(data, {"gender": 2}) = เน้นเพศ 2 เท่า
    result = GeneticAlgorithm(objective, GAConfig()).run()
    ReportGenerator(data, objective, BASE / "output").report(result)


if __name__ == "__main__":
    main()
