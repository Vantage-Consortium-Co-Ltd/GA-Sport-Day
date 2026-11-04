"""
จุดรันหลักของโปรเจกต์ GA Sport Day

    python main.py

โครงสร้าง
    main.py                      <- รันที่นี่ (ประกอบ object แล้วสั่งทำงาน)
    sportday/config.py           path ของข้อมูลและโฟลเดอร์ output
    sportday/data.py             (1)   EmployeeDataset
    sportday/fairness.py         (2,5) FairnessWeights, FairnessObjective
    sportday/population.py       (3,4) PopulationFactory
    sportday/selection.py        (6)   TournamentSelection
    sportday/crossover.py        (7)   OnePointCrossover
    sportday/mutation.py         (8)   SwapMutation
    sportday/genetic_algorithm.py(9)   GAConfig, GeneticAlgorithm
    sportday/report.py           (10)  ReportGenerator
"""
from sportday import (
    EmployeeDataset,
    FairnessObjective,
    FairnessWeights,
    GAConfig,
    GeneticAlgorithm,
    ReportGenerator,
)


def main() -> None:
    # 1) โหลดข้อมูล
    data = EmployeeDataset()

    # 2) กำหนดความแฟร์ (ปรับน้ำหนักแต่ละมิติได้ที่นี่)
    objective = FairnessObjective(data, FairnessWeights())

    # 3) ตั้งค่าและรัน GA (ปรับ pop_size, generations, mutation_rate ฯลฯ ที่นี่)
    ga = GeneticAlgorithm(objective, GAConfig())
    result = ga.run()

    # 4) สรุปผล: กราฟ, ตารางเปรียบเทียบ, CSV รายชื่อ
    ReportGenerator(data, objective).report(result)


if __name__ == "__main__":
    main()
