# GA Sport Day

จัดทีมกีฬาสี 2 สี (สี A / สี B) ให้ยุติธรรมที่สุดด้วย **Genetic Algorithm** เขียนแบบ OOP

อ่านข้อมูลพนักงาน 358 คน แล้วแบ่งเป็นสองทีม ทีมละ 179 คน โดยให้ความต่างระหว่างทีมน้อยที่สุดในมิติ
เพศ, น้ำหนัก, ส่วนสูง, อายุ (ค่าเฉลี่ยและสัดส่วนรุ่นใหม่/อาวุโส) และการกระจายตัวของแผนก

## สิ่งที่ต้องมี

- Python 3.10 ขึ้นไป
- แพ็กเกจใน `requirements.txt` (numpy, matplotlib)

```bash
pip install -r requirements.txt
```

## เตรียมข้อมูล

โปรเจกต์นี้**ไม่รวมไฟล์ข้อมูลพนักงาน** (ถูกกันไว้ใน `.gitignore`) ให้วางไฟล์ `HR.csv` ไว้ที่

```
data/HR.csv
```

รูปแบบไฟล์ (มีหัวคอลัมน์ ช่องว่างนำหน้าค่าไม่เป็นไร):

```csv
gender,department,weight,height,age
Male, R, 89, 172, 32
Female, D, 55, 160, 41
```

| คอลัมน์ | ชนิด | ค่า |
|---|---|---|
| `gender` | หมวดหมู่ | `Male` / `Female` |
| `department` | หมวดหมู่ | รหัสแผนก เช่น `A`–`S` |
| `weight` | ตัวเลข | กิโลกรัม |
| `height` | ตัวเลข | เซนติเมตร |
| `age` | ตัวเลข | ปี |

## วิธีรัน

```bash
python main.py
```

ระหว่างรันจะพิมพ์ cost ทุก 50 รุ่น และหยุดเมื่อครบ 1,000 รุ่น หรือเมื่อ cost ไม่ดีขึ้นติดต่อกัน 200 รุ่น
จากนั้นแสดงตารางเปรียบเทียบสี A กับสี B

ผลลัพธ์ถูกบันทึกในโฟลเดอร์ `output/` (สร้างให้อัตโนมัติ)

| ไฟล์ | เนื้อหา |
|---|---|
| `output/convergence.png` | กราฟ cost ที่ลดลงในแต่ละรุ่น |
| `output/team_assignment.csv` | รายชื่อพนักงานพร้อมคอลัมน์ `team` (Color A / Color B) |

## ปรับค่าได้ที่ `main.py`

```python
objective = FairnessObjective(data, {"gender": 2})   # เน้นสมดุลเพศเป็น 2 เท่า (ส่งเฉพาะมิติที่อยากเปลี่ยน)
ga = GeneticAlgorithm(objective, GAConfig(pop_size=200, mutation_rate=0.2))
```

| ค่า | ความหมาย | ค่าเริ่มต้น |
|---|---|---|
| น้ำหนัก `gender / weight / height / age / age_group / department` | ความสำคัญของแต่ละมิติ | 1 |
| น้ำหนัก `size` | โทษถ้าสองทีมคนไม่เท่ากัน | 100 |
| `GAConfig.pop_size` | จำนวนคำตอบในประชากร | 100 |
| `GAConfig.generations` | จำนวนรุ่นสูงสุด | 1000 |
| `GAConfig.elite_size` | จำนวนคำตอบที่เก็บไว้ตรงๆ ทุกรุ่น | 5 |
| `GAConfig.tournament_k` | ขนาดการแข่งขันตอนเลือกพ่อแม่ | 3 |
| `GAConfig.crossover_rate` | โอกาสเกิด crossover | 0.9 |
| `GAConfig.mutation_rate` | โอกาสเกิด mutation | 0.15 |
| `GAConfig.patience` | หยุดถ้าไม่ดีขึ้นกี่รุ่น | 200 |
| `GAConfig.seed` | seed สุ่ม (`None` = สุ่มใหม่ทุกครั้ง) | 42 |

## โครงสร้างโปรเจกต์

```
GA-Sport-Day/
├── main.py                   จุดรันหลัก
├── requirements.txt
├── data/HR.csv               ข้อมูลพนักงาน (ไม่อยู่ใน git)
├── docs/                     เอกสารแนวคิดของ GA
├── output/                   ผลลัพธ์ (ไม่อยู่ใน git)
└── sportday/
    ├── data.py               EmployeeDataset         อ่านข้อมูล + normalize
    ├── fairness.py           FairnessObjective       นิยามความแฟร์ + คำนวณ cost
    ├── operators.py          TournamentSelection / OnePointCrossover / SwapMutation
    ├── genetic_algorithm.py  GAConfig, GeneticAlgorithm  ประชากรเริ่มต้น + วนรอบวิวัฒนาการ
    └── report.py             ReportGenerator         กราฟ ตาราง และไฟล์ CSV
```

## หลักการทำงานโดยย่อ

1. **โครโมโซม** คืออาร์เรย์ 0/1 ยาวเท่าจำนวนพนักงาน (0 = สีที่ 1, 1 = สีที่ 2)
2. **ประชากรเริ่มต้น** สุ่มให้ทุกโครโมโซมมี 0 และ 1 อย่างละครึ่ง
3. **Fitness (cost)** คือผลรวมถ่วงน้ำหนักของความต่างระหว่างสองทีมในแต่ละมิติ บวกโทษหนักถ้าทีมไม่เท่ากัน ยิ่งต่ำยิ่งดี
4. **Selection** เลือกพ่อแม่แบบ Tournament และเก็บตัวที่ดีที่สุดไว้ทุกรุ่น (Elitism)
5. **Crossover** ตัดสลับที่จุดเดียว แล้ว repair ให้จำนวนสองทีมเท่ากัน
6. **Mutation** สลับสีพนักงานสองคนที่อยู่คนละทีม จำนวนคนจึงเท่าเดิมเสมอ
7. วนซ้ำจนครบจำนวนรุ่น หรือ cost หยุดดีขึ้น

รายละเอียดเชิงแนวคิดอยู่ใน[docs/ga_team_balancing_concept.md](docs/ga_team_balancing_concept.md)

## แก้ปัญหา

**Windows + Anaconda: `ImportError: DLL load failed while importing pyexpat`**
การติดตั้ง Anaconda ขาด `libexpat.dll` ซ่อมด้วย

```bash
conda install -n base --force-reinstall libexpat
```
