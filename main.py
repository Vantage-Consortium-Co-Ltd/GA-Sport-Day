"""แบ่งพนักงานเป็นทีมกีฬาสี 2 สีให้ยุติธรรมที่สุดด้วย Genetic Algorithm: python main.py"""
import os

from teamga.employees import Employees
from teamga.fitness import Fitness
from teamga.ga import GeneticAlgorithm, Settings
from teamga.report import Report

DATA = 'data/HR.csv'
OUTPUT = 'output'

os.chdir(os.path.dirname(os.path.abspath(__file__)))
emp = Employees(DATA)
print(f'พนักงาน {len(emp)} คน, {len(emp.dept_names)} แผนก')

fitness = Fitness(emp)                  # เช่น Fitness(emp, {'gender': 3}) = ให้ความสำคัญกับเพศ 3 เท่า
ga = GeneticAlgorithm(fitness, len(emp), Settings())
random_team = ga.random_team()          # ไว้เทียบว่า GA ดีกว่าการสุ่มแบ่งแค่ไหน
ga.run()

Report(emp, fitness, OUTPUT).show(ga, random_team)
