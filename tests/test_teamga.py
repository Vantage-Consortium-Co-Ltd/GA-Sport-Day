"""เทสต์ Employees / Fitness / GeneticAlgorithm ด้วยพนักงานสมมติ (ไม่ใช้ HR.csv จริง)
รัน: pytest"""
import numpy as np
import pytest

from teamga.employees import Employees
from teamga.fitness import Fitness
from teamga.ga import GeneticAlgorithm, Settings

N = 40          # จำนวนพนักงานสมมติ (เลขคู่ จะได้แบ่งครึ่งพอดี)


@pytest.fixture
def emp(tmp_path):
    """สร้าง HR.csv สมมติ มีช่องว่างนำหน้าค่าเหมือนไฟล์จริง เช่น " Male", " 65" """
    rng = np.random.default_rng(0)
    lines = ['id, gender, department, weight, height, age']
    for i in range(N):
        g = 'Male' if i % 3 else 'Female'
        d = ['HR', 'IT', 'Sales', 'Ops'][i % 4]
        lines.append(f'{i}, {g}, {d}, {rng.integers(45, 95)}, {rng.integers(150, 190)}, {rng.integers(22, 60)}')
    path = tmp_path / 'HR.csv'
    path.write_text('\n'.join(lines), encoding='utf-8-sig')
    return Employees(str(path))


def quick_settings(**kw):
    return Settings(**{'pop_size': 40, 'generations': 200, 'patience': 80, **kw})


# ---------- Employees ----------
def test_อ่านไฟล์และตัดช่องว่างนำหน้า(emp):
    assert len(emp) == N
    assert set(emp.gender) == {'Male', 'Female'}
    assert emp.dept_names.tolist() == ['HR', 'IT', 'Ops', 'Sales']
    assert emp.dept_onehot.sum(axis=1).tolist() == [1] * N      # ทุกคนอยู่ 1 แผนกพอดี
    assert emp.age_onehot.sum(axis=1).tolist() == [1] * N


def test_z_score_เฉลี่ย_0_และ_sd_1(emp):
    for v in emp.z.values():
        assert v.mean() == pytest.approx(0, abs=1e-9)
        assert v.std() == pytest.approx(1)


# ---------- Fitness ----------
def test_fitness_เป็น_0_เมื่อสองทีมเหมือนกันทุกอย่าง(tmp_path):
    # คนคู่ 2i กับ 2i+1 ข้อมูลเหมือนกันทุกช่อง -> แยกคู่ไปคนละทีม = ยุติธรรมสมบูรณ์
    lines = ['gender,department,weight,height,age']
    for i in range(10):
        row = f'{"Male" if i % 2 else "Female"},{"IT" if i < 5 else "HR"},{60 + i},{160 + i},{25 + i * 3}'
        lines += [row, row]
    path = tmp_path / 'twins.csv'
    path.write_text('\n'.join(lines), encoding='utf-8')
    e = Employees(str(path))
    team = np.tile([0, 1], 10)
    assert Fitness(e)(team)[0] == pytest.approx(0)


def test_fitness_ทั้งประชากรเท่ากับคิดทีละตัว(emp):
    f = Fitness(emp)
    ga = GeneticAlgorithm(f, len(emp), quick_settings())
    pop = np.array([ga.random_team() for _ in range(5)])
    assert np.allclose(f(pop), [f(p)[0] for p in pop])


def test_fitness_น้ำหนักมิติมีผลจริง(emp):
    team = GeneticAlgorithm(Fitness(emp), N, quick_settings()).random_team()
    base, heavy = Fitness(emp), Fitness(emp, {'gender': 3})
    gap = base.parts(team)['gender'][0]
    assert heavy(team)[0] == pytest.approx(base(team)[0] + 2 * gap)


# ---------- GA operators ----------
def test_ทุก_operator_รักษาจำนวนคนสองทีมให้เท่ากัน(emp):
    ga = GeneticAlgorithm(Fitness(emp), N, quick_settings(mutation_rate=1.0))
    for _ in range(200):
        a, b = ga.random_team(), ga.random_team()
        assert a.sum() == N // 2
        assert ga.crossover(a, b).sum() == N // 2
        assert ga.mutate(a.copy()).sum() == N // 2


def test_crossover_ไม่เปลี่ยนคนที่พ่อแม่ให้สีตรงกัน(emp):
    ga = GeneticAlgorithm(Fitness(emp), N, quick_settings(crossover_rate=1.0))
    for _ in range(50):
        a, b = ga.random_team(), ga.random_team()
        child = ga.crossover(a, b)
        same = a == b
        assert np.array_equal(child[same], a[same])


def test_selection_ตัวดีถูกเลือกบ่อยกว่าตัวแย่(emp):
    ga = GeneticAlgorithm(Fitness(emp), N, quick_settings())
    pop = np.arange(10)[:, None]            # ใช้เลขแทนตัว จะได้นับง่าย
    cost = np.arange(10, dtype=float)       # ตัว 0 ดีที่สุด, ตัว 9 แย่ที่สุด
    picks = [int(ga.select(pop, cost)[0]) for _ in range(3000)]
    assert picks.count(0) > 3 * picks.count(9)


# ---------- run ----------
def test_GA_ยุติธรรมกว่าสุ่มแบ่ง(emp, capsys):
    f = Fitness(emp)
    ga = GeneticAlgorithm(f, N, quick_settings())
    random_cost = np.mean([f(ga.random_team())[0] for _ in range(30)])
    best = ga.run()
    assert best.sum() == N // 2
    assert ga.best_cost == pytest.approx(f(best)[0])
    assert ga.best_cost < random_cost / 2
    assert all(x >= y for x, y in zip(ga.best_log, ga.best_log[1:]))   # best ไม่แย่ลง (elitism)


def test_seed_เดิมได้ทีมเดิม(emp, capsys):
    a = GeneticAlgorithm(Fitness(emp), N, quick_settings()).run()
    b = GeneticAlgorithm(Fitness(emp), N, quick_settings()).run()
    assert np.array_equal(a, b)
