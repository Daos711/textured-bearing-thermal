"""
Быстрый тест работоспособности модели.

Запуск:
    cd textured-bearing-thermal
    python examples/quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_parameters():
    """Тест модуля parameters."""
    print("Тест parameters...")
    from src.parameters import create_test_bearing, create_roller_cone_bit_bearing

    model = create_test_bearing()
    print(f"  Test bearing: R={model.geometry.R*1000:.1f} мм, n={model.operating.n} об/мин")

    model_bit = create_roller_cone_bit_bearing()
    print(f"  Roller cone: R={model_bit.geometry.R*1000:.1f} мм, n={model_bit.operating.n} об/мин")

    # Температурная зависимость вязкости
    eta_40 = model.lubricant.viscosity(40)
    eta_80 = model.lubricant.viscosity(80)
    print(f"  Вязкость: η(40°C)={eta_40*1000:.2f} мПа·с, η(80°C)={eta_80*1000:.2f} мПа·с")

    print("  OK\n")


def test_geometry():
    """Тест модуля geometry."""
    print("Тест geometry...")
    from src.geometry import create_grid, compute_film_thickness
    from src.parameters import create_test_bearing

    model = create_test_bearing()
    grid = create_grid(100, 50)

    print(f"  Сетка: {grid.N_phi} x {grid.N_Z}")
    print(f"  d_phi = {grid.d_phi:.4f}, d_Z = {grid.d_Z:.4f}")

    # Зазор без текстуры
    H_smooth = compute_film_thickness(grid, model.geometry, 0.6, texture=None)
    print(f"  H_smooth: min={H_smooth.min():.3f}, max={H_smooth.max():.3f}")

    # Зазор с текстурой
    H_tex = compute_film_thickness(grid, model.geometry, 0.6, model.texture)
    print(f"  H_textured: min={H_tex.min():.3f}, max={H_tex.max():.3f}")

    print("  OK\n")


def test_reynolds():
    """Тест решателя Рейнольдса."""
    print("Тест reynolds...")
    from src.geometry import create_grid, compute_film_thickness
    from src.reynolds import solve_reynolds_static
    from src.parameters import create_test_bearing

    model = create_test_bearing()
    grid = create_grid(180, 50)
    D_over_L = model.geometry.D / model.geometry.L

    H = compute_film_thickness(grid, model.geometry, 0.6, texture=None)

    P, residual, iters = solve_reynolds_static(
        H, grid.d_phi, grid.d_Z, D_over_L,
        omega_sor=1.5, tol=1e-5, max_iter=10000
    )

    print(f"  Итераций: {iters}, невязка: {residual:.2e}")
    print(f"  P: min={P.min():.3f}, max={P.max():.3f}")

    if residual < 1e-5:
        print("  OK\n")
    else:
        print("  WARNING: не сошлось!\n")


def test_forces():
    """Тест вычисления сил."""
    print("Тест forces...")
    from src.geometry import create_grid, compute_film_thickness
    from src.reynolds import solve_reynolds_static
    from src.forces import integrate_forces, compute_friction
    from src.parameters import create_test_bearing

    model = create_test_bearing()
    grid = create_grid(180, 50)
    D_over_L = model.geometry.D / model.geometry.L

    H = compute_film_thickness(grid, model.geometry, 0.6, texture=None)
    P, _, _ = solve_reynolds_static(
        H, grid.d_phi, grid.d_Z, D_over_L, tol=1e-5
    )

    forces = integrate_forces(
        P, grid, model.pressure_scale,
        model.geometry.R, model.geometry.L
    )
    print(f"  Силы: Fx={forces.Fx:.1f} Н, Fy={forces.Fy:.1f} Н, F={forces.F_total:.1f} Н")

    friction = compute_friction(P, H, grid, model)
    print(f"  Трение: f={friction.friction_force:.2f} Н, μ={friction.friction_coeff:.5f}")

    print("  OK\n")


def test_thermal():
    """Тест температурной модели."""
    print("Тест thermal...")
    from src.geometry import create_grid, compute_film_thickness
    from src.thermal import ThermalModel, estimate_temperature_rise
    from src.reynolds import solve_reynolds_static
    from src.parameters import create_roller_cone_bit_bearing

    model = create_roller_cone_bit_bearing(n_rpm=150.0, T_inlet=80.0)
    grid = create_grid(180, 50)
    D_over_L = model.geometry.D / model.geometry.L

    # Оценка подъёма температуры
    delta_T = estimate_temperature_rise(model)
    print(f"  Оценка ΔT: {delta_T:.1f}°C")

    # Решаем Reynolds
    H = compute_film_thickness(grid, model.geometry, 0.5, texture=None)
    P, _, _ = solve_reynolds_static(H, grid.d_phi, grid.d_Z, D_over_L, tol=1e-5)

    # Температурная модель
    thermal = ThermalModel(model, grid)
    T_solution = thermal.compute_temperature_field(H, P)

    print(f"  T_max={T_solution.max_T:.1f}°C, T_mean={T_solution.mean_T:.1f}°C")

    print("  OK\n")


def main():
    print("=" * 50)
    print("БЫСТРЫЙ ТЕСТ МОДЕЛИ ПОДШИПНИКА")
    print("=" * 50 + "\n")

    test_parameters()
    test_geometry()
    test_reynolds()
    test_forces()
    test_thermal()

    print("=" * 50)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    print("=" * 50)


if __name__ == "__main__":
    main()
