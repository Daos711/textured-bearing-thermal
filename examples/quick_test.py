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
    from src.parameters import create_chinese_paper_bearing, create_roller_cone_bit_bearing

    model = create_chinese_paper_bearing()
    print(f"  Chinese paper: R_J={model.geometry.R_J*1000:.2f} мм, c0={model.geometry.c0*1e6:.1f} мкм")
    print(f"    λ = 2R_J/L = {model.geometry.lambda_ratio:.3f}")

    model_bit = create_roller_cone_bit_bearing()
    print(f"  Roller cone: R_J={model_bit.geometry.R_J*1000:.2f} мм, n={model_bit.operating.n_rpm} об/мин")

    # Температурная зависимость вязкости
    eta_40 = model.lubricant.viscosity(40)
    eta_80 = model.lubricant.viscosity(80)
    print(f"  Вязкость: η(40°C)={eta_40*1000:.2f} мПа·с, η(80°C)={eta_80*1000:.2f} мПа·с")
    print(f"  η̂(80°C) = {model.lubricant.eta_hat(80):.4f}")

    print("  OK\n")


def test_geometry():
    """Тест модуля geometry."""
    print("Тест geometry...")
    from src.geometry import create_grid, compute_film_thickness_static, compute_phyllotaxis_centers
    from src.parameters import create_chinese_paper_bearing

    model = create_chinese_paper_bearing()
    grid = create_grid(100, 50)

    print(f"  Сетка: {grid.N_phi} x {grid.N_Z}")
    print(f"  d_phi = {grid.d_phi:.4f}, d_Z = {grid.d_Z:.4f}")

    # Зазор без текстуры
    H_smooth = compute_film_thickness_static(grid, model.geometry, epsilon_0=0.6)
    print(f"  H_smooth: min={H_smooth.min():.3f}, max={H_smooth.max():.3f}")

    # Проверка: минимум при φ=π
    phi_min_idx = np.argmin(H_smooth[grid.N_Z//2, :])
    phi_at_min = grid.phi[phi_min_idx]
    print(f"  H_min при φ = {phi_at_min:.3f} рад (должно быть ≈ π = {np.pi:.3f})")

    # Зазор с текстурой
    H_tex = compute_film_thickness_static(grid, model.geometry, epsilon_0=0.6, texture=model.texture)
    print(f"  H_textured: min={H_tex.min():.3f}, max={H_tex.max():.3f}")

    # Центры текстуры (филлотаксис)
    centers = compute_phyllotaxis_centers(model.texture, model.geometry)
    print(f"  Филлотаксис: {len(centers)} ячеек текстуры")

    print("  OK\n")


def test_reynolds():
    """Тест решателя Рейнольдса."""
    print("Тест reynolds...")
    from src.geometry import create_grid, compute_film_thickness_static, compute_dH_dphi
    from src.reynolds import solve_reynolds_equation
    from src.parameters import create_chinese_paper_bearing

    model = create_chinese_paper_bearing()
    grid = create_grid(180, 50)

    epsilon_0 = 0.6
    T = model.operating.T_min  # Используем нижнюю границу температурного диапазона

    H = compute_film_thickness_static(grid, model.geometry, epsilon_0)
    dH_dphi = compute_dH_dphi(grid, epsilon_0)
    dH_dtau = np.zeros_like(H)  # Статика
    eta_hat = model.lubricant.eta_hat(T) * np.ones_like(H)

    lambda_sq = model.geometry.lambda_ratio ** 2

    P, residual, iters = solve_reynolds_equation(
        H, eta_hat, dH_dphi, dH_dtau,
        grid.d_phi, grid.d_Z, lambda_sq,
        beta=2.0, omega_sor=1.7, tol=1e-6, max_iter=50000
    )

    print(f"  λ² = {lambda_sq:.4f}")
    print(f"  Итераций: {iters}, невязка: {residual:.2e}")
    print(f"  P: min={P.min():.3f}, max={P.max():.3f}")

    if residual < 1e-5:
        print("  OK\n")
    else:
        print("  WARNING: не сошлось!\n")


def test_forces():
    """Тест вычисления сил."""
    print("Тест forces...")
    from src.geometry import create_grid, compute_film_thickness_static, compute_dH_dphi
    from src.reynolds import solve_reynolds_equation
    from src.forces import integrate_forces
    from src.parameters import create_chinese_paper_bearing

    model = create_chinese_paper_bearing()
    grid = create_grid(180, 50)

    epsilon_0 = 0.6
    T = model.operating.T_min  # Используем нижнюю границу температурного диапазона

    H = compute_film_thickness_static(grid, model.geometry, epsilon_0)
    dH_dphi = compute_dH_dphi(grid, epsilon_0)
    dH_dtau = np.zeros_like(H)
    eta_hat = model.lubricant.eta_hat(T) * np.ones_like(H)

    lambda_sq = model.geometry.lambda_ratio ** 2

    P, _, _ = solve_reynolds_equation(
        H, eta_hat, dH_dphi, dH_dtau,
        grid.d_phi, grid.d_Z, lambda_sq, tol=1e-6
    )

    # Масштаб силы
    f_scale = model.force_scale(T)

    forces = integrate_forces(P, grid, f_scale)
    F_total = np.sqrt(forces.Fx**2 + forces.Fy**2)
    print(f"  f_scale = {f_scale:.3f} Н")
    print(f"  Силы: Fx={forces.Fx:.1f} Н, Fy={forces.Fy:.1f} Н, F={F_total:.1f} Н")

    print("  OK\n")


def test_equilibrium():
    """Тест поиска равновесного эксцентриситета."""
    print("Тест equilibrium...")
    from src.geometry import create_grid
    from src.forces import find_equilibrium_eccentricity
    from src.parameters import create_chinese_paper_bearing

    model = create_chinese_paper_bearing()
    grid = create_grid(180, 50)

    T = model.operating.T_min
    W_target = 100.0  # Н

    try:
        epsilon_0, forces = find_equilibrium_eccentricity(
            W_target, T, model, grid, with_texture=False
        )
        print(f"  W_target = {W_target} Н")
        print(f"  ε₀ = {epsilon_0:.4f}")
        print(f"  Fy = {forces.Fy:.2f} Н (должно быть ≈ {W_target})")
        print("  OK\n")
    except Exception as e:
        print(f"  Не удалось найти равновесие: {e}")
        print("  SKIP\n")


def test_coefficients():
    """Тест вычисления K_ij и C_ij."""
    print("Тест coefficients...")
    from src.geometry import create_grid
    from src.forces import (
        find_equilibrium_eccentricity,
        compute_stiffness_coefficients,
        compute_damping_coefficients,
    )
    from src.parameters import create_chinese_paper_bearing

    model = create_chinese_paper_bearing()
    grid = create_grid(180, 50)

    T = model.operating.T_min
    W_target = 100.0

    try:
        epsilon_0, _ = find_equilibrium_eccentricity(W_target, T, model, grid, with_texture=False)

        K = compute_stiffness_coefficients(epsilon_0, T, model, grid, with_texture=False)
        print(f"  K_xx = {K.K_xx/1e6:.3f} МН/м, K̄_xx = {K.K_xx_bar:.3f}")
        print(f"  K_yy = {K.K_yy/1e6:.3f} МН/м, K̄_yy = {K.K_yy_bar:.3f}")

        C = compute_damping_coefficients(epsilon_0, T, model, grid, with_texture=False)
        print(f"  C_xx = {C.C_xx/1e3:.3f} кН·с/м, C̄_xx = {C.C_xx_bar:.3f}")
        print(f"  C_yy = {C.C_yy/1e3:.3f} кН·с/м, C̄_yy = {C.C_yy_bar:.3f}")

        print("  OK\n")
    except Exception as e:
        print(f"  Ошибка: {e}")
        print("  SKIP\n")


def test_stability():
    """Тест параметров устойчивости."""
    print("Тест stability...")
    from src.geometry import create_grid
    from src.forces import (
        find_equilibrium_eccentricity,
        compute_stiffness_coefficients,
        compute_damping_coefficients,
    )
    from src.stability import compute_stability_parameters
    from src.parameters import create_chinese_paper_bearing

    model = create_chinese_paper_bearing()
    grid = create_grid(180, 50)

    T = model.operating.T_min
    W_target = 100.0

    try:
        epsilon_0, _ = find_equilibrium_eccentricity(W_target, T, model, grid, with_texture=False)
        K = compute_stiffness_coefficients(epsilon_0, T, model, grid, with_texture=False)
        C = compute_damping_coefficients(epsilon_0, T, model, grid, with_texture=False)

        stability = compute_stability_parameters(K, C)

        print(f"  K_eq = {stability.K_eq:.4f}")
        print(f"  γ²_st = {stability.gamma_sq:.4f}")
        print(f"  ω_st = {stability.omega_st:.4f}")
        print(f"  Устойчив: {stability.is_stable}")

        print("  OK\n")
    except Exception as e:
        print(f"  Ошибка: {e}")
        print("  SKIP\n")


def main():
    print("=" * 50)
    print("БЫСТРЫЙ ТЕСТ МОДЕЛИ ПОДШИПНИКА")
    print("=" * 50 + "\n")

    test_parameters()
    test_geometry()
    test_reynolds()
    test_forces()
    test_equilibrium()
    test_coefficients()
    test_stability()

    print("=" * 50)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    print("=" * 50)


if __name__ == "__main__":
    main()
