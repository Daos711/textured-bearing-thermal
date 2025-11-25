"""
Пример: Анализ опоры шарошечного долота.

Сравнение гладкого подшипника и подшипника с текстурой
при различных температурах забоя.

Запуск:
    cd textured-bearing-thermal
    python -m examples.roller_cone_bit_analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Настройка шрифтов для кириллицы
rcParams['font.family'] = 'DejaVu Sans'

from src.parameters import (
    BearingModel,
    BearingGeometry,
    TextureParameters,
    LubricantProperties,
    OperatingConditions,
    GridParameters,
)
from src.geometry import create_grid, compute_film_thickness
from src.reynolds import solve_reynolds_static, ReynoldsSolver
from src.thermal import THDSolver, estimate_temperature_rise, compute_viscosity_field
from src.forces import integrate_forces, compute_friction, BearingAnalyzer


def create_roller_cone_model(T_inlet: float = 80.0, n_rpm: float = 100.0) -> BearingModel:
    """
    Создание модели опоры шарошечного долота.

    Args:
        T_inlet: температура на забое (°C)
        n_rpm: скорость вращения (об/мин)
    """
    return BearingModel(
        geometry=BearingGeometry(
            R=0.025,        # 25 мм — радиус цапфы
            c=0.0002,       # 0.2 мм — радиальный зазор
            L=0.035,        # 35 мм — длина опоры
        ),
        texture=TextureParameters(
            enabled=True,
            h_p=0.00006,    # 60 мкм — глубина углублений
            a=0.0015,       # полуось по Z
            b=0.0012,       # полуось по φ
            N_phi=6,        # углублений по окружности
            N_Z=8,          # углублений по длине
            phi_start_deg=90.0,
            phi_end_deg=270.0,
        ),
        lubricant=LubricantProperties(
            eta_0=0.02,         # 20 мПа·с при 40°C — консистентная смазка
            T_0=40.0,
            beta=0.025,         # температурный коэффициент
            rho=900.0,
            c_p=2000.0,
            k=0.15,
        ),
        operating=OperatingConditions(
            n=n_rpm,
            epsilon=0.5,
            T_inlet=T_inlet,
            T_ambient=T_inlet + 20.0,
        ),
        grid=GridParameters(
            num_phi=360,
            num_Z=100,
        ),
    )


def analyze_temperature_effect():
    """
    Анализ влияния температуры забоя на характеристики опоры.
    """
    print("=" * 60)
    print("АНАЛИЗ ВЛИЯНИЯ ТЕМПЕРАТУРЫ НА ОПОРУ ШАРОШЕЧНОГО ДОЛОТА")
    print("=" * 60)

    temperatures = [40, 60, 80, 100, 120]  # °C

    results_smooth = []
    results_textured = []

    for T in temperatures:
        print(f"\nТемпература забоя: {T}°C")

        model = create_roller_cone_model(T_inlet=T)
        analyzer = BearingAnalyzer(model)

        # Гладкий подшипник
        res_smooth = analyzer.analyze(with_texture=False, compute_dynamics=False)
        results_smooth.append({
            'T': T,
            'F': res_smooth['forces'].F_total,
            'mu': res_smooth['friction'].friction_coeff,
            'eta': model.lubricant.viscosity(T) * 1000,  # мПа·с
        })

        # С текстурой
        res_tex = analyzer.analyze(with_texture=True, compute_dynamics=False)
        results_textured.append({
            'T': T,
            'F': res_tex['forces'].F_total,
            'mu': res_tex['friction'].friction_coeff,
        })

        print(f"  Вязкость: {results_smooth[-1]['eta']:.2f} мПа·с")
        print(f"  Гладкий:    F = {res_smooth['forces'].F_total:.1f} Н, "
              f"μ = {res_smooth['friction'].friction_coeff:.5f}")
        print(f"  С текстурой: F = {res_tex['forces'].F_total:.1f} Н, "
              f"μ = {res_tex['friction'].friction_coeff:.5f}")

        # Эффект текстуры
        delta_mu = (res_tex['friction'].friction_coeff -
                   res_smooth['friction'].friction_coeff)
        print(f"  Снижение трения: {-delta_mu/res_smooth['friction'].friction_coeff*100:.1f}%")

    return temperatures, results_smooth, results_textured


def analyze_eccentricity_effect():
    """
    Анализ влияния эксцентриситета на характеристики.
    """
    print("\n" + "=" * 60)
    print("АНАЛИЗ ВЛИЯНИЯ ЭКСЦЕНТРИСИТЕТА")
    print("=" * 60)

    model = create_roller_cone_model(T_inlet=80.0)

    epsilons = np.linspace(0.1, 0.8, 8)

    F_smooth = []
    F_textured = []
    mu_smooth = []
    mu_textured = []

    for eps in epsilons:
        model.operating.epsilon = eps
        analyzer = BearingAnalyzer(model)

        res_s = analyzer.analyze(epsilon=eps, with_texture=False, compute_dynamics=False)
        res_t = analyzer.analyze(epsilon=eps, with_texture=True, compute_dynamics=False)

        F_smooth.append(res_s['forces'].F_total)
        F_textured.append(res_t['forces'].F_total)
        mu_smooth.append(res_s['friction'].friction_coeff)
        mu_textured.append(res_t['friction'].friction_coeff)

        print(f"ε = {eps:.2f}: F_smooth = {res_s['forces'].F_total:.1f} Н, "
              f"F_tex = {res_t['forces'].F_total:.1f} Н")

    return epsilons, F_smooth, F_textured, mu_smooth, mu_textured


def plot_results(temp_data, eps_data):
    """Построение графиков результатов."""
    temperatures, results_smooth, results_textured = temp_data
    epsilons, F_smooth, F_textured, mu_smooth, mu_textured = eps_data

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Вязкость от температуры
    ax = axes[0, 0]
    eta_values = [r['eta'] for r in results_smooth]
    ax.plot(temperatures, eta_values, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Температура, °C')
    ax.set_ylabel('Вязкость, мПа·с')
    ax.set_title('Зависимость вязкости от температуры')
    ax.grid(True)

    # 2. Коэффициент трения от температуры
    ax = axes[0, 1]
    mu_s = [r['mu'] for r in results_smooth]
    mu_t = [r['mu'] for r in results_textured]
    ax.plot(temperatures, mu_s, 'o-', label='Гладкий', linewidth=2, markersize=8)
    ax.plot(temperatures, mu_t, 's--', label='С текстурой', linewidth=2, markersize=8)
    ax.set_xlabel('Температура, °C')
    ax.set_ylabel('Коэффициент трения μ')
    ax.set_title('Влияние температуры на трение')
    ax.legend()
    ax.grid(True)

    # 3. Несущая способность от эксцентриситета
    ax = axes[1, 0]
    ax.plot(epsilons, F_smooth, 'o-', label='Гладкий', linewidth=2, markersize=8)
    ax.plot(epsilons, F_textured, 's--', label='С текстурой', linewidth=2, markersize=8)
    ax.set_xlabel('Эксцентриситет ε')
    ax.set_ylabel('Несущая способность F, Н')
    ax.set_title('Несущая способность от эксцентриситета')
    ax.legend()
    ax.grid(True)

    # 4. Коэффициент трения от эксцентриситета
    ax = axes[1, 1]
    ax.plot(epsilons, mu_smooth, 'o-', label='Гладкий', linewidth=2, markersize=8)
    ax.plot(epsilons, mu_textured, 's--', label='С текстурой', linewidth=2, markersize=8)
    ax.set_xlabel('Эксцентриситет ε')
    ax.set_ylabel('Коэффициент трения μ')
    ax.set_title('Коэффициент трения от эксцентриситета')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('roller_cone_bit_results.png', dpi=150)
    print("\nГрафики сохранены в roller_cone_bit_results.png")
    plt.show()


def run_thd_analysis():
    """
    Термогидродинамический анализ с учётом нагрева в слое.
    """
    print("\n" + "=" * 60)
    print("ТЕРМОГИДРОДИНАМИЧЕСКИЙ АНАЛИЗ (THD)")
    print("=" * 60)

    model = create_roller_cone_model(T_inlet=80.0)

    # Оценка подъёма температуры
    delta_T_est = estimate_temperature_rise(model)
    print(f"\nОценка подъёма температуры: ΔT ≈ {delta_T_est:.1f}°C")

    # Создаём сетку и зазор
    grid = create_grid(model.grid.num_phi, model.grid.num_Z)

    # Гладкий подшипник
    H_smooth = compute_film_thickness(grid, model.geometry,
                                      model.operating.epsilon, texture=None)

    # С текстурой
    H_textured = compute_film_thickness(grid, model.geometry,
                                        model.operating.epsilon, model.texture)

    # THD решение
    print("\nРешение THD для гладкого подшипника...")
    thd_smooth = THDSolver(model, grid)
    P_s, T_s, eta_s, conv_s = thd_smooth.solve(H_smooth, verbose=True)

    print("\nРешение THD для подшипника с текстурой...")
    thd_textured = THDSolver(model, grid)
    P_t, T_t, eta_t, conv_t = thd_textured.solve(H_textured, verbose=True)

    print(f"\nРезультаты THD:")
    print(f"  Гладкий:     T_max = {np.max(T_s):.1f}°C, T_mean = {np.mean(T_s):.1f}°C")
    print(f"  С текстурой: T_max = {np.max(T_t):.1f}°C, T_mean = {np.mean(T_t):.1f}°C")

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    Z_idx = grid.N_Z // 2

    # Температура — гладкий
    ax = axes[0, 0]
    c = ax.contourf(grid.Phi_mesh, grid.Z_mesh, T_s, levels=30, cmap='hot')
    plt.colorbar(c, ax=ax, label='T, °C')
    ax.set_xlabel('φ, рад')
    ax.set_ylabel('Z')
    ax.set_title('Температура (гладкий)')

    # Температура — с текстурой
    ax = axes[0, 1]
    c = ax.contourf(grid.Phi_mesh, grid.Z_mesh, T_t, levels=30, cmap='hot')
    plt.colorbar(c, ax=ax, label='T, °C')
    ax.set_xlabel('φ, рад')
    ax.set_ylabel('Z')
    ax.set_title('Температура (с текстурой)')

    # Давление при Z=0
    ax = axes[1, 0]
    ax.plot(grid.phi, P_s[Z_idx, :], 'b-', label='Гладкий', linewidth=2)
    ax.plot(grid.phi, P_t[Z_idx, :], 'r--', label='С текстурой', linewidth=2)
    ax.set_xlabel('φ, рад')
    ax.set_ylabel('P (безразмерное)')
    ax.set_title('Давление при Z=0')
    ax.legend()
    ax.grid(True)

    # Температура при Z=0
    ax = axes[1, 1]
    ax.plot(grid.phi, T_s[Z_idx, :], 'b-', label='Гладкий', linewidth=2)
    ax.plot(grid.phi, T_t[Z_idx, :], 'r--', label='С текстурой', linewidth=2)
    ax.set_xlabel('φ, рад')
    ax.set_ylabel('T, °C')
    ax.set_title('Температура при Z=0')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('thd_analysis_results.png', dpi=150)
    print("\nГрафики THD сохранены в thd_analysis_results.png")
    plt.show()


def main():
    """Главная функция."""
    print("\n" + "=" * 60)
    print("МОДЕЛИРОВАНИЕ ОПОРЫ ШАРОШЕЧНОГО ДОЛОТА")
    print("С УЧЁТОМ ТЕКСТУРЫ И ТЕМПЕРАТУРЫ")
    print("=" * 60)

    # Информация о модели
    model = create_roller_cone_model()
    print(model.info())

    # Анализ влияния температуры
    temp_data = analyze_temperature_effect()

    # Анализ влияния эксцентриситета
    eps_data = analyze_eccentricity_effect()

    # Построение графиков
    plot_results(temp_data, eps_data)

    # THD анализ
    run_thd_analysis()

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 60)


if __name__ == "__main__":
    main()
