"""
Пример: Параметрический анализ подшипника.

Строит 3D поверхности K_eq(W,T), gamma_st(W,T), omega_st(W,T)
для гладкого и текстурированного подшипника.

Запуск:
    cd textured-bearing-thermal
    python examples/roller_cone_bit_analysis.py
"""

import sys
import os

# Добавляем корневую папку проекта в путь
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Папка для результатов
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Настройка шрифтов
rcParams['font.family'] = 'DejaVu Sans'

from src.parameters import create_chinese_paper_bearing, create_roller_cone_bit_bearing
from src.geometry import create_grid, compute_film_thickness_static
from src.main import run_parametric_calculation, run_full_analysis, plot_3d_surfaces


def single_point_analysis():
    """
    Анализ в одной точке (W, T).
    """
    print("=" * 60)
    print("АНАЛИЗ В ОДНОЙ ТОЧКЕ")
    print("=" * 60)

    from src.forces import (
        find_equilibrium_eccentricity,
        compute_stiffness_coefficients,
        compute_damping_coefficients,
    )
    from src.stability import compute_stability_parameters

    model = create_chinese_paper_bearing()
    grid = create_grid(180, 50)

    W = 100.0  # Н
    T = 40.0   # °C

    print(f"\nПараметры:")
    print(f"  Нагрузка W = {W} Н")
    print(f"  Температура T = {T}°C")
    print(f"  Вязкость η = {model.lubricant.viscosity(T)*1000:.2f} мПа·с")

    # Гладкий подшипник
    print("\n--- Гладкий подшипник ---")
    eps_smooth, forces_smooth = find_equilibrium_eccentricity(W, T, model, grid, with_texture=False)
    K_smooth = compute_stiffness_coefficients(eps_smooth, T, model, grid, with_texture=False)
    C_smooth = compute_damping_coefficients(eps_smooth, T, model, grid, with_texture=False)
    stab_smooth = compute_stability_parameters(K_smooth, C_smooth)

    print(f"  ε₀ = {eps_smooth:.4f}")
    print(f"  Fy = {forces_smooth.Fy:.2f} Н")
    print(f"  K̄_xx = {K_smooth.K_xx_bar:.3f}, K̄_yy = {K_smooth.K_yy_bar:.3f}")
    print(f"  C̄_xx = {C_smooth.C_xx_bar:.3f}, C̄_yy = {C_smooth.C_yy_bar:.3f}")
    print(f"  K_eq = {stab_smooth.K_eq:.4f}")
    print(f"  γ²_st = {stab_smooth.gamma_sq:.4f}")
    print(f"  ω_st = {stab_smooth.omega_st:.4f}")
    print(f"  Устойчив: {stab_smooth.is_stable}")

    # С текстурой
    print("\n--- Подшипник с текстурой ---")
    eps_tex, forces_tex = find_equilibrium_eccentricity(W, T, model, grid, with_texture=True)
    K_tex = compute_stiffness_coefficients(eps_tex, T, model, grid, with_texture=True)
    C_tex = compute_damping_coefficients(eps_tex, T, model, grid, with_texture=True)
    stab_tex = compute_stability_parameters(K_tex, C_tex)

    print(f"  ε₀ = {eps_tex:.4f}")
    print(f"  Fy = {forces_tex.Fy:.2f} Н")
    print(f"  K̄_xx = {K_tex.K_xx_bar:.3f}, K̄_yy = {K_tex.K_yy_bar:.3f}")
    print(f"  C̄_xx = {C_tex.C_xx_bar:.3f}, C̄_yy = {C_tex.C_yy_bar:.3f}")
    print(f"  K_eq = {stab_tex.K_eq:.4f}")
    print(f"  γ²_st = {stab_tex.gamma_sq:.4f}")
    print(f"  ω_st = {stab_tex.omega_st:.4f}")
    print(f"  Устойчив: {stab_tex.is_stable}")

    # Сравнение
    print("\n--- Эффект текстуры ---")
    delta_eps = (eps_tex - eps_smooth) / eps_smooth * 100
    delta_Keq = (stab_tex.K_eq - stab_smooth.K_eq) / abs(stab_smooth.K_eq) * 100 if stab_smooth.K_eq != 0 else 0
    print(f"  Δε₀ = {delta_eps:+.2f}%")
    print(f"  ΔK_eq = {delta_Keq:+.2f}%")


def parametric_analysis_small():
    """
    Небольшой параметрический расчёт для быстрой проверки.
    """
    print("\n" + "=" * 60)
    print("ПАРАМЕТРИЧЕСКИЙ РАСЧЁТ (МАЛЫЙ)")
    print("=" * 60)

    model = create_chinese_paper_bearing()

    # Малая сетка для быстрого теста
    results_smooth = run_parametric_calculation(
        model, with_texture=False, N_W=5, N_T=4, N_phi=120, N_Z=40
    )

    results_textured = run_parametric_calculation(
        model, with_texture=True, N_W=5, N_T=4, N_phi=120, N_Z=40
    )

    print("\n--- Результаты (гладкий) ---")
    print(f"  W: [{results_smooth.W_range[0]:.1f}, {results_smooth.W_range[-1]:.1f}] Н")
    print(f"  T: [{results_smooth.T_range[0]:.1f}, {results_smooth.T_range[-1]:.1f}] °C")
    print(f"  Точек: {len(results_smooth.results)}")

    # Построение 3D графиков
    fig = plot_3d_surfaces(results_smooth, results_textured, title_prefix="Малый тест")

    filepath = os.path.join(RESULTS_DIR, 'parametric_small.png')
    fig.savefig(filepath, dpi=150)
    print(f"\nГрафики сохранены в {filepath}")
    plt.show()


def parametric_analysis_full():
    """
    Полный параметрический расчёт.
    """
    print("\n" + "=" * 60)
    print("ПОЛНЫЙ ПАРАМЕТРИЧЕСКИЙ РАСЧЁТ")
    print("=" * 60)

    model = create_chinese_paper_bearing()
    print(f"\nМодель: Chinese paper bearing")
    print(f"  R_J = {model.geometry.R_J*1000:.2f} мм")
    print(f"  c₀ = {model.geometry.c0*1e6:.1f} мкм")
    print(f"  L = {model.geometry.L*1000:.1f} мм")
    print(f"  λ = {model.geometry.lambda_ratio:.3f}")

    # Полный расчёт
    results_smooth, results_textured, fig = run_full_analysis(
        model=model,
        N_W=8,
        N_T=6,
        N_phi=180,
        N_Z=50,
        save_path=os.path.join(RESULTS_DIR, 'stability_surfaces.png')
    )

    plt.show()

    return results_smooth, results_textured


def compare_temperature_effect():
    """
    Сравнение влияния температуры на гладкий и текстурированный подшипник.
    """
    print("\n" + "=" * 60)
    print("ВЛИЯНИЕ ТЕМПЕРАТУРЫ")
    print("=" * 60)

    from src.forces import find_equilibrium_eccentricity
    from src.stability import compute_stability_parameters
    from src.forces import compute_stiffness_coefficients, compute_damping_coefficients

    model = create_chinese_paper_bearing()
    grid = create_grid(180, 50)

    W = 100.0  # Н
    temperatures = [30, 40, 50, 60, 70, 80]

    eps_smooth_list = []
    eps_tex_list = []
    Keq_smooth_list = []
    Keq_tex_list = []
    gamma_smooth_list = []
    gamma_tex_list = []

    for T in temperatures:
        print(f"\nT = {T}°C, η = {model.lubricant.viscosity(T)*1000:.2f} мПа·с")

        # Гладкий
        eps_s, _ = find_equilibrium_eccentricity(W, T, model, grid, with_texture=False)
        K_s = compute_stiffness_coefficients(eps_s, T, model, grid, with_texture=False)
        C_s = compute_damping_coefficients(eps_s, T, model, grid, with_texture=False)
        stab_s = compute_stability_parameters(K_s, C_s)

        # С текстурой
        eps_t, _ = find_equilibrium_eccentricity(W, T, model, grid, with_texture=True)
        K_t = compute_stiffness_coefficients(eps_t, T, model, grid, with_texture=True)
        C_t = compute_damping_coefficients(eps_t, T, model, grid, with_texture=True)
        stab_t = compute_stability_parameters(K_t, C_t)

        eps_smooth_list.append(eps_s)
        eps_tex_list.append(eps_t)
        Keq_smooth_list.append(stab_s.K_eq)
        Keq_tex_list.append(stab_t.K_eq)
        gamma_smooth_list.append(stab_s.gamma_sq)
        gamma_tex_list.append(stab_t.gamma_sq)

        print(f"  Гладкий: ε₀={eps_s:.4f}, K_eq={stab_s.K_eq:.3f}, γ²={stab_s.gamma_sq:.3f}")
        print(f"  Текстура: ε₀={eps_t:.4f}, K_eq={stab_t.K_eq:.3f}, γ²={stab_t.gamma_sq:.3f}")

    # Графики
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(temperatures, eps_smooth_list, 'o-', label='Гладкий', linewidth=2)
    ax.plot(temperatures, eps_tex_list, 's--', label='С текстурой', linewidth=2)
    ax.set_xlabel('Температура, °C')
    ax.set_ylabel('Эксцентриситет ε₀')
    ax.set_title(f'Равновесный эксцентриситет (W={W} Н)')
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(temperatures, Keq_smooth_list, 'o-', label='Гладкий', linewidth=2)
    ax.plot(temperatures, Keq_tex_list, 's--', label='С текстурой', linewidth=2)
    ax.set_xlabel('Температура, °C')
    ax.set_ylabel('K_eq')
    ax.set_title('Эквивалентная жёсткость')
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    ax.plot(temperatures, gamma_smooth_list, 'o-', label='Гладкий', linewidth=2)
    ax.plot(temperatures, gamma_tex_list, 's--', label='С текстурой', linewidth=2)
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Температура, °C')
    ax.set_ylabel('γ²_st')
    ax.set_title('Квадрат лог. декремента')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, 'temperature_effect.png')
    fig.savefig(filepath, dpi=150)
    print(f"\nГрафики сохранены в {filepath}")
    plt.show()


def main():
    """Главная функция."""
    print("\n" + "=" * 60)
    print("АНАЛИЗ УСТОЙЧИВОСТИ ПОДШИПНИКА")
    print("С УЧЁТОМ ТЕКСТУРЫ И ТЕМПЕРАТУРЫ")
    print("=" * 60)

    # 1. Анализ в одной точке
    single_point_analysis()

    # 2. Влияние температуры
    compare_temperature_effect()

    # 3. Малый параметрический расчёт (быстрый)
    parametric_analysis_small()

    # 4. Полный параметрический расчёт (долгий)
    # Раскомментируйте для полного расчёта:
    # parametric_analysis_full()

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print(f"Результаты сохранены в папке: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
