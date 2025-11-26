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
    # Уменьшаем сетку для быстрого теста
    model.numerical.N_phi = 120
    model.numerical.N_Z = 40

    # Малая сетка для быстрого теста
    results_smooth = run_parametric_calculation(
        model, with_texture=False, N_W=5, N_T=4
    )

    results_textured = run_parametric_calculation(
        model, with_texture=True, N_W=5, N_T=4
    )

    print("\n--- Результаты (гладкий) ---")
    print(f"  W: [{results_smooth.W_arr[0]:.1f}, {results_smooth.W_arr[-1]:.1f}] Н")
    print(f"  T: [{results_smooth.T_arr[0]:.1f}, {results_smooth.T_arr[-1]:.1f}] °C")
    print(f"  Точек: {results_smooth.W_mesh.size}")

    # Построение 3D графиков
    plot_3d_surfaces(results_smooth, results_textured, save_dir=RESULTS_DIR)
    print(f"\nГрафики сохранены в {RESULTS_DIR}")
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
    results_smooth, results_textured = run_full_analysis(
        model=model,
        N_W=8,
        N_T=6,
        save_dir=RESULTS_DIR
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


def stability_vs_load_at_fixed_T():
    """
    Построение γ²_st(W) и ω_st(W) при фиксированных температурах.

    Аналог диссертационных рисунков: сравнение гладкой и текстурированной
    поверхности при нескольких значениях T.
    """
    print("\n" + "=" * 60)
    print("УСТОЙЧИВОСТЬ vs НАГРУЗКА ПРИ ФИКСИРОВАННЫХ T")
    print("=" * 60)

    from src.forces import (
        find_equilibrium_eccentricity,
        compute_stiffness_coefficients,
        compute_damping_coefficients,
    )
    from src.stability import compute_stability_parameters

    model = create_chinese_paper_bearing()
    grid = create_grid(180, 50)

    temperatures = [40, 60, 80]  # °C
    W_arr = np.linspace(500, 8000, 20)  # Н

    results = {T: {'smooth': {}, 'textured': {}} for T in temperatures}

    for T in temperatures:
        print(f"\nТемпература T = {T}°C")

        for label, with_tex in [('smooth', False), ('textured', True)]:
            gamma_list = []
            omega_list = []
            Keq_list = []
            eps_list = []

            for W in W_arr:
                try:
                    eps, _ = find_equilibrium_eccentricity(W, T, model, grid, with_texture=with_tex)
                    K = compute_stiffness_coefficients(eps, T, model, grid, with_texture=with_tex)
                    C = compute_damping_coefficients(eps, T, model, grid, with_texture=with_tex)
                    stab = compute_stability_parameters(K, C)

                    eps_list.append(eps)
                    Keq_list.append(stab.K_eq)
                    gamma_list.append(stab.gamma_sq)
                    omega_list.append(stab.omega_st if abs(stab.omega_st) < 1e6 else np.nan)
                except:
                    eps_list.append(np.nan)
                    Keq_list.append(np.nan)
                    gamma_list.append(np.nan)
                    omega_list.append(np.nan)

            results[T][label] = {
                'eps': np.array(eps_list),
                'K_eq': np.array(Keq_list),
                'gamma_sq': np.array(gamma_list),
                'omega_st': np.array(omega_list),
            }
        print(f"  Готово")

    # Графики
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = {40: 'blue', 60: 'green', 80: 'red'}
    W_kN = W_arr / 1000

    # γ²_st(W)
    for i, T in enumerate(temperatures):
        ax = axes[0, i]
        ax.plot(W_kN, results[T]['smooth']['gamma_sq'], 'o-',
                color=colors[T], label='Гладкий', linewidth=2)
        ax.plot(W_kN, results[T]['textured']['gamma_sq'], 's--',
                color=colors[T], alpha=0.7, label='С текстурой', linewidth=2)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('W, кН')
        ax.set_ylabel('γ²_st')
        ax.set_title(f'γ²_st(W) при T = {T}°C')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ω_st(W) (только положительные, где есть смысл)
    for i, T in enumerate(temperatures):
        ax = axes[1, i]
        omega_s = np.clip(results[T]['smooth']['omega_st'], -200, 200)
        omega_t = np.clip(results[T]['textured']['omega_st'], -200, 200)
        ax.plot(W_kN, omega_s, 'o-',
                color=colors[T], label='Гладкий', linewidth=2)
        ax.plot(W_kN, omega_t, 's--',
                color=colors[T], alpha=0.7, label='С текстурой', linewidth=2)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('W, кН')
        ax.set_ylabel('ω_st')
        ax.set_title(f'ω_st(W) при T = {T}°C')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, 'stability_vs_load.png')
    fig.savefig(filepath, dpi=150)
    print(f"\nГрафики сохранены в {filepath}")
    plt.show()

    return results


def texture_parameter_sweep():
    """
    Перебор параметров текстуры (h_p, a, b) для поиска оптимума.

    Ищем зону параметров, где текстура максимизирует K_eq и γ²_st.
    """
    print("\n" + "=" * 60)
    print("ПЕРЕБОР ПАРАМЕТРОВ ТЕКСТУРЫ")
    print("=" * 60)

    from src.forces import (
        find_equilibrium_eccentricity,
        compute_stiffness_coefficients,
        compute_damping_coefficients,
    )
    from src.stability import compute_stability_parameters
    from src.parameters import TextureParameters

    model = create_chinese_paper_bearing()
    grid = create_grid(120, 40)

    # Фиксированные условия
    W = 2000.0  # Н
    T = 60.0    # °C

    # Диапазоны параметров текстуры
    h_p_arr = np.array([5, 10, 15, 20, 30]) * 1e-6     # мкм -> м
    ab_arr = np.array([0.5, 1.0, 1.5, 2.0, 3.0]) * 1e-3  # мм -> м (a = b)

    # Результаты для гладкого подшипника (базовая линия)
    print("\nБазовый расчёт (гладкий)...")
    eps_smooth, _ = find_equilibrium_eccentricity(W, T, model, grid, with_texture=False)
    K_smooth = compute_stiffness_coefficients(eps_smooth, T, model, grid, with_texture=False)
    C_smooth = compute_damping_coefficients(eps_smooth, T, model, grid, with_texture=False)
    stab_smooth = compute_stability_parameters(K_smooth, C_smooth)

    print(f"  Гладкий: ε₀={eps_smooth:.4f}, K_eq={stab_smooth.K_eq:.3f}, γ²={stab_smooth.gamma_sq:.4f}")

    # Сетка результатов
    delta_Keq = np.zeros((len(h_p_arr), len(ab_arr)))
    delta_gamma = np.zeros((len(h_p_arr), len(ab_arr)))
    gamma_values = np.zeros((len(h_p_arr), len(ab_arr)))

    print(f"\nПеребор: {len(h_p_arr)} x {len(ab_arr)} = {len(h_p_arr)*len(ab_arr)} точек")

    for i, h_p in enumerate(h_p_arr):
        for j, ab in enumerate(ab_arr):
            # Создаём модель с изменёнными параметрами текстуры
            model_tex = create_chinese_paper_bearing()
            model_tex.texture.h_p = h_p
            model_tex.texture.a = ab
            model_tex.texture.b = ab

            try:
                eps_tex, _ = find_equilibrium_eccentricity(W, T, model_tex, grid, with_texture=True)
                K_tex = compute_stiffness_coefficients(eps_tex, T, model_tex, grid, with_texture=True)
                C_tex = compute_damping_coefficients(eps_tex, T, model_tex, grid, with_texture=True)
                stab_tex = compute_stability_parameters(K_tex, C_tex)

                delta_Keq[i, j] = (stab_tex.K_eq - stab_smooth.K_eq) / abs(stab_smooth.K_eq) * 100
                delta_gamma[i, j] = stab_tex.gamma_sq - stab_smooth.gamma_sq
                gamma_values[i, j] = stab_tex.gamma_sq
            except Exception as e:
                delta_Keq[i, j] = np.nan
                delta_gamma[i, j] = np.nan
                gamma_values[i, j] = np.nan

        print(f"  h_p = {h_p*1e6:.0f} мкм: готово")

    # Вывод таблицы
    print("\n--- Изменение K_eq (%) ---")
    print("h_p\\a,b:", end="")
    for ab in ab_arr:
        print(f"  {ab*1e3:.1f}мм", end="")
    print()
    for i, h_p in enumerate(h_p_arr):
        print(f"{h_p*1e6:4.0f}мкм:", end="")
        for j in range(len(ab_arr)):
            print(f"  {delta_Keq[i,j]:+5.1f}", end="")
        print()

    print("\n--- Изменение γ²_st ---")
    print("h_p\\a,b:", end="")
    for ab in ab_arr:
        print(f"  {ab*1e3:.1f}мм", end="")
    print()
    for i, h_p in enumerate(h_p_arr):
        print(f"{h_p*1e6:4.0f}мкм:", end="")
        for j in range(len(ab_arr)):
            print(f"  {delta_gamma[i,j]:+5.3f}", end="")
        print()

    # Графики
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    h_p_um = h_p_arr * 1e6
    ab_mm = ab_arr * 1e3

    # ΔK_eq
    ax = axes[0]
    H, AB = np.meshgrid(h_p_um, ab_mm, indexing='ij')
    c1 = ax.contourf(H, AB, delta_Keq, levels=15, cmap='RdBu_r')
    ax.contour(H, AB, delta_Keq, levels=[0], colors='k', linewidths=2)
    plt.colorbar(c1, ax=ax, label='ΔK_eq, %')
    ax.set_xlabel('h_p, мкм')
    ax.set_ylabel('a = b, мм')
    ax.set_title(f'Изменение K_eq (W={W/1000:.0f}кН, T={T}°C)')

    # Δγ²_st
    ax = axes[1]
    c2 = ax.contourf(H, AB, delta_gamma, levels=15, cmap='RdBu_r')
    ax.contour(H, AB, delta_gamma, levels=[0], colors='k', linewidths=2)
    plt.colorbar(c2, ax=ax, label='Δγ²_st')
    ax.set_xlabel('h_p, мкм')
    ax.set_ylabel('a = b, мм')
    ax.set_title('Изменение γ²_st')

    # Абсолютное γ²_st
    ax = axes[2]
    c3 = ax.contourf(H, AB, gamma_values, levels=15, cmap='coolwarm')
    ax.contour(H, AB, gamma_values, levels=[0], colors='k', linewidths=2)
    plt.colorbar(c3, ax=ax, label='γ²_st')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('h_p, мкм')
    ax.set_ylabel('a = b, мм')
    ax.set_title(f'γ²_st (устойчив при >0)\nГладкий: γ²={stab_smooth.gamma_sq:.3f}')

    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, 'texture_parameter_sweep.png')
    fig.savefig(filepath, dpi=150)
    print(f"\nГрафики сохранены в {filepath}")
    plt.show()


def friction_verification():
    """
    Проверка порядка величин трения: μ_f, F_f, N_f.

    Сравнение с ручной оценкой:
    - N_f ≈ μ_f · W · U
    - При W ~ несколько кН, U ~ 0.4 м/с ожидаем N_f ~ десятки-сотни Вт
    """
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ТРЕНИЯ: μ_f, F_f, N_f")
    print("=" * 60)

    from src.forces import (
        find_equilibrium_eccentricity,
        compute_friction,
    )
    from src.reynolds import solve_reynolds_static
    from src.geometry import compute_film_thickness_static

    model = create_chinese_paper_bearing()
    grid = create_grid(180, 50)

    test_points = [
        (1000.0, 40.0),
        (2000.0, 60.0),
        (5000.0, 80.0),
        (8000.0, 100.0),
    ]

    print(f"\nПараметры модели:")
    print(f"  R_J = {model.geometry.R_J*1000:.2f} мм")
    print(f"  c0 = {model.geometry.c0*1e6:.1f} мкм")
    print(f"  L = {model.geometry.L*1000:.1f} мм")
    print(f"  n = {model.operating.n_rpm:.0f} об/мин")
    print(f"  U = {model.operating.U(model.geometry.R_J):.3f} м/с")

    print(f"\n{'W, Н':>8} {'T, °C':>6} {'η, мПа·с':>10} {'ε₀':>8} {'μ_f':>10} {'F_f, Н':>10} {'N_f, Вт':>10} {'N_f≈μWU':>10}")
    print("-" * 85)

    for W, T in test_points:
        eta = model.lubricant.viscosity(T)
        U = model.operating.U(model.geometry.R_J)

        # Гладкий
        eps, _ = find_equilibrium_eccentricity(W, T, model, grid, with_texture=False)

        eta_hat = model.lubricant.eta_hat(T)
        H_T = model.material.H_T(model.geometry, T)
        lambda_sq = model.geometry.lambda_ratio ** 2

        H = compute_film_thickness_static(grid, model.geometry, eps, None, H_T)
        P, _, _ = solve_reynolds_static(
            H, eta_hat, grid.d_phi, grid.d_Z, lambda_sq,
            omega_sor=model.numerical.omega_GS,
            tol=model.numerical.tol_Re,
            max_iter=model.numerical.max_iter_Re,
        )

        friction = compute_friction(P, H, grid, model, T)

        # Ручная оценка N_f
        N_f_estimate = friction.friction_coeff * W * U

        print(f"{W:8.0f} {T:6.0f} {eta*1000:10.2f} {eps:8.4f} {friction.friction_coeff:10.4f} "
              f"{friction.friction_force:10.2f} {friction.power_loss:10.3f} {N_f_estimate:10.3f}")

    print("\nПояснение:")
    print("  μ_f — безразмерный коэффициент трения")
    print("  F_f — размерная сила трения (Н)")
    print("  N_f — мощность потерь = |F_f| × U (Вт)")
    print("  N_f≈μWU — грубая оценка мощности")

    # Проверка масштабов
    print("\n--- Анализ масштабов ---")
    T_test = 60.0
    W_test = 2000.0
    eta = model.lubricant.viscosity(T_test)
    U = model.operating.U(model.geometry.R_J)

    print(f"При T={T_test}°C, W={W_test:.0f}Н:")
    print(f"  Вязкость η = {eta*1000:.2f} мПа·с")
    print(f"  Окружная скорость U = {U:.3f} м/с")
    print(f"  Если μ_f ~ 0.01-0.1, то N_f ~ {0.01*W_test*U:.1f} - {0.1*W_test*U:.1f} Вт")
    print(f"  Для реального долота с W ~ 10кН: N_f ~ {0.05*10000*U:.0f} Вт")


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

    # === НОВЫЕ ФУНКЦИИ ===

    # 4. Графики γ²_st(W) при фиксированных T (аналог диссертации)
    stability_vs_load_at_fixed_T()

    # 5. Перебор параметров текстуры
    texture_parameter_sweep()

    # 6. Проверка порядка величин трения
    friction_verification()

    # 7. Полный параметрический расчёт (долгий)
    # Раскомментируйте для полного расчёта:
    # parametric_analysis_full()

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print(f"Результаты сохранены в папке: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
