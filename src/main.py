"""
Главный модуль параметрического расчёта подшипника.

Алгоритм из постановки задачи (раздел 11):
1. Задаётся диапазон температур T ∈ [T_min, T_max]
2. Задаётся диапазон нагрузок W ∈ [W_min, W_max]
3. Для каждой пары (W, T):
   - вычисляются η̂(T) и H_T(T)
   - находится рабочий эксцентриситет ε₀(W, T) из условия равновесия
   - решается набор задач Рейнольдса для определения K_ij, C_ij
   - коэффициенты приводятся к безразмерному виду K̄_ij, C̄_ij
   - вычисляются K_eq, γ²_st, ω_st
   - опционально — коэффициент трения μ_f и мощность потерь N_f
4. По полученным данным строятся:
   - поверхности K_eq(W, T), γ²_st(W, T), ω_st(W, T)
   - отдельно для гладкой и текстурированной опор
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Tuple, List, Optional
import os
from tqdm import tqdm

from .parameters import BearingModel, create_chinese_paper_bearing, create_roller_cone_bit_bearing
from .geometry import create_grid, Grid, get_texture_centers_count
from .forces import (
    find_equilibrium_eccentricity,
    compute_stiffness_coefficients,
    compute_damping_coefficients,
    compute_all_coefficients,
    compute_friction,
    integrate_forces,
    ForceResult,
    StiffnessCoefficients,
    DampingCoefficients,
)
from .stability import compute_stability_parameters, StabilityParameters
from .reynolds import solve_reynolds_static
from .geometry import compute_film_thickness_static


@dataclass
class PointResult:
    """Результат расчёта в одной точке (W, T)."""
    W: float                        # Нагрузка, Н
    T: float                        # Температура, °C
    epsilon_0: float                # Статический эксцентриситет
    K: StiffnessCoefficients        # Коэффициенты жёсткости
    C: DampingCoefficients          # Коэффициенты демпфирования
    stability: StabilityParameters  # Параметры устойчивости
    mu_f: Optional[float] = None    # Коэффициент трения
    N_f: Optional[float] = None     # Мощность потерь, Вт


@dataclass
class ParametricResults:
    """Результаты параметрического расчёта."""
    W_arr: np.ndarray               # Массив нагрузок
    T_arr: np.ndarray               # Массив температур
    W_mesh: np.ndarray              # Сетка нагрузок (N_T x N_W)
    T_mesh: np.ndarray              # Сетка температур (N_T x N_W)
    epsilon_0: np.ndarray           # ε₀(W, T)
    K_eq: np.ndarray                # K_eq(W, T)
    gamma_sq: np.ndarray            # γ²_st(W, T)
    omega_st: np.ndarray            # ω_st(W, T)
    mu_f: Optional[np.ndarray] = None
    N_f: Optional[np.ndarray] = None
    # Безразмерные коэффициенты
    K_xx_bar: Optional[np.ndarray] = None
    K_yy_bar: Optional[np.ndarray] = None
    C_xx_bar: Optional[np.ndarray] = None
    C_yy_bar: Optional[np.ndarray] = None


def run_parametric_calculation(
    model: BearingModel,
    with_texture: bool = True,
    N_W: Optional[int] = None,
    N_T: Optional[int] = None,
    compute_friction_flag: bool = True,
    verbose: bool = True,
) -> ParametricResults:
    """
    Выполнить параметрический расчёт по нагрузке и температуре.

    Args:
        model: модель подшипника
        with_texture: учитывать текстуру
        N_W: число точек по нагрузке (если None — из model.operating)
        N_T: число точек по температуре
        compute_friction_flag: вычислять трение
        verbose: выводить прогресс

    Returns:
        ParametricResults
    """
    # Параметры диапазонов
    if N_W is None:
        N_W = model.operating.N_W
    if N_T is None:
        N_T = model.operating.N_T

    W_arr = np.linspace(model.operating.W_min, model.operating.W_max, N_W)
    T_arr = np.linspace(model.operating.T_min, model.operating.T_max, N_T)

    # Сетка
    T_mesh, W_mesh = np.meshgrid(T_arr, W_arr, indexing='ij')

    # Массивы для результатов
    epsilon_0 = np.zeros((N_T, N_W))
    K_eq = np.zeros((N_T, N_W))
    gamma_sq = np.zeros((N_T, N_W))
    omega_st = np.zeros((N_T, N_W))
    K_xx_bar = np.zeros((N_T, N_W))
    K_yy_bar = np.zeros((N_T, N_W))
    C_xx_bar = np.zeros((N_T, N_W))
    C_yy_bar = np.zeros((N_T, N_W))

    if compute_friction_flag:
        mu_f = np.zeros((N_T, N_W))
        N_f = np.zeros((N_T, N_W))
    else:
        mu_f = None
        N_f = None

    # Расчётная сетка
    grid = create_grid(model.numerical.N_phi, model.numerical.N_Z)

    # Итератор
    total = N_T * N_W
    iterator = range(total)
    if verbose:
        print(f"Параметрический расчёт: {N_T} x {N_W} = {total} точек")
        print(f"Текстура: {'Да' if with_texture else 'Нет'}")
        if with_texture:
            n_cells = get_texture_centers_count(model.texture, model.geometry)
            print(f"  Число углублений: {n_cells}")
        iterator = tqdm(iterator, desc="Расчёт")

    for idx in iterator:
        i_T = idx // N_W
        i_W = idx % N_W

        T = T_arr[i_T]
        W = W_arr[i_W]

        try:
            # Поиск равновесного эксцентриситета
            eps_0, forces = find_equilibrium_eccentricity(
                W, T, model, grid, with_texture=with_texture
            )
            epsilon_0[i_T, i_W] = eps_0

            # Коэффициенты жёсткости и демпфирования
            K = compute_stiffness_coefficients(eps_0, T, model, grid, with_texture)
            C = compute_damping_coefficients(eps_0, T, model, grid, with_texture)

            K_xx_bar[i_T, i_W] = K.K_xx_bar
            K_yy_bar[i_T, i_W] = K.K_yy_bar
            C_xx_bar[i_T, i_W] = C.C_xx_bar
            C_yy_bar[i_T, i_W] = C.C_yy_bar

            # Параметры устойчивости
            stability = compute_stability_parameters(K, C)
            K_eq[i_T, i_W] = stability.K_eq
            gamma_sq[i_T, i_W] = stability.gamma_sq
            omega_st[i_T, i_W] = stability.omega_st

            # Трение (опционально)
            if compute_friction_flag:
                eta_hat = model.lubricant.eta_hat(T)
                H_T = model.material.H_T(model.geometry, T)
                texture = model.texture if with_texture else None

                H = compute_film_thickness_static(
                    grid, model.geometry, eps_0, texture, H_T
                )
                P, _, _ = solve_reynolds_static(
                    H, eta_hat,
                    grid.d_phi, grid.d_Z,
                    model.geometry.lambda_ratio ** 2,
                    omega_sor=model.numerical.omega_GS,
                    tol=model.numerical.tol_Re,
                    max_iter=model.numerical.max_iter_Re,
                )
                friction = compute_friction(P, H, grid, model, T)
                mu_f[i_T, i_W] = friction.friction_coeff
                N_f[i_T, i_W] = friction.power_loss

        except Exception as e:
            if verbose:
                print(f"\nОшибка в точке (W={W:.0f}, T={T:.0f}): {e}")
            epsilon_0[i_T, i_W] = np.nan
            K_eq[i_T, i_W] = np.nan
            gamma_sq[i_T, i_W] = np.nan
            omega_st[i_T, i_W] = np.nan

    return ParametricResults(
        W_arr=W_arr,
        T_arr=T_arr,
        W_mesh=W_mesh,
        T_mesh=T_mesh,
        epsilon_0=epsilon_0,
        K_eq=K_eq,
        gamma_sq=gamma_sq,
        omega_st=omega_st,
        mu_f=mu_f,
        N_f=N_f,
        K_xx_bar=K_xx_bar,
        K_yy_bar=K_yy_bar,
        C_xx_bar=C_xx_bar,
        C_yy_bar=C_yy_bar,
    )


def plot_3d_surfaces(
    results_smooth: ParametricResults,
    results_textured: ParametricResults,
    save_dir: str = "results",
) -> None:
    """
    Построение 3D-поверхностей K_eq(W,T), γ²_st(W,T), ω_st(W,T).

    Args:
        results_smooth: результаты для гладкого подшипника
        results_textured: результаты для текстурированного подшипника
        save_dir: директория для сохранения графиков
    """
    os.makedirs(save_dir, exist_ok=True)

    W = results_smooth.W_mesh / 1000  # кН
    T = results_smooth.T_mesh

    # Конвертируем W_mesh для правильного масштаба
    fig = plt.figure(figsize=(18, 12))

    # --- K_eq ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot_surface(W, T, results_smooth.K_eq, alpha=0.7, cmap='Blues', label='Гладкий')
    ax1.plot_surface(W, T, results_textured.K_eq, alpha=0.7, cmap='Reds', label='Текстурир.')
    ax1.set_xlabel('W, кН')
    ax1.set_ylabel('T, °C')
    ax1.set_zlabel('K_eq')
    ax1.set_title('Эквивалентная жёсткость K_eq')

    # --- γ²_st ---
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    # Ограничиваем γ² для визуализации
    gamma_smooth = np.clip(results_smooth.gamma_sq, -10, 10)
    gamma_textured = np.clip(results_textured.gamma_sq, -10, 10)
    ax2.plot_surface(W, T, gamma_smooth, alpha=0.7, cmap='Blues')
    ax2.plot_surface(W, T, gamma_textured, alpha=0.7, cmap='Reds')
    ax2.set_xlabel('W, кН')
    ax2.set_ylabel('T, °C')
    ax2.set_zlabel('γ²_st')
    ax2.set_title('Квадрат логарифм. декремента γ²_st')

    # --- ω_st ---
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    # Ограничиваем ω_st для визуализации
    omega_smooth = np.clip(results_smooth.omega_st, 0, 100)
    omega_textured = np.clip(results_textured.omega_st, 0, 100)
    ax3.plot_surface(W, T, omega_smooth, alpha=0.7, cmap='Blues')
    ax3.plot_surface(W, T, omega_textured, alpha=0.7, cmap='Reds')
    ax3.set_xlabel('W, кН')
    ax3.set_ylabel('T, °C')
    ax3.set_zlabel('ω_st')
    ax3.set_title('Критическая скорость ω_st')

    # --- ε₀ ---
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.plot_surface(W, T, results_smooth.epsilon_0, alpha=0.7, cmap='Blues')
    ax4.plot_surface(W, T, results_textured.epsilon_0, alpha=0.7, cmap='Reds')
    ax4.set_xlabel('W, кН')
    ax4.set_ylabel('T, °C')
    ax4.set_zlabel('ε₀')
    ax4.set_title('Статический эксцентриситет ε₀')

    # --- μ_f ---
    if results_smooth.mu_f is not None:
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        ax5.plot_surface(W, T, results_smooth.mu_f, alpha=0.7, cmap='Blues')
        ax5.plot_surface(W, T, results_textured.mu_f, alpha=0.7, cmap='Reds')
        ax5.set_xlabel('W, кН')
        ax5.set_ylabel('T, °C')
        ax5.set_zlabel('μ_f')
        ax5.set_title('Коэффициент трения μ_f')

    # --- N_f ---
    if results_smooth.N_f is not None:
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        ax6.plot_surface(W, T, results_smooth.N_f, alpha=0.7, cmap='Blues')
        ax6.plot_surface(W, T, results_textured.N_f, alpha=0.7, cmap='Reds')
        ax6.set_xlabel('W, кН')
        ax6.set_ylabel('T, °C')
        ax6.set_zlabel('N_f, Вт')
        ax6.set_title('Мощность потерь N_f')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stability_surfaces_3d.png'), dpi=150)
    print(f"Сохранено: {save_dir}/stability_surfaces_3d.png")
    plt.close()


def plot_comparison_contours(
    results_smooth: ParametricResults,
    results_textured: ParametricResults,
    save_dir: str = "results",
) -> None:
    """
    Построение контурных графиков для сравнения гладкого и текстурированного.
    """
    os.makedirs(save_dir, exist_ok=True)

    W = results_smooth.W_arr / 1000  # кН
    T = results_smooth.T_arr

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # K_eq
    ax = axes[0, 0]
    c1 = ax.contourf(W, T, results_smooth.K_eq.T, levels=20, cmap='Blues', alpha=0.8)
    ax.contour(W, T, results_textured.K_eq.T, levels=20, colors='red', linewidths=0.5)
    plt.colorbar(c1, ax=ax, label='K_eq (гладкий)')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T, °C')
    ax.set_title('K_eq')

    # γ²_st
    ax = axes[0, 1]
    gamma_s = np.clip(results_smooth.gamma_sq.T, -5, 5)
    gamma_t = np.clip(results_textured.gamma_sq.T, -5, 5)
    c2 = ax.contourf(W, T, gamma_s, levels=20, cmap='Blues', alpha=0.8)
    ax.contour(W, T, gamma_t, levels=20, colors='red', linewidths=0.5)
    plt.colorbar(c2, ax=ax, label='γ²_st (гладкий)')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T, °C')
    ax.set_title('γ²_st')

    # ω_st
    ax = axes[0, 2]
    omega_s = np.clip(results_smooth.omega_st.T, 0, 50)
    omega_t = np.clip(results_textured.omega_st.T, 0, 50)
    c3 = ax.contourf(W, T, omega_s, levels=20, cmap='Blues', alpha=0.8)
    ax.contour(W, T, omega_t, levels=20, colors='red', linewidths=0.5)
    plt.colorbar(c3, ax=ax, label='ω_st (гладкий)')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T, °C')
    ax.set_title('ω_st')

    # ε₀
    ax = axes[1, 0]
    c4 = ax.contourf(W, T, results_smooth.epsilon_0.T, levels=20, cmap='Blues', alpha=0.8)
    ax.contour(W, T, results_textured.epsilon_0.T, levels=20, colors='red', linewidths=0.5)
    plt.colorbar(c4, ax=ax, label='ε₀ (гладкий)')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T, °C')
    ax.set_title('ε₀')

    # μ_f
    if results_smooth.mu_f is not None:
        ax = axes[1, 1]
        c5 = ax.contourf(W, T, results_smooth.mu_f.T, levels=20, cmap='Blues', alpha=0.8)
        ax.contour(W, T, results_textured.mu_f.T, levels=20, colors='red', linewidths=0.5)
        plt.colorbar(c5, ax=ax, label='μ_f (гладкий)')
        ax.set_xlabel('W, кН')
        ax.set_ylabel('T, °C')
        ax.set_title('μ_f')

    # Разница (текстурированный - гладкий)
    ax = axes[1, 2]
    delta_K_eq = results_textured.K_eq - results_smooth.K_eq
    c6 = ax.contourf(W, T, delta_K_eq.T, levels=20, cmap='RdBu_r', alpha=0.8)
    plt.colorbar(c6, ax=ax, label='ΔK_eq')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T, °C')
    ax.set_title('ΔK_eq (текстурир. - гладкий)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_contours.png'), dpi=150)
    print(f"Сохранено: {save_dir}/comparison_contours.png")
    plt.close()


def run_full_analysis(
    model: Optional[BearingModel] = None,
    N_W: int = 10,
    N_T: int = 8,
    save_dir: str = "results",
) -> Tuple[ParametricResults, ParametricResults]:
    """
    Полный анализ: расчёт для гладкого и текстурированного подшипника.

    Args:
        model: модель подшипника (если None — китайская статья)
        N_W: число точек по нагрузке
        N_T: число точек по температуре
        save_dir: директория для результатов

    Returns:
        (results_smooth, results_textured)
    """
    if model is None:
        model = create_chinese_paper_bearing()

    print("=" * 60)
    print("ПАРАМЕТРИЧЕСКИЙ АНАЛИЗ ПОДШИПНИКА")
    print("=" * 60)
    print(model.info())

    # Расчёт для гладкого подшипника
    print("\n--- Гладкий подшипник ---")
    results_smooth = run_parametric_calculation(
        model, with_texture=False, N_W=N_W, N_T=N_T
    )

    # Расчёт для текстурированного подшипника
    print("\n--- Текстурированный подшипник ---")
    results_textured = run_parametric_calculation(
        model, with_texture=True, N_W=N_W, N_T=N_T
    )

    # Построение графиков
    print("\nПостроение графиков...")
    plot_3d_surfaces(results_smooth, results_textured, save_dir)
    plot_comparison_contours(results_smooth, results_textured, save_dir)

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)

    return results_smooth, results_textured


if __name__ == "__main__":
    # Запуск полного анализа
    results_smooth, results_textured = run_full_analysis(
        N_W=8,
        N_T=6,
        save_dir="results"
    )
