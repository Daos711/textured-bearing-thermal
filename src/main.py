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
from joblib import Parallel, delayed
import copy

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
    # THD версии функций
    integrate_forces_thd,
    find_equilibrium_eccentricity_thd,
    compute_stiffness_coefficients_thd,
    compute_damping_coefficients_thd,
    compute_all_coefficients_thd,
    THDForceResult,
)
from .stability import compute_stability_parameters, StabilityParameters
from .reynolds import solve_reynolds_static
from .geometry import compute_film_thickness_static
from .thermal import THDSolver, estimate_temperature_rise


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


# Допустимые режимы расчёта
# "isothermal" - изотермический (постоянная вязкость η(T_inlet))
# "thd_mean"   - THD с осреднённой температурой (быстрый приближённый режим)
# "thd_full"   - полный THD с полем вязкости η(φ,z) (самый точный)
THD_MODES = ("isothermal", "thd_mean", "thd_full")


@dataclass
class ParametricResults:
    """Результаты параметрического расчёта."""
    W_arr: np.ndarray               # Массив нагрузок
    T_arr: np.ndarray               # Массив температур (входных T_inlet или расчётных T_mean)
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
    # THD результаты
    T_mean: Optional[np.ndarray] = None   # Средняя температура из THD
    T_max: Optional[np.ndarray] = None    # Максимальная температура из THD
    # Поля η(φ,z) (только для thd_full, опционально)
    eta_field_sample: Optional[np.ndarray] = None  # Пример поля вязкости
    # Режим расчёта
    mode: str = "isothermal"              # Режим: isothermal, thd_mean, thd_full
    use_thd: bool = False                 # Использовался ли THD-расчёт (для обратной совместимости)


@dataclass
class SinglePointResult:
    """Результат расчёта одной точки для параллельной обработки."""
    i_T: int
    i_W: int
    epsilon_0: float
    K_xx_bar: float
    K_yy_bar: float
    C_xx_bar: float
    C_yy_bar: float
    K_eq: float
    gamma_sq: float
    omega_st: float
    mu_f: Optional[float] = None
    N_f: Optional[float] = None
    T_mean: Optional[float] = None
    T_max: Optional[float] = None
    error: Optional[str] = None


def _compute_single_point(
    i_T: int,
    i_W: int,
    T_inlet: float,
    W: float,
    model: BearingModel,
    grid: Grid,
    with_texture: bool,
    mode: str,
    compute_friction_flag: bool,
) -> SinglePointResult:
    """
    Вычисление одной точки (W, T) для параллельной обработки.

    THDSolver создаётся внутри функции для каждого процесса.
    """
    use_thd_any = mode in ("thd_mean", "thd_full")
    thd_solver = THDSolver(model, grid) if use_thd_any else None

    try:
        # ================================================================
        # РЕЖИМ: isothermal
        # ================================================================
        if mode == "isothermal":
            eps_0, forces = find_equilibrium_eccentricity(
                W, T_inlet, model, grid, with_texture=with_texture
            )
            T_effective = T_inlet
            T_mean_val = None
            T_max_val = None

            K = compute_stiffness_coefficients(eps_0, T_effective, model, grid, with_texture)
            C = compute_damping_coefficients(eps_0, T_effective, model, grid, with_texture)

        # ================================================================
        # РЕЖИМ: thd_mean
        # ================================================================
        elif mode == "thd_mean":
            eps_0, forces = find_equilibrium_eccentricity(
                W, T_inlet, model, grid, with_texture=with_texture
            )

            H_T = model.material.H_T(model.geometry, T_inlet)
            texture = model.texture if with_texture else None
            H = compute_film_thickness_static(
                grid, model.geometry, eps_0, texture, H_T
            )

            original_T_inlet = model.operating.T_inlet
            model.operating.T_inlet = T_inlet

            P_thd, T_field, eta_field, thd_converged = thd_solver.solve(
                H, max_iter=15, tol=0.5, verbose=False
            )

            model.operating.T_inlet = original_T_inlet

            T_mean_val = float(np.mean(T_field))
            T_max_val = float(np.max(T_field))
            T_effective = T_mean_val

            K = compute_stiffness_coefficients(eps_0, T_effective, model, grid, with_texture)
            C = compute_damping_coefficients(eps_0, T_effective, model, grid, with_texture)

        # ================================================================
        # РЕЖИМ: thd_full
        # ================================================================
        elif mode == "thd_full":
            eps_0, thd_result = find_equilibrium_eccentricity_thd(
                W, T_inlet, model, grid, thd_solver,
                with_texture=with_texture,
                thd_max_iter=15, thd_tol=0.5
            )

            T_mean_val = thd_result.T_mean
            T_max_val = thd_result.T_max
            T_effective = thd_result.T_mean

            K = compute_stiffness_coefficients_thd(
                eps_0, T_inlet, model, grid, thd_solver,
                with_texture=with_texture,
                thd_max_iter=10, thd_tol=1.0
            )
            C = compute_damping_coefficients_thd(
                eps_0, T_inlet, model, grid, thd_solver,
                with_texture=with_texture,
                thd_max_iter=10, thd_tol=1.0
            )
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

        # Параметры устойчивости
        stability = compute_stability_parameters(K, C)

        # Трение (опционально)
        mu_f_val = None
        N_f_val = None
        if compute_friction_flag:
            eta_hat = model.lubricant.eta_hat(T_effective)
            H_T = model.material.H_T(model.geometry, T_effective)
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

            friction = compute_friction(P, H, grid, model, T_effective)
            mu_f_val = friction.friction_coeff
            N_f_val = friction.power_loss

        return SinglePointResult(
            i_T=i_T,
            i_W=i_W,
            epsilon_0=eps_0,
            K_xx_bar=K.K_xx_bar,
            K_yy_bar=K.K_yy_bar,
            C_xx_bar=C.C_xx_bar,
            C_yy_bar=C.C_yy_bar,
            K_eq=stability.K_eq,
            gamma_sq=stability.gamma_sq,
            omega_st=stability.omega_st,
            mu_f=mu_f_val,
            N_f=N_f_val,
            T_mean=T_mean_val,
            T_max=T_max_val,
        )

    except Exception as e:
        return SinglePointResult(
            i_T=i_T,
            i_W=i_W,
            epsilon_0=np.nan,
            K_xx_bar=np.nan,
            K_yy_bar=np.nan,
            C_xx_bar=np.nan,
            C_yy_bar=np.nan,
            K_eq=np.nan,
            gamma_sq=np.nan,
            omega_st=np.nan,
            T_mean=np.nan if use_thd_any else None,
            T_max=np.nan if use_thd_any else None,
            error=str(e),
        )


def run_parametric_calculation(
    model: BearingModel,
    with_texture: bool = True,
    N_W: Optional[int] = None,
    N_T: Optional[int] = None,
    compute_friction_flag: bool = True,
    mode: str = "isothermal",
    use_thd: bool = False,  # Deprecated: используйте mode="thd_mean"
    verbose: bool = True,
    n_jobs: int = 1,  # Число параллельных процессов (-1 = все ядра)
) -> ParametricResults:
    """
    Выполнить параметрический расчёт по нагрузке и температуре.

    Args:
        model: модель подшипника
        with_texture: учитывать текстуру
        N_W: число точек по нагрузке (если None — из model.operating)
        N_T: число точек по температуре
        compute_friction_flag: вычислять трение
        mode: режим расчёта:
              - "isothermal": изотермический (постоянная вязкость η(T_inlet))
              - "thd_mean": THD с осреднённой температурой (быстрый режим)
              - "thd_full": полный THD с полем вязкости η(φ,z)
        use_thd: DEPRECATED - для обратной совместимости; эквивалентно mode="thd_mean"
        verbose: выводить прогресс
        n_jobs: число параллельных процессов (1 = последовательно, -1 = все ядра)

    Returns:
        ParametricResults
    """
    # Обратная совместимость: если use_thd=True, но mode не задан
    if use_thd and mode == "isothermal":
        mode = "thd_mean"

    # Проверка режима
    if mode not in THD_MODES:
        raise ValueError(f"Неизвестный режим '{mode}'. Допустимые: {THD_MODES}")

    use_thd_any = mode in ("thd_mean", "thd_full")

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

    # THD результаты
    T_mean_arr = np.zeros((N_T, N_W)) if use_thd_any else None
    T_max_arr = np.zeros((N_T, N_W)) if use_thd_any else None

    if compute_friction_flag:
        mu_f = np.zeros((N_T, N_W))
        N_f = np.zeros((N_T, N_W))
    else:
        mu_f = None
        N_f = None

    # Расчётная сетка
    grid = create_grid(model.numerical.N_phi, model.numerical.N_Z)

    # Названия режимов для вывода
    mode_names = {
        "isothermal": "Изотермический",
        "thd_mean": "THD (осреднённая температура)",
        "thd_full": "THD (полное поле вязкости η(φ,z))",
    }

    total = N_T * N_W

    if verbose:
        print(f"Параметрический расчёт: {N_T} x {N_W} = {total} точек")
        print(f"Текстура: {'Да' if with_texture else 'Нет'}")
        print(f"Режим: {mode_names[mode]}")
        if with_texture:
            n_cells = get_texture_centers_count(model.texture, model.geometry)
            print(f"  Число углублений: {n_cells}")
        if use_thd_any:
            delta_T_est = estimate_temperature_rise(model)
            print(f"  Оценка подъёма температуры: ΔT ≈ {delta_T_est:.1f}°C")
        if n_jobs != 1:
            actual_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
            print(f"  Параллельных процессов: {actual_jobs}")

    # Подготовка списка задач
    tasks = []
    for idx in range(total):
        i_T = idx // N_W
        i_W = idx % N_W
        tasks.append((i_T, i_W, T_arr[i_T], W_arr[i_W]))

    # ========================================================================
    # ПАРАЛЛЕЛЬНЫЙ РАСЧЁТ
    # ========================================================================
    if n_jobs != 1:
        # Создаём копию модели для каждого процесса (для безопасности)
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(_compute_single_point)(
                i_T, i_W, T_inlet, W,
                copy.deepcopy(model), grid,
                with_texture, mode, compute_friction_flag
            )
            for i_T, i_W, T_inlet, W in tqdm(tasks, desc="Расчёт", disable=not verbose)
        )

        # Заполняем массивы результатами
        for res in results:
            i_T, i_W = res.i_T, res.i_W
            epsilon_0[i_T, i_W] = res.epsilon_0
            K_xx_bar[i_T, i_W] = res.K_xx_bar
            K_yy_bar[i_T, i_W] = res.K_yy_bar
            C_xx_bar[i_T, i_W] = res.C_xx_bar
            C_yy_bar[i_T, i_W] = res.C_yy_bar
            K_eq[i_T, i_W] = res.K_eq
            gamma_sq[i_T, i_W] = res.gamma_sq
            omega_st[i_T, i_W] = res.omega_st

            if compute_friction_flag and res.mu_f is not None:
                mu_f[i_T, i_W] = res.mu_f
                N_f[i_T, i_W] = res.N_f

            if use_thd_any and res.T_mean is not None:
                T_mean_arr[i_T, i_W] = res.T_mean
                T_max_arr[i_T, i_W] = res.T_max

            if res.error and verbose:
                print(f"\nОшибка в точке (W={W_arr[i_W]:.0f}, T={T_arr[i_T]:.0f}): {res.error}")

    # ========================================================================
    # ПОСЛЕДОВАТЕЛЬНЫЙ РАСЧЁТ (n_jobs=1)
    # ========================================================================
    else:
        # THD solver (создаём один раз)
        thd_solver = THDSolver(model, grid) if use_thd_any else None

        iterator = tqdm(tasks, desc="Расчёт", disable=not verbose)

        for i_T, i_W, T_inlet, W in iterator:
            try:
                # ============================================================
                # РЕЖИМ: isothermal
                # ============================================================
                if mode == "isothermal":
                    eps_0, forces = find_equilibrium_eccentricity(
                        W, T_inlet, model, grid, with_texture=with_texture
                    )
                    epsilon_0[i_T, i_W] = eps_0
                    T_effective = T_inlet

                    K = compute_stiffness_coefficients(eps_0, T_effective, model, grid, with_texture)
                    C = compute_damping_coefficients(eps_0, T_effective, model, grid, with_texture)

                # ============================================================
                # РЕЖИМ: thd_mean
                # ============================================================
                elif mode == "thd_mean":
                    eps_0, forces = find_equilibrium_eccentricity(
                        W, T_inlet, model, grid, with_texture=with_texture
                    )
                    epsilon_0[i_T, i_W] = eps_0

                    H_T = model.material.H_T(model.geometry, T_inlet)
                    texture = model.texture if with_texture else None
                    H = compute_film_thickness_static(
                        grid, model.geometry, eps_0, texture, H_T
                    )

                    original_T_inlet = model.operating.T_inlet
                    model.operating.T_inlet = T_inlet

                    P_thd, T_field, eta_field, thd_converged = thd_solver.solve(
                        H, max_iter=15, tol=0.5, verbose=False
                    )

                    model.operating.T_inlet = original_T_inlet

                    T_mean_val = np.mean(T_field)
                    T_max_val = np.max(T_field)
                    T_mean_arr[i_T, i_W] = T_mean_val
                    T_max_arr[i_T, i_W] = T_max_val

                    T_effective = T_mean_val
                    K = compute_stiffness_coefficients(eps_0, T_effective, model, grid, with_texture)
                    C = compute_damping_coefficients(eps_0, T_effective, model, grid, with_texture)

                # ============================================================
                # РЕЖИМ: thd_full
                # ============================================================
                elif mode == "thd_full":
                    eps_0, thd_result = find_equilibrium_eccentricity_thd(
                        W, T_inlet, model, grid, thd_solver,
                        with_texture=with_texture,
                        thd_max_iter=15, thd_tol=0.5
                    )
                    epsilon_0[i_T, i_W] = eps_0

                    T_mean_arr[i_T, i_W] = thd_result.T_mean
                    T_max_arr[i_T, i_W] = thd_result.T_max
                    T_effective = thd_result.T_mean

                    K = compute_stiffness_coefficients_thd(
                        eps_0, T_inlet, model, grid, thd_solver,
                        with_texture=with_texture,
                        thd_max_iter=10, thd_tol=1.0
                    )
                    C = compute_damping_coefficients_thd(
                        eps_0, T_inlet, model, grid, thd_solver,
                        with_texture=with_texture,
                        thd_max_iter=10, thd_tol=1.0
                    )

                # Сохраняем коэффициенты
                K_xx_bar[i_T, i_W] = K.K_xx_bar
                K_yy_bar[i_T, i_W] = K.K_yy_bar
                C_xx_bar[i_T, i_W] = C.C_xx_bar
                C_yy_bar[i_T, i_W] = C.C_yy_bar

                # Параметры устойчивости
                stability = compute_stability_parameters(K, C)
                K_eq[i_T, i_W] = stability.K_eq
                gamma_sq[i_T, i_W] = stability.gamma_sq
                omega_st[i_T, i_W] = stability.omega_st

                # Трение
                if compute_friction_flag:
                    eta_hat = model.lubricant.eta_hat(T_effective)
                    H_T = model.material.H_T(model.geometry, T_effective)
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

                    friction = compute_friction(P, H, grid, model, T_effective)
                    mu_f[i_T, i_W] = friction.friction_coeff
                    N_f[i_T, i_W] = friction.power_loss

            except Exception as e:
                if verbose:
                    print(f"\nОшибка в точке (W={W:.0f}, T={T_inlet:.0f}): {e}")
                epsilon_0[i_T, i_W] = np.nan
                K_eq[i_T, i_W] = np.nan
                gamma_sq[i_T, i_W] = np.nan
                omega_st[i_T, i_W] = np.nan
                if use_thd_any:
                    T_mean_arr[i_T, i_W] = np.nan
                    T_max_arr[i_T, i_W] = np.nan

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
        T_mean=T_mean_arr,
        T_max=T_max_arr,
        mode=mode,
        use_thd=use_thd_any,
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


def plot_3d_surfaces_grayscale(
    results_smooth: ParametricResults,
    results_textured: ParametricResults,
    save_dir: str = "results",
) -> None:
    """
    Построение 3D-поверхностей в градациях серого для публикаций.

    Гладкий подшипник: светло-серая заливка с чёрными линиями
    Текстурированный: тёмно-серая заливка с белыми линиями (или штриховка)

    Args:
        results_smooth: результаты для гладкого подшипника
        results_textured: результаты для текстурированного подшипника
        save_dir: директория для сохранения графиков
    """
    from matplotlib.colors import LinearSegmentedColormap

    os.makedirs(save_dir, exist_ok=True)

    W = results_smooth.W_mesh / 1000  # кН
    T = results_smooth.T_mesh

    # Создаём оттенки серого colormaps
    # Гладкий: светло-серые оттенки (от белого к светло-серому)
    cmap_smooth = LinearSegmentedColormap.from_list('smooth_gray', ['#FFFFFF', '#A0A0A0'])
    # Текстурированный: тёмно-серые оттенки (от серого к тёмно-серому)
    cmap_textured = LinearSegmentedColormap.from_list('textured_gray', ['#808080', '#303030'])

    fig = plt.figure(figsize=(18, 12))

    # --- K_eq ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot_surface(W, T, results_smooth.K_eq, alpha=0.8, cmap=cmap_smooth,
                     edgecolor='black', linewidth=0.3)
    ax1.plot_surface(W, T, results_textured.K_eq, alpha=0.6, cmap=cmap_textured,
                     edgecolor='white', linewidth=0.3)
    ax1.set_xlabel('W, кН')
    ax1.set_ylabel('T, °C')
    ax1.set_zlabel('K_eq')
    ax1.set_title('Эквивалентная жёсткость K_eq')

    # --- γ²_st ---
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    gamma_smooth = np.clip(results_smooth.gamma_sq, -10, 10)
    gamma_textured = np.clip(results_textured.gamma_sq, -10, 10)
    ax2.plot_surface(W, T, gamma_smooth, alpha=0.8, cmap=cmap_smooth,
                     edgecolor='black', linewidth=0.3)
    ax2.plot_surface(W, T, gamma_textured, alpha=0.6, cmap=cmap_textured,
                     edgecolor='white', linewidth=0.3)
    ax2.set_xlabel('W, кН')
    ax2.set_ylabel('T, °C')
    ax2.set_zlabel('γ²_st')
    ax2.set_title('Квадрат логарифм. декремента γ²_st')

    # --- ω_st ---
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    omega_smooth = np.clip(results_smooth.omega_st, 0, 100)
    omega_textured = np.clip(results_textured.omega_st, 0, 100)
    ax3.plot_surface(W, T, omega_smooth, alpha=0.8, cmap=cmap_smooth,
                     edgecolor='black', linewidth=0.3)
    ax3.plot_surface(W, T, omega_textured, alpha=0.6, cmap=cmap_textured,
                     edgecolor='white', linewidth=0.3)
    ax3.set_xlabel('W, кН')
    ax3.set_ylabel('T, °C')
    ax3.set_zlabel('ω_st')
    ax3.set_title('Критическая скорость ω_st')

    # --- ε₀ ---
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.plot_surface(W, T, results_smooth.epsilon_0, alpha=0.8, cmap=cmap_smooth,
                     edgecolor='black', linewidth=0.3)
    ax4.plot_surface(W, T, results_textured.epsilon_0, alpha=0.6, cmap=cmap_textured,
                     edgecolor='white', linewidth=0.3)
    ax4.set_xlabel('W, кН')
    ax4.set_ylabel('T, °C')
    ax4.set_zlabel('ε₀')
    ax4.set_title('Статический эксцентриситет ε₀')

    # --- μ_f ---
    if results_smooth.mu_f is not None:
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        ax5.plot_surface(W, T, results_smooth.mu_f, alpha=0.8, cmap=cmap_smooth,
                         edgecolor='black', linewidth=0.3)
        ax5.plot_surface(W, T, results_textured.mu_f, alpha=0.6, cmap=cmap_textured,
                         edgecolor='white', linewidth=0.3)
        ax5.set_xlabel('W, кН')
        ax5.set_ylabel('T, °C')
        ax5.set_zlabel('μ_f')
        ax5.set_title('Коэффициент трения μ_f')

    # --- N_f ---
    if results_smooth.N_f is not None:
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        ax6.plot_surface(W, T, results_smooth.N_f, alpha=0.8, cmap=cmap_smooth,
                         edgecolor='black', linewidth=0.3)
        ax6.plot_surface(W, T, results_textured.N_f, alpha=0.6, cmap=cmap_textured,
                         edgecolor='white', linewidth=0.3)
        ax6.set_xlabel('W, кН')
        ax6.set_ylabel('T, °C')
        ax6.set_zlabel('N_f, Вт')
        ax6.set_title('Мощность потерь N_f')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stability_surfaces_3d_grayscale.png'), dpi=300)
    print(f"Сохранено: {save_dir}/stability_surfaces_3d_grayscale.png")
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
    # Данные имеют форму (N_T, N_W), что соответствует (len(T), len(W)) для contourf
    ax = axes[0, 0]
    c1 = ax.contourf(W, T, results_smooth.K_eq, levels=20, cmap='Blues', alpha=0.8)
    ax.contour(W, T, results_textured.K_eq, levels=20, colors='red', linewidths=0.5)
    plt.colorbar(c1, ax=ax, label='K_eq (гладкий)')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T, °C')
    ax.set_title('K_eq')

    # γ²_st
    ax = axes[0, 1]
    gamma_s = np.clip(results_smooth.gamma_sq, -5, 5)
    gamma_t = np.clip(results_textured.gamma_sq, -5, 5)
    c2 = ax.contourf(W, T, gamma_s, levels=20, cmap='Blues', alpha=0.8)
    ax.contour(W, T, gamma_t, levels=20, colors='red', linewidths=0.5)
    plt.colorbar(c2, ax=ax, label='γ²_st (гладкий)')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T, °C')
    ax.set_title('γ²_st')

    # ω_st
    ax = axes[0, 2]
    omega_s = np.clip(results_smooth.omega_st, 0, 50)
    omega_t = np.clip(results_textured.omega_st, 0, 50)
    c3 = ax.contourf(W, T, omega_s, levels=20, cmap='Blues', alpha=0.8)
    ax.contour(W, T, omega_t, levels=20, colors='red', linewidths=0.5)
    plt.colorbar(c3, ax=ax, label='ω_st (гладкий)')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T, °C')
    ax.set_title('ω_st')

    # ε₀
    ax = axes[1, 0]
    c4 = ax.contourf(W, T, results_smooth.epsilon_0, levels=20, cmap='Blues', alpha=0.8)
    ax.contour(W, T, results_textured.epsilon_0, levels=20, colors='red', linewidths=0.5)
    plt.colorbar(c4, ax=ax, label='ε₀ (гладкий)')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T, °C')
    ax.set_title('ε₀')

    # μ_f
    if results_smooth.mu_f is not None:
        ax = axes[1, 1]
        c5 = ax.contourf(W, T, results_smooth.mu_f, levels=20, cmap='Blues', alpha=0.8)
        ax.contour(W, T, results_textured.mu_f, levels=20, colors='red', linewidths=0.5)
        plt.colorbar(c5, ax=ax, label='μ_f (гладкий)')
        ax.set_xlabel('W, кН')
        ax.set_ylabel('T, °C')
        ax.set_title('μ_f')

    # Разница (текстурированный - гладкий)
    ax = axes[1, 2]
    delta_K_eq = results_textured.K_eq - results_smooth.K_eq
    c6 = ax.contourf(W, T, delta_K_eq, levels=20, cmap='RdBu_r', alpha=0.8)
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
    mode: str = "isothermal",
    use_thd: bool = False,  # Deprecated
    grayscale: bool = False,
    n_jobs: int = 1,  # Число параллельных процессов (-1 = все ядра)
) -> Tuple[ParametricResults, ParametricResults]:
    """
    Полный анализ: расчёт для гладкого и текстурированного подшипника.

    Args:
        model: модель подшипника (если None — китайская статья)
        N_W: число точек по нагрузке
        N_T: число точек по температуре
        save_dir: директория для результатов
        mode: режим расчёта: "isothermal", "thd_mean", "thd_full"
        use_thd: DEPRECATED - используйте mode="thd_mean"
        grayscale: использовать градации серого для графиков (для публикаций)
        n_jobs: число параллельных процессов (1 = последовательно, -1 = все ядра)

    Returns:
        (results_smooth, results_textured)
    """
    # Обратная совместимость
    if use_thd and mode == "isothermal":
        mode = "thd_mean"

    if model is None:
        model = create_chinese_paper_bearing()

    mode_names = {
        "isothermal": "Изотермический",
        "thd_mean": "THD (осреднённая температура)",
        "thd_full": "THD (полное поле вязкости)",
    }

    print("=" * 60)
    print("ПАРАМЕТРИЧЕСКИЙ АНАЛИЗ ПОДШИПНИКА")
    print(f"Режим: {mode_names.get(mode, mode)}")
    print("=" * 60)
    print(model.info())

    # Расчёт для гладкого подшипника
    print("\n--- Гладкий подшипник ---")
    results_smooth = run_parametric_calculation(
        model, with_texture=False, N_W=N_W, N_T=N_T, mode=mode, n_jobs=n_jobs
    )

    # Расчёт для текстурированного подшипника
    print("\n--- Текстурированный подшипник ---")
    results_textured = run_parametric_calculation(
        model, with_texture=True, N_W=N_W, N_T=N_T, mode=mode, n_jobs=n_jobs
    )

    # Построение графиков
    print("\nПостроение графиков...")
    if grayscale:
        plot_3d_surfaces_grayscale(results_smooth, results_textured, save_dir)
    else:
        plot_3d_surfaces(results_smooth, results_textured, save_dir)
    plot_comparison_contours(results_smooth, results_textured, save_dir)

    # Дополнительные графики для THD режимов
    if mode in ("thd_mean", "thd_full"):
        plot_thd_results(results_smooth, results_textured, save_dir)

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)

    return results_smooth, results_textured


def plot_thd_results(
    results_smooth: ParametricResults,
    results_textured: ParametricResults,
    save_dir: str = "results",
) -> None:
    """
    Построение графиков THD результатов: T_mean, T_max.
    """
    if results_smooth.T_mean is None:
        return

    os.makedirs(save_dir, exist_ok=True)

    W = results_smooth.W_arr / 1000  # кН
    T_inlet = results_smooth.T_arr

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # T_mean гладкий
    ax = axes[0, 0]
    c = ax.contourf(W, T_inlet, results_smooth.T_mean, levels=20, cmap='hot')
    plt.colorbar(c, ax=ax, label='T_mean, °C')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T_inlet, °C')
    ax.set_title('T_mean (гладкий)')

    # T_mean текстурированный
    ax = axes[0, 1]
    c = ax.contourf(W, T_inlet, results_textured.T_mean, levels=20, cmap='hot')
    plt.colorbar(c, ax=ax, label='T_mean, °C')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T_inlet, °C')
    ax.set_title('T_mean (текстурированный)')

    # ΔT = T_mean - T_inlet
    ax = axes[1, 0]
    delta_T_smooth = results_smooth.T_mean - results_smooth.T_mesh
    c = ax.contourf(W, T_inlet, delta_T_smooth, levels=20, cmap='Reds')
    plt.colorbar(c, ax=ax, label='ΔT, °C')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T_inlet, °C')
    ax.set_title('ΔT = T_mean - T_inlet (гладкий)')

    # Разница T_mean текстурир. - гладкий
    ax = axes[1, 1]
    delta_T = results_textured.T_mean - results_smooth.T_mean
    c = ax.contourf(W, T_inlet, delta_T, levels=20, cmap='RdBu_r')
    plt.colorbar(c, ax=ax, label='ΔT, °C')
    ax.set_xlabel('W, кН')
    ax.set_ylabel('T_inlet, °C')
    ax.set_title('T_mean: текстурир. - гладкий')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'thd_temperatures.png'), dpi=150)
    print(f"Сохранено: {save_dir}/thd_temperatures.png")
    plt.close()


if __name__ == "__main__":
    import sys

    # Проверяем аргументы командной строки
    # --mode=isothermal|thd_mean|thd_full
    # --grayscale для чёрно-белых графиков
    # --thd (deprecated, эквивалент --mode=thd_mean)
    mode = "isothermal"
    grayscale = "--grayscale" in sys.argv or "--bw" in sys.argv

    for arg in sys.argv:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1]
        elif arg == "--thd":
            mode = "thd_mean"
        elif arg == "--thd-full":
            mode = "thd_full"

    # Запуск полного анализа
    results_smooth, results_textured = run_full_analysis(
        N_W=8,
        N_T=6,
        save_dir="results",
        mode=mode,
        grayscale=grayscale,
    )

    # Вывод сводки по THD
    if mode in ("thd_mean", "thd_full") and results_smooth.T_mean is not None:
        print("\n--- THD Сводка ---")
        print(f"Режим: {mode}")
        print(f"Гладкий: T_mean = {np.nanmean(results_smooth.T_mean):.1f}°C, "
              f"T_max = {np.nanmax(results_smooth.T_max):.1f}°C")
        print(f"Текстур.: T_mean = {np.nanmean(results_textured.T_mean):.1f}°C, "
              f"T_max = {np.nanmax(results_textured.T_max):.1f}°C")
