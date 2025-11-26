"""
Вычисление сил, нагрузок и коэффициентов жёсткости/демпфирования.

Система координат (из постановки задачи):
- ось x по линии минимального зазора
- ось y перпендикулярна x
- нагрузка W направлена по оси y

Безразмерные силы:
    Fx* = ∫∫ P(φ,Z)·cos(φ) dφ dZ
    Fy* = ∫∫ P(φ,Z)·sin(φ) dφ dZ

Размерный масштаб:
    F_scale = p₀·R_J·L = 6η_ref·U·R_J²·L / c₀²

Коэффициенты жёсткости (конечные разности):
    K_xx = -(F_x⁺ - F_x⁻) / (2Δx)
    K_yx = -(F_y⁺ - F_y⁻) / (2Δx)
    где Δx = Δξ·c₀

Коэффициенты демпфирования (через squeeze-член):
    C_xx = -(F_x^{+ẋ} - F_x^{-ẋ}) / (2Δẋ)
    где Δẋ = Δξ'·c₀·ω

Безразмерные масштабы:
    K_scale = η·ω·L / ψ³
    C_scale = η·L / ψ³
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import brentq

from .geometry import (
    Grid, create_grid,
    compute_film_thickness_static,
    compute_film_thickness_full,
    compute_dH_dtau,
)
from .reynolds import ReynoldsSolver, solve_reynolds_static, solve_reynolds_with_squeeze, solve_reynolds_equation
from .parameters import BearingModel, BearingGeometry, TextureParameters

# Импорт THD компонентов будет выполняться локально для избежания циклических импортов


@dataclass
class ForceResult:
    """Результат вычисления сил."""
    Fx: float               # Сила по x (Н) — размерная
    Fy: float               # Сила по y (Н) — размерная
    Fx_star: float          # Безразмерная сила по x
    Fy_star: float          # Безразмерная сила по y


@dataclass
class FrictionResult:
    """Результат вычисления трения."""
    friction_force: float   # Сила трения (Н)
    friction_coeff: float   # Коэффициент трения μ_f
    power_loss: float       # Мощность потерь N_f (Вт)


@dataclass
class StiffnessCoefficients:
    """Коэффициенты жёсткости."""
    K_xx: float     # Н/м
    K_xy: float
    K_yx: float
    K_yy: float
    # Безразмерные
    K_xx_bar: float
    K_xy_bar: float
    K_yx_bar: float
    K_yy_bar: float


@dataclass
class DampingCoefficients:
    """Коэффициенты демпфирования."""
    C_xx: float     # Н·с/м
    C_xy: float
    C_yx: float
    C_yy: float
    # Безразмерные
    C_xx_bar: float
    C_xy_bar: float
    C_yx_bar: float
    C_yy_bar: float


def integrate_forces(
    P: np.ndarray,
    grid: Grid,
    force_scale: float,
) -> ForceResult:
    """
    Интегрирование давления для получения сил.

    Fx* = ∫_{-1}^{1} ∫_0^{2π} P(φ,Z)·cos(φ) dφ dZ
    Fy* = ∫_{-1}^{1} ∫_0^{2π} P(φ,Z)·sin(φ) dφ dZ

    Args:
        P: безразмерное давление (N_Z x N_phi)
        grid: расчётная сетка
        force_scale: F_scale = p₀·R_J·L

    Returns:
        ForceResult с размерными и безразмерными силами
    """
    # Безразмерные интегралы
    Fx_star = np.trapz(np.trapz(P * grid.cos_phi, grid.phi, axis=1), grid.Z)
    Fy_star = np.trapz(np.trapz(P * grid.sin_phi, grid.phi, axis=1), grid.Z)

    # Размерные силы
    Fx = Fx_star * force_scale
    Fy = Fy_star * force_scale

    return ForceResult(
        Fx=Fx,
        Fy=Fy,
        Fx_star=Fx_star,
        Fy_star=Fy_star
    )


def compute_friction(
    P: np.ndarray,
    H: np.ndarray,
    grid: Grid,
    model: BearingModel,
    T: float,
) -> FrictionResult:
    """
    Вычисление силы трения.

    Интеграл силы трения:
        I = 1/H + 3H·∂P/∂φ
        f = ∫∫ I dφ dZ

    Коэффициент трения:
        μ_f = f / W

    Мощность потерь:
        N_f = f · U

    Args:
        P: безразмерное давление
        H: безразмерная толщина
        grid: расчётная сетка
        model: параметры подшипника
        T: температура (°C)

    Returns:
        FrictionResult
    """
    # ∂P/∂φ
    dP_dphi = np.zeros_like(P)
    for i in range(grid.N_Z):
        for j in range(grid.N_phi):
            j_plus = (j + 1) % grid.N_phi
            j_minus = (j - 1) % grid.N_phi
            dP_dphi[i, j] = (P[i, j_plus] - P[i, j_minus]) / (2 * grid.d_phi)

    # Интеграл трения (безразмерный)
    # I = 1/H + 3H·∂P/∂φ  (формула из постановки)
    integrand = 1.0 / H + 3.0 * H * dP_dphi
    f_star = np.trapz(np.trapz(integrand, grid.phi, axis=1), grid.Z)

    # Масштаб силы трения
    eta = model.lubricant.viscosity(T)
    U = model.operating.U(model.geometry.R_J)
    friction_scale = eta * U * model.geometry.R_J * model.geometry.L / model.geometry.c0

    friction_force = f_star * friction_scale

    # Для коэффициента трения нужна нагрузка — получим из сил
    force_scale = model.force_scale(T)
    forces = integrate_forces(P, grid, force_scale)
    W = np.sqrt(forces.Fx**2 + forces.Fy**2)

    friction_coeff = abs(friction_force / W) if W > 1e-12 else 0.0

    # Мощность потерь
    power_loss = abs(friction_force) * U

    return FrictionResult(
        friction_force=friction_force,
        friction_coeff=friction_coeff,
        power_loss=power_loss
    )


def find_equilibrium_eccentricity(
    W_target: float,
    T: float,
    model: BearingModel,
    grid: Grid,
    with_texture: bool = True,
    eps_min: float = 0.01,
    eps_max: float = 0.98,
    tol: float = 1e-3,
) -> Tuple[float, ForceResult]:
    """
    Поиск статического эксцентриситета ε₀(W, T) из условия равновесия.

    Условие: F_y(ε₀, T) = W, F_x(ε₀, T) ≈ 0

    Алгоритм:
    1. Задаём T → вычисляем η̂(T) и H_T(T)
    2. Методом деления отрезка ищем ε₀ такой что |F_y - W| < tol

    Args:
        W_target: целевая нагрузка (Н)
        T: температура (°C)
        model: параметры подшипника
        grid: расчётная сетка
        with_texture: учитывать текстуру
        eps_min, eps_max: диапазон поиска ε₀
        tol: допуск по нагрузке (относительный)

    Returns:
        (epsilon_0, ForceResult)
    """
    eta_hat = model.lubricant.eta_hat(T)
    H_T = model.material.H_T(model.geometry, T)
    force_scale = model.force_scale(T)
    lambda_sq = model.geometry.lambda_ratio ** 2

    texture = model.texture if with_texture else None

    def residual(epsilon_0: float) -> float:
        """Невязка: F_y - W."""
        H = compute_film_thickness_static(
            grid, model.geometry, epsilon_0, texture, H_T
        )

        P, _, _ = solve_reynolds_static(
            H, eta_hat,
            grid.d_phi, grid.d_Z, lambda_sq,
            omega_sor=model.numerical.omega_GS,
            tol=model.numerical.tol_Re,
            max_iter=model.numerical.max_iter_Re,
            use_cavitation=model.numerical.use_cavitation
        )

        forces = integrate_forces(P, grid, force_scale)
        return forces.Fy - W_target

    # Проверяем границы
    r_min = residual(eps_min)
    r_max = residual(eps_max)

    # Если знаки одинаковые, расширяем диапазон или возвращаем границу
    at_boundary = False
    if r_min * r_max > 0:
        if abs(r_min) < abs(r_max):
            epsilon_0 = eps_min
        else:
            epsilon_0 = eps_max
        at_boundary = True
    else:
        # Метод Брента для поиска корня
        epsilon_0 = brentq(residual, eps_min, eps_max, xtol=1e-4)
        # Проверяем близость к границе
        if epsilon_0 > eps_max - 0.01:
            at_boundary = True

    # Вычисляем силы при найденном ε₀
    H = compute_film_thickness_static(
        grid, model.geometry, epsilon_0, texture, H_T
    )
    P, _, _ = solve_reynolds_static(
        H, eta_hat,
        grid.d_phi, grid.d_Z, lambda_sq,
        omega_sor=model.numerical.omega_GS,
        tol=model.numerical.tol_Re,
        max_iter=model.numerical.max_iter_Re,
        use_cavitation=model.numerical.use_cavitation
    )
    forces = integrate_forces(P, grid, force_scale)

    return epsilon_0, forces


def compute_stiffness_coefficients(
    epsilon_0: float,
    T: float,
    model: BearingModel,
    grid: Grid,
    with_texture: bool = True,
) -> StiffnessCoefficients:
    """
    Вычисление коэффициентов жёсткости методом конечных разностей.

    Из постановки задачи:
        K_xx = -(F_x⁺ - F_x⁻) / (2Δx),  где ξ = +Δξ и ξ = -Δξ
        K_yx = -(F_y⁺ - F_y⁻) / (2Δx)
        Δx = Δξ · c₀

    Для K_xy, K_yy аналогично со смещением по η.

    Args:
        epsilon_0: статический эксцентриситет
        T: температура (°C)
        model: параметры подшипника
        grid: расчётная сетка
        with_texture: учитывать текстуру

    Returns:
        StiffnessCoefficients
    """
    delta_xi = model.numerical.delta_xi
    eta_hat = model.lubricant.eta_hat(T)
    H_T = model.material.H_T(model.geometry, T)
    force_scale = model.force_scale(T)
    lambda_sq = model.geometry.lambda_ratio ** 2
    texture = model.texture if with_texture else None

    def get_forces(xi: float, eta: float) -> Tuple[float, float]:
        """Вычислить силы при смещении (ξ, η)."""
        H = compute_film_thickness_full(
            grid, model.geometry, epsilon_0, xi, eta, texture, H_T
        )
        P, _, _ = solve_reynolds_static(
            H, eta_hat,
            grid.d_phi, grid.d_Z, lambda_sq,
            omega_sor=model.numerical.omega_GS,
            tol=model.numerical.tol_Re,
            max_iter=model.numerical.max_iter_Re,
            use_cavitation=model.numerical.use_cavitation
        )
        forces = integrate_forces(P, grid, force_scale)
        return forces.Fx, forces.Fy

    # Смещение по x (ξ)
    Fx_plus_x, Fy_plus_x = get_forces(+delta_xi, 0.0)
    Fx_minus_x, Fy_minus_x = get_forces(-delta_xi, 0.0)

    # Смещение по y (η)
    Fx_plus_y, Fy_plus_y = get_forces(0.0, +delta_xi)
    Fx_minus_y, Fy_minus_y = get_forces(0.0, -delta_xi)

    # Размерный шаг
    delta_x = delta_xi * model.geometry.c0

    # Коэффициенты жёсткости (размерные)
    K_xx = -(Fx_plus_x - Fx_minus_x) / (2 * delta_x)
    K_yx = -(Fy_plus_x - Fy_minus_x) / (2 * delta_x)
    K_xy = -(Fx_plus_y - Fx_minus_y) / (2 * delta_x)
    K_yy = -(Fy_plus_y - Fy_minus_y) / (2 * delta_x)

    # Масштаб для безразмерных коэффициентов
    K_scale = model.K_scale(T)

    return StiffnessCoefficients(
        K_xx=K_xx, K_xy=K_xy, K_yx=K_yx, K_yy=K_yy,
        K_xx_bar=K_xx / K_scale,
        K_xy_bar=K_xy / K_scale,
        K_yx_bar=K_yx / K_scale,
        K_yy_bar=K_yy / K_scale,
    )


def compute_damping_coefficients(
    epsilon_0: float,
    T: float,
    model: BearingModel,
    grid: Grid,
    with_texture: bool = True,
) -> DampingCoefficients:
    """
    Вычисление коэффициентов демпфирования через squeeze-член.

    Из постановки задачи:
        В уравнение Рейнольдса входит ∂H/∂τ = ξ'·cos(φ) + η'·sin(φ)
        где ξ' = ẋ/(c₀·ω), η' = ẏ/(c₀·ω)

        C_xx = -(F_x^{+ẋ} - F_x^{-ẋ}) / (2Δẋ)
        Δẋ = Δξ' · c₀ · ω

    Args:
        epsilon_0: статический эксцентриситет
        T: температура (°C)
        model: параметры подшипника
        grid: расчётная сетка
        with_texture: учитывать текстуру

    Returns:
        DampingCoefficients
    """
    delta_xi_dot = model.numerical.delta_xi_dot
    eta_hat = model.lubricant.eta_hat(T)
    H_T = model.material.H_T(model.geometry, T)
    force_scale = model.force_scale(T)
    lambda_sq = model.geometry.lambda_ratio ** 2
    texture = model.texture if with_texture else None

    # Базовый зазор (при статическом положении)
    H = compute_film_thickness_static(
        grid, model.geometry, epsilon_0, texture, H_T
    )

    def get_forces_squeeze(xi_dot: float, eta_dot: float) -> Tuple[float, float]:
        """Вычислить силы при скорости (ξ', η')."""
        dH_dtau = compute_dH_dtau(grid, xi_dot, eta_dot)

        P, _, _ = solve_reynolds_with_squeeze(
            H, dH_dtau, eta_hat,
            grid.d_phi, grid.d_Z, lambda_sq,
            beta=2.0,
            omega_sor=model.numerical.omega_GS,
            tol=model.numerical.tol_Re,
            max_iter=model.numerical.max_iter_Re,
            use_cavitation=model.numerical.use_cavitation
        )
        forces = integrate_forces(P, grid, force_scale)
        return forces.Fx, forces.Fy

    # Смещение скорости по x (ξ')
    Fx_plus_xdot, Fy_plus_xdot = get_forces_squeeze(+delta_xi_dot, 0.0)
    Fx_minus_xdot, Fy_minus_xdot = get_forces_squeeze(-delta_xi_dot, 0.0)

    # Смещение скорости по y (η')
    Fx_plus_ydot, Fy_plus_ydot = get_forces_squeeze(0.0, +delta_xi_dot)
    Fx_minus_ydot, Fy_minus_ydot = get_forces_squeeze(0.0, -delta_xi_dot)

    # Размерный шаг скорости
    delta_x_dot = delta_xi_dot * model.geometry.c0 * model.operating.omega

    # Коэффициенты демпфирования (размерные)
    C_xx = -(Fx_plus_xdot - Fx_minus_xdot) / (2 * delta_x_dot)
    C_yx = -(Fy_plus_xdot - Fy_minus_xdot) / (2 * delta_x_dot)
    C_xy = -(Fx_plus_ydot - Fx_minus_ydot) / (2 * delta_x_dot)
    C_yy = -(Fy_plus_ydot - Fy_minus_ydot) / (2 * delta_x_dot)

    # Масштаб для безразмерных коэффициентов
    C_scale = model.C_scale(T)

    return DampingCoefficients(
        C_xx=C_xx, C_xy=C_xy, C_yx=C_yx, C_yy=C_yy,
        C_xx_bar=C_xx / C_scale,
        C_xy_bar=C_xy / C_scale,
        C_yx_bar=C_yx / C_scale,
        C_yy_bar=C_yy / C_scale,
    )


@dataclass
class FullCoefficients:
    """Полный набор коэффициентов K и C."""
    stiffness: StiffnessCoefficients
    damping: DampingCoefficients


@dataclass
class THDForceResult:
    """Результат вычисления сил с учётом THD."""
    forces: ForceResult           # Силы
    P: np.ndarray                 # Поле давления
    T_field: np.ndarray           # Поле температуры (°C)
    eta_field: np.ndarray         # Поле вязкости (Па·с)
    T_mean: float                 # Средняя температура
    T_max: float                  # Максимальная температура
    converged: bool               # Сошлось ли THD решение


def compute_all_coefficients(
    epsilon_0: float,
    T: float,
    model: BearingModel,
    grid: Grid,
    with_texture: bool = True,
) -> FullCoefficients:
    """
    Вычисление всех 8 коэффициентов K_ij и C_ij.

    Args:
        epsilon_0: статический эксцентриситет
        T: температура (°C)
        model: параметры подшипника
        grid: расчётная сетка
        with_texture: учитывать текстуру

    Returns:
        FullCoefficients
    """
    stiffness = compute_stiffness_coefficients(
        epsilon_0, T, model, grid, with_texture
    )
    damping = compute_damping_coefficients(
        epsilon_0, T, model, grid, with_texture
    )

    return FullCoefficients(stiffness=stiffness, damping=damping)


# =============================================================================
# THD (термогидродинамические) версии функций с полем вязкости η(φ,z)
# =============================================================================

def integrate_forces_thd(
    epsilon_0: float,
    T_inlet: float,
    model: BearingModel,
    grid: Grid,
    thd_solver,
    with_texture: bool = True,
    max_iter: int = 15,
    tol: float = 0.5,
) -> THDForceResult:
    """
    Интегрирование сил с учётом термогидродинамики (переменная вязкость η(φ,z)).

    Алгоритм:
    1. Вычислить H_T(T_inlet), применить текстуру → H(φ,z)
    2. Решить THD: P_thd, T_field, eta_field = thd_solver.solve(H, ...)
    3. Интегрировать силы: forces = integrate_forces(P_thd, grid, model)

    Args:
        epsilon_0: эксцентриситет
        T_inlet: входная температура (°C)
        model: параметры подшипника
        grid: расчётная сетка
        thd_solver: экземпляр THDSolver
        with_texture: учитывать текстуру
        max_iter: максимум итераций THD
        tol: допуск сходимости THD

    Returns:
        THDForceResult с полями давления, температуры, вязкости и силами
    """
    # Подготовка
    H_T = model.material.H_T(model.geometry, T_inlet)
    texture = model.texture if with_texture else None
    force_scale = model.force_scale(T_inlet)

    # Зазор с текстурой
    H = compute_film_thickness_static(
        grid, model.geometry, epsilon_0, texture, H_T
    )

    # Временно меняем T_inlet в модели для THD solver
    original_T_inlet = model.operating.T_inlet
    model.operating.T_inlet = T_inlet

    # Решаем связанную THD задачу
    P_thd, T_field, eta_field, converged = thd_solver.solve(
        H, max_iter=max_iter, tol=tol, verbose=False
    )

    # Восстанавливаем
    model.operating.T_inlet = original_T_inlet

    # Интегрируем силы
    forces = integrate_forces(P_thd, grid, force_scale)

    return THDForceResult(
        forces=forces,
        P=P_thd,
        T_field=T_field,
        eta_field=eta_field,
        T_mean=np.mean(T_field),
        T_max=np.max(T_field),
        converged=converged,
    )


def find_equilibrium_eccentricity_thd(
    W_target: float,
    T_inlet: float,
    model: BearingModel,
    grid: Grid,
    thd_solver,
    with_texture: bool = True,
    eps_min: float = 0.01,
    eps_max: float = 0.98,
    tol: float = 1e-3,
    thd_max_iter: int = 15,
    thd_tol: float = 0.5,
) -> Tuple[float, THDForceResult]:
    """
    Поиск статического эксцентриситета ε₀(W, T_inlet) с полным THD расчётом.

    В отличие от find_equilibrium_eccentricity, в цикле подбора ε₀
    вызывается integrate_forces_thd для учёта переменной вязкости η(φ,z).

    Args:
        W_target: целевая нагрузка (Н)
        T_inlet: входная температура (°C)
        model: параметры подшипника
        grid: расчётная сетка
        thd_solver: экземпляр THDSolver
        with_texture: учитывать текстуру
        eps_min, eps_max: диапазон поиска ε₀
        tol: допуск по нагрузке (относительный)
        thd_max_iter: максимум итераций THD на каждом шаге
        thd_tol: допуск сходимости THD

    Returns:
        (epsilon_0, THDForceResult)
    """
    # Кэш для THD результатов (чтобы не пересчитывать после brentq)
    last_thd_result = {}

    def residual(epsilon_0: float) -> float:
        """Невязка: F_y - W с полным THD расчётом."""
        thd_result = integrate_forces_thd(
            epsilon_0, T_inlet, model, grid, thd_solver,
            with_texture=with_texture,
            max_iter=thd_max_iter,
            tol=thd_tol,
        )
        last_thd_result['result'] = thd_result
        last_thd_result['eps'] = epsilon_0
        return thd_result.forces.Fy - W_target

    # Проверяем границы
    r_min = residual(eps_min)
    r_max = residual(eps_max)

    # Если знаки одинаковые, используем границу
    if r_min * r_max > 0:
        if abs(r_min) < abs(r_max):
            epsilon_0 = eps_min
        else:
            epsilon_0 = eps_max
    else:
        # Метод Брента для поиска корня
        epsilon_0 = brentq(residual, eps_min, eps_max, xtol=1e-4)

    # Используем закэшированный результат или пересчитываем
    if abs(last_thd_result.get('eps', -1) - epsilon_0) > 1e-6:
        thd_result = integrate_forces_thd(
            epsilon_0, T_inlet, model, grid, thd_solver,
            with_texture=with_texture,
            max_iter=thd_max_iter,
            tol=thd_tol,
        )
    else:
        thd_result = last_thd_result['result']

    return epsilon_0, thd_result


def compute_stiffness_coefficients_thd(
    epsilon_0: float,
    T_inlet: float,
    model: BearingModel,
    grid: Grid,
    thd_solver,
    with_texture: bool = True,
    thd_max_iter: int = 10,
    thd_tol: float = 1.0,
) -> StiffnessCoefficients:
    """
    Вычисление коэффициентов жёсткости с полным THD расчётом.

    Использует integrate_forces_thd для каждого возмущённого состояния.

    Args:
        epsilon_0: статический эксцентриситет
        T_inlet: входная температура (°C)
        model: параметры подшипника
        grid: расчётная сетка
        thd_solver: экземпляр THDSolver
        with_texture: учитывать текстуру
        thd_max_iter: максимум итераций THD
        thd_tol: допуск сходимости THD

    Returns:
        StiffnessCoefficients
    """
    delta_xi = model.numerical.delta_xi
    H_T = model.material.H_T(model.geometry, T_inlet)
    force_scale = model.force_scale(T_inlet)
    lambda_sq = model.geometry.lambda_ratio ** 2
    texture = model.texture if with_texture else None

    # Временно меняем T_inlet
    original_T_inlet = model.operating.T_inlet
    model.operating.T_inlet = T_inlet

    def get_forces_thd(xi: float, eta: float) -> Tuple[float, float]:
        """Вычислить силы при смещении (ξ, η) с THD."""
        H = compute_film_thickness_full(
            grid, model.geometry, epsilon_0, xi, eta, texture, H_T
        )

        # THD решение
        P_thd, T_field, eta_field, _ = thd_solver.solve(
            H, max_iter=thd_max_iter, tol=thd_tol, verbose=False
        )

        forces = integrate_forces(P_thd, grid, force_scale)
        return forces.Fx, forces.Fy

    # Смещение по x (ξ)
    Fx_plus_x, Fy_plus_x = get_forces_thd(+delta_xi, 0.0)
    Fx_minus_x, Fy_minus_x = get_forces_thd(-delta_xi, 0.0)

    # Смещение по y (η)
    Fx_plus_y, Fy_plus_y = get_forces_thd(0.0, +delta_xi)
    Fx_minus_y, Fy_minus_y = get_forces_thd(0.0, -delta_xi)

    # Восстанавливаем
    model.operating.T_inlet = original_T_inlet

    # Размерный шаг
    delta_x = delta_xi * model.geometry.c0

    # Коэффициенты жёсткости (размерные)
    K_xx = -(Fx_plus_x - Fx_minus_x) / (2 * delta_x)
    K_yx = -(Fy_plus_x - Fy_minus_x) / (2 * delta_x)
    K_xy = -(Fx_plus_y - Fx_minus_y) / (2 * delta_x)
    K_yy = -(Fy_plus_y - Fy_minus_y) / (2 * delta_x)

    # Масштаб для безразмерных коэффициентов
    # Используем T_inlet для масштаба (можно использовать T_mean)
    K_scale = model.K_scale(T_inlet)

    return StiffnessCoefficients(
        K_xx=K_xx, K_xy=K_xy, K_yx=K_yx, K_yy=K_yy,
        K_xx_bar=K_xx / K_scale,
        K_xy_bar=K_xy / K_scale,
        K_yx_bar=K_yx / K_scale,
        K_yy_bar=K_yy / K_scale,
    )


def compute_damping_coefficients_thd(
    epsilon_0: float,
    T_inlet: float,
    model: BearingModel,
    grid: Grid,
    thd_solver,
    with_texture: bool = True,
    thd_max_iter: int = 10,
    thd_tol: float = 1.0,
) -> DampingCoefficients:
    """
    Вычисление коэффициентов демпфирования с полным THD расчётом.

    Использует THD для базового давления и squeeze-член для возмущений.

    Args:
        epsilon_0: статический эксцентриситет
        T_inlet: входная температура (°C)
        model: параметры подшипника
        grid: расчётная сетка
        thd_solver: экземпляр THDSolver
        with_texture: учитывать текстуру
        thd_max_iter: максимум итераций THD
        thd_tol: допуск сходимости THD

    Returns:
        DampingCoefficients
    """
    delta_xi_dot = model.numerical.delta_xi_dot
    H_T = model.material.H_T(model.geometry, T_inlet)
    force_scale = model.force_scale(T_inlet)
    lambda_sq = model.geometry.lambda_ratio ** 2
    texture = model.texture if with_texture else None

    # Временно меняем T_inlet
    original_T_inlet = model.operating.T_inlet
    model.operating.T_inlet = T_inlet

    # Базовый зазор (при статическом положении)
    H = compute_film_thickness_static(
        grid, model.geometry, epsilon_0, texture, H_T
    )

    # Сначала получаем THD решение для базового состояния
    # чтобы получить поле вязкости eta_field
    P_base, T_field, eta_field, _ = thd_solver.solve(
        H, max_iter=thd_max_iter, tol=thd_tol, verbose=False
    )

    # Относительная вязкость для squeeze расчётов
    eta_ref = model.lubricant.viscosity(T_inlet)
    eta_ratio = eta_field / eta_ref

    # ∂H/∂φ численно
    N_Z, N_phi = H.shape
    dH_dphi = np.zeros_like(H)
    for i in range(N_Z):
        for j in range(N_phi):
            j_plus = (j + 1) % N_phi
            j_minus = (j - 1) % N_phi
            dH_dphi[i, j] = (H[i, j_plus] - H[i, j_minus]) / (2.0 * grid.d_phi)

    def get_forces_squeeze_thd(xi_dot: float, eta_dot: float) -> Tuple[float, float]:
        """Вычислить силы при скорости (ξ', η') с учётом THD вязкости."""
        dH_dtau = compute_dH_dtau(grid, xi_dot, eta_dot)

        # Решаем Reynolds с squeeze и полем вязкости
        P, _, _ = solve_reynolds_equation(
            H, eta_ratio, dH_dphi, dH_dtau,
            grid.d_phi, grid.d_Z, lambda_sq,
            beta=2.0,  # squeeze член
            omega_sor=model.numerical.omega_GS,
            tol=model.numerical.tol_Re,
            max_iter=model.numerical.max_iter_Re,
            use_cavitation=model.numerical.use_cavitation
        )

        forces = integrate_forces(P, grid, force_scale)
        return forces.Fx, forces.Fy

    # Смещение скорости по x (ξ')
    Fx_plus_xdot, Fy_plus_xdot = get_forces_squeeze_thd(+delta_xi_dot, 0.0)
    Fx_minus_xdot, Fy_minus_xdot = get_forces_squeeze_thd(-delta_xi_dot, 0.0)

    # Смещение скорости по y (η')
    Fx_plus_ydot, Fy_plus_ydot = get_forces_squeeze_thd(0.0, +delta_xi_dot)
    Fx_minus_ydot, Fy_minus_ydot = get_forces_squeeze_thd(0.0, -delta_xi_dot)

    # Восстанавливаем
    model.operating.T_inlet = original_T_inlet

    # Размерный шаг скорости
    delta_x_dot = delta_xi_dot * model.geometry.c0 * model.operating.omega

    # Коэффициенты демпфирования (размерные)
    C_xx = -(Fx_plus_xdot - Fx_minus_xdot) / (2 * delta_x_dot)
    C_yx = -(Fy_plus_xdot - Fy_minus_xdot) / (2 * delta_x_dot)
    C_xy = -(Fx_plus_ydot - Fx_minus_ydot) / (2 * delta_x_dot)
    C_yy = -(Fy_plus_ydot - Fy_minus_ydot) / (2 * delta_x_dot)

    # Масштаб для безразмерных коэффициентов
    C_scale = model.C_scale(T_inlet)

    return DampingCoefficients(
        C_xx=C_xx, C_xy=C_xy, C_yx=C_yx, C_yy=C_yy,
        C_xx_bar=C_xx / C_scale,
        C_xy_bar=C_xy / C_scale,
        C_yx_bar=C_yx / C_scale,
        C_yy_bar=C_yy / C_scale,
    )


def compute_all_coefficients_thd(
    epsilon_0: float,
    T_inlet: float,
    model: BearingModel,
    grid: Grid,
    thd_solver,
    with_texture: bool = True,
    thd_max_iter: int = 10,
    thd_tol: float = 1.0,
) -> FullCoefficients:
    """
    Вычисление всех коэффициентов K и C с полным THD расчётом.

    Args:
        epsilon_0: статический эксцентриситет
        T_inlet: входная температура (°C)
        model: параметры подшипника
        grid: расчётная сетка
        thd_solver: экземпляр THDSolver
        with_texture: учитывать текстуру
        thd_max_iter: максимум итераций THD
        thd_tol: допуск сходимости THD

    Returns:
        FullCoefficients
    """
    stiffness = compute_stiffness_coefficients_thd(
        epsilon_0, T_inlet, model, grid, thd_solver,
        with_texture, thd_max_iter, thd_tol
    )
    damping = compute_damping_coefficients_thd(
        epsilon_0, T_inlet, model, grid, thd_solver,
        with_texture, thd_max_iter, thd_tol
    )

    return FullCoefficients(stiffness=stiffness, damping=damping)


if __name__ == "__main__":
    from .parameters import create_chinese_paper_bearing

    # Тест
    model = create_chinese_paper_bearing()
    grid = create_grid(model.numerical.N_phi, model.numerical.N_Z)

    T = 80.0
    W_target = 2000.0  # Н

    print("=" * 60)
    print("ТЕСТ МОДУЛЯ forces.py")
    print("=" * 60)

    # Поиск равновесного эксцентриситета
    print(f"\nПоиск ε₀ для W = {W_target} Н, T = {T}°C...")
    epsilon_0, forces = find_equilibrium_eccentricity(
        W_target, T, model, grid, with_texture=True
    )
    print(f"  ε₀ = {epsilon_0:.4f}")
    print(f"  Fx = {forces.Fx:.1f} Н")
    print(f"  Fy = {forces.Fy:.1f} Н")

    # Коэффициенты жёсткости
    print("\nВычисление коэффициентов жёсткости...")
    K = compute_stiffness_coefficients(epsilon_0, T, model, grid, with_texture=True)
    print(f"  K_xx = {K.K_xx/1e6:.3f} МН/м, K̄_xx = {K.K_xx_bar:.4f}")
    print(f"  K_xy = {K.K_xy/1e6:.3f} МН/м, K̄_xy = {K.K_xy_bar:.4f}")
    print(f"  K_yx = {K.K_yx/1e6:.3f} МН/м, K̄_yx = {K.K_yx_bar:.4f}")
    print(f"  K_yy = {K.K_yy/1e6:.3f} МН/м, K̄_yy = {K.K_yy_bar:.4f}")
    print(f"  Проверка: K_xy ≈ -K_yx? {abs(K.K_xy + K.K_yx) / max(abs(K.K_xy), abs(K.K_yx), 1):.2%}")

    # Коэффициенты демпфирования
    print("\nВычисление коэффициентов демпфирования...")
    C = compute_damping_coefficients(epsilon_0, T, model, grid, with_texture=True)
    print(f"  C_xx = {C.C_xx/1e3:.3f} кН·с/м, C̄_xx = {C.C_xx_bar:.4f}")
    print(f"  C_xy = {C.C_xy/1e3:.3f} кН·с/м, C̄_xy = {C.C_xy_bar:.4f}")
    print(f"  C_yx = {C.C_yx/1e3:.3f} кН·с/м, C̄_yx = {C.C_yx_bar:.4f}")
    print(f"  C_yy = {C.C_yy/1e3:.3f} кН·с/м, C̄_yy = {C.C_yy_bar:.4f}")

    print("\n" + "=" * 60)
