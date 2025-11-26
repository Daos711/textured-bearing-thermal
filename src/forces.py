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
from .reynolds import ReynoldsSolver, solve_reynolds_static, solve_reynolds_with_squeeze
from .parameters import BearingModel, BearingGeometry, TextureParameters


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
    eps_max: float = 0.95,
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
    if r_min * r_max > 0:
        if abs(r_min) < abs(r_max):
            epsilon_0 = eps_min
        else:
            epsilon_0 = eps_max
    else:
        # Метод Брента для поиска корня
        epsilon_0 = brentq(residual, eps_min, eps_max, xtol=1e-4)

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
