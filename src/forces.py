"""
Вычисление сил, нагрузок и динамических коэффициентов подшипника.

Система координат:
- x: вдоль линии центров (от втулки к валу), φ=0
- y: перпендикулярно x, в направлении вращения, φ=π/2

Силы от давления:
    Fx = -∫∫ P·cos(φ) dφ dZ  (минус, т.к. давление действует на вал внутрь)
    Fy = -∫∫ P·sin(φ) dφ dZ

Коэффициенты жёсткости (статические):
    Kxx = -∂Fx/∂x,  Kxy = -∂Fx/∂y
    Kyx = -∂Fy/∂x,  Kyy = -∂Fy/∂y

Коэффициенты демпфирования (динамические):
    Cxx = -∂Fx/∂ẋ,  Cxy = -∂Fx/∂ẏ
    Cyx = -∂Fy/∂ẋ,  Cyy = -∂Fy/∂ẏ

Все коэффициенты вычисляются численным дифференцированием.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from joblib import Parallel, delayed

from .geometry import Grid, compute_film_thickness, compute_film_thickness_dynamic
from .reynolds import ReynoldsSolver, solve_reynolds_static, solve_reynolds_dynamic, compute_dH_dt
from .parameters import BearingModel


@dataclass
class ForceResult:
    """Результат вычисления сил."""
    Fx: float               # Сила по x (Н)
    Fy: float               # Сила по y (Н)
    F_total: float          # Полная сила (Н)
    attitude_angle: float   # Угол нагрузки (рад)


@dataclass
class FrictionResult:
    """Результат вычисления трения."""
    friction_force: float   # Сила трения (Н)
    friction_coeff: float   # Коэффициент трения μ
    power_loss: float       # Потери мощности (Вт)


@dataclass
class DynamicCoefficients:
    """Динамические коэффициенты подшипника."""
    # Жёсткость (Н/м)
    Kxx: float
    Kxy: float
    Kyx: float
    Kyy: float
    # Демпфирование (Н·с/м)
    Cxx: float
    Cxy: float
    Cyx: float
    Cyy: float


def integrate_forces(
    P: np.ndarray,
    grid: Grid,
    pressure_scale: float,
    R: float,
    L: float,
) -> ForceResult:
    """
    Интегрирование давления для получения сил.

    Fx = -∫∫ P·cos(φ) · R·dφ · (L/2)·dZ
    Fy = -∫∫ P·sin(φ) · R·dφ · (L/2)·dZ

    Args:
        P: безразмерное давление
        grid: расчётная сетка
        pressure_scale: масштаб давления (Па)
        R: радиус (м)
        L: длина (м)

    Returns:
        ForceResult с размерными силами
    """
    # Безразмерные интегралы (минус учитывает направление силы на вал)
    Fx_nd = -np.trapz(np.trapz(P * grid.cos_phi, grid.phi, axis=1), grid.Z)
    Fy_nd = -np.trapz(np.trapz(P * grid.sin_phi, grid.phi, axis=1), grid.Z)

    # Масштабирование к размерным величинам
    # ∫dφ даёт ~2π, ∫dZ даёт 2, площадь R·L/2
    force_scale = pressure_scale * R * L / 2

    Fx = Fx_nd * force_scale
    Fy = Fy_nd * force_scale

    F_total = np.sqrt(Fx**2 + Fy**2)
    attitude_angle = np.arctan2(Fy, Fx)

    return ForceResult(
        Fx=Fx,
        Fy=Fy,
        F_total=F_total,
        attitude_angle=attitude_angle
    )


def compute_friction(
    P: np.ndarray,
    H: np.ndarray,
    grid: Grid,
    model: BearingModel,
) -> FrictionResult:
    """
    Вычисление силы трения на валу.

    Касательное напряжение на поверхности вала:
        τ = η·U/h - (h/2)·∂P/∂x

    Сила трения:
        F_f = ∫∫ τ · R·dφ · dz

    В безразмерной форме:
        f = ∫∫ (1/H + (H/2)·∂P/∂φ) dφ dZ

    Args:
        P: безразмерное давление
        H: безразмерная толщина
        grid: расчётная сетка
        model: параметры подшипника

    Returns:
        FrictionResult
    """
    # Производная давления по φ
    dP_dphi = np.zeros_like(P)
    dP_dphi[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * grid.d_phi)
    dP_dphi[:, 0] = (P[:, 1] - P[:, -2]) / (2 * grid.d_phi)
    dP_dphi[:, -1] = dP_dphi[:, 0]

    # Безразмерный интеграл для силы трения
    # Первый член (Couette): 1/H
    # Второй член (Poiseuille): коррекция от градиента давления
    integrand = 1.0 / H - 0.5 * H * dP_dphi

    f_nd = np.trapz(np.trapz(integrand, grid.phi, axis=1), grid.Z)

    # Масштаб силы трения
    eta = model.lubricant.viscosity(model.operating.T_inlet)
    friction_scale = eta * model.U * model.geometry.R * model.geometry.L / model.geometry.c

    friction_force = f_nd * friction_scale

    # Коэффициент трения (нужна несущая способность)
    # Вычисляем силы для определения нагрузки
    forces = integrate_forces(
        P, grid, model.pressure_scale, model.geometry.R, model.geometry.L
    )

    friction_coeff = abs(friction_force / forces.F_total) if forces.F_total > 0 else 0.0

    # Потери мощности
    power_loss = friction_force * model.U

    return FrictionResult(
        friction_force=friction_force,
        friction_coeff=friction_coeff,
        power_loss=power_loss
    )


def compute_stiffness_coefficients(
    model: BearingModel,
    grid: Grid,
    epsilon: float,
    delta_eps: float = 1e-5,
    with_texture: bool = True,
    n_jobs: int = 4,
) -> Tuple[float, float, float, float]:
    """
    Вычисление коэффициентов жёсткости методом конечных разностей.

    Kij = -∂Fi/∂xj ≈ -(Fi(xj+Δ) - Fi(xj-Δ)) / (2Δ)

    Args:
        model: параметры подшипника
        grid: расчётная сетка
        epsilon: базовый эксцентриситет
        delta_eps: шаг для численного дифференцирования
        with_texture: учитывать текстуру
        n_jobs: число параллельных задач

    Returns:
        Kxx, Kxy, Kyx, Kyy (Н/м)
    """
    D_over_L = model.geometry.D / model.geometry.L
    texture = model.texture if with_texture else None

    def solve_and_get_forces(eps_x: float, eps_y: float) -> Tuple[float, float]:
        """Решить Reynolds и вернуть силы."""
        H = compute_film_thickness_dynamic(grid, model.geometry, eps_x, eps_y, texture)
        P, _, _ = solve_reynolds_static(
            H, grid.d_phi, grid.d_Z, D_over_L,
            omega_sor=1.5, tol=1e-6, max_iter=50000
        )
        forces = integrate_forces(
            P, grid, model.pressure_scale, model.geometry.R, model.geometry.L
        )
        return forces.Fx, forces.Fy

    # Возмущения по x (εx)
    eps_x_plus = epsilon + delta_eps
    eps_x_minus = epsilon - delta_eps

    # Возмущения по y (εy)
    eps_y_plus = delta_eps
    eps_y_minus = -delta_eps

    # Параллельное вычисление
    tasks = [
        (eps_x_plus, 0.0),   # +Δx
        (eps_x_minus, 0.0),  # -Δx
        (epsilon, eps_y_plus),   # +Δy
        (epsilon, eps_y_minus),  # -Δy
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(solve_and_get_forces)(ex, ey) for ex, ey in tasks
    )

    Fx_px, Fy_px = results[0]  # +Δx
    Fx_mx, Fy_mx = results[1]  # -Δx
    Fx_py, Fy_py = results[2]  # +Δy
    Fx_my, Fy_my = results[3]  # -Δy

    # Производные (с учётом перехода от ε к x: x = ε·c)
    c = model.geometry.c
    delta_x = delta_eps * c

    Kxx = -(Fx_px - Fx_mx) / (2 * delta_x)
    Kyx = -(Fy_px - Fy_mx) / (2 * delta_x)
    Kxy = -(Fx_py - Fx_my) / (2 * delta_x)
    Kyy = -(Fy_py - Fy_my) / (2 * delta_x)

    return Kxx, Kxy, Kyx, Kyy


def compute_damping_coefficients(
    model: BearingModel,
    grid: Grid,
    epsilon: float,
    delta_eps_dot: float = 1e-5,
    with_texture: bool = True,
    n_jobs: int = 4,
) -> Tuple[float, float, float, float]:
    """
    Вычисление коэффициентов демпфирования.

    Cij = -∂Fi/∂ẋj

    Используется динамическое уравнение Рейнольдса с ∂H/∂t.

    Args:
        model: параметры подшипника
        grid: расчётная сетка
        epsilon: базовый эксцентриситет
        delta_eps_dot: шаг по скорости
        with_texture: учитывать текстуру
        n_jobs: число параллельных задач

    Returns:
        Cxx, Cxy, Cyx, Cyy (Н·с/м)
    """
    D_over_L = model.geometry.D / model.geometry.L
    texture = model.texture if with_texture else None

    # Базовый зазор
    H = compute_film_thickness_dynamic(grid, model.geometry, epsilon, 0.0, texture)

    def solve_dynamic_and_get_forces(eps_x_dot: float, eps_y_dot: float) -> Tuple[float, float]:
        """Решить динамический Reynolds и вернуть силы."""
        dH_dt = compute_dH_dt(grid, eps_x_dot, eps_y_dot)

        P, _, _ = solve_reynolds_dynamic(
            H, dH_dt, grid.d_phi, grid.d_Z, D_over_L,
            omega_sor=1.5, tol=1e-6, max_iter=50000
        )

        forces = integrate_forces(
            P, grid, model.pressure_scale, model.geometry.R, model.geometry.L
        )
        return forces.Fx, forces.Fy

    # Возмущения по скоростям
    tasks = [
        (+delta_eps_dot, 0.0),  # +ẋ
        (-delta_eps_dot, 0.0),  # -ẋ
        (0.0, +delta_eps_dot),  # +ẏ
        (0.0, -delta_eps_dot),  # -ẏ
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(solve_dynamic_and_get_forces)(vx, vy) for vx, vy in tasks
    )

    Fx_pvx, Fy_pvx = results[0]
    Fx_mvx, Fy_mvx = results[1]
    Fx_pvy, Fy_pvy = results[2]
    Fx_mvy, Fy_mvy = results[3]

    # Производные
    # Скорость: ẋ = ε̇·c, но в безразмерной форме ε̇ = ẋ/(c·ω)
    # Поэтому масштаб: delta_v = delta_eps_dot * c * omega
    c = model.geometry.c
    omega = model.operating.omega
    delta_v = delta_eps_dot * c * omega

    Cxx = -(Fx_pvx - Fx_mvx) / (2 * delta_v)
    Cyx = -(Fy_pvx - Fy_mvx) / (2 * delta_v)
    Cxy = -(Fx_pvy - Fx_mvy) / (2 * delta_v)
    Cyy = -(Fy_pvy - Fy_mvy) / (2 * delta_v)

    return Cxx, Cxy, Cyx, Cyy


def compute_all_dynamic_coefficients(
    model: BearingModel,
    grid: Grid,
    epsilon: float,
    delta_eps: float = 1e-5,
    delta_eps_dot: float = 1e-5,
    with_texture: bool = True,
    n_jobs: int = 8,
) -> DynamicCoefficients:
    """
    Вычисление всех 8 динамических коэффициентов.

    Args:
        model: параметры подшипника
        grid: расчётная сетка
        epsilon: эксцентриситет
        delta_eps: шаг для жёсткости
        delta_eps_dot: шаг для демпфирования
        with_texture: учитывать текстуру
        n_jobs: число параллельных задач

    Returns:
        DynamicCoefficients
    """
    # Жёсткость
    Kxx, Kxy, Kyx, Kyy = compute_stiffness_coefficients(
        model, grid, epsilon, delta_eps, with_texture, n_jobs // 2
    )

    # Демпфирование
    Cxx, Cxy, Cyx, Cyy = compute_damping_coefficients(
        model, grid, epsilon, delta_eps_dot, with_texture, n_jobs // 2
    )

    return DynamicCoefficients(
        Kxx=Kxx, Kxy=Kxy, Kyx=Kyx, Kyy=Kyy,
        Cxx=Cxx, Cxy=Cxy, Cyx=Cyx, Cyy=Cyy
    )


def compute_stability_parameters(coeff: DynamicCoefficients) -> Dict[str, float]:
    """
    Вычисление параметров устойчивости ротора.

    Эквивалентная жёсткость:
        K_eq = (Kxx·Cyy + Kyy·Cxx - Kxy·Cyx - Kyx·Cxy) / (Cxx + Cyy)

    Параметр устойчивости:
        γ² = (K_eq - Kxx)(K_eq - Kyy) - Kxy·Kyx) / (Cxx·Cyy - Cxy·Cyx)

    Критическая скорость:
        ω_st = K_eq / γ²

    Args:
        coeff: динамические коэффициенты

    Returns:
        Словарь с K_eq, gamma_sq, omega_st
    """
    Kxx, Kxy, Kyx, Kyy = coeff.Kxx, coeff.Kxy, coeff.Kyx, coeff.Kyy
    Cxx, Cxy, Cyx, Cyy = coeff.Cxx, coeff.Cxy, coeff.Cyx, coeff.Cyy

    # Эквивалентная жёсткость
    denom = Cxx + Cyy
    if abs(denom) < 1e-12:
        K_eq = 0.0
    else:
        K_eq = (Kxx * Cyy + Kyy * Cxx - Kxy * Cyx - Kyx * Cxy) / denom

    # Параметр устойчивости γ²
    num = (K_eq - Kxx) * (K_eq - Kyy) - Kxy * Kyx
    det_C = Cxx * Cyy - Cxy * Cyx
    if abs(det_C) < 1e-12:
        gamma_sq = 0.0
    else:
        gamma_sq = num / det_C

    # Критическая скорость
    if abs(gamma_sq) < 1e-12:
        omega_st = 0.0
    else:
        omega_st = K_eq / gamma_sq

    return {
        "K_eq": K_eq,
        "gamma_sq": gamma_sq,
        "omega_st": omega_st
    }


class BearingAnalyzer:
    """
    Комплексный анализатор подшипника.

    Объединяет вычисление всех характеристик:
    - Несущая способность
    - Трение
    - Динамические коэффициенты
    - Устойчивость
    """

    def __init__(self, model: BearingModel):
        self.model = model
        from .geometry import create_grid
        self.grid = create_grid(model.grid.num_phi, model.grid.num_Z)
        self.D_over_L = model.geometry.D / model.geometry.L

    def analyze(
        self,
        epsilon: Optional[float] = None,
        with_texture: bool = True,
        compute_dynamics: bool = True,
        n_jobs: int = 8,
    ) -> Dict:
        """
        Полный анализ подшипника.

        Args:
            epsilon: эксцентриситет (если None — из модели)
            with_texture: учитывать текстуру
            compute_dynamics: вычислять динамические коэффициенты
            n_jobs: число параллельных задач

        Returns:
            Словарь со всеми результатами
        """
        if epsilon is None:
            epsilon = self.model.operating.epsilon

        texture = self.model.texture if with_texture else None

        # Зазор и давление
        H = compute_film_thickness(
            self.grid, self.model.geometry, epsilon, texture
        )

        P, residual, iters = solve_reynolds_static(
            H, self.grid.d_phi, self.grid.d_Z, self.D_over_L,
            omega_sor=1.5, tol=1e-6, max_iter=50000
        )

        # Силы
        forces = integrate_forces(
            P, self.grid, self.model.pressure_scale,
            self.model.geometry.R, self.model.geometry.L
        )

        # Трение
        friction = compute_friction(P, H, self.grid, self.model)

        results = {
            "epsilon": epsilon,
            "with_texture": with_texture,
            "H": H,
            "P": P,
            "converged": residual < 1e-6,
            "iterations": iters,
            "forces": forces,
            "friction": friction,
        }

        # Динамические коэффициенты
        if compute_dynamics:
            coeffs = compute_all_dynamic_coefficients(
                self.model, self.grid, epsilon,
                with_texture=with_texture, n_jobs=n_jobs
            )
            stability = compute_stability_parameters(coeffs)

            results["dynamic_coefficients"] = coeffs
            results["stability"] = stability

        return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .parameters import create_test_bearing, create_roller_cone_bit_bearing
    from .geometry import create_grid

    # Тест
    model = create_test_bearing()
    print(model.info())

    analyzer = BearingAnalyzer(model)

    # Анализ без текстуры
    results_smooth = analyzer.analyze(with_texture=False, compute_dynamics=False)
    print(f"\nГладкий подшипник:")
    print(f"  Несущая способность: {results_smooth['forces'].F_total:.1f} Н")
    print(f"  Сила трения: {results_smooth['friction'].friction_force:.2f} Н")
    print(f"  Коэффициент трения: {results_smooth['friction'].friction_coeff:.6f}")

    # Анализ с текстурой
    results_textured = analyzer.analyze(with_texture=True, compute_dynamics=False)
    print(f"\nС текстурой:")
    print(f"  Несущая способность: {results_textured['forces'].F_total:.1f} Н")
    print(f"  Сила трения: {results_textured['friction'].friction_force:.2f} Н")
    print(f"  Коэффициент трения: {results_textured['friction'].friction_coeff:.6f}")

    # Сравнение
    delta_F = (results_textured['forces'].F_total - results_smooth['forces'].F_total)
    delta_mu = (results_textured['friction'].friction_coeff - results_smooth['friction'].friction_coeff)
    print(f"\nЭффект текстуры:")
    print(f"  Изменение несущей способности: {delta_F/results_smooth['forces'].F_total*100:+.1f}%")
    print(f"  Изменение коэффициента трения: {delta_mu/results_smooth['friction'].friction_coeff*100:+.1f}%")
