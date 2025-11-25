"""
Термогидродинамическая модель подшипника.

Уравнение энергии для смазочного слоя (упрощённое, адиабатическое):

    ρ·c_p·U·h·∂T/∂x = η·(U/h)² + (h³/(12η))·|∇P|²

В безразмерной форме по окружной координате:

    Pe·H·∂θ/∂φ = Ф_v + Ф_p

где:
    θ = (T - T_0) / ΔT_ref — безразмерная температура
    Pe = ρ·c_p·U·c / k — число Пекле
    Ф_v = η·U²/(k·ΔT_ref) · 1/H — диссипация от сдвига (Couette)
    Ф_p — диссипация от течения под давлением (Poiseuille)

Для подшипника скольжения основной источник тепла — вязкое трение.

Итерационная схема THD:
    1. Начальное T(φ,Z) = T_inlet
    2. Вычислить η(T)
    3. Решить Reynolds → P(φ,Z)
    4. Вычислить тепловыделение
    5. Решить Energy → T(φ,Z)
    6. Проверить сходимость, если нет — к п.2
"""

import numpy as np
from numba import njit
from typing import Tuple, Optional
from dataclasses import dataclass

from .geometry import Grid
from .parameters import BearingModel, LubricantProperties


@dataclass
class ThermalSolution:
    """Результат решения температурной задачи."""
    T: np.ndarray           # Температура (°C) на сетке
    eta: np.ndarray         # Вязкость на сетке (Па·с)
    converged: bool
    iterations: int
    max_T: float            # Максимальная температура
    mean_T: float           # Средняя температура


def compute_viscosity_field(
    T: np.ndarray,
    lubricant: LubricantProperties,
) -> np.ndarray:
    """
    Вычисление поля вязкости по полю температуры.

    η(T) = η_0 · exp(-β·(T - T_0))

    Args:
        T: температура (°C) на сетке
        lubricant: свойства смазки

    Returns:
        eta: вязкость (Па·с) на сетке
    """
    return lubricant.eta_0 * np.exp(-lubricant.beta * (T - lubricant.T_0))


def compute_viscosity_ratio(
    T: np.ndarray,
    lubricant: LubricantProperties,
    T_ref: float,
) -> np.ndarray:
    """
    Относительная вязкость η(T)/η(T_ref).

    Используется в уравнении Рейнольдса с переменной вязкостью.
    """
    eta = compute_viscosity_field(T, lubricant)
    eta_ref = lubricant.viscosity(T_ref)
    return eta / eta_ref


@njit(cache=True)
def compute_heat_dissipation(
    H: np.ndarray,
    P: np.ndarray,
    d_phi: float,
    d_Z: float,
    D_over_L: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисление источников тепла в смазочном слое.

    Возвращает:
        Q_shear: тепловыделение от вязкого сдвига (Couette) ~ 1/H
        Q_pressure: тепловыделение от течения под давлением ~ H³·|∇P|²
    """
    N_Z, N_phi = H.shape

    Q_shear = 1.0 / H  # Безразмерное, ~ η·(U/h)²

    # Градиент давления
    dP_dphi = np.zeros_like(P)
    dP_dZ = np.zeros_like(P)

    for i in range(N_Z):
        for j in range(N_phi):
            j_plus = (j + 1) % N_phi
            j_minus = (j - 1) % N_phi

            dP_dphi[i, j] = (P[i, j_plus] - P[i, j_minus]) / (2 * d_phi)

            if i == 0:
                dP_dZ[i, j] = (P[i + 1, j] - P[i, j]) / d_Z
            elif i == N_Z - 1:
                dP_dZ[i, j] = (P[i, j] - P[i - 1, j]) / d_Z
            else:
                dP_dZ[i, j] = (P[i + 1, j] - P[i - 1, j]) / (2 * d_Z)

    # Тепловыделение от градиента давления
    Q_pressure = H ** 3 * (dP_dphi ** 2 + (D_over_L ** 2) * dP_dZ ** 2)

    return Q_shear, Q_pressure


@njit(cache=True)
def solve_energy_equation(
    H: np.ndarray,
    Q_total: np.ndarray,
    T_inlet: float,
    d_phi: float,
    Pe: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, float, int]:
    """
    Решение уравнения энергии методом прогонки по φ.

    Упрощённая 1D модель (усреднение по толщине плёнки):
        Pe·H·∂T/∂φ = Q_total

    Решается для каждого сечения Z отдельно.

    Args:
        H: безразмерная толщина
        Q_total: суммарный источник тепла
        T_inlet: температура на входе (°C)
        d_phi: шаг по φ
        Pe: число Пекле
        max_iter: максимум итераций
        tol: критерий сходимости

    Returns:
        T: температура (°C)
        residual: невязка
        iterations: число итераций
    """
    N_Z, N_phi = H.shape
    T = np.ones((N_Z, N_phi)) * T_inlet

    residual = 1.0
    iteration = 0

    while residual > tol and iteration < max_iter:
        T_old = T.copy()
        residual = 0.0

        for i in range(N_Z):
            # Прогонка вперёд по φ (от φ=0 к φ=2π)
            for j in range(1, N_phi):
                # Pe·H·(T[j] - T[j-1])/dφ = Q[j]
                # T[j] = T[j-1] + Q[j]·dφ / (Pe·H[j])
                if Pe * H[i, j] > 1e-10:
                    T[i, j] = T[i, j - 1] + Q_total[i, j] * d_phi / (Pe * H[i, j])
                else:
                    T[i, j] = T[i, j - 1]

            # Замыкание: T[0] = T[-1] для стационарного режима
            # или T[0] = T_inlet для открытой системы
            # Используем смешивание
            T[i, 0] = 0.5 * (T_inlet + T[i, -1])

        residual = np.max(np.abs(T - T_old))
        iteration += 1

    return T, residual, iteration


class ThermalModel:
    """
    Термогидродинамическая модель подшипника.

    Связывает уравнение Рейнольдса и уравнение энергии.
    """

    def __init__(self, model: BearingModel, grid: Grid):
        """
        Args:
            model: параметры подшипника
            grid: расчётная сетка
        """
        self.model = model
        self.grid = grid

        # Характерные величины
        self.D_over_L = model.geometry.D / model.geometry.L

        # Число Пекле
        # Pe = ρ·c_p·U·c / k
        self.Pe = (model.lubricant.rho * model.lubricant.c_p *
                   model.U * model.geometry.c / model.lubricant.k)

        # Масштаб температуры (подъём от диссипации)
        # ΔT ~ η·U²·L / (k·c)
        eta_ref = model.lubricant.viscosity(model.operating.T_inlet)
        self.delta_T_ref = (eta_ref * model.U ** 2 * model.geometry.L /
                           (model.lubricant.k * model.geometry.c))

    def compute_temperature_field(
        self,
        H: np.ndarray,
        P: np.ndarray,
    ) -> ThermalSolution:
        """
        Вычисление температурного поля для заданных H и P.

        Args:
            H: безразмерная толщина плёнки
            P: безразмерное давление

        Returns:
            ThermalSolution
        """
        # Источники тепла
        Q_shear, Q_pressure = compute_heat_dissipation(
            H, P, self.grid.d_phi, self.grid.d_Z, self.D_over_L
        )

        # Коэффициенты для безразмерного тепловыделения
        # Q_shear ~ η·U²/h² → масштаб η_ref·U²/c²
        # В безразмерной форме: Q* = Q / (k·ΔT_ref/c²)
        # Упрощённо берём суммарный источник
        Q_total = Q_shear + 0.1 * Q_pressure  # Poiseuille обычно меньше

        # Решаем уравнение энергии
        T, residual, iterations = solve_energy_equation(
            H, Q_total * self.delta_T_ref,
            self.model.operating.T_inlet,
            self.grid.d_phi,
            self.Pe,
        )

        # Ограничиваем температуру разумными значениями
        T = np.clip(T, self.model.operating.T_inlet, 200.0)

        # Вязкость при полученной температуре
        eta = compute_viscosity_field(T, self.model.lubricant)

        return ThermalSolution(
            T=T,
            eta=eta,
            converged=(residual < 1e-3),
            iterations=iterations,
            max_T=np.max(T),
            mean_T=np.mean(T),
        )

    def get_viscosity_ratio(self, T: np.ndarray) -> np.ndarray:
        """
        Относительная вязкость для использования в Reynolds solver.
        """
        return compute_viscosity_ratio(
            T, self.model.lubricant, self.model.operating.T_inlet
        )


class THDSolver:
    """
    Итерационный решатель термогидродинамической задачи.

    Связывает:
    - Уравнение Рейнольдса (давление)
    - Уравнение энергии (температура)
    - Зависимость вязкости от температуры
    """

    def __init__(self, model: BearingModel, grid: Grid):
        self.model = model
        self.grid = grid
        self.thermal = ThermalModel(model, grid)

        # Импортируем Reynolds solver здесь чтобы избежать циклического импорта
        from .reynolds import ReynoldsSolver
        self.reynolds = ReynoldsSolver(
            grid, model.geometry.D / model.geometry.L
        )

    def solve(
        self,
        H: np.ndarray,
        max_iter: int = 20,
        tol: float = 1e-3,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Итерационное решение связанной THD задачи.

        Args:
            H: безразмерная толщина плёнки
            max_iter: максимум итераций THD цикла
            tol: критерий сходимости по температуре
            verbose: выводить информацию о сходимости

        Returns:
            P: давление
            T: температура
            eta: вязкость
            converged: сошлось ли
        """
        # Начальные условия
        T = np.ones_like(H) * self.model.operating.T_inlet
        eta_ratio = np.ones_like(H)

        converged = False

        for iteration in range(max_iter):
            T_old = T.copy()

            # 1. Решаем Reynolds с текущей вязкостью
            solution = self.reynolds.solve_thermal(H, eta_ratio)
            P = solution.P

            # 2. Вычисляем температуру
            thermal_sol = self.thermal.compute_temperature_field(H, P)
            T = thermal_sol.T

            # 3. Обновляем вязкость
            eta_ratio = self.thermal.get_viscosity_ratio(T)

            # Проверка сходимости
            delta_T = np.max(np.abs(T - T_old))

            if verbose:
                print(f"THD iter {iteration + 1}: ΔT_max = {delta_T:.4f}°C, "
                      f"T_max = {thermal_sol.max_T:.1f}°C")

            if delta_T < tol:
                converged = True
                break

        eta = compute_viscosity_field(T, self.model.lubricant)

        return P, T, eta, converged


def estimate_temperature_rise(model: BearingModel) -> float:
    """
    Оценка подъёма температуры в подшипнике (упрощённая формула).

    ΔT ≈ η·U²·L / (ρ·c_p·Q)

    где Q — расход смазки.

    Для шарошечного долота это даёт порядок величины.

    Args:
        model: параметры подшипника

    Returns:
        ΔT: оценка подъёма температуры (°C)
    """
    eta = model.lubricant.viscosity(model.operating.T_inlet)

    # Характерный расход Q ~ U·c·L
    Q = model.U * model.geometry.c * model.geometry.L

    # Тепловыделение ~ η·U²·(объём зазора)/h
    # Объём ~ 2π·R·L·c
    volume = 2 * np.pi * model.geometry.R * model.geometry.L * model.geometry.c
    heat_gen = eta * (model.U / model.geometry.c) ** 2 * volume

    # Подъём температуры
    delta_T = heat_gen / (model.lubricant.rho * model.lubricant.c_p * Q)

    return delta_T


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .parameters import create_roller_cone_bit_bearing, create_test_bearing
    from .geometry import create_grid, compute_film_thickness

    # Тест для шарошечного долота
    model = create_roller_cone_bit_bearing()
    print(model.info())

    # Оценка подъёма температуры
    delta_T_est = estimate_temperature_rise(model)
    print(f"\nОценка подъёма температуры: ΔT ≈ {delta_T_est:.1f}°C")

    # Создаём сетку и зазор
    grid = create_grid(model.grid.num_phi, model.grid.num_Z)
    H = compute_film_thickness(grid, model.geometry, model.operating.epsilon,
                              model.texture)

    # Решаем THD задачу
    thd = THDSolver(model, grid)
    P, T, eta, converged = thd.solve(H, verbose=True)

    print(f"\nTHD решение: converged = {converged}")
    print(f"T_max = {np.max(T):.1f}°C, T_mean = {np.mean(T):.1f}°C")
    print(f"η_min = {np.min(eta)*1000:.3f} мПа·с, η_max = {np.max(eta)*1000:.3f} мПа·с")

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    Z_idx = grid.N_Z // 2

    # Температура
    ax = axes[0, 0]
    c = ax.contourf(grid.Phi_mesh, grid.Z_mesh, T, levels=30, cmap='hot')
    plt.colorbar(c, ax=ax, label='T, °C')
    ax.set_xlabel('φ, рад')
    ax.set_ylabel('Z')
    ax.set_title('Температура T(φ, Z)')

    # Вязкость
    ax = axes[0, 1]
    c = ax.contourf(grid.Phi_mesh, grid.Z_mesh, eta * 1000, levels=30, cmap='viridis')
    plt.colorbar(c, ax=ax, label='η, мПа·с')
    ax.set_xlabel('φ, рад')
    ax.set_ylabel('Z')
    ax.set_title('Вязкость η(φ, Z)')

    # Давление
    ax = axes[1, 0]
    ax.plot(grid.phi, P[Z_idx, :])
    ax.set_xlabel('φ, рад')
    ax.set_ylabel('P (безразмерное)')
    ax.set_title('Давление при Z=0')
    ax.grid(True)

    # Температура по φ
    ax = axes[1, 1]
    ax.plot(grid.phi, T[Z_idx, :])
    ax.set_xlabel('φ, рад')
    ax.set_ylabel('T, °C')
    ax.set_title('Температура при Z=0')
    ax.grid(True)

    plt.tight_layout()
    plt.show()
