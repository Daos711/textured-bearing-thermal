"""
Решатель уравнения Рейнольдса для гидродинамического подшипника.

Безразмерное уравнение Рейнольдса:
    ∂/∂φ(H³·∂P/∂φ) + (D/L)²·∂/∂Z(H³·∂P/∂Z) = ∂H/∂φ

Для динамического случая (учёт скорости сжатия плёнки):
    ∂/∂φ(H³·∂P/∂φ) + (D/L)²·∂/∂Z(H³·∂P/∂Z) = ∂H/∂φ + 2·∂H/∂τ

где:
    P — безразмерное давление
    H — безразмерная толщина плёнки
    τ = ω·t — безразмерное время

Граничные условия:
    - P(φ=0) = P(φ=2π) — периодичность по φ
    - P(Z=-1) = P(Z=1) = 0 — давление на торцах
    - P ≥ 0 — условие кавитации (модель Гюмбеля)

Численный метод: Гаусс-Зейдель с последовательной верхней релаксацией (SOR).
"""

import numpy as np
from numba import njit
from typing import Tuple, Optional
from dataclasses import dataclass

from .geometry import Grid


@dataclass
class ReynoldsSolution:
    """Результат решения уравнения Рейнольдса."""
    P: np.ndarray           # Безразмерное давление (N_Z x N_phi)
    converged: bool         # Сошлось ли решение
    iterations: int         # Количество итераций
    residual: float         # Финальная невязка


@njit(cache=True)
def solve_reynolds_static(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    D_over_L: float,
    omega_sor: float = 1.5,
    tol: float = 1e-6,
    max_iter: int = 50000,
) -> Tuple[np.ndarray, float, int]:
    """
    Решение стационарного уравнения Рейнольдса методом Гаусс-Зейделя.

    Args:
        H: безразмерная толщина плёнки (N_Z x N_phi)
        d_phi: шаг по φ
        d_Z: шаг по Z
        D_over_L: отношение D/L
        omega_sor: параметр релаксации SOR (1 < ω < 2)
        tol: критерий сходимости
        max_iter: максимум итераций

    Returns:
        P: безразмерное давление
        residual: финальная невязка
        iterations: количество итераций
    """
    N_Z, N_phi = H.shape
    P = np.zeros((N_Z, N_phi))

    # Коэффициент для члена с Z-производной
    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    # Предвычисление H на полуцелых узлах для аппроксимации производных
    # H_{i+1/2,j} = 0.5*(H_{i,j} + H_{i+1,j})
    H_i_plus = np.zeros((N_Z, N_phi))
    H_i_minus = np.zeros((N_Z, N_phi))
    H_j_plus = np.zeros((N_Z, N_phi))
    H_j_minus = np.zeros((N_Z, N_phi))

    for i in range(N_Z):
        for j in range(N_phi):
            j_plus = (j + 1) % N_phi
            j_minus = (j - 1) % N_phi

            H_i_plus[i, j] = 0.5 * (H[i, j] + H[i, j_plus])
            H_i_minus[i, j] = 0.5 * (H[i, j] + H[i, j_minus])

            if i < N_Z - 1:
                H_j_plus[i, j] = 0.5 * (H[i, j] + H[i + 1, j])
            if i > 0:
                H_j_minus[i, j] = 0.5 * (H[i, j] + H[i - 1, j])

    # Коэффициенты дискретной схемы
    A = H_i_plus ** 3       # коэфф. при P_{i,j+1}
    B = H_i_minus ** 3      # коэфф. при P_{i,j-1}
    C = alpha_sq * H_j_plus ** 3   # коэфф. при P_{i+1,j}
    D = alpha_sq * H_j_minus ** 3  # коэфф. при P_{i-1,j}
    E = A + B + C + D       # диагональный коэффициент

    # Правая часть: ∂H/∂φ ≈ (H_{i+1/2} - H_{i-1/2}) / dφ · dφ = H_{i+1/2} - H_{i-1/2}
    # Но в уравнении уже домножено на dφ², поэтому:
    F = d_phi * (H_i_plus - H_i_minus)

    # Итерационный процесс
    residual = 1.0
    iteration = 0

    while residual > tol and iteration < max_iter:
        residual = 0.0
        norm_P = 0.0

        # Внутренние узлы (границы по Z — Дирихле P=0)
        for i in range(1, N_Z - 1):
            for j in range(N_phi):
                j_plus = (j + 1) % N_phi
                j_minus = (j - 1) % N_phi

                P_old = P[i, j]

                # Обновление давления
                numerator = (A[i, j] * P[i, j_plus] +
                             B[i, j] * P[i, j_minus] +
                             C[i, j] * P[i + 1, j] +
                             D[i, j] * P[i - 1, j] -
                             F[i, j])

                P_new = numerator / E[i, j]

                # Условие кавитации
                if P_new < 0.0:
                    P_new = 0.0

                # SOR релаксация
                P[i, j] = P_old + omega_sor * (P_new - P_old)

                residual += abs(P[i, j] - P_old)
                norm_P += abs(P[i, j])

        # Относительная невязка
        if norm_P > 1e-12:
            residual /= norm_P

        iteration += 1

    return P, residual, iteration


@njit(cache=True)
def solve_reynolds_dynamic(
    H: np.ndarray,
    dH_dt: np.ndarray,
    d_phi: float,
    d_Z: float,
    D_over_L: float,
    omega_sor: float = 1.5,
    tol: float = 1e-6,
    max_iter: int = 50000,
) -> Tuple[np.ndarray, float, int]:
    """
    Решение динамического уравнения Рейнольдса.

    Учитывает скорость изменения толщины плёнки ∂H/∂τ.

    Уравнение:
        ∂/∂φ(H³·∂P/∂φ) + (D/L)²·∂/∂Z(H³·∂P/∂Z) = ∂H/∂φ + 2·∂H/∂τ

    Args:
        H: безразмерная толщина плёнки
        dH_dt: безразмерная скорость изменения толщины ∂H/∂τ
        d_phi, d_Z: шаги сетки
        D_over_L: отношение диаметра к длине
        omega_sor: параметр релаксации
        tol: критерий сходимости
        max_iter: максимум итераций

    Returns:
        P: безразмерное давление
        residual: финальная невязка
        iterations: количество итераций
    """
    N_Z, N_phi = H.shape
    P = np.zeros((N_Z, N_phi))

    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    # H на полуцелых узлах
    H_i_plus = np.zeros((N_Z, N_phi))
    H_i_minus = np.zeros((N_Z, N_phi))
    H_j_plus = np.zeros((N_Z, N_phi))
    H_j_minus = np.zeros((N_Z, N_phi))

    for i in range(N_Z):
        for j in range(N_phi):
            j_plus = (j + 1) % N_phi
            j_minus = (j - 1) % N_phi

            H_i_plus[i, j] = 0.5 * (H[i, j] + H[i, j_plus])
            H_i_minus[i, j] = 0.5 * (H[i, j] + H[i, j_minus])

            if i < N_Z - 1:
                H_j_plus[i, j] = 0.5 * (H[i, j] + H[i + 1, j])
            if i > 0:
                H_j_minus[i, j] = 0.5 * (H[i, j] + H[i - 1, j])

    A = H_i_plus ** 3
    B = H_i_minus ** 3
    C = alpha_sq * H_j_plus ** 3
    D = alpha_sq * H_j_minus ** 3
    E = A + B + C + D

    # Правая часть: ∂H/∂φ + 2·∂H/∂τ
    # ∂H/∂φ ≈ d_phi * (H_{i+1/2} - H_{i-1/2})
    F = d_phi * (H_i_plus - H_i_minus) + 2.0 * d_phi * d_phi * dH_dt

    residual = 1.0
    iteration = 0

    while residual > tol and iteration < max_iter:
        residual = 0.0
        norm_P = 0.0

        for i in range(1, N_Z - 1):
            for j in range(N_phi):
                j_plus = (j + 1) % N_phi
                j_minus = (j - 1) % N_phi

                P_old = P[i, j]

                numerator = (A[i, j] * P[i, j_plus] +
                             B[i, j] * P[i, j_minus] +
                             C[i, j] * P[i + 1, j] +
                             D[i, j] * P[i - 1, j] -
                             F[i, j])

                P_new = numerator / E[i, j]

                if P_new < 0.0:
                    P_new = 0.0

                P[i, j] = P_old + omega_sor * (P_new - P_old)

                residual += abs(P[i, j] - P_old)
                norm_P += abs(P[i, j])

        if norm_P > 1e-12:
            residual /= norm_P

        iteration += 1

    return P, residual, iteration


@njit(cache=True)
def solve_reynolds_variable_viscosity(
    H: np.ndarray,
    eta_ratio: np.ndarray,
    d_phi: float,
    d_Z: float,
    D_over_L: float,
    omega_sor: float = 1.5,
    tol: float = 1e-6,
    max_iter: int = 50000,
) -> Tuple[np.ndarray, float, int]:
    """
    Решение уравнения Рейнольдса с переменной вязкостью.

    Используется для термогидродинамической модели.

    Модифицированное уравнение:
        ∂/∂φ((H³/η*)·∂P/∂φ) + (D/L)²·∂/∂Z((H³/η*)·∂P/∂Z) = ∂H/∂φ

    где η* = η(T)/η_ref — относительная вязкость.

    Args:
        H: безразмерная толщина плёнки
        eta_ratio: относительная вязкость η/η_ref (N_Z x N_phi)
        d_phi, d_Z: шаги сетки
        D_over_L: отношение D/L
        omega_sor: параметр релаксации
        tol: критерий сходимости
        max_iter: максимум итераций

    Returns:
        P, residual, iterations
    """
    N_Z, N_phi = H.shape
    P = np.zeros((N_Z, N_phi))

    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    # Эффективный "H³" с учётом вязкости: H³/η*
    H_eff = H ** 3 / eta_ratio

    # H_eff на полуцелых узлах
    H_i_plus = np.zeros((N_Z, N_phi))
    H_i_minus = np.zeros((N_Z, N_phi))
    H_j_plus = np.zeros((N_Z, N_phi))
    H_j_minus = np.zeros((N_Z, N_phi))

    for i in range(N_Z):
        for j in range(N_phi):
            j_plus = (j + 1) % N_phi
            j_minus = (j - 1) % N_phi

            H_i_plus[i, j] = 0.5 * (H_eff[i, j] + H_eff[i, j_plus])
            H_i_minus[i, j] = 0.5 * (H_eff[i, j] + H_eff[i, j_minus])

            if i < N_Z - 1:
                H_j_plus[i, j] = 0.5 * (H_eff[i, j] + H_eff[i + 1, j])
            if i > 0:
                H_j_minus[i, j] = 0.5 * (H_eff[i, j] + H_eff[i - 1, j])

    A = H_i_plus
    B = H_i_minus
    C = alpha_sq * H_j_plus
    D = alpha_sq * H_j_minus
    E = A + B + C + D

    # Правая часть — используем исходный H (не H_eff)
    H_i_plus_orig = np.zeros((N_Z, N_phi))
    H_i_minus_orig = np.zeros((N_Z, N_phi))

    for i in range(N_Z):
        for j in range(N_phi):
            j_plus = (j + 1) % N_phi
            j_minus = (j - 1) % N_phi
            H_i_plus_orig[i, j] = 0.5 * (H[i, j] + H[i, j_plus])
            H_i_minus_orig[i, j] = 0.5 * (H[i, j] + H[i, j_minus])

    F = d_phi * (H_i_plus_orig - H_i_minus_orig)

    residual = 1.0
    iteration = 0

    while residual > tol and iteration < max_iter:
        residual = 0.0
        norm_P = 0.0

        for i in range(1, N_Z - 1):
            for j in range(N_phi):
                j_plus = (j + 1) % N_phi
                j_minus = (j - 1) % N_phi

                P_old = P[i, j]

                numerator = (A[i, j] * P[i, j_plus] +
                             B[i, j] * P[i, j_minus] +
                             C[i, j] * P[i + 1, j] +
                             D[i, j] * P[i - 1, j] -
                             F[i, j])

                P_new = numerator / E[i, j]

                if P_new < 0.0:
                    P_new = 0.0

                P[i, j] = P_old + omega_sor * (P_new - P_old)

                residual += abs(P[i, j] - P_old)
                norm_P += abs(P[i, j])

        if norm_P > 1e-12:
            residual /= norm_P

        iteration += 1

    return P, residual, iteration


class ReynoldsSolver:
    """
    Высокоуровневый класс для решения уравнения Рейнольдса.
    """

    def __init__(self, grid: Grid, D_over_L: float):
        """
        Args:
            grid: расчётная сетка
            D_over_L: отношение диаметра к длине подшипника
        """
        self.grid = grid
        self.D_over_L = D_over_L

    def solve_static(
        self,
        H: np.ndarray,
        omega_sor: float = 1.5,
        tol: float = 1e-6,
        max_iter: int = 50000,
    ) -> ReynoldsSolution:
        """
        Решение стационарного уравнения.
        """
        P, residual, iterations = solve_reynolds_static(
            H, self.grid.d_phi, self.grid.d_Z, self.D_over_L,
            omega_sor, tol, max_iter
        )

        return ReynoldsSolution(
            P=P,
            converged=(residual <= tol),
            iterations=iterations,
            residual=residual
        )

    def solve_dynamic(
        self,
        H: np.ndarray,
        dH_dt: np.ndarray,
        omega_sor: float = 1.5,
        tol: float = 1e-6,
        max_iter: int = 50000,
    ) -> ReynoldsSolution:
        """
        Решение динамического уравнения.

        Args:
            H: толщина плёнки
            dH_dt: ∂H/∂τ — безразмерная скорость изменения толщины
        """
        P, residual, iterations = solve_reynolds_dynamic(
            H, dH_dt, self.grid.d_phi, self.grid.d_Z, self.D_over_L,
            omega_sor, tol, max_iter
        )

        return ReynoldsSolution(
            P=P,
            converged=(residual <= tol),
            iterations=iterations,
            residual=residual
        )

    def solve_thermal(
        self,
        H: np.ndarray,
        eta_ratio: np.ndarray,
        omega_sor: float = 1.5,
        tol: float = 1e-6,
        max_iter: int = 50000,
    ) -> ReynoldsSolution:
        """
        Решение с переменной вязкостью (для THD модели).

        Args:
            H: толщина плёнки
            eta_ratio: η(T)/η_ref — относительная вязкость
        """
        P, residual, iterations = solve_reynolds_variable_viscosity(
            H, eta_ratio, self.grid.d_phi, self.grid.d_Z, self.D_over_L,
            omega_sor, tol, max_iter
        )

        return ReynoldsSolution(
            P=P,
            converged=(residual <= tol),
            iterations=iterations,
            residual=residual
        )


def compute_dH_dt(
    grid: Grid,
    epsilon_x_dot: float,
    epsilon_y_dot: float,
) -> np.ndarray:
    """
    Вычисление ∂H/∂τ для динамического уравнения.

    H = 1 - εx·cos(φ) - εy·sin(φ)
    ∂H/∂τ = -ε̇x·cos(φ) - ε̇y·sin(φ)

    где ε̇ = dε/dτ = (c/ω)·(dε/dt) — безразмерная скорость.

    Args:
        grid: расчётная сетка
        epsilon_x_dot: dεx/dτ
        epsilon_y_dot: dεy/dτ

    Returns:
        dH_dt: ∂H/∂τ на сетке
    """
    return -epsilon_x_dot * np.cos(grid.Phi_mesh) - epsilon_y_dot * np.sin(grid.Phi_mesh)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .geometry import create_grid, compute_film_thickness
    from .parameters import create_test_bearing

    # Тест
    model = create_test_bearing()
    grid = create_grid(model.grid.num_phi, model.grid.num_Z)

    # Зазор без текстуры
    H = compute_film_thickness(grid, model.geometry, model.operating.epsilon, texture=None)

    # Решение
    solver = ReynoldsSolver(grid, model.geometry.D / model.geometry.L)
    solution = solver.solve_static(H, tol=1e-6)

    print(f"Сходимость: {solution.converged}")
    print(f"Итераций: {solution.iterations}")
    print(f"Невязка: {solution.residual:.2e}")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Давление при Z=0
    Z_idx = grid.N_Z // 2
    axes[0].plot(grid.phi, solution.P[Z_idx, :])
    axes[0].set_xlabel('φ, рад')
    axes[0].set_ylabel('P (безразмерное)')
    axes[0].set_title('Давление при Z=0')
    axes[0].grid(True)

    # Контурный график
    c = axes[1].contourf(grid.Phi_mesh, grid.Z_mesh, solution.P, levels=50, cmap='plasma')
    plt.colorbar(c, ax=axes[1], label='P')
    axes[1].set_xlabel('φ, рад')
    axes[1].set_ylabel('Z')
    axes[1].set_title('Поле давления P(φ, Z)')

    plt.tight_layout()
    plt.show()
