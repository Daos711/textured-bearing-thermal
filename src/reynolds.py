"""
Решатель уравнения Рейнольдса для гидродинамического подшипника.

Безразмерное уравнение Рейнольдса (из постановки задачи):

    ∂/∂φ(H³/η̂ · ∂P/∂φ) + λ² · ∂/∂Z(H³/η̂ · ∂P/∂Z) = ∂H/∂φ + β·∂H/∂τ

где:
    P — безразмерное давление, P = p/p₀, p₀ = 6η_ref·U·R_J/c₀²
    H — безразмерная толщина плёнки, H = h/c₀
    η̂ = η(T)/η_ref — безразмерная вязкость
    λ = 2R_J/L — отношение длин
    β = 2 — коэффициент при squeeze-члене
    τ = ω·t — безразмерное время

Граничные условия:
    - P(φ=0) = P(φ=2π) — периодичность по φ
    - P(Z=-1) = P(Z=1) = 0 — давление на торцах
    - P ≥ 0 — условие кавитации (модель half-Sommerfeld)

Численный метод: Гаусс-Зейдель с последовательной верхней релаксацией (SOR).
"""

import numpy as np
from numba import njit
from typing import Tuple, Optional
from dataclasses import dataclass

from .geometry import Grid


# Коэффициент β в squeeze-члене (классическая формулировка)
BETA_SQUEEZE = 2.0


@dataclass
class ReynoldsSolution:
    """Результат решения уравнения Рейнольдса."""
    P: np.ndarray           # Безразмерное давление (N_Z x N_phi)
    converged: bool         # Сошлось ли решение
    iterations: int         # Количество итераций
    residual: float         # Финальная невязка


@njit(cache=True)
def solve_reynolds_equation(
    H: np.ndarray,
    eta_hat: np.ndarray,
    dH_dphi: np.ndarray,
    dH_dtau: np.ndarray,
    d_phi: float,
    d_Z: float,
    lambda_sq: float,
    beta: float = 2.0,
    omega_sor: float = 1.7,
    tol: float = 1e-6,
    max_iter: int = 10000,
    use_cavitation: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """
    Универсальный решатель уравнения Рейнольдса.

    ∂/∂φ(H³/η̂ · ∂P/∂φ) + λ² · ∂/∂Z(H³/η̂ · ∂P/∂Z) = ∂H/∂φ + β·∂H/∂τ

    Args:
        H: безразмерная толщина плёнки (N_Z x N_phi)
        eta_hat: безразмерная вязкость η̂ = η(T)/η_ref (N_Z x N_phi)
        dH_dphi: ∂H/∂φ (N_Z x N_phi)
        dH_dtau: ∂H/∂τ (N_Z x N_phi), для статики = 0
        d_phi: шаг по φ
        d_Z: шаг по Z
        lambda_sq: λ² = (2R_J/L)²
        beta: коэффициент squeeze (обычно 2)
        omega_sor: параметр релаксации SOR (1 < ω < 2)
        tol: критерий сходимости
        max_iter: максимум итераций
        use_cavitation: применять условие P ≥ 0

    Returns:
        P: безразмерное давление
        residual: финальная невязка
        iterations: количество итераций
    """
    N_Z, N_phi = H.shape
    P = np.zeros((N_Z, N_phi))

    # Эффективный коэффициент G = H³/η̂
    G = np.zeros((N_Z, N_phi))
    for i in range(N_Z):
        for j in range(N_phi):
            G[i, j] = H[i, j] ** 3 / eta_hat[i, j]

    # G на полуцелых узлах для аппроксимации производных
    G_phi_plus = np.zeros((N_Z, N_phi))   # G_{i,j+1/2}
    G_phi_minus = np.zeros((N_Z, N_phi))  # G_{i,j-1/2}
    G_Z_plus = np.zeros((N_Z, N_phi))     # G_{i+1/2,j}
    G_Z_minus = np.zeros((N_Z, N_phi))    # G_{i-1/2,j}

    for i in range(N_Z):
        for j in range(N_phi):
            j_plus = (j + 1) % N_phi
            j_minus = (j - 1) % N_phi

            G_phi_plus[i, j] = 0.5 * (G[i, j] + G[i, j_plus])
            G_phi_minus[i, j] = 0.5 * (G[i, j] + G[i, j_minus])

            if i < N_Z - 1:
                G_Z_plus[i, j] = 0.5 * (G[i, j] + G[i + 1, j])
            if i > 0:
                G_Z_minus[i, j] = 0.5 * (G[i, j] + G[i - 1, j])

    # Коэффициенты конечно-разностной схемы
    # (G·∂P/∂φ) ≈ (G_{j+1/2}·(P_{j+1}-P_j) - G_{j-1/2}·(P_j-P_{j-1})) / dφ²
    A = G_phi_plus / (d_phi * d_phi)           # коэфф. при P_{i,j+1}
    B = G_phi_minus / (d_phi * d_phi)          # коэфф. при P_{i,j-1}
    C = lambda_sq * G_Z_plus / (d_Z * d_Z)     # коэфф. при P_{i+1,j}
    D = lambda_sq * G_Z_minus / (d_Z * d_Z)    # коэфф. при P_{i-1,j}
    E = A + B + C + D                          # диагональный коэффициент

    # Правая часть: ∂H/∂φ + β·∂H/∂τ
    F = dH_dphi + beta * dH_dtau

    # Итерационный процесс
    residual = 1.0
    iteration = 0

    while residual > tol and iteration < max_iter:
        max_change = 0.0

        # Внутренние узлы (границы по Z: P=0 - Дирихле)
        for i in range(1, N_Z - 1):
            for j in range(N_phi):
                j_plus = (j + 1) % N_phi
                j_minus = (j - 1) % N_phi

                P_old = P[i, j]

                # Решение уравнения
                if E[i, j] > 1e-12:
                    numerator = (A[i, j] * P[i, j_plus] +
                                 B[i, j] * P[i, j_minus] +
                                 C[i, j] * P[i + 1, j] +
                                 D[i, j] * P[i - 1, j] -
                                 F[i, j])

                    P_new = numerator / E[i, j]
                else:
                    P_new = 0.0

                # Условие кавитации
                if use_cavitation and P_new < 0.0:
                    P_new = 0.0

                # SOR релаксация
                P[i, j] = P_old + omega_sor * (P_new - P_old)

                change = abs(P[i, j] - P_old)
                if change > max_change:
                    max_change = change

        residual = max_change
        iteration += 1

    return P, residual, iteration


@njit(cache=True)
def solve_reynolds_static(
    H: np.ndarray,
    eta_hat: float,
    d_phi: float,
    d_Z: float,
    lambda_sq: float,
    omega_sor: float = 1.7,
    tol: float = 1e-6,
    max_iter: int = 10000,
    use_cavitation: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """
    Статическое уравнение Рейнольдса (∂H/∂τ = 0, η̂ = const).

    Args:
        H: безразмерная толщина плёнки (N_Z x N_phi)
        eta_hat: безразмерная вязкость (скаляр для равномерной T)
        d_phi: шаг по φ
        d_Z: шаг по Z
        lambda_sq: λ² = (2R_J/L)²
        omega_sor: параметр релаксации
        tol: критерий сходимости
        max_iter: максимум итераций
        use_cavitation: P ≥ 0

    Returns:
        P, residual, iterations
    """
    N_Z, N_phi = H.shape

    # η̂ = const → массив
    eta_hat_arr = np.full((N_Z, N_phi), eta_hat)

    # ∂H/∂φ численно
    dH_dphi = np.zeros((N_Z, N_phi))
    for i in range(N_Z):
        for j in range(N_phi):
            j_plus = (j + 1) % N_phi
            j_minus = (j - 1) % N_phi
            dH_dphi[i, j] = (H[i, j_plus] - H[i, j_minus]) / (2.0 * d_phi)

    # ∂H/∂τ = 0 для статики
    dH_dtau = np.zeros((N_Z, N_phi))

    return solve_reynolds_equation(
        H, eta_hat_arr, dH_dphi, dH_dtau,
        d_phi, d_Z, lambda_sq,
        beta=0.0,  # нет squeeze для статики
        omega_sor=omega_sor,
        tol=tol,
        max_iter=max_iter,
        use_cavitation=use_cavitation
    )


@njit(cache=True)
def solve_reynolds_with_squeeze(
    H: np.ndarray,
    dH_dtau: np.ndarray,
    eta_hat: float,
    d_phi: float,
    d_Z: float,
    lambda_sq: float,
    beta: float = 2.0,
    omega_sor: float = 1.7,
    tol: float = 1e-6,
    max_iter: int = 10000,
    use_cavitation: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """
    Уравнение Рейнольдса со squeeze-членом (для коэффициентов демпфирования).

    ∂/∂φ(H³/η̂ · ∂P/∂φ) + λ² · ∂/∂Z(H³/η̂ · ∂P/∂Z) = ∂H/∂φ + β·∂H/∂τ

    Args:
        H: безразмерная толщина
        dH_dtau: ∂H/∂τ = ξ'·cos(φ) + η'·sin(φ)
        eta_hat: безразмерная вязкость
        d_phi, d_Z: шаги сетки
        lambda_sq: λ²
        beta: коэффициент (обычно 2)
        omega_sor, tol, max_iter, use_cavitation: параметры решателя

    Returns:
        P, residual, iterations
    """
    N_Z, N_phi = H.shape

    eta_hat_arr = np.full((N_Z, N_phi), eta_hat)

    # ∂H/∂φ численно
    dH_dphi = np.zeros((N_Z, N_phi))
    for i in range(N_Z):
        for j in range(N_phi):
            j_plus = (j + 1) % N_phi
            j_minus = (j - 1) % N_phi
            dH_dphi[i, j] = (H[i, j_plus] - H[i, j_minus]) / (2.0 * d_phi)

    return solve_reynolds_equation(
        H, eta_hat_arr, dH_dphi, dH_dtau,
        d_phi, d_Z, lambda_sq,
        beta=beta,
        omega_sor=omega_sor,
        tol=tol,
        max_iter=max_iter,
        use_cavitation=use_cavitation
    )


class ReynoldsSolver:
    """
    Высокоуровневый класс для решения уравнения Рейнольдса.
    """

    def __init__(
        self,
        grid: Grid,
        lambda_ratio: float,
        omega_sor: float = 1.7,
        tol: float = 1e-6,
        max_iter: int = 10000,
        use_cavitation: bool = True,
    ):
        """
        Args:
            grid: расчётная сетка
            lambda_ratio: λ = 2R_J/L
            omega_sor: параметр релаксации SOR
            tol: критерий сходимости
            max_iter: макс. итераций
            use_cavitation: P ≥ 0
        """
        self.grid = grid
        self.lambda_sq = lambda_ratio ** 2
        self.omega_sor = omega_sor
        self.tol = tol
        self.max_iter = max_iter
        self.use_cavitation = use_cavitation

    def solve_static(
        self,
        H: np.ndarray,
        eta_hat: float = 1.0,
    ) -> ReynoldsSolution:
        """
        Решение стационарного уравнения.

        Args:
            H: безразмерная толщина плёнки
            eta_hat: безразмерная вязкость η̂ = η(T)/η_ref

        Returns:
            ReynoldsSolution
        """
        P, residual, iterations = solve_reynolds_static(
            H, eta_hat,
            self.grid.d_phi, self.grid.d_Z,
            self.lambda_sq,
            self.omega_sor, self.tol, self.max_iter,
            self.use_cavitation
        )

        return ReynoldsSolution(
            P=P,
            converged=(iterations < self.max_iter),
            iterations=iterations,
            residual=residual
        )

    def solve_with_squeeze(
        self,
        H: np.ndarray,
        dH_dtau: np.ndarray,
        eta_hat: float = 1.0,
        beta: float = 2.0,
    ) -> ReynoldsSolution:
        """
        Решение с squeeze-членом (для демпфирования).

        Args:
            H: толщина плёнки
            dH_dtau: ∂H/∂τ = ξ'·cos(φ) + η'·sin(φ)
            eta_hat: безразмерная вязкость
            beta: коэффициент squeeze
        """
        P, residual, iterations = solve_reynolds_with_squeeze(
            H, dH_dtau, eta_hat,
            self.grid.d_phi, self.grid.d_Z,
            self.lambda_sq,
            beta,
            self.omega_sor, self.tol, self.max_iter,
            self.use_cavitation
        )

        return ReynoldsSolution(
            P=P,
            converged=(iterations < self.max_iter),
            iterations=iterations,
            residual=residual
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .geometry import create_grid, compute_film_thickness_static
    from .parameters import create_chinese_paper_bearing

    # Тест
    model = create_chinese_paper_bearing()
    grid = create_grid(model.numerical.N_phi, model.numerical.N_Z)

    epsilon_0 = 0.5
    T = 80.0
    eta_hat = model.lubricant.eta_hat(T)

    # Зазор без текстуры
    H = compute_film_thickness_static(grid, model.geometry, epsilon_0, texture=None)

    # Решение
    solver = ReynoldsSolver(
        grid,
        model.geometry.lambda_ratio,
        omega_sor=model.numerical.omega_GS,
        tol=model.numerical.tol_Re,
        max_iter=model.numerical.max_iter_Re,
    )

    solution = solver.solve_static(H, eta_hat)

    print(f"Сходимость: {solution.converged}")
    print(f"Итераций: {solution.iterations}")
    print(f"Невязка: {solution.residual:.2e}")
    print(f"P: min={solution.P.min():.4f}, max={solution.P.max():.4f}")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Давление при Z=0
    Z_idx = grid.N_Z // 2
    axes[0].plot(np.rad2deg(grid.phi), solution.P[Z_idx, :])
    axes[0].set_xlabel('φ, град')
    axes[0].set_ylabel('P (безразмерное)')
    axes[0].set_title(f'Давление при Z=0 (ε₀={epsilon_0}, T={T}°C)')
    axes[0].grid(True)

    # Контурный график
    c = axes[1].contourf(
        np.rad2deg(grid.Phi_mesh), grid.Z_mesh,
        solution.P, levels=50, cmap='plasma'
    )
    plt.colorbar(c, ax=axes[1], label='P')
    axes[1].set_xlabel('φ, град')
    axes[1].set_ylabel('Z')
    axes[1].set_title('Поле давления P(φ, Z)')

    plt.tight_layout()
    plt.savefig('reynolds_test.png', dpi=150)
    print("Сохранено в reynolds_test.png")
