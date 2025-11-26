"""
Параметры устойчивости подшипника.

Из постановки задачи (раздел 9):

1. Эквивалентная жёсткость K_eq:
    K_eq = (K̄_xx·C̄_yy + K̄_yy·C̄_xx - K̄_xy·C̄_yx - K̄_yx·C̄_xy) / (C̄_xx + C̄_yy)

2. Квадрат логарифмического декремента γ²_st:
    γ²_st = ((K_eq - K̄_xx)(K_eq - K̄_yy) - K̄_xy·K̄_yx) / (C̄_xx·C̄_yy - C̄_xy·C̄_yx)

3. Критическая скорость потери устойчивости ω_st (в безразмерной форме):
    ω_st = K_eq / γ²_st

Эти три величины строятся в виде поверхностей K_eq(W,T), γ²_st(W,T), ω_st(W,T)
для гладкого и текстурированного подшипника.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .forces import StiffnessCoefficients, DampingCoefficients, FullCoefficients


@dataclass
class StabilityParameters:
    """Параметры устойчивости."""
    K_eq: float             # Эквивалентная жёсткость (безразмерная)
    gamma_sq: float         # γ²_st — квадрат логарифмического декремента
    omega_st: float         # ω_st — критическая скорость (безразмерная)
    is_stable: bool         # Флаг устойчивости (γ²_st > 0)


def compute_stability_parameters(
    K: StiffnessCoefficients,
    C: DampingCoefficients,
) -> StabilityParameters:
    """
    Вычисление параметров устойчивости по безразмерным коэффициентам.

    Args:
        K: коэффициенты жёсткости (используются безразмерные K̄_ij)
        C: коэффициенты демпфирования (используются безразмерные C̄_ij)

    Returns:
        StabilityParameters
    """
    # Безразмерные коэффициенты
    K_xx = K.K_xx_bar
    K_xy = K.K_xy_bar
    K_yx = K.K_yx_bar
    K_yy = K.K_yy_bar

    C_xx = C.C_xx_bar
    C_xy = C.C_xy_bar
    C_yx = C.C_yx_bar
    C_yy = C.C_yy_bar

    # 1. Эквивалентная жёсткость K_eq
    denom_Keq = C_xx + C_yy
    if abs(denom_Keq) < 1e-12:
        K_eq = 0.0
    else:
        K_eq = (K_xx * C_yy + K_yy * C_xx - K_xy * C_yx - K_yx * C_xy) / denom_Keq

    # 2. Квадрат логарифмического декремента γ²_st
    numerator_gamma = (K_eq - K_xx) * (K_eq - K_yy) - K_xy * K_yx
    denom_gamma = C_xx * C_yy - C_xy * C_yx

    if abs(denom_gamma) < 1e-12:
        gamma_sq = 0.0
    else:
        gamma_sq = numerator_gamma / denom_gamma

    # 3. Критическая скорость ω_st
    if abs(gamma_sq) < 1e-12:
        omega_st = float('inf')  # Нет критической скорости в обычном смысле
    else:
        omega_st = K_eq / gamma_sq

    # Устойчивость: γ²_st > 0
    is_stable = gamma_sq > 0

    return StabilityParameters(
        K_eq=K_eq,
        gamma_sq=gamma_sq,
        omega_st=omega_st,
        is_stable=is_stable
    )


def compute_stability_from_full_coefficients(
    coeffs: FullCoefficients,
) -> StabilityParameters:
    """
    Вычисление параметров устойчивости из полного набора коэффициентов.

    Args:
        coeffs: FullCoefficients (содержит stiffness и damping)

    Returns:
        StabilityParameters
    """
    return compute_stability_parameters(coeffs.stiffness, coeffs.damping)


def compute_stability_margins(
    stability: StabilityParameters,
    omega_operating: float,
) -> dict:
    """
    Вычисление запасов устойчивости.

    Args:
        stability: параметры устойчивости
        omega_operating: рабочая угловая скорость (рад/с)

    Returns:
        Словарь с запасами
    """
    # Запас по скорости (если ω_st конечна)
    if stability.omega_st > 0 and stability.omega_st < float('inf'):
        speed_margin = stability.omega_st / omega_operating
    else:
        speed_margin = float('inf')

    return {
        'speed_margin': speed_margin,      # Запас по скорости ω_st/ω_operating
        'gamma_sq': stability.gamma_sq,    # γ² — характеризует демпфирование
        'is_stable': stability.is_stable,  # Устойчив ли при данных условиях
    }


if __name__ == "__main__":
    # Тест с примерными значениями
    K = StiffnessCoefficients(
        K_xx=1e6, K_xy=0.5e6, K_yx=-0.5e6, K_yy=2e6,
        K_xx_bar=1.0, K_xy_bar=0.5, K_yx_bar=-0.5, K_yy_bar=2.0,
    )
    C = DampingCoefficients(
        C_xx=1e3, C_xy=0.2e3, C_yx=-0.2e3, C_yy=1.5e3,
        C_xx_bar=1.0, C_xy_bar=0.2, C_yx_bar=-0.2, C_yy_bar=1.5,
    )

    stability = compute_stability_parameters(K, C)

    print("=" * 50)
    print("ТЕСТ ПАРАМЕТРОВ УСТОЙЧИВОСТИ")
    print("=" * 50)
    print(f"K_eq = {stability.K_eq:.4f}")
    print(f"γ²_st = {stability.gamma_sq:.4f}")
    print(f"ω_st = {stability.omega_st:.4f}")
    print(f"Устойчив: {stability.is_stable}")

    # Запасы при ω = 10 рад/с
    margins = compute_stability_margins(stability, omega_operating=10.0)
    print(f"\nЗапас по скорости: {margins['speed_margin']:.2f}")
