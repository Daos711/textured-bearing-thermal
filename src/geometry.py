"""
Геометрия подшипника: сетка, зазор, текстура.

Система координат (см. постановку задачи):
- ось x направлена по линии минимального зазора
- ось y перпендикулярна ей
- ось z вдоль оси подшипника
- φ отсчитывается от оси x против часовой стрелки
- Z = z/L ∈ [-1, 1] — безразмерная осевая координата

Безразмерный зазор:
    H = h/c₀ = 1 + ε₀·cos(φ) + ξ·cos(φ) + η·sin(φ) + H_tex(φ,Z) + H_T(T)

При φ=0: H = 1+ε (максимум)
При φ=π: H = 1-ε (минимум, зона нагрузки)

Текстура располагается в зоне 90°-270° — это зона высокого давления.
Поддерживается филлотаксическое расположение углублений.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

from .parameters import (
    BearingModel, BearingGeometry, TextureParameters,
    MaterialProperties, LubricantProperties
)


@dataclass
class Grid:
    """
    Расчётная сетка для подшипника.

    Attributes:
        phi: 1D массив углов [0, 2π]
        Z: 1D массив безразмерных осевых координат [-1, 1]
        Phi_mesh: 2D сетка углов (N_Z x N_phi)
        Z_mesh: 2D сетка координат Z (N_Z x N_phi)
        d_phi: шаг по φ
        d_Z: шаг по Z
    """
    phi: np.ndarray
    Z: np.ndarray
    Phi_mesh: np.ndarray
    Z_mesh: np.ndarray
    d_phi: float
    d_Z: float

    @property
    def shape(self) -> Tuple[int, int]:
        """Размер сетки (N_Z, N_phi)."""
        return self.Phi_mesh.shape

    @property
    def N_phi(self) -> int:
        return len(self.phi)

    @property
    def N_Z(self) -> int:
        return len(self.Z)

    @property
    def cos_phi(self) -> np.ndarray:
        """cos(φ) на сетке."""
        return np.cos(self.Phi_mesh)

    @property
    def sin_phi(self) -> np.ndarray:
        """sin(φ) на сетке."""
        return np.sin(self.Phi_mesh)


def create_grid(N_phi: int, N_Z: int) -> Grid:
    """
    Создание расчётной сетки.

    Args:
        N_phi: количество узлов по φ
        N_Z: количество узлов по Z

    Returns:
        Grid объект
    """
    phi = np.linspace(0, 2 * np.pi, N_phi)
    Z = np.linspace(-1, 1, N_Z)

    Phi_mesh, Z_mesh = np.meshgrid(phi, Z)

    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]

    return Grid(
        phi=phi,
        Z=Z,
        Phi_mesh=Phi_mesh,
        Z_mesh=Z_mesh,
        d_phi=d_phi,
        d_Z=d_Z,
    )


def compute_film_thickness_static(
    grid: Grid,
    geometry: BearingGeometry,
    epsilon_0: float,
    texture: Optional[TextureParameters] = None,
    H_T: float = 0.0,
) -> np.ndarray:
    """
    Статический профиль зазора (без динамических смещений).

    H_stat(φ, Z; ε₀, T) = 1 + ε₀·cos(φ) + H_tex(φ, Z) + H_T(T)

    Args:
        grid: расчётная сетка
        geometry: геометрия подшипника
        epsilon_0: статический эксцентриситет (0..1)
        texture: параметры текстуры (None = гладкий подшипник)
        H_T: безразмерная добавка от теплового расширения

    Returns:
        H: безразмерная толщина плёнки (N_Z x N_phi)
    """
    # Базовый зазор
    H = 1 + epsilon_0 * np.cos(grid.Phi_mesh)

    # Добавка от температуры
    H = H + H_T

    # Добавляем текстуру если есть
    if texture is not None and texture.enabled:
        H_tex = compute_texture_contribution(grid, geometry, texture)
        H = H + H_tex

    return H


def compute_film_thickness_full(
    grid: Grid,
    geometry: BearingGeometry,
    epsilon_0: float,
    xi: float = 0.0,
    eta: float = 0.0,
    texture: Optional[TextureParameters] = None,
    H_T: float = 0.0,
) -> np.ndarray:
    """
    Полный профиль зазора с динамическими смещениями.

    H(φ, Z, t) = 1 + ε₀·cos(φ) + ξ(t)·cos(φ) + η(t)·sin(φ) + H_tex(φ, Z) + H_T(T)

    где:
        ε₀ — статический эксцентриситет
        ξ = x/c₀, η = y/c₀ — безразмерные динамические смещения
        H_T = Δc_T/c₀ — добавка от теплового расширения

    Args:
        grid: расчётная сетка
        geometry: геометрия подшипника
        epsilon_0: статический эксцентриситет
        xi: безразмерное смещение по x
        eta: безразмерное смещение по y
        texture: параметры текстуры
        H_T: безразмерная добавка от теплового расширения

    Returns:
        H: безразмерная толщина плёнки (N_Z x N_phi)
    """
    # Полный профиль зазора
    H = 1 + (epsilon_0 + xi) * np.cos(grid.Phi_mesh) + eta * np.sin(grid.Phi_mesh)

    # Добавка от температуры
    H = H + H_T

    # Добавляем текстуру
    if texture is not None and texture.enabled:
        H_tex = compute_texture_contribution(grid, geometry, texture)
        H = H + H_tex

    return H


def compute_dH_dphi(
    grid: Grid,
    epsilon_0: float,
    xi: float = 0.0,
    eta: float = 0.0,
) -> np.ndarray:
    """
    Производная зазора по φ (для правой части уравнения Рейнольдса).

    ∂H/∂φ = -(ε₀ + ξ)·sin(φ) + η·cos(φ)

    Args:
        grid: расчётная сетка
        epsilon_0: статический эксцентриситет
        xi: безразмерное смещение по x
        eta: безразмерное смещение по y

    Returns:
        dH_dphi: производная (N_Z x N_phi)
    """
    return -(epsilon_0 + xi) * np.sin(grid.Phi_mesh) + eta * np.cos(grid.Phi_mesh)


def compute_dH_dtau(
    grid: Grid,
    xi_dot: float = 0.0,
    eta_dot: float = 0.0,
) -> np.ndarray:
    """
    Производная зазора по безразмерному времени τ (для демпфирования).

    ∂H/∂τ = ξ'(τ)·cos(φ) + η'(τ)·sin(φ)

    где ξ' = dξ/dτ = ẋ/(c₀·ω), η' = dη/dτ = ẏ/(c₀·ω)

    Args:
        grid: расчётная сетка
        xi_dot: безразмерная скорость по x
        eta_dot: безразмерная скорость по y

    Returns:
        dH_dtau: производная (N_Z x N_phi)
    """
    return xi_dot * np.cos(grid.Phi_mesh) + eta_dot * np.sin(grid.Phi_mesh)


# =============================================================================
# Текстура (эллипсоидальные углубления)
# =============================================================================

def compute_texture_contribution(
    grid: Grid,
    geometry: BearingGeometry,
    texture: TextureParameters,
) -> np.ndarray:
    """
    Вклад эллипсоидальных углублений в толщину плёнки.

    Каждое углубление описывается:
        H_tex = H_p · √(1 - (Δφ/B)² - ((Z - Z_c)/A)²)

    где:
        H_p = h_p/c₀ — безразмерная глубина
        A = 2a/L — безразмерная полуось по Z
        B = b/R_J — безразмерная полуось по φ

    Args:
        grid: расчётная сетка
        geometry: геометрия подшипника
        texture: параметры текстуры

    Returns:
        H_tex: вклад текстуры в H (N_Z x N_phi)
    """
    H_tex = np.zeros_like(grid.Phi_mesh)

    # Безразмерные параметры
    H_p = texture.h_p / geometry.c0
    A = 2 * texture.a / geometry.L   # полуось по Z
    B = texture.b / geometry.R_J     # полуось по φ

    # Получаем центры углублений
    if texture.use_phyllotaxis:
        centers = compute_phyllotaxis_centers(texture, geometry)
    else:
        centers = compute_regular_centers(texture, geometry)

    # Добавляем каждое углубление
    for phi_c, Z_c in centers:
        # Расстояние до центра с учётом периодичности по φ
        delta_phi = np.arctan2(
            np.sin(grid.Phi_mesh - phi_c),
            np.cos(grid.Phi_mesh - phi_c)
        )
        delta_Z = grid.Z_mesh - Z_c

        # Эллиптический параметр
        r_sq = (delta_phi / B) ** 2 + (delta_Z / A) ** 2

        # Внутри углубления
        mask = r_sq <= 1
        H_tex[mask] += H_p * np.sqrt(1 - r_sq[mask])

    return H_tex


def compute_phyllotaxis_centers(
    texture: TextureParameters,
    geometry: BearingGeometry,
) -> List[Tuple[float, float]]:
    """
    Филлотаксическое расположение центров углублений.

    T спиралей, равномерно распределённых по углу.
    На каждой спирали точки расположены с шагом c_step по высоте
    и угловым шагом alpha_step_deg.

    Формулы:
        φ_k,n = 2π·k/T + n·α  (угол точки n на спирали k)
        z_n = z_vals[n]        (высота точки)

    Args:
        texture: параметры текстуры
        geometry: геометрия подшипника

    Returns:
        centers: список пар (φ_c, Z_c) для каждого углубления
    """
    T = texture.T_spirals
    alpha_step = np.deg2rad(texture.alpha_step_deg)
    c_step = texture.c_step

    # Безразмерные полуоси
    A = 2 * texture.a / geometry.L  # полуось по Z
    B = texture.b / geometry.R_J    # полуось по φ

    # Эффективная зона по Z (оставляем место для полуоси углубления)
    Z_min = texture.Z_min + A
    Z_max = texture.Z_max - A

    if Z_max <= Z_min:
        Z_min = texture.Z_min
        Z_max = texture.Z_max

    # Переводим c_step в безразмерные координаты
    # z ∈ [-L/2, L/2], Z = 2z/L ∈ [-1, 1]
    # Δz соответствует ΔZ = 2·Δz/L
    dZ_step = 2 * c_step / geometry.L

    # Число точек по высоте на каждой спирали
    N_points = max(1, int((Z_max - Z_min) / dZ_step) + 1)
    Z_vals = np.linspace(Z_min, Z_max, N_points)

    centers = []
    for k in range(T):
        phi0 = 2 * np.pi * k / T  # начальный угол спирали
        for n, Z_c in enumerate(Z_vals):
            phi_c = phi0 + n * alpha_step

            # Приводим к [0, 2π)
            phi_c = phi_c % (2 * np.pi)

            # Проверяем, что центр в зоне текстуры
            if texture.phi_min <= phi_c <= texture.phi_max:
                centers.append((phi_c, Z_c))

    return centers


def compute_regular_centers(
    texture: TextureParameters,
    geometry: BearingGeometry,
) -> List[Tuple[float, float]]:
    """
    Регулярное (сеточное) расположение центров углублений.

    Углубления равномерно распределены в зоне [phi_min, phi_max] × [Z_min, Z_max]
    с учётом их размеров.

    Args:
        texture: параметры текстуры
        geometry: геометрия подшипника

    Returns:
        centers: список пар (φ_c, Z_c) для каждого углубления
    """
    # Безразмерные полуоси
    A = 2 * texture.a / geometry.L
    B = texture.b / geometry.R_J

    N_phi = texture.N_phi
    N_Z = texture.N_Z

    phi_start = texture.phi_min
    phi_end = texture.phi_max

    # === Центры по φ ===
    if N_phi > 1:
        delta_phi_gap = (phi_end - phi_start - 2 * N_phi * B) / (N_phi - 1)
        if delta_phi_gap < 0:
            delta_phi_gap = 0
        delta_phi_center = 2 * B + delta_phi_gap
        phi_center_start = phi_start + B
        phi_centers = phi_center_start + delta_phi_center * np.arange(N_phi)
    else:
        phi_centers = np.array([(phi_start + phi_end) / 2])

    # === Центры по Z ===
    Z_range = texture.Z_max - texture.Z_min
    if N_Z > 1:
        delta_Z_gap = (Z_range - 2 * N_Z * A) / (N_Z - 1)
        if delta_Z_gap < 0:
            delta_Z_gap = 0
        delta_Z_center = 2 * A + delta_Z_gap
        Z_center_start = texture.Z_min + A
        Z_centers = Z_center_start + delta_Z_center * np.arange(N_Z)
    else:
        Z_centers = np.array([(texture.Z_min + texture.Z_max) / 2])

    # Собираем все центры
    centers = []
    for phi_c in phi_centers:
        for Z_c in Z_centers:
            centers.append((phi_c, Z_c))

    return centers


def get_texture_centers_count(
    texture: TextureParameters,
    geometry: BearingGeometry,
) -> int:
    """
    Подсчёт количества углублений.
    """
    if texture.use_phyllotaxis:
        centers = compute_phyllotaxis_centers(texture, geometry)
    else:
        centers = compute_regular_centers(texture, geometry)
    return len(centers)


# =============================================================================
# Вспомогательные классы
# =============================================================================

class GeometryProcessor:
    """
    Класс для работы с геометрией подшипника.
    """

    def __init__(self, model: BearingModel):
        """
        Args:
            model: полная модель подшипника
        """
        self.model = model
        self.grid = create_grid(
            model.numerical.N_phi,
            model.numerical.N_Z
        )

    def get_H_static(
        self,
        epsilon_0: float,
        T: float,
        with_texture: bool = True,
    ) -> np.ndarray:
        """
        Статический профиль зазора.

        Args:
            epsilon_0: статический эксцентриситет
            T: температура (°C)
            with_texture: учитывать текстуру

        Returns:
            H: безразмерная толщина (N_Z x N_phi)
        """
        H_T = self.model.material.H_T(self.model.geometry, T)
        texture = self.model.texture if with_texture else None

        return compute_film_thickness_static(
            self.grid,
            self.model.geometry,
            epsilon_0,
            texture,
            H_T
        )

    def get_H_full(
        self,
        epsilon_0: float,
        xi: float,
        eta: float,
        T: float,
        with_texture: bool = True,
    ) -> np.ndarray:
        """
        Полный профиль зазора с динамическими смещениями.
        """
        H_T = self.model.material.H_T(self.model.geometry, T)
        texture = self.model.texture if with_texture else None

        return compute_film_thickness_full(
            self.grid,
            self.model.geometry,
            epsilon_0,
            xi,
            eta,
            texture,
            H_T
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .parameters import create_chinese_paper_bearing

    # Тест
    model = create_chinese_paper_bearing()
    processor = GeometryProcessor(model)

    T = 80.0
    epsilon_0 = 0.5

    # Зазор без текстуры и с текстурой
    model.texture.enabled = False
    H_smooth = processor.get_H_static(epsilon_0, T, with_texture=False)

    model.texture.enabled = True
    H_textured = processor.get_H_static(epsilon_0, T, with_texture=True)

    print(f"Число углублений: {get_texture_centers_count(model.texture, model.geometry)}")
    print(f"H_smooth: min={H_smooth.min():.3f}, max={H_smooth.max():.3f}")
    print(f"H_textured: min={H_textured.min():.3f}, max={H_textured.max():.3f}")

    # Сечение при Z=0
    Z_idx = processor.grid.N_Z // 2
    phi = processor.grid.phi

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(np.rad2deg(phi), H_smooth[Z_idx, :], 'b-', label='Гладкий')
    axes[0].plot(np.rad2deg(phi), H_textured[Z_idx, :], 'r-', label='С текстурой')
    axes[0].set_xlabel('φ, град')
    axes[0].set_ylabel('H')
    axes[0].set_title('Безразмерный зазор при Z=0')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].axvline(x=90, color='g', linestyle='--', alpha=0.5)
    axes[0].axvline(x=270, color='g', linestyle='--', alpha=0.5)

    c = axes[1].contourf(
        np.rad2deg(processor.grid.Phi_mesh),
        processor.grid.Z_mesh,
        H_textured, levels=50
    )
    plt.colorbar(c, ax=axes[1], label='H')
    axes[1].set_xlabel('φ, град')
    axes[1].set_ylabel('Z')
    axes[1].set_title('Распределение H(φ, Z)')

    plt.tight_layout()
    plt.savefig('geometry_test.png', dpi=150)
    print("Сохранено в geometry_test.png")
