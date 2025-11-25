"""
Геометрия подшипника: сетка, зазор, текстура.

Система координат:
- φ ∈ [0, 2π]: угол от линии центров
- Z ∈ [-1, 1]: безразмерная осевая координата, Z = 2z/L
- φ = 0 соответствует минимальному зазору (h_min = c(1-ε))
- φ = π соответствует максимальному зазору (h_max = c(1+ε))

Безразмерный зазор:
    H = h/c = 1 - ε·cos(φ)

Обратите внимание: здесь "-cos(φ)", а не "+cos(φ)" как в исходном коде.
Это обеспечивает минимум при φ=0, что соответствует стандартной системе координат.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from .parameters import BearingModel, BearingGeometry, TextureParameters


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


def create_grid(num_phi: int, num_Z: int) -> Grid:
    """
    Создание расчётной сетки.

    Args:
        num_phi: количество узлов по φ
        num_Z: количество узлов по Z

    Returns:
        Grid объект
    """
    phi = np.linspace(0, 2 * np.pi, num_phi)
    Z = np.linspace(-1, 1, num_Z)

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


def compute_film_thickness(
    grid: Grid,
    geometry: BearingGeometry,
    epsilon: float,
    texture: Optional[TextureParameters] = None,
) -> np.ndarray:
    """
    Вычисление безразмерной толщины смазочного слоя H = h/c.

    H(φ, Z) = 1 - ε·cos(φ) + H_texture(φ, Z)

    Args:
        grid: расчётная сетка
        geometry: геометрия подшипника
        epsilon: эксцентриситет (0..1)
        texture: параметры текстуры (None = гладкий подшипник)

    Returns:
        H: безразмерная толщина плёнки (N_Z x N_phi)
    """
    # Базовый зазор (гладкий подшипник)
    # Минус перед cos — минимум при φ=0
    H = 1 - epsilon * np.cos(grid.Phi_mesh)

    # Добавляем текстуру если есть
    if texture is not None and texture.enabled:
        H_tex = compute_texture_contribution(grid, geometry, texture)
        H = H + H_tex

    return H


def compute_texture_contribution(
    grid: Grid,
    geometry: BearingGeometry,
    texture: TextureParameters,
) -> np.ndarray:
    """
    Вклад эллипсоидальных углублений в толщину плёнки.

    Каждое углубление описывается:
        ΔH = H_p · √(1 - (Δφ/B)² - (ΔZ/A)²)

    где:
        H_p = h_p/c — безразмерная глубина
        A = 2a/L — безразмерная полуось по Z
        B = b/R — безразмерная полуось по φ

    Args:
        grid: расчётная сетка
        geometry: геометрия подшипника
        texture: параметры текстуры

    Returns:
        H_tex: вклад текстуры в H (N_Z x N_phi)
    """
    H_tex = np.zeros_like(grid.Phi_mesh)

    # Безразмерные параметры
    H_p = texture.h_p / geometry.c
    A = 2 * texture.a / geometry.L  # полуось по Z
    B = texture.b / geometry.R       # полуось по φ

    # Центры углублений
    phi_centers, Z_centers = compute_depression_centers(texture)

    # Добавляем каждое углубление
    for phi_c in phi_centers:
        for Z_c in Z_centers:
            # Расстояние до центра с учётом периодичности по φ
            delta_phi = np.arctan2(
                np.sin(grid.Phi_mesh - phi_c),
                np.cos(grid.Phi_mesh - phi_c)
            )
            delta_Z = grid.Z_mesh - Z_c

            # Эллиптический параметр
            expr = (delta_phi / B) ** 2 + (delta_Z / A) ** 2

            # Внутри углубления
            mask = expr <= 1
            H_tex[mask] += H_p * np.sqrt(1 - expr[mask])

    return H_tex


def compute_depression_centers(
    texture: TextureParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисление координат центров углублений.

    Углубления равномерно распределены в зоне [phi_start, phi_end] × [-1, 1].

    Returns:
        phi_centers: координаты центров по φ
        Z_centers: координаты центров по Z
    """
    # Безразмерные полуоси (используем для расчёта зазоров)
    # Здесь нужны абсолютные значения для расчёта расположения
    # Примем что A и B уже заданы в texture

    # Упрощённый расчёт: равномерное распределение центров
    phi_centers = np.linspace(
        texture.phi_start,
        texture.phi_end,
        texture.N_phi,
        endpoint=True
    )

    Z_centers = np.linspace(
        -0.9,  # не совсем до края
        0.9,
        texture.N_Z,
        endpoint=True
    )

    return phi_centers, Z_centers


def compute_film_thickness_dynamic(
    grid: Grid,
    geometry: BearingGeometry,
    epsilon_x: float,
    epsilon_y: float,
    texture: Optional[TextureParameters] = None,
) -> np.ndarray:
    """
    Зазор при произвольном положении центра вала.

    Используется для динамических расчётов (орбиты).

    H(φ) = 1 - εx·cos(φ) - εy·sin(φ)

    где εx, εy — компоненты эксцентриситета по осям x и y.

    Args:
        grid: расчётная сетка
        geometry: геометрия подшипника
        epsilon_x: компонента эксцентриситета по x (вдоль линии центров)
        epsilon_y: компонента эксцентриситета по y (перпендикулярно)
        texture: параметры текстуры

    Returns:
        H: безразмерная толщина плёнки
    """
    H = 1 - epsilon_x * np.cos(grid.Phi_mesh) - epsilon_y * np.sin(grid.Phi_mesh)

    if texture is not None and texture.enabled:
        H_tex = compute_texture_contribution(grid, geometry, texture)
        H = H + H_tex

    return H


class BearingGeometryProcessor:
    """
    Класс для работы с геометрией подшипника.

    Инкапсулирует создание сетки и вычисление зазоров.
    """

    def __init__(self, model: BearingModel):
        """
        Args:
            model: полная модель подшипника
        """
        self.model = model
        self.grid = create_grid(
            model.grid.num_phi,
            model.grid.num_Z
        )

    def get_film_thickness(
        self,
        epsilon: Optional[float] = None,
        with_texture: bool = True,
    ) -> np.ndarray:
        """
        Получить толщину плёнки H.

        Args:
            epsilon: эксцентриситет (если None — из модели)
            with_texture: учитывать текстуру

        Returns:
            H: безразмерная толщина (N_Z x N_phi)
        """
        if epsilon is None:
            epsilon = self.model.operating.epsilon

        texture = self.model.texture if with_texture else None

        return compute_film_thickness(
            self.grid,
            self.model.geometry,
            epsilon,
            texture
        )

    def get_film_thickness_dynamic(
        self,
        epsilon_x: float,
        epsilon_y: float,
        with_texture: bool = True,
    ) -> np.ndarray:
        """
        Толщина плёнки при произвольном положении вала.
        """
        texture = self.model.texture if with_texture else None

        return compute_film_thickness_dynamic(
            self.grid,
            self.model.geometry,
            epsilon_x,
            epsilon_y,
            texture
        )

    def get_dimensional_thickness(self, H: np.ndarray) -> np.ndarray:
        """
        Перевод безразмерной толщины в размерную (м).

        h = H * c
        """
        return H * self.model.geometry.c


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .parameters import create_test_bearing

    # Тест
    model = create_test_bearing()
    processor = BearingGeometryProcessor(model)

    # Зазор без текстуры и с текстурой
    H_smooth = processor.get_film_thickness(with_texture=False)
    H_textured = processor.get_film_thickness(with_texture=True)

    # Сечение при Z=0
    Z_idx = processor.grid.N_Z // 2
    phi = processor.grid.phi

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(phi, H_smooth[Z_idx, :], 'b-', label='Гладкий')
    plt.plot(phi, H_textured[Z_idx, :], 'r-', label='С текстурой')
    plt.xlabel('φ, рад')
    plt.ylabel('H')
    plt.title('Безразмерный зазор при Z=0')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.contourf(processor.grid.Phi_mesh, processor.grid.Z_mesh, H_textured, levels=50)
    plt.colorbar(label='H')
    plt.xlabel('φ, рад')
    plt.ylabel('Z')
    plt.title('Распределение H(φ, Z)')

    plt.tight_layout()
    plt.show()
