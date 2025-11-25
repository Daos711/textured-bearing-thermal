"""
Параметры подшипника скольжения для шарошечного долота.

Система координат:
- φ (phi): угол от линии центров (0..2π), φ=0 — минимальный зазор
- z: осевая координата (-L/2..+L/2), безразмерная Z = 2z/L ∈ [-1, 1]
- Ось x направлена по линии центров (от центра втулки к центру вала)
- Ось y перпендикулярна x в плоскости вращения
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BearingGeometry:
    """Геометрия подшипника."""
    R: float = 0.035        # Радиус вала (м)
    c: float = 0.0005       # Радиальный зазор (м)
    L: float = 0.056        # Длина подшипника (м)

    @property
    def D(self) -> float:
        """Диаметр вала."""
        return 2 * self.R

    @property
    def psi(self) -> float:
        """Относительный зазор ψ = c/R."""
        return self.c / self.R

    @property
    def L_over_D(self) -> float:
        """Отношение L/D."""
        return self.L / self.D


@dataclass
class TextureParameters:
    """Параметры текстуры (эллипсоидальные углубления)."""
    enabled: bool = True
    h_p: float = 0.0001     # Глубина углубления (м)
    a: float = 0.00241      # Полуось по Z (м)
    b: float = 0.002214     # Полуось по φ (м)
    N_phi: int = 8          # Количество углублений по φ
    N_Z: int = 11           # Количество углублений по Z
    phi_start_deg: float = 90.0   # Начальный угол зоны текстуры (°)
    phi_end_deg: float = 270.0    # Конечный угол зоны текстуры (°)

    @property
    def phi_start(self) -> float:
        """Начальный угол в радианах."""
        return np.deg2rad(self.phi_start_deg)

    @property
    def phi_end(self) -> float:
        """Конечный угол в радианах."""
        return np.deg2rad(self.phi_end_deg)


@dataclass
class LubricantProperties:
    """
    Свойства смазки с температурной зависимостью.

    Вязкость по формуле Vogel-Cameron:
        η(T) = η_0 * exp(-β * (T - T_0))

    Или по формуле Walther (более точная для масел):
        log(log(ν + 0.7)) = A - B * log(T)
    """
    eta_0: float = 0.01105      # Динамическая вязкость при T_0 (Па·с)
    T_0: float = 40.0           # Референсная температура (°C)
    beta: float = 0.03          # Коэффициент температурной зависимости (1/K)
    rho: float = 870.0          # Плотность (кг/м³)
    c_p: float = 2000.0         # Удельная теплоёмкость (Дж/(кг·K))
    k: float = 0.14             # Теплопроводность (Вт/(м·K))

    def viscosity(self, T: float) -> float:
        """
        Вязкость при температуре T.

        Args:
            T: Температура (°C)

        Returns:
            Динамическая вязкость η (Па·с)
        """
        return self.eta_0 * np.exp(-self.beta * (T - self.T_0))

    def viscosity_array(self, T: np.ndarray) -> np.ndarray:
        """Вязкость для массива температур."""
        return self.eta_0 * np.exp(-self.beta * (T - self.T_0))


@dataclass
class OperatingConditions:
    """Условия работы подшипника."""
    n: float = 100.0            # Скорость вращения (об/мин) — для долота меньше!
    epsilon: float = 0.6        # Эксцентриситет (0..1)
    T_inlet: float = 60.0       # Температура на входе (°C)
    T_ambient: float = 80.0     # Температура окружающей среды/забоя (°C)

    @property
    def omega(self) -> float:
        """Угловая скорость (рад/с)."""
        return 2 * np.pi * self.n / 60


@dataclass
class GridParameters:
    """Параметры расчётной сетки."""
    num_phi: int = 360      # Узлов по φ
    num_Z: int = 100        # Узлов по Z


@dataclass
class SolverSettings:
    """Настройки численного решателя."""
    max_iter: int = 50000       # Максимум итераций
    tol: float = 1e-6           # Критерий сходимости
    relaxation: float = 1.5     # Параметр релаксации (SOR)

    # Для термогидродинамической модели
    thd_max_iter: int = 20      # Итерации связи Reynolds-Energy
    thd_tol: float = 1e-3       # Критерий сходимости по температуре


@dataclass
class BearingModel:
    """
    Полная модель подшипника — собирает все параметры.
    """
    geometry: BearingGeometry = field(default_factory=BearingGeometry)
    texture: TextureParameters = field(default_factory=TextureParameters)
    lubricant: LubricantProperties = field(default_factory=LubricantProperties)
    operating: OperatingConditions = field(default_factory=OperatingConditions)
    grid: GridParameters = field(default_factory=GridParameters)
    solver: SolverSettings = field(default_factory=SolverSettings)

    @property
    def U(self) -> float:
        """Линейная скорость поверхности вала (м/с)."""
        return self.operating.omega * self.geometry.R

    @property
    def pressure_scale(self) -> float:
        """
        Масштаб давления для безразмерных величин.
        P* = P / pressure_scale
        """
        eta = self.lubricant.viscosity(self.operating.T_inlet)
        return (6 * eta * self.U * self.geometry.R) / (self.geometry.c ** 2)

    @property
    def load_scale(self) -> float:
        """Масштаб нагрузки (Н)."""
        return self.pressure_scale * self.geometry.R * self.geometry.L / 2

    @property
    def friction_scale(self) -> float:
        """Масштаб силы трения (Н)."""
        eta = self.lubricant.viscosity(self.operating.T_inlet)
        return (eta * self.U * self.geometry.R * self.geometry.L) / self.geometry.c

    def info(self) -> str:
        """Информация о модели."""
        eta = self.lubricant.viscosity(self.operating.T_inlet)
        lines = [
            "=" * 50,
            "ПАРАМЕТРЫ МОДЕЛИ ПОДШИПНИКА",
            "=" * 50,
            f"Геометрия:",
            f"  Радиус R = {self.geometry.R*1000:.2f} мм",
            f"  Зазор c = {self.geometry.c*1000:.3f} мм",
            f"  Длина L = {self.geometry.L*1000:.2f} мм",
            f"  L/D = {self.geometry.L_over_D:.2f}",
            f"  ψ = c/R = {self.geometry.psi:.4f}",
            "",
            f"Режим работы:",
            f"  n = {self.operating.n:.1f} об/мин",
            f"  ω = {self.operating.omega:.2f} рад/с",
            f"  U = {self.U:.3f} м/с",
            f"  ε = {self.operating.epsilon:.2f}",
            "",
            f"Смазка при T = {self.operating.T_inlet}°C:",
            f"  η = {eta*1000:.3f} мПа·с",
            f"  ρ = {self.lubricant.rho:.0f} кг/м³",
            "",
            f"Масштабы:",
            f"  Давление: {self.pressure_scale/1e6:.3f} МПа",
            f"  Нагрузка: {self.load_scale:.1f} Н",
            "",
            f"Текстура: {'Да' if self.texture.enabled else 'Нет'}",
            "=" * 50,
        ]
        return "\n".join(lines)


# Пресеты для разных применений
def create_roller_cone_bit_bearing() -> BearingModel:
    """
    Параметры для опоры шарошечного долота.

    Особенности:
    - Низкая скорость вращения (50-150 об/мин)
    - Высокая температура забоя (растёт с глубиной)
    - Тяжёлые нагрузки
    """
    return BearingModel(
        geometry=BearingGeometry(
            R=0.030,        # 30 мм
            c=0.0003,       # 0.3 мм
            L=0.040,        # 40 мм
        ),
        texture=TextureParameters(
            enabled=True,
            h_p=0.00008,    # 80 мкм
            a=0.002,
            b=0.0018,
            N_phi=6,
            N_Z=8,
        ),
        lubricant=LubricantProperties(
            eta_0=0.015,    # Более вязкое масло
            T_0=40.0,
            beta=0.025,
        ),
        operating=OperatingConditions(
            n=100.0,        # об/мин
            epsilon=0.6,
            T_inlet=80.0,   # Температура на забое
            T_ambient=100.0,
        ),
        grid=GridParameters(
            num_phi=360,
            num_Z=100,
        ),
    )


def create_test_bearing() -> BearingModel:
    """Тестовые параметры из исходного кода."""
    return BearingModel(
        geometry=BearingGeometry(R=0.035, c=0.0005, L=0.056),
        texture=TextureParameters(
            enabled=True,
            h_p=0.0001,
            a=0.00241,
            b=0.002214,
            N_phi=8,
            N_Z=11,
        ),
        lubricant=LubricantProperties(eta_0=0.01105, T_0=40.0, beta=0.03),
        operating=OperatingConditions(n=2980.0, epsilon=0.6, T_inlet=40.0),
        grid=GridParameters(num_phi=500, num_Z=500),
    )


if __name__ == "__main__":
    # Тест
    model = create_roller_cone_bit_bearing()
    print(model.info())

    # Проверка температурной зависимости вязкости
    print("\nЗависимость вязкости от температуры:")
    for T in [40, 60, 80, 100, 120]:
        eta = model.lubricant.viscosity(T)
        print(f"  T = {T}°C: η = {eta*1000:.3f} мПа·с")
