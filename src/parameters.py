"""
Параметры модели подшипника скольжения.

Структура параметров согласно постановке задачи:
- Геометрия: R_J, R_B, c0, L
- Текстура: h_p, a, b, параметры филлотаксиса
- Материалы: alpha_J, alpha_B, T_ref
- Смазка: eta_ref, beta_eta
- Режим работы: n, W_min, W_max, T_min, T_max
- Численные настройки: N_phi, N_Z, tol, omega_GS
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class BearingGeometry:
    """Геометрия подшипника."""
    R_J: float          # Радиус шейки (вала), м
    R_B: float          # Радиус вкладыша, м
    L: float            # Длина подшипника, м

    @property
    def c0(self) -> float:
        """Радиальный зазор при T_ref, м."""
        return self.R_B - self.R_J

    @property
    def D_J(self) -> float:
        """Диаметр шейки, м."""
        return 2 * self.R_J

    @property
    def psi(self) -> float:
        """Относительный зазор ψ = c0/R_J."""
        return self.c0 / self.R_J

    @property
    def lambda_ratio(self) -> float:
        """Отношение длин λ = 2*R_J/L."""
        return 2 * self.R_J / self.L


@dataclass
class TextureParameters:
    """Параметры текстуры (эллипсоидальные углубления)."""
    enabled: bool = False

    # Геометрия углубления
    h_p: float = 10e-6      # Глубина углубления, м
    a: float = 1e-3         # Полуось по z (осевая), м
    b: float = 1e-3         # Полуось по окружности, м

    # Филлотаксическое расположение
    use_phyllotaxis: bool = True
    T_spirals: int = 8          # Число спиралей
    alpha_step_deg: float = 15  # Угловой шаг вдоль спирали, градусы
    c_step: float = 4e-3        # Шаг по высоте между точками, м

    # Зона размещения текстуры (в радианах)
    phi_min: float = np.pi/2    # 90°
    phi_max: float = 3*np.pi/2  # 270°
    Z_min: float = -1.0         # Безразмерная координата
    Z_max: float = 1.0          # Безразмерная координата

    # Для регулярной сетки (если use_phyllotaxis=False)
    N_phi: int = 8      # Число углублений по окружности
    N_Z: int = 6        # Число рядов по оси z


@dataclass
class MaterialProperties:
    """Свойства материалов (тепловое расширение)."""
    alpha_J: float = 12e-6      # Коэф. линейного расширения вала, 1/°C
    alpha_B: float = 18e-6      # Коэф. линейного расширения вкладыша, 1/°C
    T_ref: float = 20.0         # Опорная температура, °C

    def clearance_change(self, geometry: BearingGeometry, T: float) -> float:
        """
        Изменение зазора из-за теплового расширения.

        Δc_T = R_B0 * α_B * ΔT - R_J0 * α_J * ΔT

        Returns:
            Δc_T: изменение зазора, м
        """
        delta_T = T - self.T_ref
        delta_c = geometry.R_B * self.alpha_B * delta_T - geometry.R_J * self.alpha_J * delta_T
        return delta_c

    def H_T(self, geometry: BearingGeometry, T: float) -> float:
        """
        Безразмерная добавка к зазору от температуры.

        H_T = Δc_T / c0
        """
        delta_c = self.clearance_change(geometry, T)
        return delta_c / geometry.c0


@dataclass
class LubricantProperties:
    """Свойства смазки."""
    eta_ref: float = 0.05       # Вязкость при T_ref, Па·с
    beta_eta: float = 0.03      # Параметр температурной зависимости, 1/°C
    T_ref: float = 40.0         # Опорная температура для вязкости, °C
    rho: float = 870.0          # Плотность, кг/м³
    c_p: float = 2000.0         # Теплоёмкость, Дж/(кг·°C)

    def viscosity(self, T: float) -> float:
        """
        Динамическая вязкость η(T) = η_ref * exp(-β_η * (T - T_ref)).
        """
        return self.eta_ref * np.exp(-self.beta_eta * (T - self.T_ref))

    def eta_hat(self, T: float) -> float:
        """
        Безразмерная вязкость η̂ = η(T) / η_ref.
        """
        return np.exp(-self.beta_eta * (T - self.T_ref))


@dataclass
class OperatingConditions:
    """Режим работы."""
    n_rpm: float = 100.0        # Частота вращения, об/мин

    # Диапазоны для параметрического расчёта
    W_min: float = 100.0        # Минимальная нагрузка, Н
    W_max: float = 5000.0       # Максимальная нагрузка, Н
    T_min: float = 40.0         # Минимальная температура, °C
    T_max: float = 120.0        # Максимальная температура, °C

    # Число точек для построения поверхностей
    N_W: int = 20               # Точек по нагрузке
    N_T: int = 15               # Точек по температуре

    @property
    def omega(self) -> float:
        """Угловая скорость, рад/с."""
        return 2 * np.pi * self.n_rpm / 60

    def U(self, R_J: float) -> float:
        """Окружная скорость поверхности шейки, м/с."""
        return self.omega * R_J


@dataclass
class NumericalSettings:
    """Численные настройки."""
    N_phi: int = 360            # Число узлов по φ
    N_Z: int = 100              # Число узлов по Z
    tol_Re: float = 1e-6        # Допуск сходимости решателя Рейнольдса
    max_iter_Re: int = 10000    # Макс. число итераций
    omega_GS: float = 1.7       # Параметр релаксации Гаусса-Зейделя
    use_cavitation: bool = True # Условие P ≥ 0

    # Для конечных разностей при вычислении K, C
    delta_xi: float = 1e-4      # Δξ для жёсткости
    delta_xi_dot: float = 1e-4  # Δξ' для демпфирования


@dataclass
class BearingModel:
    """Полная модель подшипника."""
    geometry: BearingGeometry
    texture: TextureParameters
    material: MaterialProperties
    lubricant: LubricantProperties
    operating: OperatingConditions
    numerical: NumericalSettings

    def pressure_scale(self, T: float) -> float:
        """
        Масштаб давления p0 = 6 * η(T) * U * R_J / c0².
        """
        eta = self.lubricant.viscosity(T)
        U = self.operating.U(self.geometry.R_J)
        return 6 * eta * U * self.geometry.R_J / self.geometry.c0**2

    def force_scale(self, T: float) -> float:
        """
        Масштаб силы F_scale = p0 * R_J * L = 6 * η * U * R_J² * L / c0².
        """
        return self.pressure_scale(T) * self.geometry.R_J * self.geometry.L

    def K_scale(self, T: float) -> float:
        """
        Масштаб жёсткости K_scale = η * ω * L / ψ³.
        """
        eta = self.lubricant.viscosity(T)
        psi = self.geometry.psi
        return eta * self.operating.omega * self.geometry.L / psi**3

    def C_scale(self, T: float) -> float:
        """
        Масштаб демпфирования C_scale = η * L / ψ³.
        """
        eta = self.lubricant.viscosity(T)
        psi = self.geometry.psi
        return eta * self.geometry.L / psi**3

    def info(self, T: float = 80.0) -> str:
        """Информация о модели."""
        eta = self.lubricant.viscosity(T)
        U = self.operating.U(self.geometry.R_J)
        lines = [
            "=" * 50,
            "ПАРАМЕТРЫ МОДЕЛИ ПОДШИПНИКА",
            "=" * 50,
            f"Геометрия:",
            f"  R_J = {self.geometry.R_J*1000:.2f} мм",
            f"  R_B = {self.geometry.R_B*1000:.2f} мм",
            f"  c0 = {self.geometry.c0*1e6:.1f} мкм",
            f"  L = {self.geometry.L*1000:.1f} мм",
            f"  ψ = c0/R_J = {self.geometry.psi:.6f}",
            f"  λ = 2R_J/L = {self.geometry.lambda_ratio:.3f}",
            "",
            f"Режим работы:",
            f"  n = {self.operating.n_rpm:.1f} об/мин",
            f"  ω = {self.operating.omega:.2f} рад/с",
            f"  U = {U:.3f} м/с",
            "",
            f"Смазка при T = {T}°C:",
            f"  η = {eta*1000:.3f} мПа·с",
            f"  η̂ = {self.lubricant.eta_hat(T):.4f}",
            "",
            f"Масштабы при T = {T}°C:",
            f"  p0 = {self.pressure_scale(T)/1e6:.3f} МПа",
            f"  F_scale = {self.force_scale(T):.1f} Н",
            f"  K_scale = {self.K_scale(T)/1e6:.3f} МН/м",
            f"  C_scale = {self.C_scale(T)/1e3:.3f} кН·с/м",
            "",
            f"Текстура: {'Да' if self.texture.enabled else 'Нет'}",
            f"  Филлотаксис: {'Да' if self.texture.use_phyllotaxis else 'Нет'}",
            "=" * 50,
        ]
        return "\n".join(lines)


def create_chinese_paper_bearing(
    n_rpm: float = 100.0,
    T_operating: float = 80.0,
) -> BearingModel:
    """
    Создать модель подшипника по данным китайской статьи.

    D_B = 83 мм, D_J = 82.5 мм, L = 45 мм
    c0 = (83 - 82.5)/2 = 0.25 мм = 250 мкм
    """
    geometry = BearingGeometry(
        R_J=82.5e-3 / 2,    # 41.25 мм
        R_B=83e-3 / 2,      # 41.5 мм
        L=45e-3,            # 45 мм
    )

    texture = TextureParameters(
        enabled=True,
        h_p=20e-6,              # 20 мкм глубина
        a=2e-3,                 # 2 мм полуось по z
        b=2e-3,                 # 2 мм полуось по φ
        use_phyllotaxis=True,
        T_spirals=10,           # 10 спиралей
        alpha_step_deg=12,      # 12° угловой шаг
        c_step=5e-3,            # 5 мм шаг по высоте
        phi_min=np.pi/2,        # 90°
        phi_max=3*np.pi/2,      # 270°
    )

    material = MaterialProperties(
        alpha_J=12e-6,          # Сталь
        alpha_B=18e-6,          # Бронза
        T_ref=20.0,
    )

    lubricant = LubricantProperties(
        eta_ref=0.05,           # 50 мПа·с при 40°C
        beta_eta=0.03,          # Типичное значение для минеральных масел
        T_ref=40.0,
        rho=870.0,
        c_p=2000.0,
    )

    operating = OperatingConditions(
        n_rpm=n_rpm,
        W_min=50.0,             # 50 Н (меньше для видимости зоны устойчивости)
        W_max=2000.0,           # 2 кН (уменьшено для лучшей визуализации)
        T_min=30.0,             # 30°C (понижено для большей вязкости)
        T_max=100.0,            # 100°C
        N_W=20,
        N_T=15,
    )

    numerical = NumericalSettings(
        N_phi=360,
        N_Z=100,
        tol_Re=1e-6,
        max_iter_Re=10000,
        omega_GS=1.7,
        use_cavitation=True,
        delta_xi=1e-4,
        delta_xi_dot=1e-4,
    )

    return BearingModel(
        geometry=geometry,
        texture=texture,
        material=material,
        lubricant=lubricant,
        operating=operating,
        numerical=numerical,
    )


def create_roller_cone_bit_bearing(
    n_rpm: float = 150.0,
    T_operating: float = 80.0,
) -> BearingModel:
    """
    Создать модель подшипника шарошечного долота.

    Типичные параметры для опоры шарошечного долота.
    """
    geometry = BearingGeometry(
        R_J=25e-3,          # 25 мм радиус шейки
        R_B=25.05e-3,       # 25.05 мм радиус вкладыша (зазор 50 мкм)
        L=30e-3,            # 30 мм длина
    )

    texture = TextureParameters(
        enabled=True,
        h_p=15e-6,              # 15 мкм глубина
        a=1.5e-3,               # 1.5 мм полуось по z
        b=1.5e-3,               # 1.5 мм полуось по φ
        use_phyllotaxis=True,
        T_spirals=8,
        alpha_step_deg=15,
        c_step=4e-3,
        phi_min=np.pi/2,
        phi_max=3*np.pi/2,
    )

    material = MaterialProperties(
        alpha_J=12e-6,
        alpha_B=18e-6,
        T_ref=20.0,
    )

    lubricant = LubricantProperties(
        eta_ref=0.05,
        beta_eta=0.025,
        T_ref=40.0,
        rho=870.0,
        c_p=2000.0,
    )

    operating = OperatingConditions(
        n_rpm=n_rpm,
        W_min=100.0,
        W_max=5000.0,
        T_min=40.0,
        T_max=120.0,
        N_W=20,
        N_T=15,
    )

    numerical = NumericalSettings(
        N_phi=360,
        N_Z=100,
        tol_Re=1e-6,
        max_iter_Re=10000,
        omega_GS=1.7,
        use_cavitation=True,
        delta_xi=1e-4,
        delta_xi_dot=1e-4,
    )

    return BearingModel(
        geometry=geometry,
        texture=texture,
        material=material,
        lubricant=lubricant,
        operating=operating,
        numerical=numerical,
    )


if __name__ == "__main__":
    # Тест
    model = create_chinese_paper_bearing()
    print(model.info(T=80.0))

    print("\nЗависимость вязкости от температуры:")
    for T in [40, 60, 80, 100, 120]:
        eta = model.lubricant.viscosity(T)
        eta_hat = model.lubricant.eta_hat(T)
        print(f"  T = {T}°C: η = {eta*1000:.3f} мПа·с, η̂ = {eta_hat:.4f}")

    print("\nИзменение зазора от температуры:")
    for T in [40, 60, 80, 100, 120]:
        H_T = model.material.H_T(model.geometry, T)
        delta_c = model.material.clearance_change(model.geometry, T)
        print(f"  T = {T}°C: Δc = {delta_c*1e6:.2f} мкм, H_T = {H_T:.4f}")
