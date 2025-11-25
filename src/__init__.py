"""
Textured Bearing Thermal Model
Модель гидродинамического подшипника с текстурой и учётом температуры

Модули:
- parameters: параметры подшипника, смазки, режима работы
- geometry: сетка, зазор, текстура
- reynolds: решатель уравнения Рейнольдса
- thermal: температурная модель, THD
- forces: силы, коэффициенты жёсткости/демпфирования
"""

from .parameters import (
    BearingGeometry,
    TextureParameters,
    LubricantProperties,
    OperatingConditions,
    GridParameters,
    SolverSettings,
    BearingModel,
    create_roller_cone_bit_bearing,
    create_test_bearing,
)

from .geometry import (
    Grid,
    create_grid,
    compute_film_thickness,
    compute_film_thickness_dynamic,
    BearingGeometryProcessor,
)

from .reynolds import (
    ReynoldsSolution,
    ReynoldsSolver,
    solve_reynolds_static,
    solve_reynolds_dynamic,
    solve_reynolds_variable_viscosity,
)

from .thermal import (
    ThermalSolution,
    ThermalModel,
    THDSolver,
    compute_viscosity_field,
    estimate_temperature_rise,
)

from .forces import (
    ForceResult,
    FrictionResult,
    DynamicCoefficients,
    integrate_forces,
    compute_friction,
    compute_stiffness_coefficients,
    compute_damping_coefficients,
    compute_all_dynamic_coefficients,
    BearingAnalyzer,
)

__version__ = "0.1.0"
__author__ = "Textured Bearing Research"
