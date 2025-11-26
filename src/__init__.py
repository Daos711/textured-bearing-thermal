"""
Textured Bearing Thermal Model
Модель гидродинамического подшипника с текстурой и учётом температуры

Модули:
- parameters: параметры подшипника, смазки, режима работы
- geometry: сетка, зазор, текстура (включая филлотаксис)
- reynolds: решатель уравнения Рейнольдса
- forces: силы, коэффициенты жёсткости/демпфирования
- stability: параметры устойчивости
- main: параметрический расчёт
- thermal: термогидродинамическая модель (THD)
"""

from .parameters import (
    BearingGeometry,
    TextureParameters,
    MaterialProperties,
    LubricantProperties,
    OperatingConditions,
    NumericalSettings,
    BearingModel,
    create_chinese_paper_bearing,
    create_roller_cone_bit_bearing,
)

from .geometry import (
    Grid,
    create_grid,
    compute_film_thickness_static,
    compute_film_thickness_full,
    compute_dH_dphi,
    compute_dH_dtau,
    compute_texture_contribution,
    compute_phyllotaxis_centers,
    compute_regular_centers,
    get_texture_centers_count,
    GeometryProcessor,
)

from .reynolds import (
    ReynoldsSolution,
    ReynoldsSolver,
    solve_reynolds_static,
    solve_reynolds_with_squeeze,
    solve_reynolds_equation,
)

from .forces import (
    ForceResult,
    FrictionResult,
    StiffnessCoefficients,
    DampingCoefficients,
    FullCoefficients,
    integrate_forces,
    compute_friction,
    find_equilibrium_eccentricity,
    compute_stiffness_coefficients,
    compute_damping_coefficients,
    compute_all_coefficients,
)

from .stability import (
    StabilityParameters,
    compute_stability_parameters,
    compute_stability_from_full_coefficients,
    compute_stability_margins,
)

from .main import (
    PointResult,
    ParametricResults,
    run_parametric_calculation,
    run_full_analysis,
    plot_3d_surfaces,
    plot_comparison_contours,
)

from .thermal import (
    ThermalSolution,
    ThermalModel,
    THDSolver,
    compute_viscosity_field,
    compute_viscosity_ratio,
    compute_heat_dissipation,
    solve_energy_equation,
    estimate_temperature_rise,
)

__version__ = "0.2.0"
__author__ = "Textured Bearing Research"
