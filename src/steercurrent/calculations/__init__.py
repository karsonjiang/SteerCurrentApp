"""Field calculation entry points, independent of meshing and web app."""
from .fields import FieldData
from .conduction import ConductionResult, solve_conduction
from .exterior import ExteriorResult, solve_exterior
from .charges import ChargeTrace, ChargeResult, calculate_charges
from .magnetics import MagneticResult, solve_magnetics
from .poynting import PoyntingResult, calculate_poynting
from .battery_only import BatteryOnlyResult, battery_only_field
