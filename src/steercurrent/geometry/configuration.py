"""Fixed circuit length, materials, battery, and resistor spans."""
from dataclasses import asdict, dataclass
from .validation import validate_config


@dataclass(frozen=True)
class BatterySpec:
    length: float = 2.0
    emf: float = 1.0
    resistivity: float = 0.1
    polarity: int = 1  # +1 follows loop traversal; -1 opposes it


@dataclass(frozen=True)
class ResistorSpec:
    """Span [start, end] along the sharp loop; 0 <= start < end <= L."""
    id: str
    start: float
    end: float
    resistivity: float = 20.0


@dataclass(frozen=True)
class CircuitConfig:
    """Fixed physical circuit, independent of its shape.

    Component distances follow the loop from the battery center. Changing shape
    never changes these distances or the prescribed total length.
    """
    length: float
    width: float = 1.0
    battery: BatterySpec = BatterySpec()
    resistors: tuple[ResistorSpec, ...] = ()
    wire_resistivity: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'resistors', tuple(self.resistors))
        self.validate()

    def validate(self):
        validate_config(self)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if 'battery' in data:
            data['battery'] = BatterySpec(**data['battery'])
        data['resistors'] = tuple(ResistorSpec(**r)
                                  for r in data.get('resistors', ()))
        return cls(**data)
