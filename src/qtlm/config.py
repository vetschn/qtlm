import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self

from qtlm import xp
import numpy as np

_default_num_orbitals_per_atom = {
    "C": 13,
    "B": 13,
    "N": 13,
    "S": 13,
    "W": 29,
    "Mo": 29,
}


class BiasConfig(BaseModel):
    """Options for the voltage bias."""

    model_config = ConfigDict(extra="forbid")

    bias_start: float = 0.0  # V
    bias_stop: float = 1.0  # V
    num_bias_points: PositiveInt = 10

    bias_points: None = None

    @model_validator(mode="after")
    def check_bias_points(self):
        """Checks if the bias points are set correctly."""
        self.bias_points = np.linspace(
            self.bias_start, self.bias_stop, self.num_bias_points, endpoint=True
        )

        return self


class ElectronConfig(BaseModel):
    """Options for the electronic subsystem solver."""

    model_config = ConfigDict(extra="forbid")

    eta_contact: NonNegativeFloat = 25e-3  # eV
    eta: NonNegativeFloat = 1e-12  # eV

    fermi_level: float

    temperature: PositiveFloat = 300.0  # K

    energy_start: float
    energy_stop: float
    energy_window_num: PositiveInt | None = None
    energy_step: float | None = None

    kpt_grid: tuple[PositiveInt, PositiveInt, PositiveInt]

    energy_batch_size: PositiveInt = 128
    kpt_batch_size: PositiveInt = 4096

    energies: None = None

    @model_validator(mode="after")
    def check_energies(self):
        """Check if 'energy_window_num' and 'energy_step' are set."""
        if self.energy_window_num is not None and self.energy_step is not None:
            raise ValueError(
                "Only one of 'energy_window_num' or 'energy_step' should be set."
            )

        if self.energy_window_num is not None:
            self.energies = xp.linspace(
                self.energy_start,
                self.energy_stop,
                self.energy_window_num,
            )
        elif self.energy_step is not None:
            self.energies = xp.arange(
                self.energy_start,
                self.energy_stop,
                self.energy_step,
            )

        return self

    @model_validator(mode="after")
    def kpts_size_to_tuple(self) -> Self:
        """Transforms list to tuple."""
        self.kpt_grid = tuple(self.kpt_grid)
        return self


class GrapheneCapacitorConfig(BaseModel):
    """Options for the capacitor model."""

    model_config = ConfigDict(extra="forbid")

    # --- Capacitor parameters ----------------------------------------
    plate_separation: Literal["auto"] | PositiveFloat = "auto"  # m
    fermi_velocity: PositiveFloat = 1.02e6  # m/s

    dielectric_permittivity: PositiveFloat = 3.4  # relative permittivity


class DeviceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Device geometry ---------------------------------------------
    left_contact_region: tuple[float, float]
    right_contact_region: tuple[float, float]

    transport_direction: Literal["x", "y", "z"]

    num_orbitals_per_atom: dict[str, int] = Field(
        default_factory=lambda: _default_num_orbitals_per_atom
    )

    capacitor_model : Literal["none", "graphene"] = "none"
    graphene_capacitor: GrapheneCapacitorConfig = GrapheneCapacitorConfig()

    @model_validator(mode="after")
    def region_to_tuple(self) -> Self:
        """Transforms list to tuple."""
        self.left_contact_region = tuple(self.left_contact_region)
        self.right_contact_region = tuple(self.right_contact_region)
        return self


class QTLMConfig(BaseModel):
    """Top-level simulation configuration."""

    model_config = ConfigDict(extra="forbid")

    # --- Simulation parameters ---------------------------------------
    device: DeviceConfig
    bias: BiasConfig = BiasConfig()

    electron: ElectronConfig

    # --- Directory paths ----------------------------------------------
    config_dir: Path
    simulation_dir: Path = Path("./quatrex/")
    input_dir: Path | None = None
    output_dir: Path | None = None

    @model_validator(mode="after")
    def resolve_config_path(self) -> Self:
        """Resolves the config directory path."""
        self.config_dir = Path(self.config_dir).resolve()
        return self

    @model_validator(mode="after")
    def resolve_simulation_dir(self):
        """Resolves the simulation directory path."""
        self.simulation_dir = (self.config_dir / self.simulation_dir).resolve()
        return self

    @model_validator(mode="after")
    def set_output_dir(self):
        """Resolves the simulation directory path."""
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir).resolve()
            return self

        self.output_dir = self.simulation_dir / "outputs/"
        return self

    @model_validator(mode="after")
    def set_input_dir(self) -> Path:
        """Returns the input directory path."""
        if self.input_dir is not None:
            self.input_dir = Path(self.input_dir).resolve()
            return self
        self.input_dir = self.simulation_dir / "inputs/"
        return self


def parse_config(config_file: Path) -> QTLMConfig:
    """Reads the TOML config file.

    Parameters
    ----------
    config_file : Path
        Path to the TOML configuration file.

    Returns
    -------
    QuatrexConfig
        The parsed configuration object.

    """

    config_file = Path(config_file).resolve()

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    if "simulation_dir" in config:
        simulation_dir = config["simulation_dir"]
        if not os.path.isabs(simulation_dir):
            parent_dir = os.path.dirname(os.path.abspath(config_file))
            simulation_dir = Path(os.path.join(parent_dir, simulation_dir))
            config["simulation_dir"] = simulation_dir

    config["config_dir"] = config_file.parent

    return QTLMConfig(**config)
