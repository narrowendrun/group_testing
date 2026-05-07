
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple
import math

BeamPattern = Tuple[int, ...]
BeamPatternMap = Dict[str, BeamPattern]


def generate_interleaved_group_patterns(num_beams: int, stride: int = 11) -> BeamPatternMap:
    """Create default non-contiguous probing groups.

    Each group is formed by taking one residue class modulo ``stride``.
    For ``num_beams=121`` and ``stride=11``, every group contains 11
    non-contiguous beam indices, for example:
        grp_mod11_00 -> (0, 11, 22, ..., 110)
        grp_mod11_01 -> (1, 12, 23, ..., 111)

    This is only a placeholder codebook for the early simulator. It is easy
    to replace later with professor-provided beam patterns.
    """
    if num_beams <= 0:
        raise ValueError("num_beams must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    patterns: BeamPatternMap = {}
    for offset in range(min(stride, num_beams)):
        members = tuple(range(offset, num_beams, stride))
        if members:
            patterns[f"grp_mod{stride}_{offset:02d}"] = members
    return patterns


@dataclass(slots=True)
class BeamConfig:
    sector_min_deg: float = -60.0
    sector_max_deg: float = 60.0
    num_beams: int = 121
    pencil_beamwidth_deg: float = 12.0
    group_patterns: Optional[BeamPatternMap] = None
    default_group_stride: int = 11

    def __post_init__(self) -> None:
        if self.sector_max_deg <= self.sector_min_deg:
            raise ValueError("sector_max_deg must be greater than sector_min_deg.")
        if self.num_beams < 2:
            raise ValueError("num_beams must be at least 2.")
        if self.pencil_beamwidth_deg <= 0:
            raise ValueError("pencil_beamwidth_deg must be positive.")

        # For this project, the expected grid is -60 to 60 in 1 deg steps.
        # This check keeps the config honest and catches accidental edits.
        expected_step = (self.sector_max_deg - self.sector_min_deg) / (self.num_beams - 1)
        if not math.isclose(expected_step, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Beam grid must currently be 1 degree per step. "
                f"Got step={expected_step:.6f} degrees."
            )

        if self.group_patterns is None:
            self.group_patterns = generate_interleaved_group_patterns(
                num_beams=self.num_beams,
                stride=self.default_group_stride,
            )

        self._validate_group_patterns()

    @property
    def beam_step_deg(self) -> float:
        return (self.sector_max_deg - self.sector_min_deg) / (self.num_beams - 1)

    @property
    def beam_centers_deg(self) -> Tuple[float, ...]:
        return tuple(
            self.sector_min_deg + i * self.beam_step_deg
            for i in range(self.num_beams)
        )

    @property
    def beam_indices(self) -> Tuple[int, ...]:
        return tuple(range(self.num_beams))

    def _validate_group_patterns(self) -> None:
        assert self.group_patterns is not None  # for type checkers
        valid_indices = set(range(self.num_beams))

        for name, members in self.group_patterns.items():
            if not members:
                raise ValueError(f"Group pattern '{name}' is empty.")

            if len(set(members)) != len(members):
                raise ValueError(f"Group pattern '{name}' contains duplicate beam indices.")

            illegal = [idx for idx in members if idx not in valid_indices]
            if illegal:
                raise ValueError(
                    f"Group pattern '{name}' contains out-of-range beam indices: {illegal}"
                )

    def is_non_contiguous_group(self, name: str) -> bool:
        members = self.group_patterns[name]
        return any((b - a) != 1 for a, b in zip(members[:-1], members[1:]))

    def summary(self) -> str:
        num_groups = len(self.group_patterns or {})
        non_contig = sum(
            1 for name in (self.group_patterns or {})
            if self.is_non_contiguous_group(name)
        )
        return (
            f"BeamConfig(num_beams={self.num_beams}, "
            f"sector=[{self.sector_min_deg}, {self.sector_max_deg}] deg, "
            f"beamwidth={self.pencil_beamwidth_deg} deg, "
            f"group_patterns={num_groups}, non_contiguous={non_contig})"
        )


@dataclass(slots=True)
class SceneConfig:
    ue_range_min_m: float = 10.0
    ue_range_max_m: float = 100.0
    range_sampling: Literal["area_uniform", "uniform_radius"] = "area_uniform"
    blockage_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.ue_range_min_m <= 0:
            raise ValueError("ue_range_min_m must be positive.")
        if self.ue_range_max_m <= self.ue_range_min_m:
            raise ValueError("ue_range_max_m must be greater than ue_range_min_m.")
        if not (0.0 <= self.blockage_probability <= 1.0):
            raise ValueError("blockage_probability must lie in [0, 1].")
        if self.range_sampling not in {"area_uniform", "uniform_radius"}:
            raise ValueError(
                "range_sampling must be 'area_uniform' or 'uniform_radius'."
            )


@dataclass(slots=True)
class ChannelConfig:
    tx_power_dbm: float = 20.0
    rx_gain_dbi: float = 0.0
    bandwidth_hz: float = 100e6
    num_active_subcarriers: int = 128
    nfft: int = 128
    num_pilot_symbols: int = 1
    subcarrier_spacing_hz: Optional[float] = None
    num_taps: int = 8
    nlos_tap_decay: float = 0.6
    noise_figure_db: float = 7.0
    rician_k_db: float = 10.0
    pathloss_ref_distance_m: float = 1.0
    pathloss_ref_db: float = 61.4
    pathloss_exponent_los: float = 2.0
    blockage_loss_db: float = 20.0

    def __post_init__(self) -> None:
        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive.")
        if self.num_active_subcarriers <= 0:
            raise ValueError("num_active_subcarriers must be positive.")
        if self.nfft < self.num_active_subcarriers:
            raise ValueError("nfft must be at least num_active_subcarriers.")
        if self.num_pilot_symbols <= 0:
            raise ValueError("num_pilot_symbols must be positive.")
        if self.subcarrier_spacing_hz is not None and self.subcarrier_spacing_hz <= 0:
            raise ValueError("subcarrier_spacing_hz must be positive when provided.")
        if self.num_taps <= 0:
            raise ValueError("num_taps must be positive.")
        if self.num_taps > self.nfft:
            raise ValueError("num_taps must be no larger than nfft.")
        if self.nlos_tap_decay <= 0:
            raise ValueError("nlos_tap_decay must be positive.")
        if self.pathloss_ref_distance_m <= 0:
            raise ValueError("pathloss_ref_distance_m must be positive.")
        if self.pathloss_exponent_los <= 0:
            raise ValueError("pathloss_exponent_los must be positive.")

    @property
    def effective_subcarrier_spacing_hz(self) -> float:
        if self.subcarrier_spacing_hz is not None:
            return self.subcarrier_spacing_hz
        return self.bandwidth_hz / self.num_active_subcarriers

    @property
    def active_bandwidth_hz(self) -> float:
        return self.num_active_subcarriers * self.effective_subcarrier_spacing_hz

    @property
    def thermal_noise_dbm(self) -> float:
        # -174 dBm/Hz is the standard thermal noise density at room temperature.
        return -174.0 + 10.0 * math.log10(self.bandwidth_hz) + self.noise_figure_db

    @property
    def subcarrier_noise_dbm(self) -> float:
        # Noise power for one OFDM pilot subcarrier: N0 * delta_f.
        return -174.0 + 10.0 * math.log10(self.effective_subcarrier_spacing_hz) + self.noise_figure_db

    @property
    def rician_k_linear(self) -> float:
        return 10.0 ** (self.rician_k_db / 10.0)


@dataclass(slots=True)
class MeasurementConfig:
    detection_threshold_db: float = -5.0
    feedback_noise_std_db: float = 0.0

    def __post_init__(self) -> None:
        if self.feedback_noise_std_db < 0:
            raise ValueError("feedback_noise_std_db must be non-negative.")


@dataclass(slots=True)
class RACHConfig:
    coarse_message_names: Tuple[str, str, str] = ("MSG1", "MSG2", "MSG3")
    fine_lock_state_name: str = "FINE_LOCK"
    final_message_name: str = "MSG4"
    require_detection_each_msg: bool = True

    @property
    def all_state_names(self) -> Tuple[str, ...]:
        return (
            "SEARCH",
            *self.coarse_message_names,
            self.fine_lock_state_name,
            self.final_message_name,
            "ATTACHED",
            "FAIL",
        )


@dataclass(slots=True)
class SimulationConfig:
    seed: int = 7
    num_episodes: int = 1000
    beam: BeamConfig = field(default_factory=BeamConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    rach: RACHConfig = field(default_factory=RACHConfig)

    def __post_init__(self) -> None:
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")

    def summary(self) -> str:
        return (
            f"SimulationConfig(seed={self.seed}, episodes={self.num_episodes}, "
            f"{self.beam.summary()}, "
            f"range=[{self.scene.ue_range_min_m}, {self.scene.ue_range_max_m}] m, "
            f"blockage_prob={self.scene.blockage_probability})"
        )


def make_config(**overrides) -> SimulationConfig:
    """Convenience factory for quick experiments.

    Example
    -------
    cfg = make_config(num_episodes=200)

    Nested dataclasses should be overridden directly after construction:
        cfg = make_config()
        cfg.scene.blockage_probability = 0.1
        cfg.channel.rician_k_db = 8.0
    """
    return SimulationConfig(**overrides)
