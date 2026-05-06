
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union
import math

import numpy as np

from config import BeamConfig, SimulationConfig

PatternKind = Literal["pencil", "group"]
GroupCombineMode = Literal["equal_power", "sum", "max"]


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap angle to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def angular_difference_deg(a_deg: float, b_deg: float) -> float:
    """Signed smallest-angle difference a-b in degrees."""
    return wrap_angle_deg(a_deg - b_deg)


def linear_to_db(value_linear: float, floor_db: float = -300.0) -> float:
    """Convert a positive linear power/gain value to dB safely."""
    if value_linear <= 0.0:
        return floor_db
    return 10.0 * math.log10(value_linear)


@dataclass(frozen=True, slots=True)
class BeamPatternSpec:
    """Description of either a pencil beam or a predefined group-test pattern."""

    name: str
    kind: PatternKind
    member_indices: Tuple[int, ...]
    nominal_beamwidth_deg: float
    center_deg: Optional[float] = None

    @property
    def size(self) -> int:
        return len(self.member_indices)

    @property
    def is_non_contiguous(self) -> bool:
        return any(
            (b - a) != 1 for a, b in zip(self.member_indices[:-1], self.member_indices[1:])
        )


class BeamCodebook:
    """Analytic beam/codebook model for the minimal attach simulator.

    The file supports:
    1. 121 pencil beams with Gaussian-like angular gain.
    2. Predefined group-test patterns containing arbitrary, possibly
       non-contiguous, beam indices.
    3. A binary membership matrix and a more physical response matrix
       evaluated at the pencil-beam center angles.

    Notes
    -----
    Group patterns are modeled as multi-lobe beams synthesized from their
    member pencil beams. Because we do not have true PAAM vectors yet, the
    coarse group gain is approximated using combinations of pencil-beam
    gains:

    - ``equal_power``:
          mean(member_gains)
      This is the default and is the most physically conservative of the
      three because total probe power is effectively split across the active
      lobes.

    - ``sum``:
          sum(member_gains)
      Useful as an optimistic upper bound or diagnostic tool.

    - ``max``:
          max(member_gains)
      Useful if you want the group test to behave like a set-membership
      indicator rather than a true power split.
    """

    def __init__(self, config: Union[SimulationConfig, BeamConfig]) -> None:
        if isinstance(config, SimulationConfig):
            self.beam_config = config.beam
        else:
            self.beam_config = config

        self._beam_centers_deg = self.beam_config.beam_centers_deg
        self._pencil_patterns = self._build_pencil_patterns()
        self._group_patterns = self._build_group_patterns()

    @property
    def num_beams(self) -> int:
        return self.beam_config.num_beams

    @property
    def beam_centers_deg(self) -> Tuple[float, ...]:
        return self._beam_centers_deg

    @property
    def pencil_beamwidth_deg(self) -> float:
        return self.beam_config.pencil_beamwidth_deg

    @property
    def pencil_patterns(self) -> Dict[int, BeamPatternSpec]:
        return self._pencil_patterns

    @property
    def group_patterns(self) -> Dict[str, BeamPatternSpec]:
        return self._group_patterns

    @property
    def group_names(self) -> Tuple[str, ...]:
        return tuple(self._group_patterns.keys())

    def _build_pencil_patterns(self) -> Dict[int, BeamPatternSpec]:
        patterns: Dict[int, BeamPatternSpec] = {}
        for idx, center_deg in enumerate(self._beam_centers_deg):
            patterns[idx] = BeamPatternSpec(
                name=f"b{idx:03d}",
                kind="pencil",
                member_indices=(idx,),
                nominal_beamwidth_deg=self.pencil_beamwidth_deg,
                center_deg=center_deg,
            )
        return patterns

    def _build_group_patterns(self) -> Dict[str, BeamPatternSpec]:
        group_patterns = self.beam_config.group_patterns or {}
        patterns: Dict[str, BeamPatternSpec] = {}

        for name, member_indices in group_patterns.items():
            centers = [self._beam_centers_deg[idx] for idx in member_indices]
            center_deg = float(sum(centers) / len(centers))
            patterns[name] = BeamPatternSpec(
                name=name,
                kind="group",
                member_indices=tuple(member_indices),
                nominal_beamwidth_deg=self.pencil_beamwidth_deg,
                center_deg=center_deg,
            )

        return patterns

    def get_pencil(self, beam_index: int) -> BeamPatternSpec:
        if beam_index not in self._pencil_patterns:
            raise KeyError(f"Unknown pencil beam index: {beam_index}")
        return self._pencil_patterns[beam_index]

    def get_group(self, group_name: str) -> BeamPatternSpec:
        if group_name not in self._group_patterns:
            raise KeyError(f"Unknown group pattern: {group_name}")
        return self._group_patterns[group_name]

    def get_pattern(
        self,
        pattern: Union[int, str, BeamPatternSpec],
    ) -> BeamPatternSpec:
        if isinstance(pattern, BeamPatternSpec):
            return pattern
        if isinstance(pattern, int):
            return self.get_pencil(pattern)
        if isinstance(pattern, str):
            if pattern in self._group_patterns:
                return self.get_group(pattern)
            if pattern.startswith("b") and pattern[1:].isdigit():
                return self.get_pencil(int(pattern[1:]))
            raise KeyError(f"Unknown pattern name: {pattern}")
        raise TypeError("pattern must be an int, str, or BeamPatternSpec")

    def gaussian_gain_linear(
        self,
        theta_deg: float,
        center_deg: float,
        beamwidth_deg: float,
        peak_gain_linear: float = 1.0,
    ) -> float:
        """Gaussian-like angular gain model in linear scale.

        The chosen form makes the gain equal to half-power when the angular
        offset magnitude equals ``beamwidth_deg``.
        """
        if beamwidth_deg <= 0.0:
            raise ValueError("beamwidth_deg must be positive.")
        delta_deg = angular_difference_deg(theta_deg, center_deg)
        exponent = -4.0 * math.log(2.0) * (delta_deg / beamwidth_deg) ** 2
        return peak_gain_linear * math.exp(exponent)

    def pencil_gain_linear(
        self,
        beam_index: int,
        theta_deg: float,
        peak_gain_linear: float = 1.0,
    ) -> float:
        center_deg = self._beam_centers_deg[beam_index]
        return self.gaussian_gain_linear(
            theta_deg=theta_deg,
            center_deg=center_deg,
            beamwidth_deg=self.pencil_beamwidth_deg,
            peak_gain_linear=peak_gain_linear,
        )

    def group_gain_linear(
        self,
        group_name: str,
        theta_deg: float,
        combine: GroupCombineMode = "equal_power",
    ) -> float:
        """Evaluate a group-test pattern at the requested angle.

        Parameters
        ----------
        group_name:
            Name of the predefined probing group.
        theta_deg:
            Target angle.
        combine:
            How member pencil-beam gains are combined into a multi-lobe
            coarse probing pattern.
        """
        spec = self.get_group(group_name)
        member_gains = np.array(
            [self.pencil_gain_linear(idx, theta_deg) for idx in spec.member_indices],
            dtype=float,
        )

        if combine == "equal_power":
            return float(member_gains.mean())
        if combine == "sum":
            return float(member_gains.sum())
        if combine == "max":
            return float(member_gains.max())

        raise ValueError(
            "combine must be one of {'equal_power', 'sum', 'max'}"
        )

    def gain_linear(
        self,
        pattern: Union[int, str, BeamPatternSpec],
        theta_deg: float,
        *,
        combine: GroupCombineMode = "equal_power",
    ) -> float:
        spec = self.get_pattern(pattern)
        if spec.kind == "pencil":
            return self.pencil_gain_linear(spec.member_indices[0], theta_deg)
        return self.group_gain_linear(spec.name, theta_deg, combine=combine)

    def gain_db(
        self,
        pattern: Union[int, str, BeamPatternSpec],
        theta_deg: float,
        *,
        combine: GroupCombineMode = "equal_power",
        floor_db: float = -300.0,
    ) -> float:
        return linear_to_db(
            self.gain_linear(pattern, theta_deg, combine=combine),
            floor_db=floor_db,
        )

    def oracle_best_pencil_beam(self, theta_deg: float) -> int:
        """Return the pencil beam with the maximum analytic gain."""
        gains = np.array(
            [self.pencil_gain_linear(idx, theta_deg) for idx in range(self.num_beams)],
            dtype=float,
        )
        return int(np.argmax(gains))

    def candidate_indices_from_group(self, group_name: str) -> Tuple[int, ...]:
        return self.get_group(group_name).member_indices

    def binary_group_membership_matrix(
        self,
        order: Optional[Sequence[str]] = None,
    ) -> tuple[np.ndarray, Tuple[str, ...]]:
        """Return the binary group-test matrix A in {0,1}^{T x 121}."""
        group_names = tuple(order) if order is not None else self.group_names
        matrix = np.zeros((len(group_names), self.num_beams), dtype=int)

        for row, name in enumerate(group_names):
            for col in self.get_group(name).member_indices:
                matrix[row, col] = 1

        return matrix, group_names

    def group_response_matrix_linear(
        self,
        order: Optional[Sequence[str]] = None,
        *,
        combine: GroupCombineMode = "equal_power",
        theta_grid_deg: Optional[Sequence[float]] = None,
    ) -> tuple[np.ndarray, Tuple[str, ...], np.ndarray]:
        """Return a physical response matrix evaluated on an angle grid.

        The default grid is the 121 pencil-beam center angles. The returned
        matrix has shape [num_groups, num_grid_angles].
        """
        group_names = tuple(order) if order is not None else self.group_names
        angle_grid = np.asarray(
            self.beam_centers_deg if theta_grid_deg is None else theta_grid_deg,
            dtype=float,
        )

        matrix = np.zeros((len(group_names), angle_grid.size), dtype=float)
        for row, name in enumerate(group_names):
            for col, theta_deg in enumerate(angle_grid):
                matrix[row, col] = self.group_gain_linear(
                    name,
                    float(theta_deg),
                    combine=combine,
                )

        return matrix, group_names, angle_grid

    def pencil_response_matrix_linear(
        self,
        theta_grid_deg: Optional[Sequence[float]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the pencil-beam response matrix on an angle grid.

        Shape: [num_beams, num_grid_angles].
        """
        angle_grid = np.asarray(
            self.beam_centers_deg if theta_grid_deg is None else theta_grid_deg,
            dtype=float,
        )
        matrix = np.zeros((self.num_beams, angle_grid.size), dtype=float)

        for row in range(self.num_beams):
            for col, theta_deg in enumerate(angle_grid):
                matrix[row, col] = self.pencil_gain_linear(row, float(theta_deg))

        return matrix, angle_grid

    def group_score_from_measurements(
        self,
        measurements: Sequence[float],
        order: Optional[Sequence[str]] = None,
        *,
        normalize_by_group_size: bool = False,
    ) -> tuple[np.ndarray, Tuple[str, ...]]:
        """Convert a vector of group-test measurements into beam scores.

        If ``y`` contains one scalar response per group test and ``A`` is the
        binary group-membership matrix, then this method computes either:

            score = A^T y

        or a size-normalized variant. This is a direct implementation of the
        simple coarse decoding rule discussed for the project.
        """
        A, group_names = self.binary_group_membership_matrix(order=order)
        y = np.asarray(measurements, dtype=float).reshape(-1)

        if y.size != A.shape[0]:
            raise ValueError(
                f"Expected {A.shape[0]} measurements, got {y.size}."
            )

        scores = A.T @ y

        if normalize_by_group_size:
            group_sizes = A.sum(axis=0)
            group_sizes = np.maximum(group_sizes, 1)
            scores = scores / group_sizes

        return np.asarray(scores, dtype=float), group_names

    def describe_group(self, group_name: str) -> str:
        spec = self.get_group(group_name)
        return (
            f"{spec.name}: size={spec.size}, non_contiguous={spec.is_non_contiguous}, "
            f"members={spec.member_indices}"
        )

    def summary(self) -> str:
        non_contiguous = sum(
            1 for spec in self._group_patterns.values() if spec.is_non_contiguous
        )
        return (
            f"BeamCodebook(num_beams={self.num_beams}, "
            f"groups={len(self._group_patterns)}, "
            f"non_contiguous_groups={non_contiguous}, "
            f"beamwidth={self.pencil_beamwidth_deg} deg)"
        )


def make_codebook(config: Union[SimulationConfig, BeamConfig]) -> BeamCodebook:
    return BeamCodebook(config)
