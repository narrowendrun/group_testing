
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import random

from config import SceneConfig, SimulationConfig


@dataclass(frozen=True, slots=True)
class Point2D:
    x_m: float
    y_m: float


@dataclass(frozen=True, slots=True)
class UEState:
    position: Point2D
    r_m: float
    theta_deg: float
    blocked: bool
    los: bool

    @property
    def x_m(self) -> float:
        return self.position.x_m

    @property
    def y_m(self) -> float:
        return self.position.y_m


def deg2rad(angle_deg: float) -> float:
    return math.radians(angle_deg)


def rad2deg(angle_rad: float) -> float:
    return math.degrees(angle_rad)


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap angle to [-180, 180)."""
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    return wrapped


def polar_to_cartesian(r_m: float, theta_deg: float) -> Point2D:
    theta_rad = deg2rad(theta_deg)
    return Point2D(
        x_m=r_m * math.cos(theta_rad),
        y_m=r_m * math.sin(theta_rad),
    )


def cartesian_to_polar(x_m: float, y_m: float) -> tuple[float, float]:
    r_m = math.hypot(x_m, y_m)
    theta_deg = rad2deg(math.atan2(y_m, x_m))
    return r_m, wrap_angle_deg(theta_deg)


class SectorGeometry:
    """Geometry helper for a single-gNB, single-UE sector simulation.

    The gNB is fixed at the origin. The UE is sampled inside the angular sector
    and radial annulus defined by ``SceneConfig`` and the beam sector limits.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.scene = config.scene
        self.beam = config.beam

    @property
    def sector_min_deg(self) -> float:
        return self.beam.sector_min_deg

    @property
    def sector_max_deg(self) -> float:
        return self.beam.sector_max_deg

    @property
    def ue_range_min_m(self) -> float:
        return self.scene.ue_range_min_m

    @property
    def ue_range_max_m(self) -> float:
        return self.scene.ue_range_max_m

    def contains_angle(self, theta_deg: float) -> bool:
        return self.sector_min_deg <= theta_deg <= self.sector_max_deg

    def contains_range(self, r_m: float) -> bool:
        return self.ue_range_min_m <= r_m <= self.ue_range_max_m

    def contains_point(self, point: Point2D) -> bool:
        r_m, theta_deg = cartesian_to_polar(point.x_m, point.y_m)
        return self.contains_range(r_m) and self.contains_angle(theta_deg)

    def sample_theta_deg(self, rng: random.Random) -> float:
        return rng.uniform(self.sector_min_deg, self.sector_max_deg)

    def sample_range_m(self, rng: random.Random) -> float:
        r_min = self.ue_range_min_m
        r_max = self.ue_range_max_m

        if self.scene.range_sampling == "uniform_radius":
            return rng.uniform(r_min, r_max)

        # Area-uniform sampling is usually the better default in a sector:
        # the UE is uniform over area, not biased toward the origin.
        u = rng.random()
        return math.sqrt((r_max ** 2 - r_min ** 2) * u + r_min ** 2)

    def sample_blocked_flag(
        self,
        rng: random.Random,
        blocked: Optional[bool] = None,
    ) -> bool:
        if blocked is not None:
            return blocked
        return rng.random() < self.scene.blockage_probability

    def sample_ue(
        self,
        rng: Optional[random.Random] = None,
        *,
        force_theta_deg: Optional[float] = None,
        force_range_m: Optional[float] = None,
        blocked: Optional[bool] = None,
    ) -> UEState:
        """Sample one static UE for an episode.

        Parameters
        ----------
        rng:
            Optional ``random.Random`` instance for reproducibility.
        force_theta_deg:
            If provided, use this exact UE angle instead of sampling.
        force_range_m:
            If provided, use this exact UE range instead of sampling.
        blocked:
            If provided, override the blockage draw.

        Returns
        -------
        UEState
            The sampled UE position and basic geometry tags.
        """
        rng = rng or random.Random(self.config.seed)

        theta_deg = self.sample_theta_deg(rng) if force_theta_deg is None else force_theta_deg
        r_m = self.sample_range_m(rng) if force_range_m is None else force_range_m

        if not self.contains_angle(theta_deg):
            raise ValueError(
                f"UE angle {theta_deg:.3f} deg lies outside the sector "
                f"[{self.sector_min_deg}, {self.sector_max_deg}] deg."
            )
        if not self.contains_range(r_m):
            raise ValueError(
                f"UE range {r_m:.3f} m lies outside the allowed interval "
                f"[{self.ue_range_min_m}, {self.ue_range_max_m}] m."
            )

        blocked_flag = self.sample_blocked_flag(rng, blocked=blocked)
        position = polar_to_cartesian(r_m=r_m, theta_deg=theta_deg)

        return UEState(
            position=position,
            r_m=r_m,
            theta_deg=theta_deg,
            blocked=blocked_flag,
            los=not blocked_flag,
        )

    def oracle_nearest_beam_index(self, theta_deg: float) -> int:
        """Return the nearest beam center to the given UE angle.

        This is a purely geometric oracle based on angular proximity.
        It is useful for diagnostics and later evaluation, even before the
        channel/beam gain model is implemented.
        """
        centers = self.beam.beam_centers_deg
        return min(range(len(centers)), key=lambda idx: abs(centers[idx] - theta_deg))


def make_geometry(config: SimulationConfig) -> SectorGeometry:
    return SectorGeometry(config)
