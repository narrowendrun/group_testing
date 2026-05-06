
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from channel import LinkBudgetResult, MinimalRicianChannel
from config import MeasurementConfig, SimulationConfig
from geometry import UEState


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    """UE feedback for one probe transmission."""

    pattern_name: str
    true_snr_db: float
    measured_snr_db: Optional[float]
    detected: bool

    def numeric(self, no_detection_fill_db: float) -> float:
        return self.measured_snr_db if self.detected else no_detection_fill_db


class MeasurementModel:
    """Thresholded SNR-only feedback model.

    This module sits between the physical channel and the probing policy.
    It takes true probe SNR values from the channel and converts them into
    the limited UE feedback used by the simulator:

    - if detected: return measured SNR
    - else: return no detection
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.cfg: MeasurementConfig = config.measurement

    @property
    def detection_threshold_db(self) -> float:
        return self.cfg.detection_threshold_db

    @property
    def feedback_noise_std_db(self) -> float:
        return self.cfg.feedback_noise_std_db

    def observe_snr(
        self,
        true_snr_db: float,
        *,
        pattern_name: str,
        rng: Optional[np.random.Generator] = None,
    ) -> MeasurementResult:
        rng = rng if rng is not None else np.random.default_rng(self.config.seed)

        if true_snr_db < self.detection_threshold_db:
            return MeasurementResult(
                pattern_name=pattern_name,
                true_snr_db=true_snr_db,
                measured_snr_db=None,
                detected=False,
            )

        measured = true_snr_db
        if self.feedback_noise_std_db > 0.0:
            measured += float(rng.normal(loc=0.0, scale=self.feedback_noise_std_db))

        return MeasurementResult(
            pattern_name=pattern_name,
            true_snr_db=true_snr_db,
            measured_snr_db=measured,
            detected=True,
        )

    def observe_link_budget(
        self,
        link_budget: LinkBudgetResult,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> MeasurementResult:
        return self.observe_snr(
            link_budget.snr_db,
            pattern_name=link_budget.pattern_name,
            rng=rng,
        )

    def observe_many(
        self,
        snr_values_db: Sequence[float],
        pattern_names: Sequence[str],
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> list[MeasurementResult]:
        if len(snr_values_db) != len(pattern_names):
            raise ValueError("snr_values_db and pattern_names must have the same length.")

        rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        return [
            self.observe_snr(float(snr_db), pattern_name=name, rng=rng)
            for snr_db, name in zip(snr_values_db, pattern_names)
        ]

    def numeric_vector(
        self,
        results: Sequence[MeasurementResult],
        *,
        no_detection_fill_db: Optional[float] = None,
    ) -> np.ndarray:
        """Convert sparse UE feedback into a numeric vector for policy logic."""
        fill_value = (
            self.detection_threshold_db - 10.0
            if no_detection_fill_db is None
            else float(no_detection_fill_db)
        )
        return np.asarray([r.numeric(fill_value) for r in results], dtype=float)

    def any_detected(self, results: Sequence[MeasurementResult]) -> bool:
        return any(r.detected for r in results)

    def count_detected(self, results: Sequence[MeasurementResult]) -> int:
        return sum(1 for r in results if r.detected)

    def measure_patterns(
        self,
        channel: MinimalRicianChannel,
        ue_state: UEState,
        pattern_names: Sequence[str],
        *,
        fading=None,
        group_combine: str = "equal_power",
        rng: Optional[np.random.Generator] = None,
    ) -> list[MeasurementResult]:
        """Convenience wrapper for measuring a list of named patterns."""
        rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        results: list[MeasurementResult] = []

        for pattern_name in pattern_names:
            lb = channel.evaluate(
                pattern_name,
                ue_state,
                fading=fading,
                group_combine=group_combine,
                rng=rng,
            )
            results.append(self.observe_link_budget(lb, rng=rng))
        return results


def make_measurement_model(config: SimulationConfig) -> MeasurementModel:
    return MeasurementModel(config)
