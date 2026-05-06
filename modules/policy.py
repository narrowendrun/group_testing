
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

from beams import BeamCodebook
from channel import FadingSample, MinimalRicianChannel
from config import SimulationConfig
from geometry import UEState
from measurement import MeasurementModel, MeasurementResult


CoarseDecodeMode = Literal["max_group", "beam_score"]


@dataclass(frozen=True, slots=True)
class ExhaustiveSweepResult:
    selected_beam_index: int
    selected_beam_true_snr_db: float
    selected_beam_measured_snr_db: Optional[float]
    selected_beam_detected: bool
    oracle_best_beam_index: int
    oracle_best_snr_db: float
    all_true_snr_db: np.ndarray
    all_measurements: tuple[MeasurementResult, ...]
    num_probes: int
    success: bool


@dataclass(frozen=True, slots=True)
class GroupTestingResult:
    selected_group_name: str
    candidate_indices: tuple[int, ...]
    selected_beam_index: int
    selected_beam_true_snr_db: float
    selected_beam_measured_snr_db: Optional[float]
    selected_beam_detected: bool
    oracle_best_beam_index: int
    oracle_best_snr_db: float
    coarse_true_snr_db: np.ndarray
    coarse_measurements: tuple[MeasurementResult, ...]
    beam_scores: np.ndarray
    fine_true_snr_db: np.ndarray
    fine_measurements: tuple[MeasurementResult, ...]
    num_coarse_probes: int
    num_fine_probes: int
    num_probes: int
    success: bool


class BeamSearchPolicies:
    """Simple probing policies for the minimal attach simulator.

    Implemented policies
    --------------------
    1. Exhaustive sweep over all 121 pencil beams.
    2. Two-stage group testing:
       - coarse probing over predefined non-contiguous group patterns
       - shortlist one candidate set
       - fine sweep only inside that set
    """

    def __init__(
        self,
        config: SimulationConfig,
        *,
        codebook: Optional[BeamCodebook] = None,
        channel: Optional[MinimalRicianChannel] = None,
        measurement: Optional[MeasurementModel] = None,
    ) -> None:
        self.config = config
        self.codebook = codebook if codebook is not None else BeamCodebook(config)
        self.channel = (
            channel if channel is not None else MinimalRicianChannel(config, codebook=self.codebook)
        )
        self.measurement = (
            measurement if measurement is not None else MeasurementModel(config)
        )

    def _sample_shared_fading(
        self,
        ue_state: UEState,
        rng: Optional[np.random.Generator] = None,
    ) -> FadingSample:
        rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        return self.channel.sample_fading(
            rng=rng,
            range_m=ue_state.r_m,
            blocked=ue_state.blocked,
        )

    @staticmethod
    def _pick_best_measurement(
        results: Sequence[MeasurementResult],
        numeric_values: np.ndarray,
    ) -> int:
        if len(results) == 0:
            raise ValueError("results must be non-empty.")
        detected_indices = [i for i, r in enumerate(results) if r.detected]
        if detected_indices:
            detected_scores = numeric_values[detected_indices]
            return int(detected_indices[int(np.argmax(detected_scores))])
        return int(np.argmax(numeric_values))

    def exhaustive_sweep(
        self,
        ue_state: UEState,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> ExhaustiveSweepResult:
        rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        fading = self._sample_shared_fading(ue_state, rng=rng)

        true_snr_db = self.channel.oracle_pencil_snr_vector_db(
            ue_state,
            fading=fading,
        )
        oracle_best_beam_index = int(np.argmax(true_snr_db))
        oracle_best_snr_db = float(true_snr_db[oracle_best_beam_index])

        pattern_names = [self.codebook.get_pencil(i).name for i in range(self.codebook.num_beams)]
        measurements = self.measurement.observe_many(
            true_snr_db,
            pattern_names,
            rng=rng,
        )
        numeric = self.measurement.numeric_vector(measurements)
        selected_idx = self._pick_best_measurement(measurements, numeric)

        selected_measurement = measurements[selected_idx]
        return ExhaustiveSweepResult(
            selected_beam_index=selected_idx,
            selected_beam_true_snr_db=float(true_snr_db[selected_idx]),
            selected_beam_measured_snr_db=selected_measurement.measured_snr_db,
            selected_beam_detected=selected_measurement.detected,
            oracle_best_beam_index=oracle_best_beam_index,
            oracle_best_snr_db=oracle_best_snr_db,
            all_true_snr_db=np.asarray(true_snr_db, dtype=float),
            all_measurements=tuple(measurements),
            num_probes=self.codebook.num_beams,
            success=selected_measurement.detected,
        )

    def group_testing_search(
        self,
        ue_state: UEState,
        *,
        rng: Optional[np.random.Generator] = None,
        coarse_decode: CoarseDecodeMode = "max_group",
        normalize_beam_scores: bool = False,
        group_combine: str = "equal_power",
        no_detection_fill_db: Optional[float] = None,
    ) -> GroupTestingResult:
        rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        fading = self._sample_shared_fading(ue_state, rng=rng)

        # Oracle reference over all pencil beams for evaluation.
        oracle_true_snr_db = self.channel.oracle_pencil_snr_vector_db(
            ue_state,
            fading=fading,
        )
        oracle_best_beam_index = int(np.argmax(oracle_true_snr_db))
        oracle_best_snr_db = float(oracle_true_snr_db[oracle_best_beam_index])

        # Stage 1: coarse group tests.
        coarse_true_snr_db, group_names = self.channel.group_snr_vector_db(
            ue_state,
            fading=fading,
            group_combine=group_combine,
        )
        coarse_measurements = self.measurement.observe_many(
            coarse_true_snr_db,
            group_names,
            rng=rng,
        )
        coarse_numeric = self.measurement.numeric_vector(
            coarse_measurements,
            no_detection_fill_db=no_detection_fill_db,
        )

        if coarse_decode == "max_group":
            best_group_idx = self._pick_best_measurement(coarse_measurements, coarse_numeric)
            selected_group_name = group_names[best_group_idx]
            beam_scores = np.zeros(self.codebook.num_beams, dtype=float)
            beam_scores[list(self.codebook.candidate_indices_from_group(selected_group_name))] = coarse_numeric[best_group_idx]
        elif coarse_decode == "beam_score":
            beam_scores, _ = self.codebook.group_score_from_measurements(
                coarse_numeric,
                order=group_names,
                normalize_by_group_size=normalize_beam_scores,
            )
            top_beam = int(np.argmax(beam_scores))
            containing_groups = [
                name
                for name in group_names
                if top_beam in self.codebook.candidate_indices_from_group(name)
            ]
            if not containing_groups:
                raise RuntimeError("No group contains the top-scoring beam.")
            selected_group_name = containing_groups[0]
        else:
            raise ValueError("coarse_decode must be 'max_group' or 'beam_score'.")

        candidate_indices = self.codebook.candidate_indices_from_group(selected_group_name)

        # Stage 2: fine sweep only over shortlisted candidate beams.
        fine_true_snr_db = np.asarray(
            [oracle_true_snr_db[idx] for idx in candidate_indices],
            dtype=float,
        )
        fine_pattern_names = [self.codebook.get_pencil(idx).name for idx in candidate_indices]
        fine_measurements = self.measurement.observe_many(
            fine_true_snr_db,
            fine_pattern_names,
            rng=rng,
        )
        fine_numeric = self.measurement.numeric_vector(
            fine_measurements,
            no_detection_fill_db=no_detection_fill_db,
        )
        best_fine_local_idx = self._pick_best_measurement(fine_measurements, fine_numeric)
        selected_beam_index = int(candidate_indices[best_fine_local_idx])

        selected_measurement = fine_measurements[best_fine_local_idx]
        return GroupTestingResult(
            selected_group_name=selected_group_name,
            candidate_indices=tuple(candidate_indices),
            selected_beam_index=selected_beam_index,
            selected_beam_true_snr_db=float(oracle_true_snr_db[selected_beam_index]),
            selected_beam_measured_snr_db=selected_measurement.measured_snr_db,
            selected_beam_detected=selected_measurement.detected,
            oracle_best_beam_index=oracle_best_beam_index,
            oracle_best_snr_db=oracle_best_snr_db,
            coarse_true_snr_db=np.asarray(coarse_true_snr_db, dtype=float),
            coarse_measurements=tuple(coarse_measurements),
            beam_scores=np.asarray(beam_scores, dtype=float),
            fine_true_snr_db=np.asarray(fine_true_snr_db, dtype=float),
            fine_measurements=tuple(fine_measurements),
            num_coarse_probes=len(group_names),
            num_fine_probes=len(candidate_indices),
            num_probes=len(group_names) + len(candidate_indices),
            success=selected_measurement.detected,
        )


def make_policies(
    config: SimulationConfig,
    *,
    codebook: Optional[BeamCodebook] = None,
    channel: Optional[MinimalRicianChannel] = None,
    measurement: Optional[MeasurementModel] = None,
) -> BeamSearchPolicies:
    return BeamSearchPolicies(
        config=config,
        codebook=codebook,
        channel=channel,
        measurement=measurement,
    )
