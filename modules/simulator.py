
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from beams import BeamCodebook
from channel import MinimalRicianChannel
from config import SimulationConfig
from geometry import SectorGeometry, UEState, make_geometry
from measurement import MeasurementModel
from policy import (
    BeamSearchPolicies,
    ExhaustiveSweepResult,
    GroupTestingResult,
)
from rach import RACHRunResult, SimplifiedRACH


MethodName = Literal["exhaustive", "group_testing"]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    method: MethodName
    ue_state: UEState
    search_result: Union[ExhaustiveSweepResult, GroupTestingResult]
    rach_result: RACHRunResult
    oracle_best_beam_index: int
    selected_beam_index: int
    beam_index_error: int
    final_snr_db: float
    num_search_probes: int
    num_total_probes: int
    attached: bool

    @property
    def selected_equals_oracle(self) -> bool:
        return self.selected_beam_index == self.oracle_best_beam_index


class AttachEpisodeSimulator:
    """Top-level episode runner for the minimal attach simulator."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        geometry: Optional[SectorGeometry] = None,
        codebook: Optional[BeamCodebook] = None,
        channel: Optional[MinimalRicianChannel] = None,
        measurement: Optional[MeasurementModel] = None,
        policies: Optional[BeamSearchPolicies] = None,
        rach: Optional[SimplifiedRACH] = None,
    ) -> None:
        self.config = config
        self.geometry = geometry if geometry is not None else make_geometry(config)
        self.codebook = codebook if codebook is not None else BeamCodebook(config)
        self.channel = (
            channel if channel is not None else MinimalRicianChannel(config, codebook=self.codebook)
        )
        self.measurement = (
            measurement if measurement is not None else MeasurementModel(config)
        )
        self.policies = (
            policies
            if policies is not None
            else BeamSearchPolicies(
                config,
                codebook=self.codebook,
                channel=self.channel,
                measurement=self.measurement,
            )
        )
        self.rach = rach if rach is not None else SimplifiedRACH(config)

    def sample_ue(
        self,
        *,
        rng: Optional[np.random.Generator] = None,
        force_theta_deg: Optional[float] = None,
        force_range_m: Optional[float] = None,
        blocked: Optional[bool] = None,
    ) -> UEState:
        if rng is None:
            import random
            py_rng = random.Random(self.config.seed)
        else:
            import random
            py_rng = random.Random(int(rng.integers(0, 2**31 - 1)))

        return self.geometry.sample_ue(
            rng=py_rng,
            force_theta_deg=force_theta_deg,
            force_range_m=force_range_m,
            blocked=blocked,
        )

    def run_episode(
        self,
        method: MethodName,
        *,
        rng: Optional[np.random.Generator] = None,
        force_theta_deg: Optional[float] = None,
        force_range_m: Optional[float] = None,
        blocked: Optional[bool] = None,
        coarse_decode: str = "max_group",
        normalize_beam_scores: bool = False,
        group_combine: str = "equal_power",
        no_detection_fill_db: Optional[float] = None,
    ) -> EpisodeResult:
        rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        ue_state = self.sample_ue(
            rng=rng,
            force_theta_deg=force_theta_deg,
            force_range_m=force_range_m,
            blocked=blocked,
        )

        if method == "exhaustive":
            search_result = self.policies.exhaustive_sweep(ue_state, rng=rng)
            rach_result = self.rach.run(search_result)
            return EpisodeResult(
                method="exhaustive",
                ue_state=ue_state,
                search_result=search_result,
                rach_result=rach_result,
                oracle_best_beam_index=search_result.oracle_best_beam_index,
                selected_beam_index=search_result.selected_beam_index,
                beam_index_error=abs(
                    search_result.selected_beam_index - search_result.oracle_best_beam_index
                ),
                final_snr_db=search_result.selected_beam_true_snr_db,
                num_search_probes=search_result.num_probes,
                num_total_probes=search_result.num_probes,
                attached=rach_result.attached,
            )

        if method == "group_testing":
            search_result = self.policies.group_testing_search(
                ue_state,
                rng=rng,
                coarse_decode=coarse_decode,
                normalize_beam_scores=normalize_beam_scores,
                group_combine=group_combine,
                no_detection_fill_db=no_detection_fill_db,
            )
            rach_result = self.rach.run(search_result)
            return EpisodeResult(
                method="group_testing",
                ue_state=ue_state,
                search_result=search_result,
                rach_result=rach_result,
                oracle_best_beam_index=search_result.oracle_best_beam_index,
                selected_beam_index=search_result.selected_beam_index,
                beam_index_error=abs(
                    search_result.selected_beam_index - search_result.oracle_best_beam_index
                ),
                final_snr_db=search_result.selected_beam_true_snr_db,
                num_search_probes=search_result.num_probes,
                num_total_probes=search_result.num_probes,
                attached=rach_result.attached,
            )

        raise ValueError("method must be 'exhaustive' or 'group_testing'")

    def run_exhaustive_episode(self, **kwargs) -> EpisodeResult:
        return self.run_episode("exhaustive", **kwargs)

    def run_group_testing_episode(self, **kwargs) -> EpisodeResult:
        return self.run_episode("group_testing", **kwargs)


def make_simulator(config: SimulationConfig) -> AttachEpisodeSimulator:
    return AttachEpisodeSimulator(config)
