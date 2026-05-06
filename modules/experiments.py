from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from beams import BeamCodebook
from channel import MinimalRicianChannel
from config import SimulationConfig
from geometry import make_geometry
from measurement import MeasurementModel
from metrics import summarize_threshold_frame
from policy import BeamSearchPolicies
from rach import SimplifiedRACH
from simulator import AttachEpisodeSimulator, EpisodeResult, MethodName


DEFAULT_METHODS: tuple[MethodName, ...] = ("exhaustive", "group_testing")


def build_simulator(
    config: Optional[SimulationConfig] = None,
    *,
    threshold_db: Optional[float] = None,
) -> AttachEpisodeSimulator:
    """Build a simulator with explicitly shared component instances."""
    cfg = config if config is not None else SimulationConfig(seed=7, num_episodes=300)
    if threshold_db is not None:
        cfg.measurement.detection_threshold_db = float(threshold_db)

    codebook = BeamCodebook(cfg)
    geometry = make_geometry(cfg)
    channel = MinimalRicianChannel(cfg, codebook=codebook)
    measurement = MeasurementModel(cfg)
    policies = BeamSearchPolicies(
        cfg,
        codebook=codebook,
        channel=channel,
        measurement=measurement,
    )
    rach = SimplifiedRACH(cfg)

    return AttachEpisodeSimulator(
        cfg,
        geometry=geometry,
        codebook=codebook,
        channel=channel,
        measurement=measurement,
        policies=policies,
        rach=rach,
    )


def episode_to_dict(ep: EpisodeResult) -> dict[str, object]:
    """Flatten one episode result into the notebook's result-table schema."""
    row: dict[str, object] = {
        "method": ep.method,
        "ue_theta_deg": ep.ue_state.theta_deg,
        "ue_range_m": ep.ue_state.r_m,
        "blocked": ep.ue_state.blocked,
        "selected_beam_index": ep.selected_beam_index,
        "oracle_best_beam_index": ep.oracle_best_beam_index,
        "beam_index_error": ep.beam_index_error,
        "selected_equals_oracle": ep.selected_equals_oracle,
        "final_snr_db": ep.final_snr_db,
        "num_search_probes": ep.num_search_probes,
        "num_total_probes": ep.num_total_probes,
        "attached": ep.attached,
        "rach_fail_state": ep.rach_result.fail_state,
    }

    if ep.method == "group_testing":
        sr = ep.search_result
        row.update(
            {
                "selected_group_name": sr.selected_group_name,
                "num_coarse_probes": sr.num_coarse_probes,
                "num_fine_probes": sr.num_fine_probes,
                "candidate_size": len(sr.candidate_indices),
            }
        )
    else:
        row.update(
            {
                "selected_group_name": None,
                "num_coarse_probes": np.nan,
                "num_fine_probes": np.nan,
                "candidate_size": np.nan,
            }
        )

    return row


def run_many(
    sim: AttachEpisodeSimulator,
    method: MethodName,
    n_episodes: int,
    *,
    seed: int = 0,
    coarse_decode: str = "max_group",
    normalize_beam_scores: bool = False,
    group_combine: str = "equal_power",
    no_detection_fill_db: Optional[float] = None,
) -> pd.DataFrame:
    """Run many independent episodes and return a flat results table."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for episode_idx in range(n_episodes):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        episode_rng = np.random.default_rng(episode_seed)

        ep = sim.run_episode(
            method,
            rng=episode_rng,
            coarse_decode=coarse_decode,
            normalize_beam_scores=normalize_beam_scores,
            group_combine=group_combine,
            no_detection_fill_db=no_detection_fill_db,
        )

        row = episode_to_dict(ep)
        row["episode"] = episode_idx
        rows.append(row)

    columns = [
        "episode",
        "method",
        "ue_theta_deg",
        "ue_range_m",
        "blocked",
        "selected_beam_index",
        "oracle_best_beam_index",
        "beam_index_error",
        "selected_equals_oracle",
        "final_snr_db",
        "num_search_probes",
        "num_total_probes",
        "attached",
        "rach_fail_state",
        "selected_group_name",
        "num_coarse_probes",
        "num_fine_probes",
        "candidate_size",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_baseline_comparison(
    sim: AttachEpisodeSimulator,
    n_episodes: int,
    *,
    seed: int = 100,
    coarse_decode: str = "max_group",
    normalize_beam_scores: bool = False,
    group_combine: str = "equal_power",
    no_detection_fill_db: Optional[float] = None,
) -> pd.DataFrame:
    """Run exhaustive and group-testing baselines with matched episode seeds."""
    df_ex = run_many(sim, "exhaustive", n_episodes, seed=seed)
    df_gt = run_many(
        sim,
        "group_testing",
        n_episodes,
        seed=seed,
        coarse_decode=coarse_decode,
        normalize_beam_scores=normalize_beam_scores,
        group_combine=group_combine,
        no_detection_fill_db=no_detection_fill_db,
    )
    return pd.concat([df_ex, df_gt], ignore_index=True)


def run_threshold_sweep(
    thresholds_db: Sequence[float],
    *,
    n_episodes: int = 200,
    seed: int = 200,
    config_seed: int = 7,
    config_num_episodes: int = 300,
    methods: Sequence[MethodName] = DEFAULT_METHODS,
    return_episode_results: bool = False,
) -> pd.DataFrame:
    """Run the notebook's threshold sweep as reusable experiment code."""
    frames: list[pd.DataFrame] = []

    for threshold_db in thresholds_db:
        cfg = SimulationConfig(seed=config_seed, num_episodes=config_num_episodes)
        sim = build_simulator(cfg, threshold_db=float(threshold_db))

        for method in methods:
            df_method = run_many(sim, method, n_episodes, seed=seed)
            df_method["threshold_db"] = float(threshold_db)
            frames.append(df_method)

    if not frames:
        columns = ["threshold_db", "method"]
        return pd.DataFrame(columns=columns)

    results = pd.concat(frames, ignore_index=True)
    if return_episode_results:
        return results
    return summarize_threshold_frame(results)
