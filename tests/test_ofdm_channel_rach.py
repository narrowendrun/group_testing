from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "modules"
if str(MODULES) not in sys.path:
    sys.path.insert(0, str(MODULES))

from channel import MinimalRicianChannel
from config import SimulationConfig
from geometry import UEState, polar_to_cartesian
from simulator import make_simulator


def make_fixed_ue() -> UEState:
    return UEState(
        position=polar_to_cartesian(r_m=25.0, theta_deg=0.0),
        r_m=25.0,
        theta_deg=0.0,
        blocked=False,
        los=True,
    )


def test_ofdm_fading_is_deterministic_for_fixed_seed() -> None:
    cfg = SimulationConfig(seed=7, num_episodes=1)
    channel = MinimalRicianChannel(cfg)

    first = channel.sample_fading(
        rng=np.random.default_rng(123),
        range_m=25.0,
        blocked=False,
    )
    second = channel.sample_fading(
        rng=np.random.default_rng(123),
        range_m=25.0,
        blocked=False,
    )

    assert first.h_taps.shape == (cfg.channel.num_taps,)
    assert first.frequency_response.shape == (cfg.channel.num_active_subcarriers,)
    assert np.allclose(first.h_taps, second.h_taps)
    assert np.allclose(first.frequency_response, second.frequency_response)
    assert first.avg_channel_power_linear == second.avg_channel_power_linear


def test_channel_vectors_keep_policy_shapes() -> None:
    cfg = SimulationConfig(seed=7, num_episodes=1)
    channel = MinimalRicianChannel(cfg)
    ue = make_fixed_ue()
    fading = channel.sample_fading(
        rng=np.random.default_rng(456),
        range_m=ue.r_m,
        blocked=ue.blocked,
    )

    pencil_snr = channel.oracle_pencil_snr_vector_db(ue, fading=fading)
    group_snr, group_names = channel.group_snr_vector_db(ue, fading=fading)

    assert pencil_snr.shape == (cfg.beam.num_beams,)
    assert group_snr.shape == (len(cfg.beam.group_patterns or {}),)
    assert len(group_names) == len(cfg.beam.group_patterns or {})
    assert np.isfinite(pencil_snr).all()
    assert np.isfinite(group_snr).all()


def test_effective_snr_matches_received_power_minus_subcarrier_noise() -> None:
    cfg = SimulationConfig(seed=7, num_episodes=1)
    channel = MinimalRicianChannel(cfg)
    ue = make_fixed_ue()
    fading = channel.sample_fading(
        rng=np.random.default_rng(789),
        range_m=ue.r_m,
        blocked=ue.blocked,
    )

    snr_db = channel.snr_db(60, ue, fading=fading)
    rx_power_dbm = channel.received_power_dbm(60, ue, fading=fading)

    assert np.isclose(snr_db, rx_power_dbm - channel.noise_dbm)


def test_probe_counts_and_simple_rach_fsm_are_preserved() -> None:
    cfg = SimulationConfig(seed=7, num_episodes=1)
    cfg.measurement.detection_threshold_db = -300.0
    sim = make_simulator(cfg)

    exhaustive = sim.run_episode(
        "exhaustive",
        rng=np.random.default_rng(111),
        force_theta_deg=0.0,
        force_range_m=25.0,
        blocked=False,
    )
    group_testing = sim.run_episode(
        "group_testing",
        rng=np.random.default_rng(111),
        force_theta_deg=0.0,
        force_range_m=25.0,
        blocked=False,
    )

    expected_states = ("SEARCH", "MSG1", "MSG2", "MSG3", "FINE_LOCK", "MSG4", "ATTACHED")

    assert exhaustive.num_search_probes == 121
    assert group_testing.num_search_probes == 22
    assert exhaustive.rach_result.state_names == expected_states
    assert group_testing.rach_result.state_names == expected_states
    assert exhaustive.attached
    assert group_testing.attached


if __name__ == "__main__":
    test_ofdm_fading_is_deterministic_for_fixed_seed()
    test_channel_vectors_keep_policy_shapes()
    test_effective_snr_matches_received_power_minus_subcarrier_noise()
    test_probe_counts_and_simple_rach_fsm_are_preserved()
