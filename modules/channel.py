
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union
import math

import numpy as np

from beams import BeamCodebook, BeamPatternSpec
from config import ChannelConfig, SimulationConfig
from geometry import UEState


PatternLike = Union[int, str, BeamPatternSpec]


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 10.0)


def linear_to_db(value_linear: float, floor_db: float = -300.0) -> float:
    if value_linear <= 0.0:
        return floor_db
    return 10.0 * math.log10(value_linear)


@dataclass(frozen=True, slots=True)
class FadingSample:
    """One OFDM Rician fading realization for a static UE during an episode."""

    h_taps: np.ndarray
    frequency_response: np.ndarray
    avg_channel_power_linear: float

    @property
    def h_complex(self) -> complex:
        """Compatibility view of the dominant time-domain tap."""
        return complex(self.h_taps[0])

    @property
    def abs_h_sq(self) -> float:
        """Compatibility view used by older scalar-channel callers."""
        return self.avg_channel_power_linear


@dataclass(frozen=True, slots=True)
class LinkBudgetResult:
    """Detailed link-budget breakdown for a single probe pattern."""

    pattern_name: str
    theta_deg: float
    range_m: float
    blocked: bool
    tx_power_dbm: float
    beam_gain_db: float
    rx_gain_db: float
    pathloss_db: float
    blockage_loss_db: float
    fading_power_db: float
    noise_dbm: float
    rx_power_dbm: float
    snr_db: float
    detected: bool


class MinimalRicianChannel:
    """Minimal OFDM pilot Rician channel model for the attach simulator.

    This backend follows the project design:
    - distance-based path loss
    - optional blockage penalty
    - frequency-selective Rician fading
    - effective SNR from known OFDM pilot symbols

    It intentionally does not model waveform-level effects, scheduling, or
    full 5G PHY behavior.
    """

    def __init__(
        self,
        config: SimulationConfig,
        codebook: Optional[BeamCodebook] = None,
    ) -> None:
        self.config = config
        self.channel_cfg: ChannelConfig = config.channel
        self.measurement_cfg = config.measurement
        self.codebook = codebook if codebook is not None else BeamCodebook(config)

    @property
    def noise_dbm(self) -> float:
        return self.channel_cfg.subcarrier_noise_dbm

    @property
    def tx_power_dbm(self) -> float:
        return self.channel_cfg.tx_power_dbm

    @property
    def rx_gain_dbi(self) -> float:
        return self.channel_cfg.rx_gain_dbi

    @property
    def detection_threshold_db(self) -> float:
        return self.measurement_cfg.detection_threshold_db

    def pathloss_db(self, range_m: float, blocked: bool = False) -> float:
        """Large-scale path loss in dB.

        Model:
            PL(d) = PL(d0) + 10 * alpha * log10(d / d0) + blockage_penalty
        """
        if range_m <= 0.0:
            raise ValueError("range_m must be positive.")

        cfg = self.channel_cfg
        pathloss = (
            cfg.pathloss_ref_db
            + 10.0 * cfg.pathloss_exponent_los * math.log10(range_m / cfg.pathloss_ref_distance_m)
        )
        if blocked:
            pathloss += cfg.blockage_loss_db
        return pathloss

    def large_scale_gain_linear(self, range_m: float, blocked: bool = False) -> float:
        """Large-scale channel power gain beta(r) in linear scale."""
        return db_to_linear(-self.pathloss_db(range_m, blocked=blocked))

    def nlos_tap_powers(self) -> np.ndarray:
        """Normalized exponential power-delay profile for NLOS taps."""
        cfg = self.channel_cfg
        tap_indices = np.arange(cfg.num_taps, dtype=float)
        powers = np.exp(-tap_indices / cfg.nlos_tap_decay)
        return powers / powers.sum()

    def sample_fading(
        self,
        rng: Optional[np.random.Generator] = None,
        *,
        range_m: float,
        blocked: bool = False,
    ) -> FadingSample:
        """Sample one frequency-selective Rician OFDM fading realization.

        h[l] = sqrt(beta) * (
            sqrt(K/(K+1)) * h_los[l]
            + sqrt(1/(K+1)) * h_nlos[l]
        )

        The LOS component occupies the first tap. The NLOS component uses a
        normalized exponential power-delay profile, then the active
        subcarrier response H[k] is computed by an NFFT-point FFT.
        """
        rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        cfg = self.channel_cfg

        beta = self.large_scale_gain_linear(range_m, blocked=blocked)
        k_lin = cfg.rician_k_linear

        h_los = np.zeros(cfg.num_taps, dtype=complex)
        psi = rng.uniform(0.0, 2.0 * math.pi)
        h_los[0] = complex(math.cos(psi), math.sin(psi))

        tap_powers = self.nlos_tap_powers()
        nlos_real = rng.normal(size=cfg.num_taps)
        nlos_imag = rng.normal(size=cfg.num_taps)
        h_nlos = (nlos_real + 1j * nlos_imag) / math.sqrt(2.0)
        h_nlos *= np.sqrt(tap_powers)

        h_taps = math.sqrt(beta) * (
            math.sqrt(k_lin / (k_lin + 1.0)) * h_los
            + math.sqrt(1.0 / (k_lin + 1.0)) * h_nlos
        )
        frequency_response = np.fft.fft(h_taps, n=cfg.nfft)[: cfg.num_active_subcarriers]
        avg_channel_power = float(np.mean(np.abs(frequency_response) ** 2))

        return FadingSample(
            h_taps=h_taps,
            frequency_response=frequency_response,
            avg_channel_power_linear=avg_channel_power,
        )

    def beam_gain_linear(
        self,
        pattern: PatternLike,
        theta_deg: float,
        *,
        group_combine: str = "equal_power",
    ) -> float:
        return self.codebook.gain_linear(
            pattern,
            theta_deg,
            combine=group_combine,
        )

    def received_power_dbm(
        self,
        pattern: PatternLike,
        ue_state: UEState,
        *,
        fading: Optional[FadingSample] = None,
        group_combine: str = "equal_power",
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Average received pilot-subcarrier signal power in dBm."""
        fading = (
            fading
            if fading is not None
            else self.sample_fading(rng=rng, range_m=ue_state.r_m, blocked=ue_state.blocked)
        )

        beam_gain_lin = self.beam_gain_linear(
            pattern,
            ue_state.theta_deg,
            group_combine=group_combine,
        )
        rx_power_mw = (
            db_to_linear(self.tx_power_dbm)
            * beam_gain_lin
            * db_to_linear(self.rx_gain_dbi)
            * fading.avg_channel_power_linear
        )
        return linear_to_db(rx_power_mw)

    def effective_pilot_snr_linear(
        self,
        pattern: PatternLike,
        ue_state: UEState,
        *,
        fading: Optional[FadingSample] = None,
        group_combine: str = "equal_power",
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Effective OFDM pilot SNR from the PDF's mean-subcarrier rule."""
        fading = (
            fading
            if fading is not None
            else self.sample_fading(rng=rng, range_m=ue_state.r_m, blocked=ue_state.blocked)
        )
        beam_gain_lin = self.beam_gain_linear(
            pattern,
            ue_state.theta_deg,
            group_combine=group_combine,
        )
        signal_mw = (
            db_to_linear(self.tx_power_dbm)
            * beam_gain_lin
            * db_to_linear(self.rx_gain_dbi)
            * fading.avg_channel_power_linear
        )
        noise_mw = db_to_linear(self.noise_dbm)
        return signal_mw / noise_mw

    def snr_db(
        self,
        pattern: PatternLike,
        ue_state: UEState,
        *,
        fading: Optional[FadingSample] = None,
        group_combine: str = "equal_power",
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        snr_linear = self.effective_pilot_snr_linear(
            pattern,
            ue_state,
            fading=fading,
            group_combine=group_combine,
            rng=rng,
        )
        return linear_to_db(snr_linear)

    def evaluate(
        self,
        pattern: PatternLike,
        ue_state: UEState,
        *,
        fading: Optional[FadingSample] = None,
        group_combine: str = "equal_power",
        rng: Optional[np.random.Generator] = None,
    ) -> LinkBudgetResult:
        """Return a full link-budget result for one beam probe."""
        spec = self.codebook.get_pattern(pattern)
        fading = (
            fading
            if fading is not None
            else self.sample_fading(rng=rng, range_m=ue_state.r_m, blocked=ue_state.blocked)
        )

        beam_gain_lin = self.beam_gain_linear(
            spec,
            ue_state.theta_deg,
            group_combine=group_combine,
        )
        beam_gain_db = linear_to_db(beam_gain_lin)
        pathloss_db = self.pathloss_db(ue_state.r_m, blocked=False)
        blockage_loss_db = self.channel_cfg.blockage_loss_db if ue_state.blocked else 0.0
        fading_power_db = linear_to_db(fading.abs_h_sq) - linear_to_db(
            self.large_scale_gain_linear(ue_state.r_m, blocked=ue_state.blocked)
        )

        rx_power_dbm = self.received_power_dbm(
            spec,
            ue_state,
            fading=fading,
            group_combine=group_combine,
        )
        snr_db = rx_power_dbm - self.noise_dbm

        return LinkBudgetResult(
            pattern_name=spec.name,
            theta_deg=ue_state.theta_deg,
            range_m=ue_state.r_m,
            blocked=ue_state.blocked,
            tx_power_dbm=self.tx_power_dbm,
            beam_gain_db=beam_gain_db,
            rx_gain_db=self.rx_gain_dbi,
            pathloss_db=pathloss_db,
            blockage_loss_db=blockage_loss_db,
            fading_power_db=fading_power_db,
            noise_dbm=self.noise_dbm,
            rx_power_dbm=rx_power_dbm,
            snr_db=snr_db,
            detected=(snr_db >= self.detection_threshold_db),
        )

    def oracle_pencil_snr_vector_db(
        self,
        ue_state: UEState,
        *,
        fading: Optional[FadingSample] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """SNRs for all 121 pencil beams under one fixed fading realization."""
        fading = (
            fading
            if fading is not None
            else self.sample_fading(rng=rng, range_m=ue_state.r_m, blocked=ue_state.blocked)
        )

        values = np.zeros(self.codebook.num_beams, dtype=float)
        for beam_index in range(self.codebook.num_beams):
            values[beam_index] = self.snr_db(
                beam_index,
                ue_state,
                fading=fading,
                group_combine="equal_power",
            )
        return values

    def group_snr_vector_db(
        self,
        ue_state: UEState,
        *,
        group_names: Optional[Sequence[str]] = None,
        fading: Optional[FadingSample] = None,
        group_combine: str = "equal_power",
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """SNRs for all coarse group tests under one fixed fading realization."""
        names = tuple(group_names) if group_names is not None else self.codebook.group_names
        fading = (
            fading
            if fading is not None
            else self.sample_fading(rng=rng, range_m=ue_state.r_m, blocked=ue_state.blocked)
        )

        values = np.zeros(len(names), dtype=float)
        for i, name in enumerate(names):
            values[i] = self.snr_db(
                name,
                ue_state,
                fading=fading,
                group_combine=group_combine,
            )
        return values, names


def make_channel(
    config: SimulationConfig,
    codebook: Optional[BeamCodebook] = None,
) -> MinimalRicianChannel:
    return MinimalRicianChannel(config=config, codebook=codebook)
