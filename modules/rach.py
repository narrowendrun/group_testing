
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

from config import RACHConfig, SimulationConfig
from policy import ExhaustiveSweepResult, GroupTestingResult


RACHStatus = Literal["PENDING", "SUCCESS", "FAIL"]


@dataclass(frozen=True, slots=True)
class RACHStep:
    state_name: str
    beam_label: str
    snr_db: Optional[float]
    detected: bool
    status: RACHStatus
    note: str = ""


@dataclass(frozen=True, slots=True)
class RACHRunResult:
    method: str
    steps: tuple[RACHStep, ...]
    attached: bool
    fail_state: Optional[str]
    final_beam_index: int
    final_snr_db: float

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(step.state_name for step in self.steps)


class SimplifiedRACH:
    """Minimal logical RACH state machine.

    This is intentionally not a standards-accurate NR RACH implementation.
    It simply maps the project workflow into the logical stages:

        SEARCH -> MSG1 -> MSG2 -> MSG3 -> FINE_LOCK -> MSG4 -> ATTACHED/FAIL

    For group testing:
    - SEARCH corresponds to coarse group probing.
    - MSG1/MSG2/MSG3 use the selected coarse group beam.
    - FINE_LOCK uses the selected pencil beam after local refinement.
    - MSG4 uses the final locked pencil beam.

    For exhaustive sweep:
    - SEARCH corresponds to exhaustive probing over pencil beams.
    - MSG1/MSG2/MSG3/FINE_LOCK/MSG4 all use the selected pencil beam.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.cfg: RACHConfig = config.rach
        self.threshold_db = config.measurement.detection_threshold_db

    def _step(
        self,
        state_name: str,
        beam_label: str,
        snr_db: Optional[float],
        detected: bool,
        status: RACHStatus,
        note: str = "",
    ) -> RACHStep:
        return RACHStep(
            state_name=state_name,
            beam_label=beam_label,
            snr_db=snr_db,
            detected=detected,
            status=status,
            note=note,
        )

    def _msg_detected(self, snr_db: float) -> bool:
        return snr_db >= self.threshold_db

    def _coarse_msg_detected(self, snr_db: float) -> bool:
        if not self.cfg.require_detection_each_msg:
            return True
        return self._msg_detected(snr_db)

    def run_after_group_testing(self, result: GroupTestingResult) -> RACHRunResult:
        steps: list[RACHStep] = []

        selected_group_idx = list(result.coarse_measurements[i].pattern_name for i in range(len(result.coarse_measurements))).index(result.selected_group_name)
        coarse_selected_measurement = result.coarse_measurements[selected_group_idx]
        coarse_snr_db = float(result.coarse_true_snr_db[selected_group_idx])

        # SEARCH
        if not coarse_selected_measurement.detected:
            steps.append(
                self._step(
                    "SEARCH",
                    beam_label=result.selected_group_name,
                    snr_db=coarse_snr_db,
                    detected=False,
                    status="FAIL",
                    note="Selected coarse group was not detectable.",
                )
            )
            return RACHRunResult(
                method="group_testing",
                steps=tuple(steps),
                attached=False,
                fail_state="SEARCH",
                final_beam_index=result.selected_beam_index,
                final_snr_db=result.selected_beam_true_snr_db,
            )

        steps.append(
            self._step(
                "SEARCH",
                beam_label=result.selected_group_name,
                snr_db=coarse_snr_db,
                detected=True,
                status="SUCCESS",
                note="Coarse group probing completed.",
            )
        )

        # MSG1 / MSG2 / MSG3 on the coarse group beam.
        for state_name in self.cfg.coarse_message_names:
            msg_ok = self._coarse_msg_detected(coarse_snr_db)
            steps.append(
                self._step(
                    state_name,
                    beam_label=result.selected_group_name,
                    snr_db=coarse_snr_db,
                    detected=msg_ok,
                    status="SUCCESS" if msg_ok else "FAIL",
                    note="Logical coarse-beam RACH step.",
                )
            )
            if not msg_ok:
                return RACHRunResult(
                    method="group_testing",
                    steps=tuple(steps),
                    attached=False,
                    fail_state=state_name,
                    final_beam_index=result.selected_beam_index,
                    final_snr_db=result.selected_beam_true_snr_db,
                )

        # FINE_LOCK on final selected pencil beam.
        fine_ok = result.selected_beam_detected
        steps.append(
            self._step(
                self.cfg.fine_lock_state_name,
                beam_label=f"b{result.selected_beam_index:03d}",
                snr_db=result.selected_beam_true_snr_db,
                detected=fine_ok,
                status="SUCCESS" if fine_ok else "FAIL",
                note="Fine beam lock within shortlisted candidate set.",
            )
        )
        if not fine_ok:
            return RACHRunResult(
                method="group_testing",
                steps=tuple(steps),
                attached=False,
                fail_state=self.cfg.fine_lock_state_name,
                final_beam_index=result.selected_beam_index,
                final_snr_db=result.selected_beam_true_snr_db,
            )

        # MSG4 on the final pencil beam.
        msg4_ok = self._msg_detected(result.selected_beam_true_snr_db)
        steps.append(
            self._step(
                self.cfg.final_message_name,
                beam_label=f"b{result.selected_beam_index:03d}",
                snr_db=result.selected_beam_true_snr_db,
                detected=msg4_ok,
                status="SUCCESS" if msg4_ok else "FAIL",
                note="Final ACK on locked pencil beam.",
            )
        )
        if not msg4_ok:
            return RACHRunResult(
                method="group_testing",
                steps=tuple(steps),
                attached=False,
                fail_state=self.cfg.final_message_name,
                final_beam_index=result.selected_beam_index,
                final_snr_db=result.selected_beam_true_snr_db,
            )

        steps.append(
            self._step(
                "ATTACHED",
                beam_label=f"b{result.selected_beam_index:03d}",
                snr_db=result.selected_beam_true_snr_db,
                detected=True,
                status="SUCCESS",
                note="Attach completed successfully.",
            )
        )
        return RACHRunResult(
            method="group_testing",
            steps=tuple(steps),
            attached=True,
            fail_state=None,
            final_beam_index=result.selected_beam_index,
            final_snr_db=result.selected_beam_true_snr_db,
        )

    def run_after_exhaustive(self, result: ExhaustiveSweepResult) -> RACHRunResult:
        steps: list[RACHStep] = []
        beam_label = f"b{result.selected_beam_index:03d}"
        snr_db = result.selected_beam_true_snr_db

        # SEARCH
        if not result.selected_beam_detected:
            steps.append(
                self._step(
                    "SEARCH",
                    beam_label=beam_label,
                    snr_db=snr_db,
                    detected=False,
                    status="FAIL",
                    note="Exhaustive search selected an undetectable beam.",
                )
            )
            return RACHRunResult(
                method="exhaustive",
                steps=tuple(steps),
                attached=False,
                fail_state="SEARCH",
                final_beam_index=result.selected_beam_index,
                final_snr_db=snr_db,
            )

        steps.append(
            self._step(
                "SEARCH",
                beam_label=beam_label,
                snr_db=snr_db,
                detected=True,
                status="SUCCESS",
                note="Exhaustive sweep completed.",
            )
        )

        for state_name in self.cfg.coarse_message_names:
            msg_ok = self._coarse_msg_detected(snr_db)
            steps.append(
                self._step(
                    state_name,
                    beam_label=beam_label,
                    snr_db=snr_db,
                    detected=msg_ok,
                    status="SUCCESS" if msg_ok else "FAIL",
                    note="Logical RACH step on selected pencil beam.",
                )
            )
            if not msg_ok:
                return RACHRunResult(
                    method="exhaustive",
                    steps=tuple(steps),
                    attached=False,
                    fail_state=state_name,
                    final_beam_index=result.selected_beam_index,
                    final_snr_db=snr_db,
                )

        steps.append(
            self._step(
                self.cfg.fine_lock_state_name,
                beam_label=beam_label,
                snr_db=snr_db,
                detected=True,
                status="SUCCESS",
                note="No additional refinement needed after exhaustive search.",
            )
        )

        msg4_ok = self._msg_detected(snr_db)
        steps.append(
            self._step(
                self.cfg.final_message_name,
                beam_label=beam_label,
                snr_db=snr_db,
                detected=msg4_ok,
                status="SUCCESS" if msg4_ok else "FAIL",
                note="Final ACK on selected pencil beam.",
            )
        )
        if not msg4_ok:
            return RACHRunResult(
                method="exhaustive",
                steps=tuple(steps),
                attached=False,
                fail_state=self.cfg.final_message_name,
                final_beam_index=result.selected_beam_index,
                final_snr_db=snr_db,
            )

        steps.append(
            self._step(
                "ATTACHED",
                beam_label=beam_label,
                snr_db=snr_db,
                detected=True,
                status="SUCCESS",
                note="Attach completed successfully.",
            )
        )
        return RACHRunResult(
            method="exhaustive",
            steps=tuple(steps),
            attached=True,
            fail_state=None,
            final_beam_index=result.selected_beam_index,
            final_snr_db=snr_db,
        )

    def run(
        self,
        result: Union[ExhaustiveSweepResult, GroupTestingResult],
    ) -> RACHRunResult:
        if isinstance(result, GroupTestingResult):
            return self.run_after_group_testing(result)
        if isinstance(result, ExhaustiveSweepResult):
            return self.run_after_exhaustive(result)
        raise TypeError("Unsupported result type for RACH execution.")


def make_rach(config: SimulationConfig) -> SimplifiedRACH:
    return SimplifiedRACH(config)
