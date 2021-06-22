from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Any, Dict, Mapping, Optional, Set

logger = logging.getLogger(__name__)


class TrialPerf:
    """Perf info of a single trial"""

    def __init__(self):
        self.iter_speed: Optional[float] = None

    @property
    def is_complete(self) -> bool:
        return self.iter_speed is not None


class PerfManager:
    """
    Tracks trial performance in a trial group
    """

    def __init__(self, trial_ids: Set[str]):
        self.known_trials: Set[str] = trial_ids
        # map trial id to its perf info
        self.perf_infos: Dict[str, TrialPerf] = {}

        # TODO: find a way to pass budget info from algo to here

    @property
    def trials_missing_info(self) -> Set[str]:
        """Return set of trial ids that are missing perf info"""
        # completely unknown
        unknown = self.known_trials.difference(set(self.perf_infos.keys()))
        # incomplete
        incomplete = {
            trial_id
            for trial_id, perf in self.perf_infos.items()
            if not perf.is_complete
        }
        return unknown.union(incomplete)

    def get_height(self, trial_id: str, width: float) -> float:
        return self.perf_infos[trial_id].iter_speed / width

    def on_trial_result(self, trial_id: str, result: Mapping[str, Any]) -> None:
        if trial_id not in self.perf_infos:
            self.perf_infos[trial_id] = TrialPerf()
        self.perf_infos[trial_id].iter_speed = result["time_this_iter_s"]
