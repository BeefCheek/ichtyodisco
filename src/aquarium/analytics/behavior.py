"""Per-fish behavioural state with thresholds that adapt to the tank's own population."""

from __future__ import annotations

from dataclasses import dataclass

from aquarium.analytics.stats import DecayingHistogram

STATES = ("rest", "cruise", "explore", "burst")
STATE_LABELS_FR = {"rest": "repos", "cruise": "croisière", "explore": "exploration",
                   "burst": "sprint"}


@dataclass(slots=True)
class _FishState:
    state: str = "cruise"
    candidate: str = "cruise"
    candidate_since: float = 0.0
    entered_at: float = 0.0


class BehaviorClassifier:
    """rest / cruise / explore / burst, from speed (body lengths/s) and path tortuosity.

    * thresholds are quantiles of a decaying histogram of *all* fish speeds in this tank,
      with absolute floors so a tank that is asleep all day doesn't call 0.05 BL/s a sprint.
    * a short dwell time (hysteresis) prevents state flicker on noisy frames.
    """

    def __init__(self, dwell_s: float = 0.6, half_life_s: float = 6 * 3600.0):
        self.dwell_s = dwell_s
        self.speed_hist = DecayingHistogram(0.01, 50.0, bins=80, half_life=half_life_s, log=True)
        self._fish: dict[int, _FishState] = {}

    def thresholds(self) -> tuple[float, float]:
        rest = self.speed_hist.quantile(0.20, 0.3) or 0.3
        burst = self.speed_hist.quantile(0.95, 4.0) or 4.0
        rest = min(max(rest, 0.15), 1.0)
        burst = max(burst, rest * 4, 2.0)
        return rest, burst

    def learn(self, speed_bls: float, t: float) -> None:
        self.speed_hist.update(max(speed_bls, 0.011), t)

    def classify(self, fish_id: int, speed_bls: float, tortuosity: float, t: float) -> str:
        rest, burst = self.thresholds()
        if speed_bls < rest:
            raw = "rest"
        elif speed_bls > burst:
            raw = "burst"
        elif tortuosity > 1.8:
            raw = "explore"
        else:
            raw = "cruise"
        st = self._fish.get(fish_id)
        if st is None:
            st = _FishState(state=raw, candidate=raw, candidate_since=t, entered_at=t)
            self._fish[fish_id] = st
            return raw
        if raw != st.candidate:
            st.candidate, st.candidate_since = raw, t
        # bursts are short by nature: accept them immediately
        if st.candidate != st.state and (raw == "burst" or t - st.candidate_since >= self.dwell_s):
            st.state, st.entered_at = st.candidate, t
        return st.state

    def time_in_state(self, fish_id: int, t: float) -> float:
        st = self._fish.get(fish_id)
        return 0.0 if st is None else t - st.entered_at

    def forget(self, fish_id: int) -> None:
        self._fish.pop(fish_id, None)

    def to_dict(self) -> dict:
        return {"speed_hist": self.speed_hist.to_dict()}

    def load_dict(self, d: dict) -> None:
        if "speed_hist" in d:
            self.speed_hist = DecayingHistogram.from_dict(d["speed_hist"])
