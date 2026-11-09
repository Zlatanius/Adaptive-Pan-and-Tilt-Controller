import math
from dataclasses import dataclass

@dataclass
class ScheduleSlot: 
    target_name : str
    start_time  : float
    end_time    : float

class TargetSchedule:
    def __init__(
        self,
        slots: list[ScheduleSlot],
        dt: float,
        duration: float,
        tol: float = 1e-9,
    ) -> None:
        slots = sorted(slots, key=lambda s: s.start_time)
        self._validate(slots, duration, tol)

        n_total = round(duration / dt)
        self.schedule: list[str] = []
        prev_step = 0

        for i, slot in enumerate(slots):
            is_last = i == len(slots) - 1
            end_step = n_total if is_last else round(slot.end_time / dt)
            self.schedule.extend([slot.target_name] * (end_step - prev_step))
            prev_step = end_step

        assert len(self.schedule) == n_total

    @staticmethod
    def _validate(slots: list[ScheduleSlot], duration: float, tol: float) -> None:
        if not slots:
            raise ValueError("Schedule needs at least one slot")

        for s in slots:
            if s.end_time <= s.start_time:
                raise ValueError(
                    f"Slot '{s.target_name}' ends at {s.end_time} "
                    f"but starts at {s.start_time}"
                )

        if not math.isclose(slots[0].start_time, 0.0, abs_tol=tol):
            raise ValueError(
                f"Schedule must start at 0, but first slot starts at "
                f"{slots[0].start_time}"
            )

        for prev, cur in zip(slots, slots[1:]):
            if not math.isclose(cur.start_time, prev.end_time, abs_tol=tol):
                kind = "Gap" if cur.start_time > prev.end_time else "Overlap"
                raise ValueError(
                    f"{kind} between '{prev.target_name}' (ends {prev.end_time}) "
                    f"and '{cur.target_name}' (starts {cur.start_time})"
                )

        if not math.isclose(slots[-1].end_time, duration, abs_tol=tol):
            raise ValueError(
                f"Schedule ends at {slots[-1].end_time}, "
                f"but duration is {duration}"
            )
