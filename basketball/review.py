"""The owner's own calls on shots, kept next to the automatic ones.

No model gets every shot right. The owner can mark a shot "make", "miss" or
"not_a_shot" (or clear the mark). Marks are saved with the job (review.json) and
applied when results are served: each shot keeps its automatic call as
``auto_result``/``auto_reason`` and gets ``reviewed`` (the mark, or None); its
``result`` is the mark when there is one. Stats are recomputed from the final
results: a shot marked "not_a_shot" leaves attempts and FG%; unknown shots are
still excluded from FG%, as everywhere else. Pure functions, no I/O except the
explicit load/save/CSV helpers.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

OUTCOMES = ("make", "miss", "not_a_shot")
FILE = "review.json"
CSV = "shots_reviewed.csv"


def load(job_dir: Path) -> dict[str, str]:
    path = job_dir / FILE
    try:
        data = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        return {}
    return {str(k): v for k, v in data.items() if v in OUTCOMES}


def set_mark(job_dir: Path, shot: int, outcome: str | None, shot_numbers: list[int]) -> dict:
    """Save one mark (None clears it); returns all marks. Raises ValueError if invalid."""
    if shot not in shot_numbers:
        raise ValueError(f"no shot {shot} in this job")
    if outcome is not None and outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES} or empty")
    marks = load(job_dir)
    if outcome is None:
        marks.pop(str(shot), None)
    else:
        marks[str(shot)] = outcome
    tmp = job_dir / (FILE + ".partial")
    tmp.write_text(json.dumps(marks, indent=1, sort_keys=True))
    tmp.replace(job_dir / FILE)
    return marks


def tally(shots: list[dict]) -> dict:
    """Makes, misses, unknown and FG% over the final results (not-a-shot excluded)."""
    counted = [s for s in shots if s["result"] != "not_a_shot"]
    makes = sum(1 for s in counted if s["result"] == "make")
    misses = sum(1 for s in counted if s["result"] == "miss")
    unknown = sum(1 for s in counted if s["result"] == "unknown")
    judged = makes + misses
    return {"attempts": len(counted), "makes": makes, "misses": misses, "unknown": unknown,
            "judged": judged, "fg_pct": round(makes / judged, 4) if judged else None,
            "fg_denominator": "makes + misses (unknown shots excluded)",
            "reviewed": sum(1 for s in shots if s.get("reviewed")),
            "marked_not_a_shot": len(shots) - len(counted)}


def apply(result: dict, marks: dict[str, str]) -> dict:
    """A copy of a basketball *result* with the owner's marks applied."""
    out = copy.deepcopy(result)
    for s in out.get("shots") or []:
        s.setdefault("auto_result", s["result"])
        s.setdefault("auto_reason", s.get("reason", ""))
        mark = marks.get(str(s["shot"]))
        s["reviewed"] = mark
        s["result"] = mark if mark else s["auto_result"]
    if marks:
        out["auto_stats"] = result.get("stats")
        out["stats"] = tally(out["shots"])
    return out


def write_csv(out_dir: Path, result: dict) -> str:
    """``shots_reviewed.csv``: the automatic and the owner's call side by side."""
    with (out_dir / CSV).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["shot", "release_s", "auto_result", "auto_reason", "reviewed_result",
                    "final_result"])
        for s in result.get("shots") or []:
            w.writerow([s["shot"], s.get("release_s"), s.get("auto_result", s["result"]),
                        s.get("auto_reason", s.get("reason", "")), s.get("reviewed") or "",
                        s["result"]])
    return CSV
