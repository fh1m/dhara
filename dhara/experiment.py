"""Controlled poses isolate the unit error without neural inference or hardware."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np

from .baseline import archived_velocity_update
from .velocity import velocity_between


def archived_field(previous_m, current_m):
    previous_pose = np.eye(4)
    previous_pose[:3, 3] = previous_m
    state = SimpleNamespace(position=np.asarray(current_m, dtype=np.float64), velocity=np.zeros(3))
    archived_velocity_update(SimpleNamespace(state=state, pose_history=[previous_pose]))
    return state.velocity


def run():
    origin = np.array([3.0, -2.0, 1.0])
    truth = np.array([0.5, 0.0, 0.0])
    start_ns = 1_000_000_000
    uniform = []
    for hz in (10, 25, 50, 60):
        elapsed_ns = round(1_000_000_000 / hz)
        position = origin + truth * (elapsed_ns / 1_000_000_000)
        uniform.append({
            "nominal_accepted_pose_hz": hz, "interval_ns": elapsed_ns,
            "archived_field_m_per_update": archived_field(origin, position).tolist(),
            "corrected_interval_average_mps": velocity_between(origin, position, start_ns, start_ns + elapsed_ns).tolist(),
        })
    offsets_ns = [0, 20_000_000, 60_000_000, 260_000_000, 310_000_000, 1_000_000_000]
    irregular = []
    for a, b in zip(offsets_ns, offsets_ns[1:]):
        previous = origin + truth * (a / 1_000_000_000)
        current = origin + truth * (b / 1_000_000_000)
        irregular.append({
            "capture_pair_ns": [start_ns + a, start_ns + b],
            "archived_field_m_per_update": archived_field(previous, current).tolist(),
            "corrected_interval_average_mps": velocity_between(previous, current, start_ns + a, start_ns + b).tolist(),
        })
    position = origin + np.array([0.05, 0.0, 0.0])
    capture_pair = [start_ns, start_ns + 100_000_000]
    completion_pair = [start_ns + 30_000_000, start_ns + 300_000_000]
    wrong_clock_event = {
        "capture_pair_ns": capture_pair, "completion_pair_ns": completion_pair,
        "capture_time_result_mps": velocity_between(origin, position, *capture_pair).tolist(),
        "completion_time_result_mps": velocity_between(origin, position, *completion_pair).tolist(),
        "meaning": "Positive completion intervals still use the wrong event times for pose velocity",
    }
    refusals = []
    for label, pair in (("duplicate", [start_ns, start_ns]), ("backwards", [start_ns + 1, start_ns])):
        try:
            velocity_between(origin, position, *pair)
        except ValueError as error:
            refusals.append({"case": label, "status": "refused", "error": str(error)})
        else:
            raise AssertionError(f"{label} capture interval was not refused")
    return {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "scope": "Controlled metric positions; unchanged archived numerical block; no image, model, camera, ROS runtime or vehicle validation",
        "baseline_source": "origin/perception-system/odometry/visual_odometry.py:167-169",
        "state_contract_source": "origin/perception-system/core/types.py:226-227",
        "truth_velocity_mps": truth.tolist(),
        "uniform_cadences": uniform, "irregular_capture_intervals": irregular,
        "capture_vs_completion": wrong_clock_event, "invalid_intervals": refusals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/controlled.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")
