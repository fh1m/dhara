"""An interval-average velocity from metric positions and capture timestamps."""
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray


def velocity_between(
    previous_position_m: ArrayLike,
    position_m: ArrayLike,
    previous_capture_ns: int,
    capture_ns: int,
) -> NDArray[np.float64]:
    """Return a new 3-vector in m/s; never mutate either input position.

    Positions must already be metric and expressed in the same fixed frame.
    Integer-nanosecond timestamps must describe their capture instants in one
    clock domain. The caller owns those semantic guarantees: completion times
    and two unrelated clock epochs can also look numerically well-ordered.
    Nonpositive intervals are refused, not replaced with a nominal frame rate.
    This is an interval average, not an instantaneous-velocity estimator.
    """
    elapsed_ns = index(capture_ns) - index(previous_capture_ns)
    if elapsed_ns <= 0:
        raise ValueError("Capture timestamps must strictly increase")
    previous = np.asarray(previous_position_m, dtype=np.float64)
    current = np.asarray(position_m, dtype=np.float64)
    if previous.shape != (3,) or current.shape != (3,):
        raise ValueError("Positions must be three-component vectors")
    if not np.isfinite(previous).all() or not np.isfinite(current).all():
        raise ValueError("Positions must be finite")
    with np.errstate(over="raise", invalid="raise"):
        velocity = np.subtract(current, previous)
        velocity *= 1_000_000_000 / elapsed_ns
    return velocity
