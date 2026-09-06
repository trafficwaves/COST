#!/usr/bin/env python3
"""Trajectory -> (speed, acceleration, VSP, operating mode), MOVES conventions.

This is the one implementation shared by the generator (`what_if_scale_sweep.py`)
and the aggregator (`fig_data.py`). It used to live inside the generator, which
meant the aggregator could not recompute what the generator had deliberately not
stored; splitting it out is what makes `--store minimal` and sub-bin pricing
compatible.

MULTI-CLASS, AND WHY THE SPLIT IS WHERE IT IS. Speed and acceleration are
kinematics: they are properties of the trajectory and are identical for every
vehicle class. VSP and the operating mode are not -- they run the same kinematics
through a class-specific road-load polynomial. So the kinematics are computed
once and the per-class work is one polynomial plus one classification per class.
That is the whole reason a five-class sweep costs barely more than a one-class
sweep: the expensive part (the QP that produces `space`) is class-independent and
is never repeated.

Units, identical to the rest of the repo
    time          s
    space         miles
    speed         mph
    acceleration  miles/s^2   (multiply by 1609.344 for m/s^2)
    vsp           kW/tonne
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from opMode_utils import opMode_array
from vehicle_classes import VehicleClass
from vsp_utils import VSP

MPH_TO_MS = 0.44704
MILES_S2_TO_MS2 = 1609.344
MILES_TO_METERS = 1609.344


# Finite-difference scheme used to turn speed into ACCELERATION.
#
# "backward" is what the real MOVES software does. In
# LinkOperatingModeDistributionGenerator.calculateOpModeFractionsCore() MOVES
# emits
#
#     At0 = coalesce(speed[t]   - speed[t-1], 0.0)     -- mph/sec
#     At1 = coalesce(speed[t-1] - speed[t-2], 0.0)
#     At2 = coalesce(speed[t-2] - speed[t-3], 0.0)
#
# and the same speed[t] - speed[t-1] appears again inside the VSP equation as
# the sourceMass * v * a term. So it is a *backward* difference, it is shared
# by the VSP calculation and the braking test, and it is 0 at the first sample.
#
# "centered" is the np.gradient scheme this pipeline used before, kept only so
# that previously published numbers can be reproduced exactly.
#
# Only the ACCELERATION is affected. SPEED remains a centered np.gradient of
# position, deliberately: MOVES is *handed* speed as a per-second input from a
# drive schedule and never differentiates a position record, whereas we have to
# recover speed from a position trace, and there the centered difference is the
# better (second-order, phase-unbiased) estimator of instantaneous speed. What
# the MOVES source actually pins down is the definition of acceleration, so
# that -- and only that -- is what we match here.
#
# sweep_max_gap_scale_mp.py carries a byte-identical constant and helper in its
# own pandas implementation. The two must stay in step; change one, change the
# other.
ACCEL_SCHEME = "backward"


def acceleration(
    speed: np.ndarray, time_s: np.ndarray, scheme: str | None = None
) -> np.ndarray:
    """Differentiate speed with respect to time under the selected scheme.

    Unit-agnostic: the result is whatever `speed` is, per second. Passing speed
    in miles/s therefore returns miles/s^2, which is this module's internal
    acceleration unit.

    The backward branch divides by the actual sample spacing rather than
    assuming dt = 1 s. On the 1 Hz data MOVES is defined over that division is
    a no-op, so the result is literally MOVES' `speed[t] - speed[t-1]`; on any
    other sampling rate it is the correct rate of change instead of a raw
    per-sample increment. The leading zero is MOVES' `coalesce(..., 0.0)`.
    """
    scheme = ACCEL_SCHEME if scheme is None else scheme
    if scheme == "centered":
        return np.gradient(speed, time_s)
    if scheme != "backward":
        raise ValueError(
            f"unknown ACCEL_SCHEME {scheme!r}; expected 'backward' or 'centered'"
        )
    acc = np.zeros(len(speed), dtype=float)
    if len(speed) > 1:
        acc[1:] = np.diff(speed) / np.diff(time_s)
    return acc


_acceleration = acceleration          # the generator's historical private name


def _lagged(acc_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MOVES' At1/At2, reproducing the original shift(1)/shift(2) + fillna chain.

    t_1 = [a0, a0, a1, ...]; t_2 = [t_1[0], t_1[1], a0, a1, ...].
    """
    a1 = np.empty_like(acc_ms)
    a1[0] = acc_ms[0]
    a1[1:] = acc_ms[:-1]
    a2 = np.empty_like(acc_ms)
    a2[0] = a1[0]
    if len(acc_ms) > 1:
        a2[1] = a1[1]
    if len(acc_ms) > 2:
        a2[2:] = acc_ms[:-2]
    return a1, a2


def _as_tuple(vehicles) -> tuple[VehicleClass, ...]:
    if isinstance(vehicles, VehicleClass):
        return (vehicles,)
    return tuple(vehicles)


# --------------------------------------------------------------------------- #
# one trajectory, any number of classes
# --------------------------------------------------------------------------- #
def compute_state(
    time_s: np.ndarray,
    space_miles: np.ndarray,
    vehicles,
    scheme: str | None = None,
) -> dict[str, np.ndarray]:
    """Kinematics once, then VSP and operating mode for every vehicle class.

    Returns `{"speed", "acceleration", "vsp_st<ID>", "op_mode_ID_st<ID>", ...}`.
    Speed is mph and acceleration miles/s^2; the per-class keys follow
    `VehicleClass.vsp_column` / `.op_mode_column`.

    Numerically identical to the pandas implementation in
    `sweep_max_gap_scale_mp` for the passenger-car class (same derivative
    schemes, the same lagged-acceleration fill rule), but works on numpy arrays
    and replaces a row-wise DataFrame.apply with a vectorised classifier. At 26
    gaps x ~1.9M samples per lane the pandas row-Series overhead dominates
    everything else, including the QP solves.

    Acceleration follows ACCEL_SCHEME (see the note on that constant); speed
    stays a centered difference of position regardless.
    """
    vehicles = _as_tuple(vehicles)
    speed_miles_s = np.gradient(space_miles, time_s)
    speed_mph = speed_miles_s * 3600.0
    accel_miles_s2 = acceleration(speed_miles_s, time_s, scheme)

    speed_ms = speed_mph * MPH_TO_MS
    acc_ms = accel_miles_s2 * MILES_S2_TO_MS2
    a1, a2 = _lagged(acc_ms)

    out: dict[str, np.ndarray] = {"speed": speed_mph, "acceleration": accel_miles_s2}
    for v in vehicles:
        vsp = np.asarray(VSP(speed_ms, acc_ms, **v.vsp_kwargs()), dtype=float)
        # Vectorised classifier: verified bit-identical to the scalar opMode() on
        # an exhaustive 236,544-case boundary grid and on 1.2M real samples, at
        # 44x the throughput (1.92 -> 0.043 us/sample).
        out[v.vsp_column] = vsp
        out[v.op_mode_column] = opMode_array(speed_mph, vsp, acc_ms, a1, a2)
    return out


# --------------------------------------------------------------------------- #
# a drive schedule, which is what MOVES is natively handed
# --------------------------------------------------------------------------- #
def cycle_state(
    speed_mph: np.ndarray,
    time_s: np.ndarray,
    vehicles,
) -> pd.DataFrame:
    """VSP and operating mode for a drive SCHEDULE, MOVES conventions exactly.

    Distinct from `compute_state` in one deliberate way: speed is GIVEN, not
    differentiated out of a position record. That is MOVES' own situation -- a
    dynamometer cycle hands it speed per second -- so no centered difference is
    taken and the only derivative is the backward acceleration:
    `(dss.speed - dss2.speed) * 0.44704` of OperatingModeDistributionGenerator.java.

    Used to measure the sub-bin VSP centroids on the EPA certification cycles
    (`make_subbin_centroids.py`), where introducing a position round trip would
    put a smoothing error into the calibration itself.

    Returns one row per second with `time`, `speed_mph`, and `vsp_st<ID>` /
    `op_mode_ID_st<ID>` per class.
    """
    vehicles = _as_tuple(vehicles)
    speed_mph = np.asarray(speed_mph, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    v_ms = speed_mph * MPH_TO_MS
    a_ms2 = np.zeros_like(v_ms)
    if len(v_ms) > 1:
        a_ms2[1:] = np.diff(v_ms) / np.diff(time_s)
    a1, a2 = _lagged(a_ms2)

    out = pd.DataFrame({"time": time_s, "speed_mph": speed_mph})
    for v in vehicles:
        vsp = np.asarray(VSP(v_ms, a_ms2, **v.vsp_kwargs()), dtype=float)
        out[v.vsp_column] = vsp
        out[v.op_mode_column] = opMode_array(speed_mph, vsp, a_ms2, a1, a2)
    return out


def boundary_conditions(
    time_s: np.ndarray, space_miles: np.ndarray
) -> tuple[float, float, float, float]:
    """Endpoint speed/accel in optimizer-internal units (miles/s, miles/s^2)."""
    if len(time_s) < 4:
        raise ValueError("Need at least 4 samples to compute boundary conditions")

    v_start = (space_miles[1] - space_miles[0]) / (time_s[1] - time_s[0])
    v_end = (space_miles[-1] - space_miles[-2]) / (time_s[-1] - time_s[-2])
    a_start = (
        (space_miles[2] - space_miles[1]) / (time_s[2] - time_s[1])
        - (space_miles[1] - space_miles[0]) / (time_s[1] - time_s[0])
    ) / (time_s[1] - time_s[0])
    a_end = (
        (space_miles[-1] - space_miles[-2]) / (time_s[-1] - time_s[-2])
        - (space_miles[-2] - space_miles[-3]) / (time_s[-2] - time_s[-3])
    ) / (time_s[-1] - time_s[-2])
    return v_start, v_end, a_start, a_end


# --------------------------------------------------------------------------- #
# a stored table, recovered
# --------------------------------------------------------------------------- #
def add_state_columns(
    df: pd.DataFrame,
    vehicles,
    *,
    group_cols: Sequence[str] = ("vehicle_index",),
    time_col: str = "time",
    space_col: str = "space",
    scheme: str | None = None,
    with_vsp: bool = True,
    with_op_mode: bool = False,
) -> pd.DataFrame:
    """Recover speed (and optionally VSP / operating mode) from space and time.

    This is the counterpart of `--store minimal`, which drops speed, acceleration
    and VSP because they are smooth-function-of-noise float32 columns that zstd
    cannot compress, and recovers them exactly from `space` and `time` by the same
    calls that produced them. Measured round-trip error from the stored float32
    space is 3.5e-4 m/s^2 on acceleration -- 0.012% of the 3.0 m/s^2 limit.

    Sub-bin pricing is the reason this exists: it needs speed AND VSP per second,
    not just the stored operating mode, so an aggregator reading a minimal store
    has to rebuild them. `with_op_mode=True` additionally re-derives the operating
    mode, which is only useful for verifying the stored one.

    `group_cols` must identify one trajectory per group -- for a counterfactual
    table that is `("vehicle_index", "max_gap")`, since one vehicle appears once
    per gap. Rows within a group are differenced in the order given, so the frame
    must already be sorted by `group_cols + [time_col]`; that is how both the
    empirical and counterfactual parts are written.
    """
    vehicles = _as_tuple(vehicles)
    group_cols = list(group_cols)
    missing = [c for c in group_cols + [time_col, space_col] if c not in df.columns]
    if missing:
        raise KeyError(f"add_state_columns needs column(s) {missing}")

    n = len(df)
    out_speed = np.empty(n, dtype=np.float64)
    out_vsp = {v.vsp_column: np.empty(n, dtype=np.float64) for v in vehicles} \
        if with_vsp else {}
    out_mode = {v.op_mode_column: np.empty(n, dtype=np.int16) for v in vehicles} \
        if with_op_mode else {}

    # Position-based slicing rather than a groupby-apply: the groups are long
    # (hundreds of samples) and there are millions of them across a lane, so the
    # per-group DataFrame construction a groupby-apply performs dominates the
    # arithmetic. `sort=False` keeps the row order as given.
    t_all = df[time_col].to_numpy(dtype=float)
    s_all = df[space_col].to_numpy(dtype=float)
    for _key, idx in df.groupby(group_cols, sort=False, observed=True).indices.items():
        t = t_all[idx]
        s = s_all[idx]
        if len(t) < 2:
            # A one-sample group has no derivative. np.gradient would raise; the
            # honest value is 0, which is what a stalled sample means anyway.
            out_speed[idx] = 0.0
            for v in vehicles:
                if with_vsp:
                    out_vsp[v.vsp_column][idx] = 0.0
                if with_op_mode:
                    out_mode[v.op_mode_column][idx] = 1
            continue
        st = compute_state(t, s, vehicles, scheme)
        out_speed[idx] = st["speed"]
        for v in vehicles:
            if with_vsp:
                out_vsp[v.vsp_column][idx] = st[v.vsp_column]
            if with_op_mode:
                out_mode[v.op_mode_column][idx] = st[v.op_mode_column]

    df = df.copy()
    df["speed"] = out_speed.astype(np.float32)
    for col, arr in out_vsp.items():
        df[col] = arr.astype(np.float32)
    for col, arr in out_mode.items():
        df[col] = arr
    return df


def op_mode_columns(df: pd.DataFrame) -> dict[int, str]:
    """{sourceTypeID: column} for every per-class operating-mode column present.

    A table written before the multi-class sweep carries a bare `op_mode_ID` with
    no suffix. That was always the passenger car, so it is reported as source type
    21 and old caches keep working.
    """
    found: dict[int, str] = {}
    for col in df.columns:
        if col.startswith("op_mode_ID_st"):
            try:
                found[int(col[len("op_mode_ID_st"):])] = col
            except ValueError:
                continue
    if not found and "op_mode_ID" in df.columns:
        found[21] = "op_mode_ID"
    return found


def parquet_op_mode_columns(path: str) -> dict[int, str]:
    """`op_mode_columns` from a parquet file's schema, without reading any rows."""
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    return op_mode_columns(pd.DataFrame(columns=list(names)))


__all__ = [
    "ACCEL_SCHEME", "MPH_TO_MS", "MILES_S2_TO_MS2", "MILES_TO_METERS",
    "acceleration", "compute_state", "cycle_state", "boundary_conditions",
    "add_state_columns", "op_mode_columns", "parquet_op_mode_columns",
]
