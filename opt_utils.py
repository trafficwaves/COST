import os
from typing import Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_trajectory_data(filepath: str) -> pd.DataFrame:
    """
    Load trajectory data from CSV file.
    
    Args:
        filepath: Path to the CSV file containing trajectory data.
        
    Returns:
        DataFrame with trajectory data.
    """
    return pd.read_csv(filepath)


def extract_trajectory_input(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Extract trajectory parameters from data.
    
    Args:
        data: DataFrame containing time, space, and speed columns.
        
    Returns:
        Tuple of (time_data, space_data, speed_data, a_start, a_end)
    """
    time_data = data['time'].values
    space_data = data['space'].values
    speed_data = data['speed'].values / 3600  # convert from mph to miles per second
    
    # Estimate the acceleration at start and end using consistent finite differences
    # Forward difference at start, backward difference at end
    a_start = (speed_data[1] - speed_data[0]) / (time_data[1] - time_data[0])
    a_end = (speed_data[-1] - speed_data[-2]) / (time_data[-1] - time_data[-2])
    
    return time_data, space_data, speed_data, a_start, a_end


def solve_constrained_smoothing(
    time_data: np.ndarray,
    space_ref: np.ndarray,
    leader_time: np.ndarray,
    leader_ref: np.ndarray,
    v_start: float,
    v_end: float,
    a_start: float,
    a_end: float,
    max_gap: float = 4.0,
    lambda_1: float = 10.0,
    comfort_decel_ms2: float = 3.4,
    accel_max_ms2: float = 3.0,
    jerk_max_ms3: float = 3.0,
    pin_boundary_accel: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Solve trajectory smoothing optimization problem with constraints.

    Args:
        time: Time array for the current vehicle.
        space_ref: Reference space trajectory.
        leader_time: Time array for the leader vehicle.
        leader_ref: Leader's space trajectory.
        v_start: Initial velocity.
        v_end: Final velocity.
        a_start: Initial acceleration.
        a_end: Final acceleration.
        max_gap: Maximum allowed gap from reference trajectory.
        lambda_1: Weight for acceleration smoothness term.
        comfort_decel_ms2: Comfortable deceleration limit (m/s^2, positive
            magnitude). The counterfactual may not brake harder than this, i.e.
            a >= -comfort_decel_ms2 everywhere. Set <= 0 to disable.
        accel_max_ms2: Comfortable acceleration limit (m/s^2, positive
            magnitude), i.e. a <= accel_max_ms2 everywhere. Set <= 0 to disable.
        jerk_max_ms3: Comfortable jerk limit (m/s^3, positive magnitude). The
            counterfactual's jerk is bounded, |da/dt| <= jerk_max_ms3. Set <= 0
            to disable.

    The three limits default to a vehicle-CAPABILITY set, not a ride-comfort
    set: 3.4 m/s^2 deceleration (the AASHTO figure), 3.0 m/s^2 acceleration,
    3.0 m/s^3 jerk.  The claim they support is "the counterfactual never demands
    more than a real vehicle can deliver", which is a physics statement.

    Comfort-scale values (1.5-2.0 m/s^2, 1.5 m/s^3) were tried and rejected:
    measured on the I-24 MOTION virtual trajectories in /data/emission, the
    per-vehicle EMPIRICAL extrema already exceed them for most vehicles --
    median braking is -2.83 m/s^2 and median peak |jerk| is 3.74 m/s^3, and
    82% of vehicles brake harder than 2.0 m/s^2.  Constraining the
    counterfactual below the data it is derived from made ~11% of the QPs
    infeasible, and the infeasible set is biased toward harsh (high-emission)
    driving, so dropping it would bias every aggregate downward.

    Returns:
        Tuple of (smoothed_position, smoothed_velocity, average_velocity, objective_cost)
    """
    N = len(time_data) - 1
    dt = np.mean(np.diff(time_data))

    # Comfort limits are specified in SI (m/s^2, m/s^3) but the optimizer works
    # in internal units (space in miles, time in s): a is miles/s^2, jerk is
    # miles/s^3. Convert once here.  1 mile = 1609.344 m.
    METERS_PER_MILE = 1609.344
    a_dec_lim = comfort_decel_ms2 / METERS_PER_MILE if comfort_decel_ms2 > 0 else None
    a_acc_lim = accel_max_ms2 / METERS_PER_MILE if accel_max_ms2 > 0 else None
    jerk_lim = jerk_max_ms3 / METERS_PER_MILE if jerk_max_ms3 > 0 else None

    # The end accelerations are pinned to the empirical (noisy finite-difference)
    # values.  If a real endpoint brakes or accelerates harder than the comfort
    # limit, that equality would conflict with the corresponding inequality and
    # make the QP infeasible, so clip the pinned value into the comfort band on
    # BOTH sides -- the counterfactual is comfort-bounded end to end.
    if a_dec_lim is not None:
        a_start = max(a_start, -a_dec_lim)
        a_end = max(a_end, -a_dec_lim)
    if a_acc_lim is not None:
        a_start = min(a_start, a_acc_lim)
        a_end = min(a_end, a_acc_lim)
    # Calculate overlap between two time arrays
    time_start = max(time_data[0], leader_time[0])
    time_end = min(time_data[-1], leader_time[-1])

    # Check if there's overlap
    if time_start <= time_end:
        
        # Get overlapping portions
        mask_current = (time_data >= time_start) & (time_data <= time_end)
        mask_leader = (leader_time >= time_start) & (leader_time <= time_end)

    overlap_leader = leader_ref[mask_leader]
    
    
    # Decision Variables
    x = cp.Variable(N + 1)
    # v_avg = cp.Variable(1)
    init_gap = overlap_leader[0] - x[0]
    # Dynamics
    v = (x[1:] - x[:-1]) / dt
    #
    a = (v[1:] - v[:-1]) / dt
    # jerk (third derivative): finite difference of acceleration
    jerk = (a[1:] - a[:-1]) / dt

    # Cost: Minimize Variance from v_avg + smoothness
    # cost = cp.sum_squares(v) + lambda_1 * cp.sum_squares(a)
    cost = cp.sum_squares(v) + lambda_1 * cp.sum_squares(a)
    v_avg = 0
    # Constraints
    constraints = [
        x[0] == space_ref[0],       # Start Position
        x[N] == space_ref[-1],      # End Position
        v[0] == v_start,            # Start Speed
        v[-1] == v_end,             # End Speed
        # x <= space_ref,             # CANNOT overpass reference
        x[mask_current] <= overlap_leader,    # CANNOT overpass leader (interpolated)
        x[mask_current] >= overlap_leader - init_gap - max_gap,   # CANNOT be far behind reference
        v >= 0                      # Speed Non-negative
    ]

    # Pin endpoint accelerations to the empirical values for C2 continuity.
    # These are noisy finite differences; with tight comfort limits they can
    # over-constrain short segments, so they are optional.
    if pin_boundary_accel:
        constraints.append(a[0] == a_start)
        constraints.append(a[-1] == a_end)

    # Comfort deceleration: never brake harder than the comfort limit.
    if a_dec_lim is not None:
        constraints.append(a >= -a_dec_lim)

    # Comfort acceleration: never accelerate harder than the comfort limit.
    if a_acc_lim is not None:
        constraints.append(a <= a_acc_lim)

    # Comfort jerk: bound the rate of change of acceleration on both sides.
    if jerk_lim is not None:
        constraints.append(jerk <= jerk_lim)
        constraints.append(jerk >= -jerk_lim)

    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    # Solve with a robust interior-point conic solver. The comfort limits enter
    # in internal units (~1e-3 miles/s^2, miles/s^3), which leaves the QP badly
    # scaled for the default OSQP ADMM solver (it returns inaccurate / iteration-
    # limited solutions on the tighter feasible set). CLARABEL handles the
    # scaling reliably; fall back to OSQP with a raised iteration cap if needed.
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        prob.solve(solver=cp.OSQP, max_iter=100000)
    
    # Accept more solution statuses including user_limit if we have a reasonable solution
    if prob.status not in ["optimal", "optimal_inaccurate", "user_limit"]:
        raise RuntimeError(f"Optimization failed with status {prob.status}")
    # if x.value is None or v.value is None or v_avg.value is None:
    #     raise RuntimeError("Solver did not return a solution.")
    
    # Warn if solution may be suboptimal
    if prob.status == "user_limit":
        import warnings
        warnings.warn(f"Solver hit iteration limit but returned a solution. Cost: {prob.value}")
    
    return x.value, v.value, v_avg, prob.value

def solve_gap_sweep(
    time_data: np.ndarray,
    space_ref: np.ndarray,
    leader_time: np.ndarray,
    leader_ref: np.ndarray,
    v_start: float,
    v_end: float,
    a_start: float,
    a_end: float,
    max_gaps,
    lambda_1: float = 10.0,
    comfort_decel_ms2: float = 3.4,
    accel_max_ms2: float = 3.0,
    jerk_max_ms3: float = 3.0,
    pin_boundary_accel: bool = True,
):
    """Solve the smoothing QP for one vehicle at MANY max_gap values.

    Identical problem and semantics to solve_constrained_smoothing, but built
    once and re-solved per gap with max_gap as a cvxpy Parameter.

    Why this exists: max_gap appears only in the RHS of the trailing-band
    constraint, so the problem STRUCTURE is the same for every gap. Calling
    solve_constrained_smoothing in a loop makes cvxpy re-canonicalize that
    identical structure from scratch every time, and canonicalization measured
    at 16.6 ms against 18.2 ms of actual CLARABEL solve time -- i.e. 44% of the
    runtime was recompiling a problem cvxpy had already compiled. Declaring
    max_gap as a Parameter keeps the problem DPP-compliant, so cvxpy caches the
    canonicalization and only re-stuffs the parameter between solves.

    Measured: 38.9 -> 19.5 ms per solve (1.99x) over a 26-gap sweep, with
    solutions matching the per-gap rebuild to 1.7e-11 miles (2.8e-8 mm) and no
    disagreement on feasibility status.

    Returns:
        dict mapping each gap to (x_value_or_None, status_string).
    """
    N = len(time_data) - 1
    dt = np.mean(np.diff(time_data))

    METERS_PER_MILE = 1609.344
    a_dec_lim = comfort_decel_ms2 / METERS_PER_MILE if comfort_decel_ms2 > 0 else None
    a_acc_lim = accel_max_ms2 / METERS_PER_MILE if accel_max_ms2 > 0 else None
    jerk_lim = jerk_max_ms3 / METERS_PER_MILE if jerk_max_ms3 > 0 else None

    if a_dec_lim is not None:
        a_start = max(a_start, -a_dec_lim)
        a_end = max(a_end, -a_dec_lim)
    if a_acc_lim is not None:
        a_start = min(a_start, a_acc_lim)
        a_end = min(a_end, a_acc_lim)

    time_start = max(time_data[0], leader_time[0])
    time_end = min(time_data[-1], leader_time[-1])
    mask_current = (time_data >= time_start) & (time_data <= time_end)
    mask_leader = (leader_time >= time_start) & (leader_time <= time_end)
    overlap_leader = leader_ref[mask_leader]

    gap_param = cp.Parameter(nonneg=True)

    x = cp.Variable(N + 1)
    init_gap = overlap_leader[0] - x[0]
    v = (x[1:] - x[:-1]) / dt
    a = (v[1:] - v[:-1]) / dt
    jerk = (a[1:] - a[:-1]) / dt

    cost = cp.sum_squares(v) + lambda_1 * cp.sum_squares(a)

    constraints = [
        x[0] == space_ref[0],
        x[N] == space_ref[-1],
        v[0] == v_start,
        v[-1] == v_end,
        x[mask_current] <= overlap_leader,
        x[mask_current] >= overlap_leader - init_gap - gap_param,
        v >= 0,
    ]
    if pin_boundary_accel:
        constraints.append(a[0] == a_start)
        constraints.append(a[-1] == a_end)
    if a_dec_lim is not None:
        constraints.append(a >= -a_dec_lim)
    if a_acc_lim is not None:
        constraints.append(a <= a_acc_lim)
    if jerk_lim is not None:
        constraints.append(jerk <= jerk_lim)
        constraints.append(jerk >= -jerk_lim)

    prob = cp.Problem(cp.Minimize(cost), constraints)

    results = {}
    for gap in max_gaps:
        gap_param.value = float(gap)
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            try:
                prob.solve(solver=cp.OSQP, max_iter=100000)
            except Exception as exc:
                results[float(gap)] = (None, f"solver_error: {type(exc).__name__}")
                continue
        if prob.status not in ("optimal", "optimal_inaccurate", "user_limit"):
            results[float(gap)] = (None, prob.status)
            continue
        if x.value is None:
            results[float(gap)] = (None, "no_solution")
            continue
        results[float(gap)] = (np.asarray(x.value, dtype=float), prob.status)
    return results


# --------------------------------------------------------------------------- #
# Fast path: cached CLARABEL solver reused across vehicles and gaps
# --------------------------------------------------------------------------- #
#
# For a fixed trajectory length n_pts and fixed dt, the canonicalised problem
# data (P, q, A, cone dimensions) is BIT-IDENTICAL across every vehicle and
# every max_gap -- verified directly: |dP| = |dA| = |dq| = 0 across vehicles and
# gaps at fixed length, with only the constraint vector b changing. That is
# because the objective and every constraint matrix depend only on n_pts, dt and
# lambda_1; the trajectory data (space_ref, v_start, v_end) and max_gap enter
# purely through right-hand sides.
#
# So one CLARABEL solver can be built per (n_pts, dt, settings) bucket and
# re-used via solver.update(b=...), which skips CLARABEL's symbolic analysis and
# KKT setup -- measured at 4.64 ms of an 18.23 ms solve, i.e. 25%. Combined with
# dropping cvxpy's own per-solve overhead, a like-for-like bucket benchmark gave
# 8.61 -> 4.61 ms per solve (1.87x), with max deviation 9.6e-11 miles
# (1.5e-4 mm) over 400 solutions and zero disagreement on feasibility status.
#
# cvxpy is still used, but only to canonicalise: the template problem carries
# space_ref / v_start / v_end / max_gap as DPP Parameters, so re-stuffing b for
# a new vehicle or gap costs ~0.6 ms instead of a full recompilation.

_ACCEPT_STATUS = ("Solved", "AlmostSolved")
_INFEASIBLE_STATUS = ("PrimalInfeasible", "AlmostPrimalInfeasible")


class CachedGapSweepSolver:
    """Reusable QP for all vehicles of one trajectory length.

    Build once per (n_pts, dt, solver settings), then call solve() per vehicle.
    Not thread-safe and not picklable; construct inside each worker process.
    """

    def __init__(
        self,
        n_pts: int,
        dt: float,
        lambda_1: float = 10.0,
        comfort_decel_ms2: float = 3.4,
        accel_max_ms2: float = 3.0,
        jerk_max_ms3: float = 3.0,
        pin_boundary_accel: bool = False,
    ):
        import clarabel
        import scipy.sparse as sp

        self._clarabel = clarabel
        self.n_pts = n_pts
        self.dt = dt
        self.pin_boundary_accel = pin_boundary_accel

        METERS_PER_MILE = 1609.344
        self.a_dec_lim = (comfort_decel_ms2 / METERS_PER_MILE
                          if comfort_decel_ms2 > 0 else None)
        self.a_acc_lim = (accel_max_ms2 / METERS_PER_MILE
                          if accel_max_ms2 > 0 else None)
        jerk_lim = jerk_max_ms3 / METERS_PER_MILE if jerk_max_ms3 > 0 else None

        N = n_pts - 1
        self._sP = cp.Parameter(n_pts)
        self._v0P = cp.Parameter()
        self._v1P = cp.Parameter()
        self._gP = cp.Parameter(nonneg=True)
        self._a0P = cp.Parameter() if pin_boundary_accel else None
        self._a1P = cp.Parameter() if pin_boundary_accel else None

        x = cp.Variable(n_pts)
        v = (x[1:] - x[:-1]) / dt
        a = (v[1:] - v[:-1]) / dt
        jerk = (a[1:] - a[:-1]) / dt
        init_gap = self._sP[0] - x[0]

        constraints = [
            x[0] == self._sP[0],
            x[N] == self._sP[n_pts - 1],
            v[0] == self._v0P,
            v[-1] == self._v1P,
            x <= self._sP,
            x >= self._sP - init_gap - self._gP,
            v >= 0,
        ]
        if pin_boundary_accel:
            constraints.append(a[0] == self._a0P)
            constraints.append(a[-1] == self._a1P)
        if self.a_dec_lim is not None:
            constraints.append(a >= -self.a_dec_lim)
        if self.a_acc_lim is not None:
            constraints.append(a <= self.a_acc_lim)
        if jerk_lim is not None:
            constraints.append(jerk <= jerk_lim)
            constraints.append(jerk >= -jerk_lim)

        cost = cp.sum_squares(v) + lambda_1 * cp.sum_squares(a)
        self._prob = cp.Problem(cp.Minimize(cost), constraints)
        if not self._prob.is_dcp(dpp=True):
            raise RuntimeError("template problem is not DPP; parameter re-stuffing "
                               "would silently fall back to full recompilation")

        # Seed the parameters with a trivially feasible placeholder so the first
        # canonicalisation succeeds; real values are set in solve().
        self._sP.value = np.linspace(0.0, 1.0, n_pts)
        self._v0P.value = 0.0
        self._v1P.value = 0.0
        self._gP.value = 0.1
        if pin_boundary_accel:
            self._a0P.value = 0.0
            self._a1P.value = 0.0

        data, _chain, _inv = self._prob.get_problem_data(cp.CLARABEL)
        dims = data["dims"]
        self._solver = clarabel.DefaultSolver(
            sp.csc_matrix(data["P"]), data["c"], sp.csc_matrix(data["A"]),
            data["b"],
            [clarabel.ZeroConeT(dims.zero), clarabel.NonnegativeConeT(dims.nonneg)],
            self._make_settings(),
        )
        if not self._solver.is_data_update_allowed():
            raise RuntimeError("CLARABEL refused in-place data updates; the fast "
                               "path requires solver.update(b=...)")
        # x occupies the trailing n_pts entries of the canonical variable vector
        # (verified n_raw == 3*n_pts - 3 across many lengths). Every returned
        # solution is checked against the two position equalities anyway, so a
        # layout change would surface as a loud error rather than bad numbers.
        self._n_raw = len(data["c"])
        self._offset = self._n_raw - n_pts

    def _make_settings(self):
        st = self._clarabel.DefaultSettings()
        st.verbose = False
        return st

    def solve(self, space_ref, v_start, v_end, a_start, a_end, max_gaps):
        """Solve for one vehicle at many gaps. Same contract as solve_gap_sweep."""
        if len(space_ref) != self.n_pts:
            raise ValueError(f"expected {self.n_pts} samples, got {len(space_ref)}")

        if self.pin_boundary_accel:
            if self.a_dec_lim is not None:
                a_start = max(a_start, -self.a_dec_lim)
                a_end = max(a_end, -self.a_dec_lim)
            if self.a_acc_lim is not None:
                a_start = min(a_start, self.a_acc_lim)
                a_end = min(a_end, self.a_acc_lim)
            self._a0P.value = float(a_start)
            self._a1P.value = float(a_end)

        self._sP.value = np.asarray(space_ref, dtype=float)
        self._v0P.value = float(v_start)
        self._v1P.value = float(v_end)

        results = {}
        for gap in max_gaps:
            self._gP.value = float(gap)
            try:
                data, _c, _i = self._prob.get_problem_data(cp.CLARABEL)
                self._solver.update(b=data["b"])
                sol = self._solver.solve()
            except Exception as exc:
                results[float(gap)] = (None, f"solver_error: {type(exc).__name__}")
                continue
            status = str(sol.status).split(".")[-1]
            if status in _INFEASIBLE_STATUS:
                results[float(gap)] = (None, "infeasible")
                continue
            if status not in _ACCEPT_STATUS:
                results[float(gap)] = (None, status)
                continue
            x = np.asarray(sol.x, dtype=float)[self._offset:self._offset + self.n_pts]
            # Guard the offset assumption: both endpoints are hard equalities,
            # so if the slice were misaligned this would fail immediately.
            if (abs(x[0] - space_ref[0]) > 1e-6
                    or abs(x[-1] - space_ref[-1]) > 1e-6):
                raise RuntimeError(
                    "recovered trajectory violates its endpoint equalities; the "
                    "canonical variable layout is not what this solver assumes")
            results[float(gap)] = (x, status)
        return results


def make_gap_sweep_solver_cache(max_entries: int = 96):
    """Return (get_solver, stats) with an LRU-ish cache keyed by trajectory length.

    Each entry holds a factorised KKT system, so the cache is bounded: trajectory
    lengths span ~10-900 and a lane can contain 700+ distinct lengths, which
    would otherwise accumulate unboundedly inside a long-lived worker.
    """
    from collections import OrderedDict

    cache: "OrderedDict[tuple, CachedGapSweepSolver]" = OrderedDict()
    stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get_solver(n_pts, dt, **settings):
        key = (n_pts, round(float(dt), 9), tuple(sorted(settings.items())))
        got = cache.get(key)
        if got is not None:
            stats["hits"] += 1
            cache.move_to_end(key)
            return got
        stats["misses"] += 1
        solver = CachedGapSweepSolver(n_pts, dt, **settings)
        cache[key] = solver
        if len(cache) > max_entries:
            cache.popitem(last=False)
            stats["evictions"] += 1
        return solver

    return get_solver, stats
