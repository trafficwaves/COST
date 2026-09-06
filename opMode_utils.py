import argparse
import os

import numpy as np
import pandas as pd


def opMode(speed, vsp, accelerations=[0,0,0]):
    """
    Classifies the operating mode based on speed, acceleration over three time steps, and VSP values.
    Returns the corresponding mode ID.

    Follows the MOVES running-exhaust operating-mode definitions
    (EPA-420-R-23-028, "Exhaust Emission Rates for Light-Duty Onroad Vehicles in
    MOVES4", Table 2-5).

    The order of the tests matters, and follows the MOVES *software* rather than
    the report's prose, because the two disagree in one place. In
    LinkOperatingModeDistributionGenerator.calculateOpModeFractionsCore(), MOVES
    emits this CASE statement:

        when (speed = 0) then 501   -- remapped to 1 for every non-PM polProcessID
        when (speed < 1) then 1     -- "force idle if speed < 1"
        when (At0 <= -2 or (At0 < -1 and At1 < -1 and At2 < -1)) then 0
        ... speed/VSP bins ...

    so **idle is tested before braking**. The report states that "the
    deceleration/braking categorization takes precedence over other definitions",
    which reads as though braking should win everywhere; read alongside the code,
    that precedence applies to the speed/VSP cruise bins, not to idle. Testing
    braking first -- as this function originally did -- misclassifies the last
    second or two of a hard stop as opMode 0 rather than opMode 1. The numerical
    effect is small (the CO2 rates differ by ~8%, and on the MiTra virtual
    trajectories the benefit moved by 0.01 pp), but it was a real divergence.

    Parameters:
    - speed: Current vehicle speed in mph (MOVES opMode speed bins: 1/25/50 mph).
    - vsp: Vehicle Specific Power (VSP) value in kW/tonne.
    - accelerations: List of 3 acceleration values [a_t, a_(t-1), a_(t-2)] in m/s².
      MOVES defines these as *backward* differences of speed, At_k = v[t-k] - v[t-k-1].

    Returns:
    - mode ID based on MOVES model logic.
    """

    # Idling, tested first (see the ordering note above). MOVES defines idle as
    # -1.0 <= v < 1.0; speeds below -1 mph fall through to the remaining tests.
    if -1.0 <= speed < 1.0:
        return 1  # Idling

    # Convert acceleration values from m/s² to mph/sec (1 m/s² = 2.237 mph/sec)
    accel_mph = [a * 2.237 for a in accelerations]

    # Braking condition: Check if any of the two conditions are satisfied
    if (
        accel_mph[0] <= -2.0 or  # Condition 1: Current acceleration <= -2.0
        all(acc < -1.0 for acc in accel_mph)  # Condition 2: All three accelerations < -1.0
    ):
        return 0  # Braking

    # Define operating modes based on VSP and speed ranges
    if speed < 25:
        if vsp < 0:
            return 11  # Low Speed Coasting
        elif 0 <= vsp < 3:
            return 12  # Cruise / Acceleration (0 ≤ VSP < 3)
        elif 3 <= vsp < 6:
            return 13  # Cruise / Acceleration (3 ≤ VSP < 6)
        elif 6 <= vsp < 9:
            return 14  # Cruise / Acceleration (6 ≤ VSP < 9)
        elif 9 <= vsp < 12:
            return 15  # Cruise / Acceleration (9 ≤ VSP < 12)
        else:
            return 16  # Cruise / Acceleration (12 ≤ VSP)
    elif speed < 50:
        if vsp < 0:
            return 21  # Moderate Speed Coasting
        elif 0 <= vsp < 3:
            return 22  # Cruise / Acceleration (0 ≤ VSP < 3)
        elif 3 <= vsp < 6:
            return 23  # Cruise / Acceleration (3 ≤ VSP < 6)
        elif 6 <= vsp < 9:
            return 24  # Cruise / Acceleration (6 ≤ VSP < 9)
        elif 9 <= vsp < 12:
            return 25  # Cruise / Acceleration (9 ≤ VSP < 12)
        elif 12 <= vsp < 18:
            return 27  # Cruise / Acceleration (12 ≤ VSP < 18)
        elif 18 <= vsp < 24:
            return 28  # Cruise / Acceleration (18 ≤ VSP < 24)
        elif 24 <= vsp < 30:
            return 29  # Cruise / Acceleration (24 ≤ VSP < 30)
        else:
            return 30  # Cruise / Acceleration (30 ≤ VSP)
    else:
        if vsp < 6:
            return 33  # Cruise / Acceleration (VSP < 6)
        elif 6 <= vsp < 12:
            return 35  # Cruise / Acceleration (6 ≤ VSP < 12)
        elif 12 <= vsp < 18:
            return 37  # Cruise / Acceleration (12 ≤ VSP < 18)
        elif 18 <= vsp < 24:
            return 38  # Cruise / Acceleration (18 ≤ VSP < 24)
        elif 24 <= vsp < 30:
            return 39  # Cruise / Acceleration (24 ≤ VSP < 30)
        else:
            return 40  # Cruise / Acceleration (30 ≤ VSP)


def opMode_PM(speed, vsp, accelerations=[0,0,0]):
    """
    Classifies the operating mode for PM (pollutant/process 11609).
    This is a special version that assigns opModeID 501 for zero speed seconds.
    
    Operating mode 501 is a special case for zero speed seconds. When used for pollutant/process
    11609 (PM), it should stay operating mode 501. When used for any other pollutant/process,
    it should be converted to opModeID 1 and summed into its fraction.

    Parameters:
    - speed: Current vehicle speed in mph (MOVES opMode speed bins: 1/25/50 mph).
    - vsp: Vehicle Specific Power (VSP) value in kW/tonne.
    - accelerations: List of 3 acceleration values [a_t, a_(t-1), a_(t-2)] in m/s².

    Returns:
    - mode ID based on MOVES model logic (with 501 for zero speed).
    """
    
    # Special case for PM (polProcessID 11609): Zero speed gets opModeID 501
    # OpModeID=IF(speed=0 and polProcessID=11609,501,if(speed<1.0,1,opModeID))
    if speed == 0.0:
        return 501

    # Idle before braking, matching the MOVES CASE statement -- see opMode().
    if -1.0 <= speed < 1.0:
        return 1  # Idling

    # Convert acceleration values from m/s² to mph/sec (1 m/s² = 2.237 mph/sec)
    accel_mph = [a * 2.237 for a in accelerations]

    # Braking condition: Check if any of the two conditions are satisfied
    if (
        accel_mph[0] <= -2.0 or  # Condition 1: Current acceleration <= -2.0
        all(acc < -1.0 for acc in accel_mph)  # Condition 2: All three accelerations < -1.0
    ):
        return 0  # Braking

    # Define operating modes based on VSP and speed ranges
    if speed < 25:
        if vsp < 0:
            return 11  # Low Speed Coasting
        elif 0 <= vsp < 3:
            return 12  # Cruise / Acceleration (0 ≤ VSP < 3)
        elif 3 <= vsp < 6:
            return 13  # Cruise / Acceleration (3 ≤ VSP < 6)
        elif 6 <= vsp < 9:
            return 14  # Cruise / Acceleration (6 ≤ VSP < 9)
        elif 9 <= vsp < 12:
            return 15  # Cruise / Acceleration (9 ≤ VSP < 12)
        else:
            return 16  # Cruise / Acceleration (12 ≤ VSP)
    elif speed < 50:
        if vsp < 0:
            return 21  # Moderate Speed Coasting
        elif 0 <= vsp < 3:
            return 22  # Cruise / Acceleration (0 ≤ VSP < 3)
        elif 3 <= vsp < 6:
            return 23  # Cruise / Acceleration (3 ≤ VSP < 6)
        elif 6 <= vsp < 9:
            return 24  # Cruise / Acceleration (6 ≤ VSP < 9)
        elif 9 <= vsp < 12:
            return 25  # Cruise / Acceleration (9 ≤ VSP < 12)
        elif 12 <= vsp < 18:
            return 27  # Cruise / Acceleration (12 ≤ VSP < 18)
        elif 18 <= vsp < 24:
            return 28  # Cruise / Acceleration (18 ≤ VSP < 24)
        elif 24 <= vsp < 30:
            return 29  # Cruise / Acceleration (24 ≤ VSP < 30)
        else:
            return 30  # Cruise / Acceleration (30 ≤ VSP)
    else:
        if vsp < 6:
            return 33  # Cruise / Acceleration (VSP < 6)
        elif 6 <= vsp < 12:
            return 35  # Cruise / Acceleration (6 ≤ VSP < 12)
        elif 12 <= vsp < 18:
            return 37  # Cruise / Acceleration (12 ≤ VSP < 18)
        elif 18 <= vsp < 24:
            return 38  # Cruise / Acceleration (18 ≤ VSP < 24)
        elif 24 <= vsp < 30:
            return 39  # Cruise / Acceleration (24 ≤ VSP < 30)
        else:
            return 40  # Cruise / Acceleration (30 ≤ VSP)

# --------------------------------------------------------------------------- #
# Vectorised operating-mode classification
# --------------------------------------------------------------------------- #
#
# opMode()/opMode_PM() above are the readable reference implementations and stay
# the source of truth. They are scalar, and the pipeline calls them once per
# sample -- 49.2 million times per lane -- where the Python call and branch
# overhead measured 1.75 us/sample, about 86 core-seconds per lane and ~5% of
# total pipeline runtime.
#
# The classification is a pure function of (speed, vsp, a_t, a_t-1, a_t-2) built
# from comparisons and half-open VSP bins, so it maps exactly onto searchsorted
# over the bin edges plus two boolean overrides. No approximation is involved:
# opMode_array is verified to reproduce the scalar function bit-for-bit,
# including at every bin boundary and on non-finite input.
#
# Precedence, which is what the boolean override ORDER encodes (see the ordering
# note in opMode's docstring): speed/VSP bins are computed for everything first,
# then braking overwrites, then idle overwrites braking, then -- for PM only --
# zero speed overwrites idle. Applying them in that sequence reproduces the
# scalar function's early-return chain read in reverse.

_LOW_EDGES = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
_LOW_CODES = np.array([11, 12, 13, 14, 15, 16], dtype=np.int16)
_MID_EDGES = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 30.0])
_MID_CODES = np.array([21, 22, 23, 24, 25, 27, 28, 29, 30], dtype=np.int16)
_HIGH_EDGES = np.array([6.0, 12.0, 18.0, 24.0, 30.0])
_HIGH_CODES = np.array([33, 35, 37, 38, 39, 40], dtype=np.int16)

MS2_TO_MPH_S = 2.237


def opMode_array(speed, vsp, a_t, a_t1, a_t2, pm=False):
    """Vectorised equivalent of opMode() (or opMode_PM() when pm=True).

    Args:
        speed: vehicle speed, mph.
        vsp:   vehicle specific power, kW/tonne.
        a_t, a_t1, a_t2: acceleration at t, t-1, t-2 in m/s^2.
        pm:    if True, apply the PM rule that speed == 0 maps to opMode 501.

    Returns:
        int16 array of opModeIDs, same shape as speed.
    """
    speed = np.asarray(speed, dtype=float)
    vsp = np.asarray(vsp, dtype=float)
    a0 = np.asarray(a_t, dtype=float) * MS2_TO_MPH_S
    a1 = np.asarray(a_t1, dtype=float) * MS2_TO_MPH_S
    a2 = np.asarray(a_t2, dtype=float) * MS2_TO_MPH_S

    out = np.empty(speed.shape, dtype=np.int16)

    # Speed tiers. NaN fails every comparison, so it lands in `high`, which is
    # exactly where the scalar if/elif/else chain sends it.
    low = speed < 25.0
    mid = (~low) & (speed < 50.0)
    high = ~(low | mid)

    # side='right' makes searchsorted return the count of edges <= v, which is
    # precisely the half-open bin index: v < 0 -> 0, v == 0 -> 1, v == 3 -> 2 ...
    if low.any():
        out[low] = _LOW_CODES[np.searchsorted(_LOW_EDGES, vsp[low], side="right")]
    if mid.any():
        out[mid] = _MID_CODES[np.searchsorted(_MID_EDGES, vsp[mid], side="right")]
    if high.any():
        out[high] = _HIGH_CODES[np.searchsorted(_HIGH_EDGES, vsp[high], side="right")]

    # Braking overrides the cruise bins.
    np.copyto(out, 0, where=(a0 <= -2.0) | ((a0 < -1.0) & (a1 < -1.0) & (a2 < -1.0)))
    # Idle overrides braking (MOVES tests idle first; see opMode's docstring).
    np.copyto(out, 1, where=(speed >= -1.0) & (speed < 1.0))
    if pm:
        np.copyto(out, 501, where=(speed == 0.0))
    return out


# --------------------------------------------------------------------------- #
# Sub-bin resolution of the MOVES VSP bins
# --------------------------------------------------------------------------- #
#
# THE PROBLEM. MOVES prices a second by the AVERAGE rate of the operating-mode
# bin it falls in. For light-duty cruise below 25 mph that bin is opMode 12,
# spanning 0 <= VSP < 3 kW/tonne -- 0 to 4.4 kW at the wheels for a source type
# 21 / reg class 20 car. A vehicle holding a CONSTANT speed has a = 0, so its VSP
# is road load alone: 0.54 kW/tonne (0.80 kW) at 10.2 mph. That sits on the FLOOR
# of the bin while being charged the bin's average, and the bin's average reflects
# whatever populates it -- measured on the EPA certification cycles, opMode 12 is
# populated at a mean VSP of ~1.5 kW/tonne, roughly three times that operating
# point. So steady low-speed cruising is systematically overcharged, and MOVES
# gives the SAME rate for a steady 5 mph and a steady 20 mph.
#
# THE FIX, using nothing but published MOVES rates. A bin's rate is
# R_b = E[R(VSP) | VSP in b] under the activity the rates were built from, i.e. it
# is the value of the rate function at the bin's activity-weighted CENTROID, not
# at every point of the bin. Measure each centroid, attach the published rate
# there, and interpolate between anchors with a shape-preserving spline. Rate
# VALUES are untouched, as are idle (opMode 1), braking (opMode 0), the braking
# test, the speed-class boundaries and the VSP equation. Only the assumption that
# a bin's average applies uniformly across the bin is dropped.
#
# In MOVES' own project-level interface this is not a new model: MOVES accepts an
# operating-mode DISTRIBUTION per link, so assigning a second fractionally between
# adjacent modes such that the activity-weighted mean VSP is preserved is a legal
# input, and interpolating between bin centroids is that fractional assignment.
#
# VALIDATION (R1Q4/moves_subbin.py). Every EPA certification cycle stays within
# 1.1% of MOVES as published, so this is not a recalibration -- it changes the
# answer only where a trajectory sits away from its bin centroids. A steady
# 10.2 mph cruise moves from 4002 to ~3670 g CO2/h, inside the [2898, 4002]
# bracket that monotonicity alone guarantees.
#
# MONOTONICITY. CO2, energy and NOx rates increase strictly with VSP in every
# speed class, so `bound_rate_array` brackets those from published rates and the
# ordering alone. HC, CO, EC and nonEC are NOT monotone (the low class has HC
# 0.217 -> 0.166 -> 0.314 across opModes 11, 12, 13), so no bound exists for them;
# the interpolation still applies, since PCHIP is shape-preserving rather than
# monotone-forcing, but it carries no ordering guarantee. See is_rate_monotone().
#
# REFERENCE ACTIVITY. MOVES' rates derive from in-use and dynamometer data, not
# from the certification cycles specifically, so those cycles are a PROXY for the
# activity behind each bin. The centroid table records which pooling was used and
# R1Q4/moves_subbin.py reports the sensitivity (all cycles pooled vs UDDS alone).
#
# PER VEHICLE CLASS. Everything above is a statement about a bin of a PARTICULAR
# vehicle: the centroid is the activity-weighted mean VSP inside the bin, and VSP
# runs the reference activity through that vehicle's road-load polynomial. Two
# classes driving the identical drive schedule therefore populate the same bin at
# different mean VSP -- a Class 8 tractor's inertia term is about twice a car's
# per tonne while its rolling and drag terms per tonne are lower, so its
# centroids move, and not all in the same direction. The published rate anchored
# at a car's centroid is simply the wrong anchor for a truck.
#
# So the centroid table is keyed by (source type, reg class, model year, pooling),
# `make_subbin_centroids.py` measures one set per class, and every function below
# takes `source_type_id` / `reg_class_id`. The defaults are the passenger car,
# which is what the single-class table on disk always meant, so existing callers
# and existing published numbers are unaffected.

_HERE = os.path.dirname(os.path.abspath(__file__))

# MOVES speed classes and the VSP-binned cruise/coast modes in each, VSP order.
SPEED_CLASSES = {
    "low": {"range": (1.0, 25.0), "modes": [11, 12, 13, 14, 15, 16]},
    "mid": {"range": (25.0, 50.0), "modes": [21, 22, 23, 24, 25, 27, 28, 29, 30]},
    "high": {"range": (50.0, np.inf), "modes": [33, 35, 37, 38, 39, 40]},
}
# Half-open VSP bin edges (kW/tonne) of every VSP-binned mode. Idle (1), braking
# (0) and the PM zero-speed mode (501) are absent by design: they are not VSP bins.
VSP_BINS = {
    11: (-np.inf, 0.0), 12: (0.0, 3.0), 13: (3.0, 6.0), 14: (6.0, 9.0),
    15: (9.0, 12.0), 16: (12.0, np.inf),
    21: (-np.inf, 0.0), 22: (0.0, 3.0), 23: (3.0, 6.0), 24: (6.0, 9.0),
    25: (9.0, 12.0), 27: (12.0, 18.0), 28: (18.0, 24.0), 29: (24.0, 30.0),
    30: (30.0, np.inf),
    33: (-np.inf, 6.0), 35: (6.0, 12.0), 37: (12.0, 18.0), 38: (18.0, 24.0),
    39: (24.0, 30.0), 40: (30.0, np.inf),
}
MIN_SECONDS_FOR_CENTROID = 5

# Where the measured centroids live, in search order. The tables are small, plain
# CSV and auditable; they are the whole calibration.
#
# CENTROID_PATHS holds the original single-class table, which carries no class
# columns and has always meant the passenger car. CENTROID_BY_CLASS_PATHS holds
# the class-keyed table written by make_subbin_centroids.py. For the passenger car
# the single-class table is searched FIRST, so previously published light-duty
# numbers keep reproducing byte-for-byte even after the class-keyed table appears;
# every other class can only come from the class-keyed one.
CENTROID_PATHS = (
    os.path.join(_HERE, "moves_subbin_centroids.csv"),
    os.path.join(_HERE, "R1Q4", "moves_subbin_centroids.csv"),
)
CENTROID_BY_CLASS_PATHS = (
    os.path.join(_HERE, "moves_subbin_centroids_by_class.csv"),
    os.path.join(_HERE, "R1Q4", "moves_subbin_centroids_by_class.csv"),
)
DEFAULT_RATE_TABLE = os.path.join(_HERE, "emissionRateMicro2026.csv")

# The class the single-class table describes, and so the default everywhere here.
DEFAULT_SOURCE_TYPE_ID = 21
DEFAULT_REG_CLASS_ID = 20

# One row per VSP-binned mode is what an interpolation needs.
N_BINNED_MODES = sum(len(s["modes"]) for s in SPEED_CLASSES.values())

# Default ON. Override globally with the environment variable MOVES_SUBBIN=0, per
# process with set_subbin(), or per run with the --no-subbin CLI flag.
_SUBBIN = os.environ.get("MOVES_SUBBIN", "1").strip().lower() not in ("0", "false", "no", "off")


def subbin_enabled() -> bool:
    """Whether sub-bin resolution is currently active (default True)."""
    return _SUBBIN


def set_subbin(flag: bool) -> bool:
    """Set the process-wide default. Returns the previous value."""
    global _SUBBIN
    prev, _SUBBIN = _SUBBIN, bool(flag)
    return prev


def add_subbin_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add `--subbin / --no-subbin` (default: --subbin) to a CLI.

    Call apply_subbin_arg(args) after parsing so the process-wide default follows
    the flag; that way any helper that does not take an explicit `subbin=`
    argument still honours it.
    """
    parser.add_argument(
        "--subbin", action=argparse.BooleanOptionalAction, default=subbin_enabled(),
        help="resolve VSP within each MOVES operating-mode bin by anchoring each "
             "published rate at that bin's measured VSP centroid instead of "
             "applying the bin average across the whole bin. Corrects the "
             "overcharging of steady low-speed cruise. Use --no-subbin for MOVES "
             "exactly as published. (default: --subbin)")
    return parser


def apply_subbin_arg(args) -> bool:
    """Push a parsed --subbin/--no-subbin onto the process-wide default."""
    flag = bool(getattr(args, "subbin", subbin_enabled()))
    set_subbin(flag)
    return flag


# --------------------------------------------------------------------------- #
# centroids
# --------------------------------------------------------------------------- #
def fit_bin_centroids(activity: pd.DataFrame) -> pd.DataFrame:
    """Activity-weighted mean VSP of every VSP-binned operating mode.

    `activity` needs columns `vsp` (kW/tonne) and `op_mode_ID`, one row per
    second of reference driving. A centroid depends only on the activity, never
    on the pollutant, so one table serves every rate column.
    """
    rows = []
    for cls, spec in SPEED_CLASSES.items():
        for m in spec["modes"]:
            sub = activity[activity["op_mode_ID"] == m]
            lo, hi = VSP_BINS[m]
            if len(sub) >= MIN_SECONDS_FOR_CENTROID:
                centroid, src = float(sub["vsp"].mean()), f"measured ({len(sub)} s)"
            else:
                # No usable activity: fall back to the bin midpoint, or for an
                # open-ended bin to one half-width past its finite edge.
                if np.isinf(lo):
                    centroid = hi - 1.5
                elif np.isinf(hi):
                    centroid = lo + 3.0
                else:
                    centroid = 0.5 * (lo + hi)
                src = f"fallback ({len(sub)} s)"
            rows.append({"speed_class": cls, "op_mode_ID": m, "vsp_lo": lo,
                         "vsp_hi": hi, "centroid_vsp": centroid, "source": src})
    return pd.DataFrame(rows)


def load_bin_centroids(path: str | None = None, pooled: bool = True,
                       source_type_id: int | None = None,
                       reg_class_id: int | None = None) -> pd.DataFrame:
    """Measured bin centroids for one vehicle class.

    The `rate` column of the stored table, if present, is IGNORED: rates always
    come from the live rate table so a different model year, calendar year or
    reg class needs no recalibration of the centroids.

    A table with no `source_type_id` column is the original single-class one and
    describes the passenger car; it is used only when the passenger car is what
    was asked for. Asking for any other class and getting a class-less table back
    would silently anchor a truck's rates at a car's centroids, which is the
    error this keying exists to prevent, so that case falls through to the
    class-keyed table or raises.
    """
    st = DEFAULT_SOURCE_TYPE_ID if source_type_id is None else int(source_type_id)
    is_default = (st == DEFAULT_SOURCE_TYPE_ID
                  and reg_class_id in (None, DEFAULT_REG_CLASS_ID))
    if path:
        candidates: tuple = (path,)
    elif is_default:
        candidates = CENTROID_PATHS + CENTROID_BY_CLASS_PATHS
    else:
        candidates = CENTROID_BY_CLASS_PATHS + CENTROID_PATHS

    for p in candidates:
        if not (p and os.path.exists(p)):
            continue
        cent = pd.read_csv(p)
        if "pooled" in cent:
            cent = cent[cent["pooled"] == pooled]
        if "source_type_id" in cent:
            cent = cent[cent["source_type_id"] == st]
            if reg_class_id is not None and "reg_class_id" in cent:
                cent = cent[cent["reg_class_id"] == int(reg_class_id)]
            # More than one reg class (or model year) for this source type means
            # the caller has to say which; picking one would be a guess about
            # vehicle mass, and mass is exactly what a centroid measures.
            for col in ("reg_class_id", "model_year"):
                if col in cent and cent[col].nunique() > 1:
                    raise ValueError(
                        f"{os.path.basename(p)} holds {cent[col].nunique()} "
                        f"distinct {col} values for source type {st} "
                        f"({sorted(cent[col].unique())}); pass {col} explicitly")
        elif not is_default:
            continue
        cent = cent.drop(columns=[c for c in ("rate", "pooled") if c in cent])
        if cent.empty:
            continue
        if len(cent) != N_BINNED_MODES:
            raise ValueError(
                f"{os.path.basename(p)}: expected {N_BINNED_MODES} centroid rows "
                f"for source type {st} (one per VSP-binned operating mode), found "
                f"{len(cent)}. The table is malformed or was written for a "
                f"different mode set.")
        return cent.copy()

    where = ", ".join(str(p) for p in candidates)
    if is_default:
        raise FileNotFoundError(
            f"no MOVES bin-centroid table found (looked in {where}). Run "
            "make_subbin_centroids.py with the EPA certification cycles present "
            "to measure it, or pass --no-subbin to use MOVES exactly as "
            "published.")
    raise FileNotFoundError(
        f"no bin centroids for source type {st}"
        + (f" / reg class {reg_class_id}" if reg_class_id is not None else "")
        + f" (looked in {where}). Run `python make_subbin_centroids.py` to "
          "measure every registered vehicle class; the light-duty table is not a "
          "substitute, since a bin's centroid depends on the vehicle's road load. "
          "Or pass --no-subbin to use MOVES exactly as published.")


def _rate_series(rates) -> pd.DataFrame:
    if rates is None:
        rates = DEFAULT_RATE_TABLE
    if isinstance(rates, (str, os.PathLike)):
        return pd.read_csv(rates).set_index("opModeID")
    if rates.index.name != "opModeID" and "opModeID" in rates:
        return rates.set_index("opModeID")
    return rates


_CURVE_CACHE: dict = {}


def rate_curves(pollutant: str = "CO2", rates=None, pooled: bool = True,
                centroid_path: str | None = None,
                source_type_id: int | None = None,
                reg_class_id: int | None = None):
    """One shape-preserving rate-vs-VSP interpolant per speed class.

    Returns {class: (interpolator, x_lo, x_hi, y_lo, y_hi, monotone, xs, ys)}.
    Outside the anchor range the curve is CLAMPED, not extrapolated: beyond the
    extreme centroids there is no MOVES information to extend with, and clamping
    keeps the top and bottom bins at exactly their published rates.

    Both halves of the anchoring are class-specific and must match: the centroids
    come from `source_type_id`/`reg_class_id`, and `rates` must be that class's
    own rate table (see vehicle_classes.find_rate_table). The cache key includes
    the class and the rate values, so mixing them up produces a different curve
    rather than a stale hit.
    """
    from scipy.interpolate import PchipInterpolator

    table = _rate_series(rates)
    key = (pollutant, pooled, centroid_path, source_type_id, reg_class_id,
           tuple(np.asarray(table[pollutant], dtype=float).tolist()))
    hit = _CURVE_CACHE.get(key)
    if hit is not None:
        return hit

    cent = load_bin_centroids(centroid_path, pooled=pooled,
                              source_type_id=source_type_id,
                              reg_class_id=reg_class_id)
    cent["rate"] = cent["op_mode_ID"].map(table[pollutant]).astype(float)
    curves = {}
    for cls in SPEED_CLASSES:
        sub = cent[cent.speed_class == cls].sort_values("centroid_vsp")
        xs = sub["centroid_vsp"].to_numpy(dtype=float)
        ys = sub["rate"].to_numpy(dtype=float)
        if np.any(np.diff(xs) <= 0):
            raise RuntimeError(f"{cls}: bin centroids are not strictly increasing "
                               f"({xs}); the interpolation needs distinct anchors")
        curves[cls] = (PchipInterpolator(xs, ys), xs[0], xs[-1], ys[0], ys[-1],
                       bool(np.all(np.diff(ys) > 0)), xs, ys)
    _CURVE_CACHE[key] = curves
    return curves


def is_rate_monotone(pollutant: str = "CO2", rates=None, pooled: bool = True,
                     source_type_id: int | None = None,
                     reg_class_id: int | None = None) -> bool:
    """True when the rate increases with VSP in EVERY speed class.

    Only then is bound_rate_array() meaningful. For the light-duty table this is
    true for CO2, energy and NOx and false for HC, CO, EC and nonEC -- but it is a
    property of the rate table, so it must be re-checked per class rather than
    assumed from the passenger car.
    """
    return all(c[5] for c in rate_curves(pollutant, rates, pooled, None,
                                         source_type_id, reg_class_id).values())


def speed_class_of(speed_mph) -> np.ndarray:
    speed_mph = np.asarray(speed_mph, dtype=float)
    low = speed_mph < 25.0
    mid = (~low) & (speed_mph < 50.0)
    return np.where(low, "low", np.where(mid, "mid", "high"))


# --------------------------------------------------------------------------- #
# rates
# --------------------------------------------------------------------------- #
_DENSE_CACHE: dict = {}


def _dense_rates(table: pd.DataFrame, pollutant: str) -> np.ndarray:
    """Rate table as a dense array indexed by opModeID, NaN where undefined.

    Operating mode IDs are small dense integers, so a fancy-index into a float
    array is the whole lookup. Measured, this is NOT faster than the `Series.map`
    it replaces -- 332 ms vs 319 ms on 20M int16 modes, i.e. a wash; pandas'
    hash-based take is already efficient at this cardinality. The reason to keep
    it is that it lets the aggregator, which prices hundreds of millions of rows
    per lane once per class and accounting, use THIS function instead of carrying
    its own dense-lookup copy. One pricing implementation, one place for the
    published-vs-sub-bin choice to be made.

    Keyed on the table's own values rather than its identity, so each per-class
    table gets its own entry and a mutated or re-read table cannot return a stale
    array.
    """
    if pollutant not in table.columns:
        raise KeyError(f"rate table has no column {pollutant!r}; "
                       f"has {list(table.columns)}")
    modes = np.asarray(table.index, dtype=np.int64)
    vals = np.asarray(table[pollutant], dtype=float)
    key = (pollutant, modes.tobytes(), vals.tobytes())
    hit = _DENSE_CACHE.get(key)
    if hit is not None:
        return hit
    arr = np.full(int(modes.max()) + 1, np.nan)
    arr[modes] = vals
    _DENSE_CACHE[key] = arr
    return arr


def published_rate_array(op_mode, pollutant: str = "CO2", rates=None) -> np.ndarray:
    """MOVES as published: the bin's average applied across the whole bin.

    An operating mode absent from the table yields NaN, which is what the
    Series.map this replaces did, and lets a caller decide whether an
    unpriceable mode is an error or expected (mode 501 for non-PM, say).
    """
    arr = _dense_rates(_rate_series(rates), pollutant)
    codes = np.asarray(op_mode)
    out = np.full(codes.shape, np.nan)
    inside = (codes >= 0) & (codes < len(arr))
    out[inside] = arr[codes[inside].astype(np.intp)]
    return out


def subbin_rate_array(op_mode, speed_mph, vsp, pollutant: str = "CO2", rates=None,
                      pooled: bool = True, source_type_id: int | None = None,
                      reg_class_id: int | None = None) -> np.ndarray:
    """MOVES with VSP-bin resolution restored. Idle and braking are untouched.

    `vsp` must be the VSP of the SAME class the centroids and rates are for --
    i.e. the `vsp_st<ID>` column, or a rebuild of it (vt_state.add_state_columns).
    """
    curves = rate_curves(pollutant, rates, pooled, None,
                         source_type_id, reg_class_id)
    op_mode = np.asarray(op_mode)
    speed_mph = np.asarray(speed_mph, dtype=float)
    vsp = np.asarray(vsp, dtype=float)
    out = published_rate_array(op_mode, pollutant, rates)

    cls = speed_class_of(speed_mph)
    binned = np.isin(op_mode, list(VSP_BINS))
    for c in SPEED_CLASSES:
        sel = binned & (cls == c)
        if not sel.any():
            continue
        f, x_lo, x_hi, y_lo, y_hi, _mono, _xs, _ys = curves[c]
        v = vsp[sel]
        r = f(np.clip(v, x_lo, x_hi))
        out[sel] = np.where(v <= x_lo, y_lo, np.where(v >= x_hi, y_hi, r))
    return out


def bound_rate_array(op_mode, speed_mph, vsp, pollutant: str = "CO2", rates=None,
                     pooled: bool = True, source_type_id: int | None = None,
                     reg_class_id: int | None = None):
    """(R_low, R_high) per second from published rates and monotonicity alone.

    Each is a monotone STEP function through the same anchors, so either one
    applied to a whole trajectory is a coherent accounting, and every monotone
    rate function through the anchors lies between them pointwise. Meaningless
    unless is_rate_monotone(pollutant) for this class.
    """
    curves = rate_curves(pollutant, rates, pooled, None,
                         source_type_id, reg_class_id)
    op_mode = np.asarray(op_mode)
    speed_mph = np.asarray(speed_mph, dtype=float)
    vsp = np.asarray(vsp, dtype=float)
    base = published_rate_array(op_mode, pollutant, rates)
    lo, hi = base.copy(), base.copy()

    cls = speed_class_of(speed_mph)
    binned = np.isin(op_mode, list(VSP_BINS))
    for c in SPEED_CLASSES:
        sel = binned & (cls == c)
        if not sel.any():
            continue
        _f, _xlo, _xhi, _ylo, _yhi, _mono, xs, ys = curves[c]
        idx = np.searchsorted(xs, vsp[sel])          # first anchor >= v
        lo[sel] = ys[np.clip(idx - 1, 0, len(ys) - 1)]
        hi[sel] = ys[np.clip(idx, 0, len(ys) - 1)]
    return np.minimum(lo, hi), np.maximum(lo, hi)


def emission_rate_array(op_mode, pollutant: str = "CO2", rates=None, *,
                        speed_mph=None, vsp=None, subbin: bool | None = None,
                        pooled: bool = True, source_type_id: int | None = None,
                        reg_class_id: int | None = None) -> np.ndarray:
    """Per-second emission rate. THE entry point callers should use.

    `subbin=None` follows the process-wide default (True unless MOVES_SUBBIN=0 or
    --no-subbin). Sub-bin resolution needs `speed_mph` and `vsp`; without them
    this falls back to the published bin rates and says so, because silently
    returning a different accounting than the caller asked for is worse than a
    warning.

    `source_type_id`/`reg_class_id` select the centroid set. They do NOT select
    the rate table -- pass that as `rates`, from vehicle_classes.find_rate_table,
    and keep the two consistent. Under the published (non-sub-bin) accounting the
    class only enters through `rates`, so these are ignored there.
    """
    use = subbin_enabled() if subbin is None else bool(subbin)
    if use and (speed_mph is None or vsp is None):
        import warnings
        warnings.warn("sub-bin rates need speed_mph and vsp; falling back to MOVES "
                      "as published for this call")
        use = False
    if not use:
        return published_rate_array(op_mode, pollutant, rates)
    return subbin_rate_array(op_mode, speed_mph, vsp, pollutant, rates, pooled,
                             source_type_id, reg_class_id)


def map_emissions(df: pd.DataFrame, rates=None, pollutants=("CO2",), *,
                  subbin: bool | None = None, pooled: bool = True,
                  speed_col: str = "speed", vsp_col: str = "vsp",
                  mode_col: str = "op_mode_ID", suffix: str = "",
                  source_type_id: int | None = None,
                  reg_class_id: int | None = None) -> pd.DataFrame:
    """DataFrame convenience: add one rate column per pollutant.

    Expects speed in mph and vsp in kW/tonne -- the units the rest of the repo
    uses. Returns a copy.

    For a multi-class sweep output, point `mode_col`/`vsp_col` at that class's
    columns and pass its `source_type_id`, e.g.

        cls = vehicle_classes.MOVES_CLASSES["combination_short_haul"]
        map_emissions(df, vehicle_classes.find_rate_table(cls), ["CO2"],
                      mode_col=cls.op_mode_column, vsp_col=cls.vsp_column,
                      source_type_id=cls.sourceTypeID)
    """
    df = df.copy()
    speed = df[speed_col].to_numpy(dtype=float) if speed_col in df else None
    vsp = df[vsp_col].to_numpy(dtype=float) if vsp_col in df else None
    for pol in pollutants:
        df[f"{pol}{suffix}"] = emission_rate_array(
            df[mode_col].to_numpy(), pol, rates, speed_mph=speed, vsp=vsp,
            subbin=subbin, pooled=pooled, source_type_id=source_type_id,
            reg_class_id=reg_class_id)
    return df
