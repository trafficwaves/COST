#!/usr/bin/env python3
"""Sweep MAX_GAP values and analyze emissions benefit changes.

This script reproduces the workflow in demo2vsp.ipynb:
- Load demo trajectory (time/space)
- Compute empirical speed/acceleration/VSP/opMode/emissions totals
- For each max_gap, solve constrained smoothing, then compute benchmark emissions totals
- Plot benefit change vs max_gap for CO2, CO, NOx, HC, EC, nonEC

Outputs:
- figures/maxGap_benefit_all.png (2x3 panel)
- figures/maxGap_benefit_{POLLUTANT}.png for each pollutant

Notes:
- Emission rates are mapped by opModeID from emissionRateMicro2026.csv.
- Totals are computed as a simple sum of mapped rates across rows, matching the notebook.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import opt_utils
import opMode_utils
from opMode_utils import opMode
from vsp_utils import VSP

# The acceleration finite difference is NOT redefined here. `vt_state` owns it
# for the whole repo -- ACCEL_SCHEME is "backward", MOVES' own definition -- and
# this module imports it so a single edit there moves every pipeline at once.
# This file used to call np.gradient directly, i.e. a CENTERED difference, which
# made it and everything importing its helpers (R2Q4/pollutant_response_demo.py,
# figure1-demo.ipynb) disagree with the production sweep and with MOVES. See the
# note on vt_state.ACCEL_SCHEME.
from vt_state import ACCEL_SCHEME, acceleration as _acceleration  # noqa: F401


MPH_TO_MS = 0.44704
MILES_S2_TO_MS2 = 1609.344
MILES_TO_METERS = 1609.344


POLLUTANTS = ["CO2", "CO", "NOx", "HC", "EC", "nonEC"]


# Match notebook styling preference
plt.rcParams["font.family"] = "serif"


@dataclass(frozen=True)
class VehicleConfig:
    sourceTypeID: int = 21
    regClassID: int = 20
    model_year: int = 2020


def _boundary_conditions(time_s: np.ndarray, space_miles: np.ndarray) -> tuple[float, float, float, float]:
    """Compute boundary speed/accel in optimizer's internal units (miles/s and miles/s^2)."""
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


def _add_speed_acc_vsp(df: pd.DataFrame, vehicle: VehicleConfig,
                       scheme: str | None = None) -> pd.DataFrame:
    """Add speed (mph), acceleration (miles/s^2), and vsp (kW/tonne)."""
    time_s = df["time"].to_numpy()

    # speed in miles/s -- centered difference of position, which stays centered
    # deliberately: MOVES is handed speed and never differentiates a position
    # record, and for that recovery the centered estimator is the better one.
    speed_miles_s = np.gradient(df["space"].to_numpy(), time_s)
    df["speed"] = speed_miles_s * 3600.0

    # acceleration in miles/s^2 -- MOVES' backward difference (vt_state)
    df["acceleration"] = _acceleration(speed_miles_s, time_s, scheme)

    df["vsp"] = VSP(
        df["speed"].to_numpy() * MPH_TO_MS,
        df["acceleration"].to_numpy() * MILES_S2_TO_MS2,
        sourceTypeID=vehicle.sourceTypeID,
        regClassID=vehicle.regClassID,
        model_year=vehicle.model_year,
    )
    return df


def _calculate_opmode(df: pd.DataFrame) -> pd.DataFrame:
    """Compute op_mode_ID using speed (mph), vsp, and a 3-step accel history in m/s^2."""
    df = df.copy()
    df["speed_ms"] = df["speed"] * MPH_TO_MS
    df["acc_ms"] = df["acceleration"] * MILES_S2_TO_MS2

    df["acc_ms_t_1"] = df["acc_ms"].shift(1)
    df["acc_ms_t_2"] = df["acc_ms"].shift(2)

    # Fill the leading NaNs. Under the backward scheme acc_ms[0] is 0 (MOVES'
    # coalesce), so this shift/fill chain reproduces MOVES' At1/At2 exactly:
    # writing a = [0, a1, a2, ...], At1 = [0, 0, a1, ...] and At2 = [0, 0, 0, a1,
    # ...], i.e. coalesce(speed[t-1]-speed[t-2], 0) and
    # coalesce(speed[t-2]-speed[t-3], 0).
    df["acc_ms_t_1"] = df["acc_ms_t_1"].fillna(df["acc_ms"])
    df["acc_ms_t_2"] = df["acc_ms_t_2"].fillna(df["acc_ms_t_1"])

    df["op_mode_ID"] = df.apply(
        lambda row: opMode(
            row["speed"],
            row["vsp"],
            accelerations=[row["acc_ms"], row["acc_ms_t_1"], row["acc_ms_t_2"]],
        ),
        axis=1,
    )
    return df


def _map_emissions(df: pd.DataFrame, emission_rate_by_opmode: pd.DataFrame) -> pd.DataFrame:
    """Attach per-second rates, honouring the --subbin / --no-subbin setting.

    With sub-bin resolution on (the default) each published bin rate is anchored
    at that bin's measured VSP centroid instead of being applied flat across the
    bin, which stops steady low-speed cruise being overcharged. See the sub-bin
    section of opMode_utils for the construction and its validation.
    """
    return opMode_utils.map_emissions(df, emission_rate_by_opmode, POLLUTANTS,
                                      speed_col="speed", vsp_col="vsp")


def _totals(df: pd.DataFrame) -> dict[str, float]:
    return {pol: float(df[pol].sum()) for pol in POLLUTANTS}


def _parse_gaps(values: Iterable[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demo_data.csv", help="CSV with time, space")
    parser.add_argument(
        "--emission-rate",
        default="emissionRateMicro2026.csv",
        help="CSV with opModeID and pollutant columns",
    )
    parser.add_argument(
        "--max-gaps",
        nargs="*",
        default=None,
        help=(
            "List or comma-separated list of max_gap values. "
            "If omitted, uses a dense sweep from 0.02 to 0.50 miles (step 0.005). "
            "The sweep deliberately starts at 0.02 rather than 0: at max_gap=0 the "
            "gap band collapses, the counterfactual is pinned to the empirical "
            "trajectory, and the benefit is identically zero -- while any comfort "
            "limit the raw data violates makes that QP infeasible."
        ),
    )
    parser.add_argument("--outdir", default="figures", help="Directory to save figures")
    parser.add_argument("--model-year", type=int, default=2020)
    parser.add_argument("--source-type-id", type=int, default=21)
    parser.add_argument("--reg-class-id", type=int, default=20)
    opMode_utils.add_subbin_arg(parser)
    args = parser.parse_args()
    opMode_utils.apply_subbin_arg(args)

    if args.max_gaps is None or len(args.max_gaps) == 0:
        # Start at 0.02 mi, not 0: see --max-gaps help. 0.02 is already on the
        # 0.005 grid, so this just trims the degenerate low end.
        max_gaps = np.round(np.arange(0.02, 0.50 + 1e-12, 0.005), 4).tolist()
    else:
        max_gaps = _parse_gaps(args.max_gaps)
        if not max_gaps:
            raise ValueError("No max_gap values provided")

    os.makedirs(args.outdir, exist_ok=True)

    vehicle = VehicleConfig(
        sourceTypeID=args.source_type_id,
        regClassID=args.reg_class_id,
        model_year=args.model_year,
    )

    # Load data
    data = pd.read_csv(args.data)
    time_data = data["time"].to_numpy()
    space_data = data["space"].to_numpy()

    # Load emission rates
    emission_rate = pd.read_csv(args.emission_rate).set_index("opModeID")

    # Empirical pipeline
    empirical = pd.DataFrame({"time": time_data, "space": space_data})
    empirical = _add_speed_acc_vsp(empirical, vehicle)
    empirical = _calculate_opmode(empirical)
    empirical = _map_emissions(empirical, emission_rate)
    empirical_totals = _totals(empirical)

    v_start, v_end, a_start, a_end = _boundary_conditions(time_data, space_data)

    rows: list[dict[str, float]] = []
    for gap in max_gaps:
        x_smooth, v_smooth, v_avg, cost_value = opt_utils.solve_constrained_smoothing(
            time_data,
            space_data,
            time_data,
            space_data,
            v_start=v_start,
            v_end=v_end,
            a_start=a_start,
            a_end=a_end,
            max_gap=gap,
        )

        benchmark = pd.DataFrame({"time": time_data, "space": x_smooth})
        benchmark = _add_speed_acc_vsp(benchmark, vehicle)
        benchmark = _calculate_opmode(benchmark)
        benchmark = _map_emissions(benchmark, emission_rate)
        bench_totals = _totals(benchmark)

        row: dict[str, float] = {"max_gap": float(gap)}
        for pol in POLLUTANTS:
            emp = empirical_totals[pol]
            bench = bench_totals[pol]
            # benefit: positive means benchmark is lower than empirical
            benefit_pct = ((emp - bench) / emp) * 100.0 if emp != 0 else np.nan
            row[f"{pol}_benchmark"] = bench
            row[f"{pol}_benefit_pct"] = benefit_pct
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("max_gap")
    results["max_gap_m"] = results["max_gap"] * MILES_TO_METERS

    # Plot: all pollutants (benefit %)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for ax, pol in zip(axes, POLLUTANTS):
        ax.plot(results["max_gap_m"], results[f"{pol}_benefit_pct"], marker="o")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(pol)
        ax.set_xlabel("max_gap (m)")
        ax.set_ylabel("Benefit (%)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Emissions benefit vs max_gap (positive = lower than empirical)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "maxGap_benefit_all.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Plot: individual figures
    for pol in POLLUTANTS:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(results["max_gap_m"], results[f"{pol}_benefit_pct"], marker="o")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{pol} benefit vs max_gap")
        ax.set_xlabel("max_gap (m)")
        ax.set_ylabel("Benefit (%)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"maxGap_benefit_{pol}.pdf"), bbox_inches="tight")
        plt.close(fig)

    # Print quick table
    print("Empirical totals (sum of mapped rates):")
    for pol in POLLUTANTS:
        print(f"  {pol}: {empirical_totals[pol]:.4f}")

    print("\nBenefit (%) vs max_gap (positive = benchmark lower than empirical):")
    display_cols = ["max_gap"] + [f"{pol}_benefit_pct" for pol in POLLUTANTS]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(results[display_cols].to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\nSaved figures to:")
    print(f"  {os.path.join(args.outdir, 'maxGap_benefit_all.pdf')}")
    for pol in POLLUTANTS:
        print(f"  {os.path.join(args.outdir, f'maxGap_benefit_{pol}.pdf')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
