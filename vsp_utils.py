from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, overload

import numpy as np


@dataclass(frozen=True, slots=True)
class RoadLoadCoefficients:
	"""Road load coefficients used by the MOVES VSP equation.

	A: rollingTermA
	B: rotatingTermB
	C: dragTermC
	sourceMass: sourceMass (metric tons)
	fixedMassFactor: fixedMassFactor (metric tons)
	"""

	A: float
	B: float
	C: float
	sourceMass: float
	fixedMassFactor: float


def _default_source_use_type_physics_path() -> Path:
	# Repo root is typically the directory containing this file.
	return Path(__file__).resolve().with_name("sourceusetypephysics.csv")


@lru_cache(maxsize=8)
def load_source_use_type_physics(csv_path: str | Path | None = None) -> tuple[dict[str, Any], ...]:
	"""Load MOVES 'sourceUseTypePhysics' table export.

	Expected columns include:
	- regClassID
	- beginModelYearID
	- endModelYearID
	- rollingTermA
	- rotatingTermB
	- dragTermC
	- fixedMassFactor
	"""

	path = Path(csv_path) if csv_path is not None else _default_source_use_type_physics_path()
	if not path.exists():
		raise FileNotFoundError(
			f"sourceUseTypePhysics CSV not found at: {path}. "
			"Pass csv_path=... to load_source_use_type_physics()/get_road_load_coeffs()."
		)

	with path.open(newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		rows = list(reader)

	if not rows:
		raise ValueError(f"CSV at {path} appears to be empty")

	required = {
		"sourceTypeID",
		"regClassID",
		"beginModelYearID",
		"endModelYearID",
		"rollingTermA",
		"rotatingTermB",
		"dragTermC",
		"sourceMass",
		"fixedMassFactor",
	}
	missing = required.difference(rows[0].keys())
	if missing:
		raise ValueError(
			f"CSV at {path} is missing required columns: {sorted(missing)}. "
			"Make sure you're using the MOVES 'sourceUseTypePhysics' export."
		)

	def to_num(value: Any) -> float:
		# Treat empty strings as NaN.
		if value is None:
			return float("nan")
		if isinstance(value, (int, float)):
			return float(value)
		s = str(value).strip()
		if s == "":
			return float("nan")
		try:
			return float(s)
		except ValueError:
			return float("nan")

	normalized: list[dict[str, Any]] = []
	for r in rows:
		rr: dict[str, Any] = dict(r)
		for col in required:
			rr[col] = to_num(rr.get(col))
		normalized.append(rr)

	return tuple(normalized)


@lru_cache(maxsize=256)
def get_road_load_coeffs(
	regClassID: int,
	*,
	sourceTypeID: int | None = None,
	model_year: int | None = None,
	csv_path: str | Path | None = None,
) -> RoadLoadCoefficients:
	"""Resolve coefficients for a given regClassID (+ optional sourceTypeID).

	If model_year is provided, selects the row where beginModelYearID <= model_year <= endModelYearID.
	If multiple rows match, prefers the one with the latest beginModelYearID.
	If model_year is omitted and multiple rows exist, prefers the latest beginModelYearID.

	Note: The official MOVES implementation is keyed by (sourceTypeID, regClassID, model year).
	For backward compatibility, sourceTypeID is optional here; if omitted and multiple sourceTypeID
	rows exist for the regClassID, the most recent model-year range is chosen.

	Cached: resolving coefficients means three list comprehensions and a sort over
	the 206-row physics table, which measured 45.7 us -- 52.6% of a VSP() call on
	a 500-sample trajectory. The pipeline calls VSP once per (vehicle, gap), about
	105,550 times per lane with identical arguments every time, so this was ~4.8 s
	of CPU per lane spent re-deriving the same five constants. All arguments are
	hashable and RoadLoadCoefficients is a frozen slots dataclass, so sharing one
	instance across callers is safe.
	"""

	df = load_source_use_type_physics(csv_path)

	subset = [r for r in df if int(r["regClassID"]) == int(regClassID)]
	if not subset:
		raise KeyError(f"regClassID={regClassID} not found in sourceUseTypePhysics")

	if sourceTypeID is not None:
		subset = [r for r in subset if int(r["sourceTypeID"]) == int(sourceTypeID)]
		if not subset:
			raise KeyError(
				f"No row found for sourceTypeID={sourceTypeID}, regClassID={regClassID} in sourceUseTypePhysics"
			)

	if model_year is not None:
		subset = [
			r
			for r in subset
			if (r["beginModelYearID"] <= model_year) and (model_year <= r["endModelYearID"])
		]
		if not subset:
			raise KeyError(
				f"No row found for regClassID={regClassID} covering model_year={model_year} "
				"(check beginModelYearID/endModelYearID ranges)."
			)

	# Prefer most recent range.
	subset = sorted(
		subset,
		key=lambda r: (r["beginModelYearID"], r["endModelYearID"]),
		reverse=True,
	)
	row = subset[0]

	A = float(row["rollingTermA"])
	B = float(row["rotatingTermB"])
	C = float(row["dragTermC"])
	sourceMass = float(row["sourceMass"])
	fixedMassFactor = float(row["fixedMassFactor"])

	if not np.isfinite([A, B, C, sourceMass, fixedMassFactor]).all():
		raise ValueError(
			f"Non-numeric coefficient(s) for regClassID={regClassID} (model_year={model_year}). "
			"Check the CSV values for rollingTermA/rotatingTermB/dragTermC/sourceMass/fixedMassFactor."
		)
	if fixedMassFactor == 0:
		raise ValueError(
			f"fixedMassFactor is 0 for regClassID={regClassID} (sourceTypeID={sourceTypeID}); cannot divide"
		)

	return RoadLoadCoefficients(A=A, B=B, C=C, sourceMass=sourceMass, fixedMassFactor=fixedMassFactor)


@overload
def VSP(
	v: float,
	a: float,
	*,
	sourceTypeID: int | None = None,
	regClassID: int | None = None,
	model_year: int | None = None,
	theta: float | np.ndarray = 0.0,
	rollingTermA: float | None = None,
	rotatingTermB: float | None = None,
	dragTermC: float | None = None,
	sourceMass: float | None = None,
	fixedMassFactor: float | None = None,
	csv_path: str | Path | None = None,
) -> float: ...


@overload
def VSP(
	v: np.ndarray,
	a: np.ndarray,
	*,
	sourceTypeID: int | None = None,
	regClassID: int | None = None,
	model_year: int | None = None,
	theta: float | np.ndarray = 0.0,
	rollingTermA: float | None = None,
	rotatingTermB: float | None = None,
	dragTermC: float | None = None,
	sourceMass: float | None = None,
	fixedMassFactor: float | None = None,
	csv_path: str | Path | None = None,
) -> np.ndarray: ...


def VSP(
	v: float | np.ndarray,
	a: float | np.ndarray,
	*,
	sourceTypeID: int | None = None,
	regClassID: int | None = None,
	model_year: int | None = None,
	theta: float | np.ndarray = 0.0,
	rollingTermA: float | None = None,
	rotatingTermB: float | None = None,
	dragTermC: float | None = None,
	sourceMass: float | None = None,
	fixedMassFactor: float | None = None,
	csv_path: str | Path | None = None,
) -> float | np.ndarray:
	"""Compute VSP per the official MOVES implementation.

	Java reference:
	  VSP = (rollingTermA*speed + rotatingTermB*speed^2 + dragTermC*speed^3 + sourceMass*speed*accel)
			/ fixedMassFactor

	Grade handling:
	  theta can be a scalar or an array broadcastable to v/a.
	  Uses effective acceleration: accel_eff = accel + 9.8*sin(theta)
	  (i.e., grade contributes an additional longitudinal acceleration component).

	Provide coefficients either:
	- via (sourceTypeID, regClassID, model_year), loaded from sourceusetypephysics.csv, or
	- directly via rollingTermA/rotatingTermB/dragTermC/sourceMass/fixedMassFactor.

	Parameters:
	- v: speed (m/s)
	- a: acceleration (m/s^2)
	"""

	if regClassID is not None:
		coeffs = get_road_load_coeffs(
			regClassID,
			sourceTypeID=sourceTypeID,
			model_year=model_year,
			csv_path=csv_path,
		)
		rollingTermA = coeffs.A
		rotatingTermB = coeffs.B
		dragTermC = coeffs.C
		sourceMass = coeffs.sourceMass
		fixedMassFactor = coeffs.fixedMassFactor

	if (
		rollingTermA is None
		or rotatingTermB is None
		or dragTermC is None
		or sourceMass is None
		or fixedMassFactor is None
	):
		raise ValueError(
			"Provide either regClassID=... (plus optional sourceTypeID/model_year) "
			"or explicit rollingTermA/rotatingTermB/dragTermC/sourceMass/fixedMassFactor"
		)
	if fixedMassFactor == 0:
		raise ValueError("fixedMassFactor cannot be 0")

	v_arr = np.asarray(v)
	a_arr = np.asarray(a)

	# Add grade contribution (theta in radians). Skipped entirely on flat grade:
	# adding a scalar zero to the array is a full extra pass plus an allocation,
	# and a + 0.0 == a for every finite a. (It also normalises -0.0 to +0.0, which
	# cannot change VSP's value or any downstream comparison against 0.)
	if np.ndim(theta) == 0 and theta == 0.0:
		a_eff = a_arr
	else:
		a_eff = a_arr + 9.8 * np.sin(theta)

	# v_arr * v_arr is bit-identical to v_arr**2 and roughly twice as fast, so the
	# square is reused. The CUBE is deliberately left as v_arr**3 rather than
	# v2 * v_arr: those differ by 1 ulp on ~26% of inputs, and VSP feeds the
	# operating-mode classifier, whose VSP bin edges are exact comparisons -- a
	# 1-ulp shift could flip a mode for a sample sitting exactly on a boundary.
	# The extra ~8 us per call buys bit-for-bit stability of every emission number.
	v2 = v_arr * v_arr
	vsp = (
		rollingTermA * v_arr
		+ rotatingTermB * v2
		+ dragTermC * v_arr**3
		+ sourceMass * v_arr * a_eff
	) / fixedMassFactor

	# Preserve scalar return if inputs were scalar.
	if np.isscalar(v) and np.isscalar(a):
		return float(np.asarray(vsp))
	return vsp



def wheel_power_kw(
	v: float | np.ndarray,
	a: float | np.ndarray,
	*,
	sourceTypeID: int | None = None,
	regClassID: int | None = None,
	model_year: int | None = None,
	theta: float | np.ndarray = 0.0,
	csv_path: str | Path | None = None,
	vsp: float | np.ndarray | None = None,
) -> float | np.ndarray:
	"""Tractive power at the wheels, kW. VSP times the vehicle mass in tonnes.

	VSP is mass-normalised (kW/tonne), which makes rate bins comparable across
	vehicles but hides how much power is actually being asked for. The absolute
	figure is what makes a MOVES bin interpretable: opMode 12 spans
	0 <= VSP < 3 kW/tonne, which for a 1.4788 t passenger car is 0 to 4.4 kW, and
	a steady 10.2 mph cruise needs 0.80 kW of that -- the bin floor. That
	comparison is the basis of the sub-bin rate resolution in
	`opMode_utils.subbin_rate_array`, and of the diagnostics in
	R1Q4/moves_subbin.py.

	The conversion is exactly VSP * fixedMassFactor: MOVES divides the power
	polynomial by fixedMassFactor, so multiplying back recovers power in kW.
	Rakha et al.'s VT-CPFM writes the same quantity as (m / 1000) * VSP.

	Pass `vsp=` to convert an already-computed VSP series and skip re-evaluating
	the polynomial; `v` and `a` are then ignored.
	"""
	if regClassID is None:
		raise ValueError("wheel_power_kw needs regClassID (plus optional "
		                 "sourceTypeID / model_year) to resolve fixedMassFactor")
	coeffs = get_road_load_coeffs(
		regClassID, sourceTypeID=sourceTypeID, model_year=model_year, csv_path=csv_path
	)
	if vsp is None:
		vsp = VSP(v, a, sourceTypeID=sourceTypeID, regClassID=regClassID,
		          model_year=model_year, theta=theta, csv_path=csv_path)
	return np.asarray(vsp) * coeffs.fixedMassFactor if not np.isscalar(vsp) \
		else float(vsp) * coeffs.fixedMassFactor


def road_load_kw(
	v: float | np.ndarray,
	*,
	sourceTypeID: int | None = None,
	regClassID: int | None = None,
	model_year: int | None = None,
	csv_path: str | Path | None = None,
) -> float | np.ndarray:
	"""Steady-speed road load, kW: the a = 0 case of wheel_power_kw.

	This is what a vehicle holding a CONSTANT speed demands -- rolling, rotating
	and aerodynamic terms only, with no inertia term. It is the quantity that
	makes the steady-speed bin-floor problem concrete: 0.80 kW at 10.2 mph
	against an opMode-12 bin that reaches 4.4 kW.
	"""
	return wheel_power_kw(v, 0.0, sourceTypeID=sourceTypeID, regClassID=regClassID,
	                      model_year=model_year, csv_path=csv_path)


__all__ = [
	"RoadLoadCoefficients",
	"load_source_use_type_physics",
	"get_road_load_coeffs",
	"VSP",
	"wheel_power_kw",
	"road_load_kw",
]

