"""Empirical quantile forecaster for Grand Theft Auto VI launch-day Twitch CCU.

Bounty: gta6-twitch-ccu-forecast-01 on make.fun.
Computes multi-quantile projections across hours 2..24 based on empirical
relative trajectories derived from historical game launch comparables.
"""

import json
import math
import os
import sys
from pathlib import Path

# Ensure local imports work under python -P
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
HOURS = list(range(2, 25))


def round2(x: float) -> float | int:
    """Rounds float x to 2 decimals using half-up arithmetic.
    
    Returns an int if the rounded value is integral, producing cleaner JSON output
    that matches JavaScript Math.round(x * 100) / 100 behavior.
    """
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5 - 1e-12:
        whole += 1
    val = whole / 100.0
    return int(val) if val.is_integer() else round(val, 2)


def compute_empirical_quantiles(values: list[float], levels: list[float]) -> list[float]:
    """Computes empirical quantiles using continuous piecewise-linear interpolation.
    
    Optimized for sample sizes N >= 2 with pre-sorted arrays and strict boundary
    handling across the specified probability levels.
    """
    if not values:
        return [1.0] * len(levels)
    
    values_sorted = sorted(values)
    n = len(values_sorted)
    if n == 1:
        return [values_sorted[0]] * len(levels)

    max_idx = n - 1
    result = []
    for q in levels:
        # Clamp probability level to [0.0, 1.0]
        q_clamped = max(0.0, min(1.0, q))
        pos = q_clamped * max_idx
        idx = int(pos)
        frac = pos - idx
        if idx >= max_idx:
            result.append(values_sorted[-1])
        else:
            v0 = values_sorted[idx]
            v1 = values_sorted[idx + 1]
            interp = v0 + frac * (v1 - v0)
            result.append(interp)
    return result


def extract_comparable_ratios(comparables: list[dict]) -> dict[int, list[float]]:
    ratios_per_hour: dict[int, list[float]] = {h: [] for h in HOURS}
    for comp in comparables:
        day_start = comp.get("day_start")
        if not day_start:
            continue
        day_prefix = day_start[:10]
        launch_rows = [
            h for h in comp.get("hourly", [])
            if h.get("from", "").startswith(day_prefix)
        ]
        if len(launch_rows) == 24 and launch_rows[0].get("viewers", 0) > 0:
            v1 = max(1, launch_rows[0]["viewers"])
            for h in HOURS:
                vh = max(1, launch_rows[h - 1].get("viewers", 1))
                ratios_per_hour[h].append(vh / v1)
    return ratios_per_hour


def forecast_case(case: dict) -> dict:
    known = case.get("known", [])
    first = next((k for k in known if k.get("hour") == 1), None)
    if not first:
        raise ValueError("Missing hour-1 observation in case data")
    p1 = max(1, first.get("viewers", 1))

    comparables = case.get("comparables", [])
    ratios_per_hour = extract_comparable_ratios(comparables)

    hours_forecast = []
    for h in HOURS:
        ratios = ratios_per_hour.get(h, [])
        if ratios:
            q_ratios = compute_empirical_quantiles(ratios, LEVELS)
            raw_quantiles = [p1 * r for r in q_ratios]
        else:
            # Baseline fallback if no comparable data exists
            z_levels = [-1.645, -0.674, 0.0, 0.674, 1.645]
            raw_quantiles = [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in z_levels]

        # Enforce non-decreasing and valid ranges
        q_clean: list[float | int] = []
        last_val = 0.0
        for val in raw_quantiles:
            bounded = max(0.01, min(1_000_000_000.0, val))
            if bounded < last_val:
                bounded = last_val
            last_val = bounded
            q_clean.append(round2(bounded))

        hours_forecast.append({"hour": h, "q": q_clean})

    return {"hours": hours_forecast}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_files = sorted(INPUT_DIR.glob("*.in"))
    for in_path in in_files:
        with open(in_path, "r", encoding="utf-8") as f:
            case = json.load(f)
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / f"{in_path.stem}.out"
        out_content = json.dumps(forecast, separators=(",", ":")) + "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_content)


if __name__ == "__main__":
    main()
