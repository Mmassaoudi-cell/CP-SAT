from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario
from REALDATA.src.real_traces import load_real_price_series, load_real_wind_series, resample_to_15min


def autocorr(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    n = len(x)
    if lag >= n:
        return float("nan")
    num = np.sum(x[: n - lag] * x[lag:])
    den = np.sum(x**2)
    return float(num / den) if den > 0 else float("nan")


def diurnal_amplitude(values: np.ndarray, slots_per_day: int = 96) -> float:
    n_days = len(values) // slots_per_day
    if n_days < 2:
        return float("nan")
    trimmed = values[: n_days * slots_per_day].reshape(n_days, slots_per_day)
    hourly_mean = trimmed.mean(axis=0)
    return float(hourly_mean.max() - hourly_mean.min())


def summarize(values: np.ndarray, slots_per_day: int = 96) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    mean = float(values.mean())
    std = float(values.std())
    return {
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean != 0 else float("nan"),
        "lag1_autocorr": autocorr(values, 1),
        "lag96_autocorr": autocorr(values, slots_per_day),
        "diurnal_amplitude": diurnal_amplitude(values, slots_per_day),
        "min": float(values.min()),
        "max": float(values.max()),
        "n_slots": int(len(values)),
    }


def synthetic_price_cef_pools(n_scenarios: int = 30, seed_start: int = 1000):
    """Pooled (cross-DC, cross-scenario) samples for marginal-distribution
    statistics (mean/std/cv/min/max) only -- NOT used for autocorrelation,
    since concatenating different DCs' curves end-to-end is not a real
    time series. See synthetic_multiday_series for the autocorrelation-valid
    construction."""
    prices, cefs = [], []
    for offset in range(n_scenarios):
        scenario = generate_scenario(seed_start + offset, f"calib_synth_{offset}")
        prices.append(scenario.prices.flatten())
        cefs.append(scenario.cefs.flatten())
    return np.concatenate(prices), np.concatenate(cefs)


def synthetic_multiday_series(dc_index: int, n_days: int = 30, seed_start: int = 1000):
    """Chain n_days independently-seeded scenarios' single-DC price/CEF curves
    into one continuous multi-day series, valid for lag-96 (day-to-day)
    autocorrelation: the generator's deterministic diurnal shape is
    seed-invariant (only the AR noise differs per draw), so this reproduces
    the structure a real stationary-diurnal-plus-noise process would have."""
    prices, cefs = [], []
    for offset in range(n_days):
        scenario = generate_scenario(seed_start + offset, f"calib_multiday_{offset}")
        prices.append(scenario.prices[dc_index])
        cefs.append(scenario.cefs[dc_index])
    return np.concatenate(prices), np.concatenate(cefs)


def main():
    real_price = resample_to_15min(load_real_price_series()).dropna().to_numpy()
    real_wind_zones = [resample_to_15min(load_real_wind_series(z)).dropna().to_numpy() for z in (1, 5, 9)]

    synth_price_pooled, synth_cef_pooled = synthetic_price_cef_pools()
    synth_price_series, synth_cef_series = synthetic_multiday_series(dc_index=0)

    rows = {
        "real_price_$/MWh": summarize(real_price),
        "real_wind_zone1_fraction": summarize(real_wind_zones[0]),
        "real_wind_zone5_fraction": summarize(real_wind_zones[1]),
        "real_wind_zone9_fraction": summarize(real_wind_zones[2]),
        "synthetic_price_$/kWh_pooled_marginal": summarize(synth_price_pooled),
        "synthetic_cef_kgCO2/kWh_pooled_marginal": summarize(synth_cef_pooled),
        "synthetic_price_$/kWh_DC1_multiday_series": summarize(synth_price_series),
        "synthetic_cef_kgCO2/kWh_DC1_multiday_series": summarize(synth_cef_series),
    }
    df = pd.DataFrame(rows).T
    out = Path("REALDATA/results")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "calibration_statistics.csv")
    print(df.to_string())

    # Coefficient-of-variation and lag-96 (24h) autocorrelation comparability check:
    # real price CV/autocorr vs synthetic price CV/autocorr, real wind (renewable
    # proxy) CV/autocorr vs synthetic CEF CV/autocorr (CEF moves inversely with
    # renewable share, so autocorrelation magnitude is the comparable quantity).
    comparison = {
        "price_cv_real": rows["real_price_$/MWh"]["cv"],
        "price_cv_synthetic": rows["synthetic_price_$/kWh_pooled_marginal"]["cv"],
        "price_lag96_autocorr_real": rows["real_price_$/MWh"]["lag96_autocorr"],
        "price_lag96_autocorr_synthetic_DC1_series": rows["synthetic_price_$/kWh_DC1_multiday_series"]["lag96_autocorr"],
        "renewable_lag96_autocorr_real_mean": float(np.mean([
            rows["real_wind_zone1_fraction"]["lag96_autocorr"],
            rows["real_wind_zone5_fraction"]["lag96_autocorr"],
            rows["real_wind_zone9_fraction"]["lag96_autocorr"],
        ])),
        "cef_lag96_autocorr_synthetic_DC1_series": rows["synthetic_cef_kgCO2/kWh_DC1_multiday_series"]["lag96_autocorr"],
    }
    with open(out / "calibration_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
