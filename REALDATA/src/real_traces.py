from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(r"C:\Users\MMASSAOUDI\Desktop\Data\Load Data\GEFCom2014_Dataset\GEFCom2014 Data")
PRICE_CSV = DATA_ROOT / "GEFCom2014-P_V2" / "Price" / "Task 15" / "Task15_P.csv"
WIND_DIR = DATA_ROOT / "GEFCom2014-W_V2" / "Wind" / "Task 15" / "Task15_W_Zone1_10" / "Task15_W_Zone1_10"


def load_real_price_series() -> pd.Series:
    """Real hourly zonal locational marginal price, GEFCom2014 Task 15 (Zone 1),
    2011-01-01 through 2013-12-17. Units: $/MWh. Source: GEFCom2014 electricity
    price forecasting track (public competition dataset)."""
    df = pd.read_csv(PRICE_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m%d%Y %H:%M")
    df = df.dropna(subset=["Zonal Price"]).sort_values("timestamp")
    series = df.set_index("timestamp")["Zonal Price"]
    return series[~series.index.duplicated(keep="first")]


def load_real_wind_series(zone: int) -> pd.Series:
    """Real hourly normalized wind power output (0-1) for one of 10 GEFCom2014
    wind zones, 2012-01-01 through 2013-11-30."""
    path = WIND_DIR / f"Task15_W_Zone{zone}.csv"
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], format="%Y%m%d %H:%M")
    df = df.sort_values("timestamp")
    series = df.set_index("timestamp")["TARGETVAR"]
    return series[~series.index.duplicated(keep="first")]


def resample_to_15min(series: pd.Series) -> pd.Series:
    """Hourly -> 15-minute via linear interpolation (matches the environment's
    quarter-hour resolution)."""
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="15min")
    return series.reindex(series.index.union(full_index)).interpolate("time").reindex(full_index)


def extract_day(series_15min: pd.Series, day_start: pd.Timestamp, horizon: int = 96) -> np.ndarray:
    window = series_15min.loc[day_start : day_start + pd.Timedelta(minutes=15 * (horizon - 1))]
    if len(window) < horizon or window.isna().any():
        return None
    return window.to_numpy(dtype=np.float64)


def rescale_to_range(values: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    """Documented linear rescaling: preserves the real trace's temporal shape
    (autocorrelation, relative volatility, diurnal pattern) while matching the
    target mean/std used elsewhere in this study, since absolute $/kWh levels
    from a wholesale LMP market are not directly comparable to this study's
    (also-arbitrary, since no author data exists) synthetic price levels."""
    src_mean, src_std = float(values.mean()), float(values.std())
    if src_std < 1e-9:
        return np.full_like(values, target_mean)
    z = (values - src_mean) / src_std
    return target_mean + target_std * z


def cef_from_renewable_fraction(renewable_fraction: np.ndarray, cef_min: float, cef_max: float) -> np.ndarray:
    """Documented linear emissions-factor model: CEF decreases linearly with
    renewable generation share, consistent with the marginal-emissions
    intuition used in carbon-aware-computing literature (higher renewable
    penetration -> lower grid carbon intensity). renewable_fraction in [0,1]
    (real GEFCom2014 wind-zone normalized power output used as the driver)."""
    renewable_fraction = np.clip(renewable_fraction, 0.0, 1.0)
    return cef_max - (cef_max - cef_min) * renewable_fraction
