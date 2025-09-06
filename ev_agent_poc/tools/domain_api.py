# tools/domain_api.py
import time
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class ToolCallLog:
    name: str
    args: Dict[str, Any]
    ms: float
    out_shape: Optional[str] = None
    note: Optional[str] = None

def _now_ms() -> float:
    """Returns the current time in milliseconds."""
    return time.perf_counter() * 1000

# ---------------- Core helpers ----------------
def quality_report(df: pd.DataFrame, energy_col: str, dedupe_keys: List[str]) -> (dict, ToolCallLog):
    """Generate a report on the data quality (e.g., missing values, duplicates)."""
    t0 = _now_ms()
    n_rows = len(df)
    n_dupes = df.duplicated(subset=dedupe_keys).sum() if dedupe_keys else 0
    missing_energy = int(df[energy_col].isna().sum()) if energy_col in df.columns else None
    outliers = None
    if energy_col in df.columns:
        s = df[energy_col].dropna()
        if len(s) > 0:
            z = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1)
            outliers = int((abs(z) > 3).sum())
    report = {
        "rows": n_rows,
        "duplicates_by_keys": int(n_dupes),
        "missing_energy": missing_energy,
        "energy_z>3_outliers": outliers,
    }
    log = ToolCallLog(
        name="quality_report",
        args={"energy_col": energy_col, "dedupe_keys": dedupe_keys},
        ms=_now_ms() - t0,
        out_shape="dict"
    )
    return report, log

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the dataset (e.g., column names, missing time columns, etc.)."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if 'month' not in df.columns:
        # Inferred from file name or default
        df['month'] = pd.to_datetime('2025-08-01')  # default if missing
    else:
        df['month'] = pd.to_datetime(df['month'], errors='coerce')
    
    if 'energy_kwh' not in df.columns and 'units' in df.columns:
        df['energy_kwh'] = df['units']
    df['energy_kwh'] = pd.to_numeric(df['energy_kwh'], errors="coerce")

    if 'station_name' not in df.columns:
        df['station_name'] = df.get('area', 'Unknown Station')
    
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset (e.g., remove duplicates, handle missing values)."""
    df = df.drop_duplicates(subset=["station_name", "month"])  # dedupe by station_name and month
    df['energy_kwh'] = df['energy_kwh'].fillna(0)  # fill missing energy with 0
    return df

# ---------------- INSIGHTS ----------------
def compute_insights(df: pd.DataFrame) -> (float, pd.DataFrame, pd.DataFrame):
    """Compute basic insights like total energy, top 5 stations, and under-utilized stations."""
    total_energy = df['energy_kwh'].sum()
    top5 = df.groupby('station_name')['energy_kwh'].sum().sort_values(ascending=False).head(5).reset_index()
    underutilized = df[df['energy_kwh'] < 100]  # Stations with less than 100 kWh
    return total_energy, top5, underutilized

def timeseries(df: pd.DataFrame, metric: str, time_col: str, group_by: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None, agg: str = "sum") -> (pd.DataFrame, ToolCallLog):
    """Aggregate the metric (e.g., energy_kwh) over time (e.g., monthly)."""
    t0 = _now_ms()
    data = df.copy()
    
    # Apply filters if provided
    if filters:
        for k, v in filters.items():
            if k in data.columns:
                if isinstance(v, list):
                    data = data[data[k].isin(v)]
                else:
                    data = data[data[k] == v]

    # Ensure time_col exists and convert to datetime if necessary
    if time_col in data.columns:
        data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
        data['_period'] = data[time_col].dt.to_period("M").dt.to_timestamp()
    else:
        data['_period'] = pd.NaT
    
    group_cols = ['_period'] + (group_by or [])
    out = data.groupby(group_cols)[metric].agg(agg).reset_index()

    log = ToolCallLog(
        name="timeseries",
        args={"metric": metric, "time_col": time_col, "group_by": group_by, "filters": filters, "agg": agg},
        ms=_now_ms() - t0,
        out_shape=f"{out.shape}"
    )
    return out, log

def topk(df: pd.DataFrame, metric: str, by: str = "station_name", k: int = 5, filters: Optional[Dict[str, Any]] = None) -> (pd.DataFrame, ToolCallLog):
    """Get the top k stations (or other categories) by the selected metric."""
    t0 = _now_ms()
    data = df.copy()
    
    if filters:
        for kf, vf in filters.items():
            if kf in data.columns:
                if isinstance(vf, list):
                    data = data[data[kf].isin(vf)]
                else:
                    data = data[data[kf] == vf]
    
    top_k = data.groupby(by)[metric].sum().sort_values(ascending=False).head(k).reset_index()
    
    log = ToolCallLog(
        name="topk",
        args={"metric": metric, "by": by, "k": k, "filters": filters},
        ms=_now_ms() - t0,
        out_shape=f"{top_k.shape}"
    )
    return top_k, log

def month_over_month(df: pd.DataFrame, metric: str, time_col: str, filters: Optional[Dict[str, Any]] = None) -> (dict, ToolCallLog):
    """Compute the month-over-month change for a metric."""
    t0 = _now_ms()
    data = df.copy()
    
    # Apply filters if provided
    if filters:
        for k, v in filters.items():
            if k in data.columns:
                if isinstance(v, list):
                    data = data[data[k].isin(v)]
                else:
                    data = data[data[k] == v]
    
    if time_col in data.columns:
        data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
        data['_period'] = data[time_col].dt.to_period("M").dt.to_timestamp()
    else:
        data['_period'] = pd.Timestamp("1970-01-01")
    
    totals = data.groupby('_period')[metric].sum().sort_index()
    if len(totals) == 0:
        result = {"current_month": None, "prev_month": None, "current_total": 0.0, "prev_total": 0.0, "mom_pct": None}
    else:
        current_month = totals.index.max()
        prev_month = current_month - pd.DateOffset(months=1)
        current_total = float(totals.loc[current_month]) if current_month in totals.index else 0.0
        prev_total = float(totals.loc[prev_month]) if prev_month in totals.index else 0.0
        mom_pct = ((current_total - prev_total) / prev_total * 100.0) if prev_total else None
        result = {"current_month": current_month.strftime("%Y-%m"), "prev_month": prev_month.strftime("%Y-%m") if prev_total else None, "current_total": current_total, "prev_total": prev_total, "mom_pct": mom_pct}

    log = ToolCallLog(
        name="month_over_month",
        args={"metric": metric, "time_col": time_col, "filters": filters},
        ms=_now_ms() - t0,
        out_shape="dict"
    )
    return result, log
