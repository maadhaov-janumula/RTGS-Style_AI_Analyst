import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# ---------------- Small utility ----------------
def _now_ms() -> float:
    return time.perf_counter() * 1000


# ---------------- ToolCallLog ----------------
@dataclass
class ToolCallLog:
    name: str
    args: Dict[str, Any]
    ms: float
    out_shape: Optional[str] = None
    note: Optional[str] = None


# ---------------- Core helpers ----------------
def quality_report(
    df: pd.DataFrame, energy_col: str, dedupe_keys: List[str]
) -> Tuple[dict, ToolCallLog]:
    """Generate a report on the data quality (e.g., missing values, duplicates)."""
    t0 = _now_ms()
    n_rows = len(df)
    n_dupes = df.duplicated(subset=dedupe_keys).sum() if dedupe_keys else 0
    missing_energy = int(df[energy_col].isna().sum()) if energy_col in df.columns else None
    outliers = None
    if energy_col in df.columns:
        s = df[energy_col].dropna()
        if len(s) > 0:
            std = s.std(ddof=0)
            z = (s - s.mean()) / (std if std else 1)
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
        out_shape="dict",
    )
    return report, log


# ---------------- Domain functions (DataFrame in, DataFrame/values out) ----------------
def standardize_tool(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the dataset (column names, month dtype, energy column)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame, but got {type(df)} instead.")

    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Month
    if "month" not in df.columns:
        df["month"] = pd.to_datetime("2025-08-01")  # default if missing
    else:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")

    # Energy -> energy_kwh
    if "energy_kwh" not in df.columns and "units" in df.columns:
        df["energy_kwh"] = df["units"]
    if "energy_kwh" not in df.columns:
        df["energy_kwh"] = pd.NA
    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")

    # Station name
    if "station_name" not in df.columns:
        df["station_name"] = df.get("area", "Unknown Station")

    return df


def clean_tool(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset (dedupe, fill NA)."""
    df = df.copy()
    # dedupe by station_name + month if available
    subset_cols = [c for c in ["station_name", "month"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)
    else:
        df = df.drop_duplicates()

    if "energy_kwh" in df.columns:
        df["energy_kwh"] = df["energy_kwh"].fillna(0)
    return df


def compute_insights_tool(
    df: pd.DataFrame,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """Compute total energy, top5 stations, and under-utilized rows."""
    if "energy_kwh" not in df.columns:
        raise ValueError("Column 'energy_kwh' is required for insights.")
    if "station_name" not in df.columns:
        raise ValueError("Column 'station_name' is required for insights.")

    total_energy = float(df["energy_kwh"].sum())
    top5 = (
        df.groupby("station_name", as_index=False)["energy_kwh"]
        .sum()
        .sort_values("energy_kwh", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    # under-utilized threshold: <100 kWh (simple heuristic)
    underutilized = df[df["energy_kwh"] < 100].copy()
    return total_energy, top5, underutilized


# ---------------- Agent-facing wrappers (accept a CSV path string) ----------------
DATA_OUT_DIR = Path("outputs")
DATA_OUT_DIR.mkdir(exist_ok=True)


class PathInput(BaseModel):
    csv_path: str = Field(..., description="Path to the CSV with EV usage data")


def _standardize_from_path(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    df = standardize_tool(df)
    out_path = DATA_OUT_DIR / "standardized.csv"
    df.to_csv(out_path, index=False)
    return str(out_path)


def _clean_from_path(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    df = standardize_tool(df)
    df = clean_tool(df)
    out_path = DATA_OUT_DIR / "cleaned.csv"
    df.to_csv(out_path, index=False)
    return str(out_path)


def _compute_from_path(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    df = standardize_tool(df)
    df = clean_tool(df)
    total_energy, top5, underutilized = compute_insights_tool(df)
    top5_path = DATA_OUT_DIR / "top5.csv"
    under_path = DATA_OUT_DIR / "underutilized.csv"
    top5.to_csv(top5_path, index=False)
    underutilized.to_csv(under_path, index=False)
    return f"total_energy_kwh={total_energy:.0f};top5={top5_path};underutilized={under_path}"


def _pipeline(csv_path: str) -> str:
    """Standardize -> Clean -> Compute, and persist artifacts."""
    try:
        # Read the CSV file, ensuring no extra quotes or issues in path handling
        csv_path = csv_path.strip("'")  # Remove any extra single quotes from the path
        df = pd.read_csv(csv_path)
        
        # Standardize the data
        df_std = standardize_tool(df)
        std_path = DATA_OUT_DIR / "standardized.csv"
        df_std.to_csv(std_path, index=False)

        # Clean the data
        df_clean = clean_tool(df_std)
        clean_path = DATA_OUT_DIR / "cleaned.csv"
        df_clean.to_csv(clean_path, index=False)

        # Compute insights
        total_energy, top5, underutilized = compute_insights_tool(df_clean)
        top5_path = DATA_OUT_DIR / "top5.csv"
        under_path = DATA_OUT_DIR / "underutilized.csv"
        top5.to_csv(top5_path, index=False)
        underutilized.to_csv(under_path, index=False)

        # Return the paths and insights
        return (
            f"standardized={std_path};cleaned={clean_path};"
            f"total_energy_kwh={total_energy:.0f};top5={top5_path};underutilized={under_path}"
        )
    
    except FileNotFoundError as e:
        return f"File not found: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# ---------------- LangChain Tools (Structured) ----------------
tools = [
    StructuredTool.from_function(
        name="StandardizeData",
        func=_standardize_from_path,
        args_schema=PathInput,
        description="Standardize EV CSV columns/types/units given a CSV path.",
    ),
    StructuredTool.from_function(
        name="CleanData",
        func=_clean_from_path,
        args_schema=PathInput,
        description="Clean standardized EV CSV given a CSV path.",
    ),
    StructuredTool.from_function(
        name="ComputeInsights",
        func=_compute_from_path,
        args_schema=PathInput,
        description="Compute total energy, top stations, and under-utilized given a CSV path.",
    ),
    StructuredTool.from_function(
        name="EVUsagePipeline",
        func=_pipeline,
        args_schema=PathInput,
        description="Run Standardize -> Clean -> Compute in one go for a CSV path; returns file paths and metrics.",
    ),
]

__all__ = [
    "quality_report",
    "standardize_tool",
    "clean_tool",
    "compute_insights_tool",
    "tools",
]
