import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import re

# ---------------- CONFIG ----------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data"]["path"]
TIME_COL = config["data"]["time_col"]
ENERGY_COL = config["data"]["energy_col"]
GEO_COLS = config["data"]["geo_cols"]
OUTPUTS_DIR = Path(config["outputs_dir"])
OUTPUTS_DIR.mkdir(exist_ok=True)

# ---------------- LOAD ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def _infer_month_from_filename(path_str: str) -> pd.Timestamp | None:
    """Extract _MM_YYYY from filename and return YYYY-MM-01."""
    fname = Path(path_str).stem
    m = re.search(r'_(\d{2})_(\d{4})', fname)
    if m:
        mm, yyyy = m.groups()
        return pd.to_datetime(f"{yyyy}-{mm}-01")
    return None

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    # normalize headers
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # --- TIME: ensure TIME_COL exists (create from filename if missing) ---
    if TIME_COL not in df.columns:
        inferred = _infer_month_from_filename(DATA_PATH)
        if inferred is None:
            # last-resort default; keep it simple
            inferred = pd.Timestamp("2025-08-01")
        df[TIME_COL] = inferred
    else:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    # --- ENERGY: coerce to numeric (config already points to 'units') ---
    if ENERGY_COL not in df.columns and "units" in df.columns:
        # align to configured name
        df[ENERGY_COL] = df["units"]
    df[ENERGY_COL] = pd.to_numeric(df[ENERGY_COL], errors="coerce")

    # --- NAME: ensure 'station_name' exists for groupby/chart ---
    if "station_name" not in df.columns:
        if "area" in df.columns:
            df["station_name"] = df["area"]
        else:
            df["station_name"] = "Unknown Station"

    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # drop duplicates
    df = df.drop_duplicates(subset=config["cleaning"]["dedupe_keys"])
    # fill missing energy with 0
    df[ENERGY_COL] = df[ENERGY_COL].fillna(config["cleaning"]["fill_missing_energy"])
    return df

# ---------------- INSIGHTS ----------------
def compute_insights(df: pd.DataFrame):
    total_energy = df[ENERGY_COL].sum()
    top5 = (
        df.groupby("station_name")[ENERGY_COL]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    underutilized = df[df[ENERGY_COL] < 100]  # threshold = 100 kWh
    return total_energy, top5, underutilized

# ---------------- UI ----------------
st.title("🔌 SPDCL EV Charging Analyst (Phase-1 POC)")

df_raw = load_data()
st.subheader("Raw Data Preview")
st.write(df_raw.head())

df_std = standardize(df_raw)
df_clean = clean(df_std)

st.subheader("Cleaned Data Preview")
st.write(df_clean.head())

total_energy, top5, underutilized = compute_insights(df_clean)

# derive month label from TIME_COL (keeps header text simple)
try:
    month_label = pd.to_datetime(df_clean[TIME_COL].iloc[0]).strftime("%b %Y")
except Exception:
    month_label = "Selected Period"

st.subheader("Summary")
st.markdown(f"- **Total Energy ({month_label}):** {total_energy:,.0f} kWh")
st.markdown(f"- **Top 5 Stations:**")
st.table(top5)

st.markdown(f"- **Under-utilized Stations (<100 kWh):** {len(underutilized)}")

# Chart
st.subheader("Energy by Top 5 Stations")
fig, ax = plt.subplots()
ax.bar(top5["station_name"], top5[ENERGY_COL])
ax.set_ylabel("Energy (kWh)")
ax.set_xlabel("Station")
plt.xticks(rotation=45)
st.pyplot(fig)

# Download
st.subheader("Download Cleaned Data")
csv_path = OUTPUTS_DIR / f"cleaned_ev_{month_label.replace(' ', '_').lower()}.csv"
df_clean.to_csv(csv_path, index=False)
st.download_button(
    label="Download Cleaned CSV",
    data=df_clean.to_csv(index=False).encode("utf-8"),
    file_name=csv_path.name,
    mime="text/csv",
)
