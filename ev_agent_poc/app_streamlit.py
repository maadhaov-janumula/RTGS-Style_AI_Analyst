# app_streamlit.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from tools import domain_api  # Import the new Domain API layer

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
    """Load the raw data from CSV."""
    df = pd.read_csv(DATA_PATH)
    return df

# ---------------- INSIGHTS ----------------
def compute_insights(df: pd.DataFrame):
    """Compute basic insights like total energy, top 5 stations, and under-utilized stations."""
    total_energy = df[ENERGY_COL].sum()
    top5 = (
        df.groupby("station_name")[ENERGY_COL]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    underutilized = df[df[ENERGY_COL] < 100]  # Threshold = 100 kWh
    return total_energy, top5, underutilized

# ---------------- UI ----------------
st.title("🔌 SPDCL EV Charging Analyst (Phase-2 POC)")

df_raw = load_data()
st.subheader("Raw Data Preview")
st.write(df_raw.head())

# --- Standardization & Cleaning ---
df_std = domain_api.standardize(df_raw)  # Use the Domain API for standardization
df_clean = domain_api.clean(df_std)  # Use the Domain API for cleaning

st.subheader("Cleaned Data Preview")
st.write(df_clean.head())

# Initialize tool_calls to store log data
tool_calls = []

# --- Domain API calls ---
qr, log = domain_api.quality_report(df_clean, ENERGY_COL, config["cleaning"]["dedupe_keys"])
tool_calls.append(log.__dict__)

mom, log = domain_api.month_over_month(df_clean, ENERGY_COL, TIME_COL)
tool_calls.append(log.__dict__)

top5, log = domain_api.topk(df_clean, metric=ENERGY_COL, by="station_name", k=5)
tool_calls.append(log.__dict__)

ts, log = domain_api.timeseries(df_clean, metric=ENERGY_COL, time_col=TIME_COL, group_by=[], filters=None, agg="sum")
tool_calls.append(log.__dict__)

# Derive month label from TIME_COL (keeps header text simple)
try:
    month_label = pd.to_datetime(df_clean[TIME_COL].iloc[0]).strftime("%b %Y")
except Exception:
    month_label = "Selected Period"


# ---- Summary card ----
st.subheader("Summary")
st.markdown(f"- **Total Energy ({month_label}):** {ts[ENERGY_COL].sum():,.0f} kWh")
if mom["mom_pct"] is not None:
    st.markdown(f"- **MoM change:** {mom['mom_pct']:+.2f}% "
                f"(Prev: {mom['prev_total']:,.0f} kWh → Curr: {mom['current_total']:,.0f} kWh)")
else:
    st.markdown("- **MoM change:** n/a (no previous month in data)")

st.markdown("**Top 5 Stations (by total energy):**")
st.table(top5)

# ---- Chart ----
st.subheader("Energy by Top 5 Stations")
fig, ax = plt.subplots()
ax.bar(top5["station_name"], top5[ENERGY_COL])
ax.set_ylabel("Energy (kWh)")
ax.set_xlabel("Station")
plt.xticks(rotation=45)
st.pyplot(fig)

# ---- Quality box ----
st.subheader("Data Quality")
st.json(qr)

# ---- Downloads ----
st.subheader("Download Cleaned Data")
csv_path = OUTPUTS_DIR / f"cleaned_ev_{month_label.replace(' ', '_').lower()}.csv"
df_clean.to_csv(csv_path, index=False)
st.download_button(
    label="Download Cleaned CSV",
    data=df_clean.to_csv(index=False).encode("utf-8"),
    file_name=csv_path.name,
    mime="text/csv",
)

# ---- Tool call trace ----
with st.expander("🔍 Tool Calls (Phase-2 trace)"):
    st.json(tool_calls)
