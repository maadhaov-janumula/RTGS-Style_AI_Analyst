import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# If your project structure is:
# ev_agent_poc/
#   tools/
#     domain_api.py
# then this import works:
from tools import domain_api

# ---------------- LOAD ENV ----------------
load_dotenv()  # read .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to your .env file.")
    st.stop()

# ---------------- CONFIG ----------------
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data"]["path"]          # e.g., "data/TG-SPDCL_consumption_detail_ev_charging_stations_08_2025.csv"
TIME_COL = config["data"]["time_col"]       # might be "month"
ENERGY_COL = config["data"]["energy_col"]   # expected "energy_kwh" after standardize
GEO_COLS = config["data"]["geo_cols"]
OUTPUTS_DIR = Path(config["outputs_dir"])
OUTPUTS_DIR.mkdir(exist_ok=True)

# ---------------- LOAD RAW DATA ----------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Loaded data is not a DataFrame.")
    return df

# ---------------- LLM / AGENT ----------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

agent = initialize_agent(
    tools=domain_api.tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # okay for now
    verbose=True,
)

# ---------------- UI ----------------
st.title("🔌 SPDCL EV Charging Analyst (Phase-3: LangChain Agent)")

df_raw = load_data(DATA_PATH)
st.write(f"Data type of df_raw: {type(df_raw)}")

st.subheader("Raw Data Preview")
st.write(df_raw.head())

# --- Agent run (use invoke and pass CSV path, not a DataFrame) ---
st.subheader("Agent's Thought Process and Output:")
user_goal = (
    "Summarize August 2025 EV usage for the SPDCL stations. "
    f"Use EVUsagePipeline with csv_path='{DATA_PATH}'."
)
agent_result = agent.invoke({"input": user_goal})
# AgentExecutor returns a dict-like; "output" is the final answer string
st.write(agent_result.get("output", agent_result))

# ---- Deterministic local pipeline for charts and downloads ----
df_std = domain_api.standardize_tool(df_raw)
df_clean = domain_api.clean_tool(df_std)
total_energy, top5, underutilized = domain_api.compute_insights_tool(df_clean)

st.subheader("Cleaned Data Preview")
st.write(df_clean.head())

st.subheader("Summary")
st.markdown(f"- **Total Energy (August 2025):** {total_energy:,.0f} kWh")
st.markdown("- **Top 5 Stations:**")
st.table(top5)

# ---- Chart ----
st.subheader("Energy by Top 5 Stations")
fig, ax = plt.subplots()
ax.bar(top5["station_name"], top5["energy_kwh"])
ax.set_ylabel("Energy (kWh)")
ax.set_xlabel("Station")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

# ---- Data Quality ----
st.subheader("Data Quality")
qr, _log = domain_api.quality_report(
    df_clean,
    ENERGY_COL if ENERGY_COL in df_clean.columns else "energy_kwh",
    config.get("cleaning", {}).get("dedupe_keys", ["station_name", "month"]),
)
st.json(qr)

# ---- Downloads ----
st.subheader("Download Cleaned Data")
csv_path = OUTPUTS_DIR / "cleaned_ev_august_2025.csv"
df_clean.to_csv(csv_path, index=False)
st.download_button(
    label="Download Cleaned CSV",
    data=df_clean.to_csv(index=False).encode("utf-8"),
    file_name=csv_path.name,
    mime="text/csv",
)
