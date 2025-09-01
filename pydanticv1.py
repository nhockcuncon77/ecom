import os
import streamlit as st
import re
import json
from fuzzywuzzy import process
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import pandas as pd
from datetime import datetime, timedelta
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 0) CREDENTIALS & PATHS                                                      │
# └──────────────────────────────────────────────────────────────────────────────┘
PINECONE_API_KEY    = "pcsk_6MvBEW_Lzrqekycsbj5snYqthzSVTc8aiUvCUrjQ3SQ9nNPn4WvVPAcJBHVakncsPf4vH"
OPENAI_API_KEY      = "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"
PINECONE_INDEX_NAME = "aiboost-v3"
METADATA_PATH       = "metadata_cache.json"
ASIN_REPORT_PATH    = "full_detailed_aggregated_report.xlsx"
BRAND_REPORT_PATH   = "full_detailed_aggregated_report_brand.xlsx"
GM_REPORT_PATH      = "full_detailed_aggregated_report_gm.xlsx"
UNDERPERF_REPORT    = "underperforming_ats_report_V2.xlsx"
BUSINESS_INV_PATH   = "20250527 Business and Inventory reports.xlsx"
ORDER_PATH          = "202502 Brands Reports v5.xlsx"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 1) DEFINE PERCENT & METRIC COLUMNS                                          │
# └──────────────────────────────────────────────────────────────────────────────┘
PERCENT_COLS = {
    "Gross Margin",
    "Featured Offer (Buy Box) Percentage",
    "Unit Session Percentage",
}

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 2) LOAD & CLEAN ORDER REPORT                                                │
# └──────────────────────────────────────────────────────────────────────────────┘
order_df    = pd.read_excel(ORDER_PATH, sheet_name="Transaction Report", skiprows=7)
business_df = pd.read_excel(ORDER_PATH, sheet_name="Business Report")
unit_df     = pd.read_excel(ORDER_PATH, sheet_name="Unit Financial", skiprows=3)

# Compute settlement period
date_col = next(c for c in order_df.columns if "date" in c.lower())
order_df[date_col] = (
    order_df[date_col].astype(str)
                   .str.replace(r"\s[A-Z]{3,4}$", "", regex=True)
)
order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")
start_date = order_df[date_col].min()
end_date   = order_df[date_col].max()
def fmt_date(dt: datetime) -> str:
    # e.g. “May 1, 2025” (no leading zero on the day)
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
settlement_period = f"{fmt_date(start_date)} – {fmt_date(end_date)}"

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 3) LOAD & CLEAN ASIN-LEVEL REPORT                                           │
# └──────────────────────────────────────────────────────────────────────────────┘
asin_df = pd.read_excel(ASIN_REPORT_PATH)
if len(asin_df):
    asin_df = asin_df.drop(asin_df.tail(1).index)  # drop trailing total row
asin_df["quantity"] = pd.to_numeric(
    asin_df["quantity"].astype(str).str.replace(",", ""), errors="coerce"
)
for pct in PERCENT_COLS:
    if pct in asin_df.columns:
        asin_df[pct] = pd.to_numeric(
            asin_df[pct].astype(str).str.rstrip("%"), errors="coerce"
        )

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 4) LOAD & CLEAN BRAND-LEVEL REPORT                                          │
# └──────────────────────────────────────────────────────────────────────────────┘
brand_df = pd.read_excel(BRAND_REPORT_PATH)
brand_df["quantity"]    = pd.to_numeric(brand_df["quantity"].astype(str).str.replace(",", ""), errors="coerce")
for pct in PERCENT_COLS:
    if pct in brand_df.columns:
        brand_df[pct] = pd.to_numeric(brand_df[pct].astype(str).str.rstrip("%"), errors="coerce")

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 5) LOAD & CLEAN GM-LEVEL REPORT                                             │
# └──────────────────────────────────────────────────────────────────────────────┘
gm_df = pd.read_excel(GM_REPORT_PATH)
gm_df["quantity"] = pd.to_numeric(gm_df["quantity"].astype(str).str.replace(",", ""), errors="coerce")
for pct in PERCENT_COLS:
    if pct in gm_df.columns:
        gm_df[pct] = pd.to_numeric(gm_df[pct].astype(str).str.rstrip("%"), errors="coerce")

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 6) LOAD & CLEAN UNDERPERFORMING-ASIN REPORT                                 │
# └──────────────────────────────────────────────────────────────────────────────┘
under_df = pd.read_excel(UNDERPERF_REPORT)
for col in ("average sales", "expected_total_ats"):
    under_df[col] = pd.to_numeric(
        under_df[col].astype(str).str.replace(r"[\$,]", "", regex=True),
        errors="coerce"
    )

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 7) LOAD GM↔Brand↔ASIN MAPPING WORKBOOK                                       │
# └──────────────────────────────────────────────────────────────────────────────┘
gm_map_df = pd.read_excel("GM, Brand and ASIN.xlsx")
gm_map_df.columns = gm_map_df.columns.str.strip()
if "Brand" in gm_map_df and "Brands" not in gm_map_df:
    gm_map_df.rename(columns={"Brand":"Brands"}, inplace=True)
gm_map_df["ASIN"]   = gm_map_df["ASIN"].astype(str).str.upper()
gm_map_df["Brands"] = gm_map_df["Brands"].astype(str).str.upper()

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 8) LOAD “Manage FBA Inventory” SHEETS                                       │
# └──────────────────────────────────────────────────────────────────────────────┘
inv_xls      = pd.ExcelFile(BUSINESS_INV_PATH)
prod_inv_df  = pd.read_excel(inv_xls, sheet_name="Products Manage FBA Inventory")
brand_inv_df = pd.read_excel(inv_xls, sheet_name="Brands Manage FBA Inventory")
tw_inv_df    = pd.read_excel(inv_xls, sheet_name="Ten West Manage FBA Inventory")

prod_asins  = set(prod_inv_df["asin"].astype(str).str.upper())
brand_asins = set(brand_inv_df["asin"].astype(str).str.upper())
tw_asins    = set(tw_inv_df["asin"].astype(str).str.upper())

sheet_name_by_store_and_days = {
    "Product": {7:"Products Page Sales& Traffic(7)", 14:"Products Page Sales&Traffic(14)", 30:"Products Page Sales&Traffic(30)"},
    "Brand":   {7:"Brands Page Sales & Traffic(7)",  14:"Brands Page Sales&Traffic(14)",   30:"Brands Page Sales&Traffic(30)"},
    "TenWest": {7:"Ten West Page Sales & Traffic(7)",14:"Ten West Page Sales&Traffic(14)", 30:"Ten West Page Sales&Traffic(30)"}
}

def choose_store_for_asin(asin: str) -> str:
    a = asin.strip().upper()
    if a in tw_asins:    return "TenWest"
    if a in brand_asins: return "Brand"
    return "Product"

def fetch_asin_metrics(asin: str, days: int) -> dict:
    target = asin.strip().upper()
    for store_key in sheet_name_by_store_and_days:
        sheet = sheet_name_by_store_and_days[store_key].get(days)
        if not sheet:
            continue
        try:
            df = pd.read_excel(inv_xls, sheet_name=sheet)
        except:
            continue
        col_map = {str(c).lower():c for c in df.columns}
        candidates = [orig for low, orig in col_map.items() if "asin" in low and "parent" not in low]
        if not candidates:
            continue
        asin_col = candidates[0]
        mask = df[asin_col].astype(str).str.upper() == target
        if not mask.any():
            continue
        row = df.loc[mask].iloc[0]
        def find_col(*keywords):
            kws = [k.lower() for k in keywords]
            for low, orig in col_map.items():
                if all(k in low for k in kws):
                    return orig
            return None
        return {
            "sales":       row.get(find_col("ordered","product","sales"), None),
            "units":       row.get(find_col("units","ordered"), None),
            "buy_box_pct": row.get(find_col("buy","box","percentage"), None),
            "sessions":    row.get(find_col("sessions"), None),
            "unit_sess_pct":row.get(find_col("unit","session","percentage"), None)
        }
    return None

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 9) INITIALIZE PINECONE + LANGCHAIN                                           │
# └──────────────────────────────────────────────────────────────────────────────┘
pc         = Pinecone(api_key=PINECONE_API_KEY)
index      = pc.Index(PINECONE_INDEX_NAME)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

with open(METADATA_PATH, "r") as f:
    cache = json.load(f)
known_brands = [b.lower() for b in cache.get("brands", [])]
known_gms    = [g.lower() for g in cache.get("gms", [])]
known_asins  = cache.get("asins", [])

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 10) Pydantic model & Agent setup                                             │
# └──────────────────────────────────────────────────────────────────────────────┘
class ShowBelowPlanResult(BaseModel):
    period: str
    rows: list[dict]

agent = Agent(
    "openai:gpt-4o",
    deps_type=None,
    output_type=ShowBelowPlanResult,
    system_prompt=(
        "When asked 'Show ASINs with average sales price below plan,' "
        "call the tool `show_asins_below_plan` and return its JSON."
    )
)

@agent.tool
def show_asins_below_plan(ctx: RunContext[None]) -> ShowBelowPlanResult:
    df = asin_df[["ASIN","Brands","Amazon Top-line Sales (ATS)","quantity"]].copy()
    df["Total_ATS"] = pd.to_numeric(
        df["Amazon Top-line Sales (ATS)"].astype(str).str.replace(r"[^\d\.]", "", regex=True),
        errors="coerce"
    )
    df["Units_Sold"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["Actual_Avg_Price"] = df["Total_ATS"] / df["Units_Sold"].replace(0, pd.NA)

    # ← here’s the fix: use to_numeric instead of .astype(float)
    planned = unit_df[["ASIN","Amazon Top-line Sales (ATS)"]].rename(
        columns={"Amazon Top-line Sales (ATS)":"Planned_Per_Unit"}
    )
    planned["Planned_Per_Unit"] = pd.to_numeric(
        planned["Planned_Per_Unit"]
               .astype(str)
               .str.replace(r"[^\d\.]", "", regex=True),
        errors="coerce"
    )

    df = df.merge(planned, on="ASIN", how="left")
    df["Delta"]             = (df["Planned_Per_Unit"] - df["Actual_Avg_Price"]).round(2)
    df["Total Lost Revenue"] = (df["Delta"] * df["Units_Sold"]).round(2)
    out_df = df[df["Delta"] > 0].copy()

    rows = []
    for _, row in out_df.iterrows():
        rows.append({
            "Brand":               row["Brands"],
            "ASIN":                row["ASIN"],
            "Planned Sales Price": f"${row['Planned_Per_Unit']:.2f}",
            "Average Sales Price": f"${row['Actual_Avg_Price']:.2f}",
            "Delta":               f"${row['Delta']:.2f}",
            "Units Sold":          int(row["Units_Sold"]),
            "Total Lost Revenue":  f"${row['Total Lost Revenue']:.2f}",
        })
    return ShowBelowPlanResult(period=settlement_period, rows=rows)

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ 11) Streamlit chat UI                                                       │
# └──────────────────────────────────────────────────────────────────────────────┘
st.set_page_config(page_title="Amazon Business Analyst Assistant")
st.title("🤖 Amazon Business Analyst Assistant")

st.session_state.setdefault("messages", [
    SystemMessage(content="You are a helpful Amazon business analyst.")
])

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

prompt = st.chat_input("Ask about ASINs, brands, GMs, metrics…")
if not prompt:
    st.stop()

st.chat_message("user").markdown(prompt)
st.session_state.messages.append(HumanMessage(content=prompt))

if re.search(r"show\s+asins?\s+with\s+average\s+sales\s+price\s+below\s+plan", prompt, flags=re.IGNORECASE):
    result = agent.run_sync(prompt)
    df_out = pd.DataFrame(result.output.rows)

    # Rename the one column for header consistency
    df_out = df_out.rename(
        columns={"Average Sales Price": "Average sales price"}
    )

    with st.chat_message("assistant"):
        # Your requested lead‐in sentence
        st.markdown(
            f"The following ASINs average sales price below the planned sales price "
            f"for the settlement period {result.output.period}"
        )
        # Show only the exact columns, in order, with the exact headers
        st.table(
            df_out[
                [
                    "Brand",
                    "ASIN",
                    "Planned Sales Price",
                    "Average sales price",
                    "Delta",
                    "Units Sold",
                    "Total Lost Revenue",
                ]
            ]
        )
        st.download_button(
            "Download CSV",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name="below_plan_report.csv",
            mime="text/csv"
        )

    st.session_state.messages.append(
        AIMessage(content=f"Displayed {len(df_out)} rows.")
    )
else:
    with st.chat_message("assistant"):
        st.markdown("Sorry, I only support showing ASINs below plan right now.")
    st.session_state.messages.append(AIMessage(content="Unhandled question."))

