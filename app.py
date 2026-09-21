
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


st.set_page_config(
    page_title="Global Superstore | Revival Strategy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Data
# -----------------------------
@st.cache_data
def load_data():
    path = "Global_Superstore2.csv"
    df = pd.read_csv(path, encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.strftime("%b")
    df["Quarter"] = "Q" + df["Order Date"].dt.quarter.astype(str)
    df["Profit Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], 0)
    df["Shipping Ratio"] = np.where(df["Sales"] != 0, df["Shipping Cost"] / df["Sales"], 0)
    return df

df = load_data()

# -----------------------------
# Helpers
# -----------------------------
def money(x):
    return f"${x:,.0f}"

def pct(x):
    return f"{x*100:.1f}%"

def aggregate(data, by):
    g = data.groupby(by, dropna=False).agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique"),
        Avg_Discount=("Discount", "mean"),
        Shipping_Cost=("Shipping Cost", "sum"),
    ).reset_index()
    g["Margin"] = np.where(g["Sales"] != 0, g["Profit"] / g["Sales"], 0)
    g["Shipping_Ratio"] = np.where(g["Sales"] != 0, g["Shipping_Cost"] / g["Sales"], 0)
    return g

def year_summary(data):
    g = aggregate(data, "Year").sort_values("Year")
    g["Sales YoY"] = g["Sales"].pct_change()
    g["Profit YoY"] = g["Profit"].pct_change()
    return g

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.title("🎯 Dashboard Filters")
st.sidebar.caption("Global Superstore — Revival Strategy")

years = sorted(df["Year"].dropna().unique())
selected_years = st.sidebar.multiselect("Year", years, default=years)

markets = sorted(df["Market"].dropna().unique())
selected_markets = st.sidebar.multiselect("Market", markets, default=markets)

segments = sorted(df["Segment"].dropna().unique())
selected_segments = st.sidebar.multiselect("Segment", segments, default=segments)

filtered = df[
    df["Year"].isin(selected_years)
    & df["Market"].isin(selected_markets)
    & df["Segment"].isin(selected_segments)
].copy()

if filtered.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini.")
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.title("📊 Global Superstore: The Revival Strategy")
st.markdown(
    "**Tujuan:** menentukan apakah perusahaan benar-benar underperforming, "
    "menemukan konsentrasi masalah, mengidentifikasi root cause, dan menerjemahkannya "
    "menjadi prioritas aksi 6–12 bulan."
)

# -----------------------------
# Executive KPI
# -----------------------------
total_sales = filtered["Sales"].sum()
total_profit = filtered["Profit"].sum()
margin = total_profit / total_sales if total_sales else 0
orders = filtered["Order ID"].nunique()
avg_discount = filtered["Discount"].mean()
loss_lines = (filtered["Profit"] < 0).sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sales", money(total_sales))
c2.metric("Profit", money(total_profit))
c3.metric("Profit Margin", pct(margin))
c4.metric("Orders", f"{orders:,}")
c5.metric("Avg. Discount", pct(avg_discount))

st.caption(
    f"Periode/filter aktif: {', '.join(map(str, selected_years))} | "
    f"{len(selected_markets)} market | {len(selected_segments)} segment | "
    f"{loss_lines:,} loss-making order lines"
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Performance",
    "2. Geography",
    "3. Product",
    "4. Root Cause",
    "5. AI Recommendation",
])

# ============================================================
# TAB 1 — Performance
# ============================================================
with tab1:
    st.subheader("Apakah Global Superstore benar-benar underperforming?")

    ys = year_summary(filtered)

    if len(ys) > 0:
        a, b = st.columns(2)

        with a:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ys["Year"], y=ys["Sales"], name="Sales"))
            fig.add_trace(go.Bar(x=ys["Year"], y=ys["Profit"], name="Profit"))
            fig.update_layout(
                title="Sales & Profit by Year",
                barmode="group",
                yaxis_title="USD",
                xaxis_title="Year",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with b:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ys["Year"], y=ys["Margin"] * 100,
                    mode="lines+markers", name="Profit Margin"
                )
            )
            fig.update_layout(
                title="Profit Margin Trend",
                yaxis_title="Margin (%)",
                xaxis_title="Year",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            ys[["Year", "Sales", "Sales YoY", "Profit", "Profit YoY", "Margin"]]
            .style.format({
                "Sales": "${:,.0f}",
                "Profit": "${:,.0f}",
                "Sales YoY": "{:.1%}",
                "Profit YoY": "{:.1%}",
                "Margin": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    monthly = (
        filtered.groupby(["Year", "Month", "Month Name"], as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .sort_values(["Year", "Month"])
    )
    monthly["Period"] = pd.to_datetime(
        monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01"
    )

    fig = px.line(
        monthly, x="Period", y=["Sales", "Profit"],
        markers=True, title="Monthly Sales & Profit Trend"
    )
    fig.update_layout(height=400, yaxis_title="USD", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Interpretasi utama dari data penuh: Sales dan Profit meningkat dari 2011–2014. "
        "Namun, margin mencapai puncak pada 2013 lalu turun pada 2014. "
        "Jadi isu utamanya bukan penurunan penjualan secara keseluruhan, melainkan "
        "perlindungan profitabilitas saat perusahaan tumbuh."
    )

# ============================================================
# TAB 2 — Geography
# ============================================================
with tab2:
    st.subheader("Di mana masalah profit terkonsentrasi?")

    geo_level = st.radio(
        "Drill-down geography",
        ["Market", "Region", "Country"],
        horizontal=True,
    )
    geo = aggregate(filtered, geo_level).sort_values("Profit")

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            geo.head(12),
            x="Profit", y=geo_level,
            orientation="h",
            title=f"Bottom {min(12, len(geo))} {geo_level} by Profit",
            color="Margin",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            geo,
            x="Sales", y="Profit",
            size="Quantity",
            color="Margin",
            hover_name=geo_level,
            title=f"Sales vs Profit — {geo_level}",
            color_continuous_scale="RdYlGn",
        )
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        geo.sort_values("Profit").head(20).style.format({
            "Sales": "${:,.0f}",
            "Profit": "${:,.0f}",
            "Margin": "{:.1%}",
            "Avg_Discount": "{:.1%}",
            "Shipping_Cost": "${:,.0f}",
            "Shipping_Ratio": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Data penuh menunjukkan Southeast Asia sebagai trouble spot regional: "
        "profit sekitar $17.9K dari sales $884.4K, dengan margin sekitar 2.0% "
        "dan average discount sekitar 27.2%."
    )

# ============================================================
# TAB 3 — Product
# ============================================================
with tab3:
    st.subheader("Produk mana yang menggerus profit?")

    p_level = st.radio(
        "Product drill-down",
        ["Category", "Sub-Category", "Product Name"],
        horizontal=True,
    )
    prod = aggregate(filtered, p_level).sort_values("Profit")

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            prod.head(15),
            x="Profit", y=p_level,
            orientation="h",
            title=f"Bottom {min(15, len(prod))} {p_level} by Profit",
            color="Margin",
            color_continuous_scale="RdYlGn",
        )
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            prod,
            x="Avg_Discount", y="Margin",
            size="Sales",
            color="Profit",
            hover_name=p_level,
            title=f"Discount vs Margin — {p_level}",
            color_continuous_scale="RdYlGn",
        )
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(
            height=550,
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        prod.head(25).style.format({
            "Sales": "${:,.0f}",
            "Profit": "${:,.0f}",
            "Margin": "{:.1%}",
            "Avg_Discount": "{:.1%}",
            "Shipping_Cost": "${:,.0f}",
            "Shipping_Ratio": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Pada data penuh, Tables merupakan sub-category dengan profit negatif terbesar "
        "(sekitar -$64.1K) dan margin sekitar -8.5%."
    )

# ============================================================
# TAB 4 — Root Cause
# ============================================================
with tab4:
    st.subheader("Mengapa profit tertekan?")

    # Discount bands
    bands = pd.cut(
        filtered["Discount"],
        bins=[-0.001, 0, .10, .20, .30, .40, .50, .60, 1],
        labels=["0%", "1–10%", "11–20%", "21–30%", "31–40%", "41–50%", "51–60%", ">60%"],
    )
    d = (
        filtered.assign(Discount_Band=bands)
        .groupby("Discount_Band", observed=True)
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Lines=("Row ID", "count"),
        )
        .reset_index()
    )
    d["Margin"] = np.where(d["Sales"] != 0, d["Profit"] / d["Sales"], 0)

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            d, x="Discount_Band", y="Profit",
            title="Profit by Discount Band",
            text_auto=".2s",
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        ship = aggregate(filtered, "Ship Mode")
        fig = px.bar(
            ship, x="Ship Mode", y=["Profit", "Shipping_Cost"],
            barmode="group",
            title="Profit vs Shipping Cost by Ship Mode",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        d.style.format({
            "Sales": "${:,.0f}",
            "Profit": "${:,.0f}",
            "Margin": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Root-cause evidence")

    evidence = [
        ["Discount escalation", "Margin turns negative around the 21–30% discount band and becomes increasingly negative at higher discounts."],
        ["Regional concentration", "Southeast Asia has very low margin despite meaningful sales volume; average discount is high."],
        ["Product mix", "Tables is the largest loss-making sub-category, indicating a product-level pricing/discount problem rather than only a volume problem."],
        ["Country concentration", "Several countries have deeply negative margins, including Turkey and Nigeria, indicating localized commercial-policy issues."],
        ["Shipping", "Shipping cost is a material cost driver and should be assessed together with order economics, especially in low-margin markets."],
    ]
    st.table(pd.DataFrame(evidence, columns=["Potential Root Cause", "Evidence from Data"]))

# ============================================================
# TAB 5 — AI Recommendation
# ============================================================
with tab5:
    st.subheader("🤖 AI Executive Recommendation")
    st.caption(
        "Rekomendasi dibuat dari KPI dan pola data yang dihitung di dashboard. "
        "AI tidak menggantikan keputusan manajemen; gunakan output sebagai bahan diskusi."
    )

    # api_key = st.text_input(
    #     "Gemini API Key",
    #     value=os.getenv("GOOGLE_API_KEY", ""),
    #     type="password",
    #     help="Gunakan GOOGLE_API_KEY di environment/secrets untuk deployment.",
    # )
    
    # api_key = os.getenv("GOOGLE_API_KEY")
    api_key = st.secrets("GOOGLE_API_KEY")

    if st.button("Generate AI Recommendation", type="primary"):
        if not api_key:
            st.error("Masukkan Gemini API Key atau set environment variable GOOGLE_API_KEY.")
            st.stop()

        # Build compact evidence from current filter
        ys = year_summary(filtered)
        geo = aggregate(filtered, "Region").sort_values("Profit")
        prod = aggregate(filtered, "Sub-Category").sort_values("Profit")
        disc = (
            filtered.assign(
                Discount_Band=pd.cut(
                    filtered["Discount"],
                    bins=[-0.001, 0, .10, .20, .30, .40, .50, .60, 1],
                    labels=["0%", "1–10%", "11–20%", "21–30%", "31–40%", "41–50%", "51–60%", ">60%"],
                )
            )
            .groupby("Discount_Band", observed=True)
            .agg(Sales=("Sales","sum"), Profit=("Profit","sum"))
            .reset_index()
        )
        disc["Margin"] = np.where(disc["Sales"] != 0, disc["Profit"]/disc["Sales"], 0)

        context = {
            "overall": {
                "sales": float(filtered["Sales"].sum()),
                "profit": float(filtered["Profit"].sum()),
                "margin": float(filtered["Profit"].sum()/filtered["Sales"].sum()),
            },
            "yearly": ys[["Year","Sales","Profit","Margin","Sales YoY","Profit YoY"]].round(4).to_dict("records"),
            "bottom_regions": geo.head(5).round(4).to_dict("records"),
            "bottom_subcategories": prod.head(7).round(4).to_dict("records"),
            "discount_bands": disc.round(4).to_dict("records"),
        }

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a business analytics advisor helping a Board of Directors.
Use only the supplied Global Superstore data. Do not invent facts.
Be concise, practical, and evidence-based.
The company is growing overall, so distinguish growth problems from profitability problems.
Produce:
1) one executive key message,
2) three findings with numbers,
3) three priority actions for the next 6–12 months.
For each action include owner, timeline, KPI, and expected impact direction.
Explicitly connect each action to a root cause.
Mention uncertainty where causality cannot be proven from observational sales data.
Do not recommend arbitrary discount thresholds as absolute policy; frame them as a test/guardrail to validate.
Answer in Indonesian."""
    # """ Kamu ada bisnis analis. Coba simpulkan berdasarkan konteks ini. 3 bagian paragarf campur poin-point misal 1 paragraf 1 poin 1 paragraf
    # """
            ),
            (
                "human",
                "Dashboard evidence:\n{context}"
            )
        ])

        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                google_api_key=api_key,
                temperature=0.2,
            )
            chain = prompt | llm
            response = chain.invoke({"context": context})
            st.markdown(response.content)
        except Exception as e:
            st.error(f"Gagal memanggil Gemini: {e}")

    st.markdown("### Recommended Board storyline")
    st.markdown(
        """
**Growth is not the main issue. Profit quality is.**

1. **Performance:** Sales and profit rise strongly from 2011–2014, but margin slips in 2014.
2. **Concentration:** A small number of markets/countries and product groups create disproportionate losses.
3. **Root cause:** High discounts are strongly associated with negative margins; shipping and product mix can amplify the problem.
4. **Revival:** Protect unit economics first, then selectively rebuild growth in profitable segments/markets.
        """
    )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Global Superstore Revival Strategy Dashboard • Built with Streamlit + Plotly + LangChain + Gemini"
)
