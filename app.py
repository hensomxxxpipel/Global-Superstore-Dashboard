
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Performance",
    "2. Geography",
    "3. Product",
    "4. Root Cause",
    "5. 12-Month Revival Plan",
    "6. AI Recommendation",
    "7. AI Chart Maker",
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
# TAB 5 — 12-Month Revival Plan
# ============================================================
with tab5:

    st.subheader("🚀 12-Month Revival Plan")

    st.caption(
        "Rekomendasi berikut menggunakan full dataset Global Superstore. "
        "Angka simulasi adalah scenario analysis, bukan forecast pasti."
    )

    # --------------------------------------------------------
    # USE FULL DATASET
    # --------------------------------------------------------
    strategic_df = df.copy()

    total_sales = strategic_df["Sales"].sum()
    total_profit = strategic_df["Profit"].sum()
    global_margin = total_profit / total_sales if total_sales else 0

    st.info(
        f"""
        **Strategic Direction**

        Global Superstore tidak menunjukkan masalah utama pada pertumbuhan
        sales. Fokus revival adalah memperbaiki **profitability** melalui
        pengendalian discount, perbaikan product economics, dan regional
        profitability.

        - Current Global Sales: {money(total_sales)}  
        - Current Global Profit: {money(total_profit)}  
        - Current Global Margin: {pct(global_margin)}
        """
    )

    # ========================================================
    # PRIORITY 1
    # ========================================================
    st.markdown("## 1️⃣ Control High-Discount Transactions")

    st.markdown(
        """
        ### Problem

        Transaksi dengan discount tinggi menghasilkan sales yang besar,
        tetapi secara agregat justru menghasilkan negative profit.
        """
    )

    # High discount >= 21%
    high_discount_df = strategic_df[
        strategic_df["Discount"] >= 0.21
    ].copy()

    hd_sales = high_discount_df["Sales"].sum()
    hd_profit = high_discount_df["Profit"].sum()
    hd_margin = hd_profit / hd_sales if hd_sales else 0
    hd_lines = len(high_discount_df)
    hd_avg_discount = high_discount_df["Discount"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Sales ≥21% Discount",
        money(hd_sales)
    )

    c2.metric(
        "Current Profit",
        money(hd_profit)
    )

    c3.metric(
        "Current Margin",
        pct(hd_margin)
    )

    c4.metric(
        "Order Lines",
        f"{hd_lines:,}"
    )

    st.markdown(
        f"""
        **Current evidence:**

        - Sales: **{money(hd_sales)}**
        - Profit: **{money(hd_profit)}**
        - Margin: **{pct(hd_margin)}**
        - Average discount: **{pct(hd_avg_discount)}**
        - Order lines: **{hd_lines:,}**

        Dengan demikian, discount ≥21% merupakan area yang perlu
        dikendalikan karena menghasilkan **profit leakage** yang material.
        """
    )

    # --------------------------------------------------------
    # SIMULATION: REDUCE DISCOUNT TO 20%
    # --------------------------------------------------------

    st.markdown("### Recommended Pilot: Maximum 20% Discount")

    st.markdown(
        """
        **Rekomendasi:** lakukan pilot dengan maximum discount **20%**
        untuk transaksi high-discount yang menghasilkan negative profit.

        20% digunakan sebagai **pilot guardrail**, bukan berarti seluruh
        transaksi perusahaan harus langsung dibatasi 20%.
        """
    )

    # Calculate base price and cost
    hd_sim = high_discount_df.copy()

    hd_sim["Base_Price"] = np.where(
        hd_sim["Discount"] < 1,
        hd_sim["Sales"] / (1 - hd_sim["Discount"]),
        hd_sim["Sales"]
    )

    hd_sim["Estimated_Cost"] = (
        hd_sim["Sales"] - hd_sim["Profit"]
    )

    # New sales at 20%
    hd_sim["Scenario_Sales_20"] = (
        hd_sim["Base_Price"] * 0.80
    )

    hd_sim["Scenario_Profit_20"] = (
        hd_sim["Scenario_Sales_20"]
        - hd_sim["Estimated_Cost"]
    )

    scenario_sales_20 = hd_sim["Scenario_Sales_20"].sum()
    scenario_profit_20 = hd_sim["Scenario_Profit_20"].sum()

    sales_change_20 = scenario_sales_20 - hd_sales
    profit_change_20 = scenario_profit_20 - hd_profit

    scenario_margin_20 = (
        scenario_profit_20 / scenario_sales_20
        if scenario_sales_20
        else 0
    )

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Current Sales",
        money(hd_sales)
    )

    s2.metric(
        "Scenario Sales",
        money(scenario_sales_20),
        delta=money(sales_change_20)
    )

    s3.metric(
        "Current Profit",
        money(hd_profit)
    )

    s4.metric(
        "Scenario Profit",
        money(scenario_profit_20),
        delta=money(profit_change_20)
    )

    st.markdown(
        f"""
        ### Scenario Result

        Jika seluruh transaksi discount ≥21% diasumsikan dapat dipertahankan
        volumenya setelah discount diturunkan menjadi **20%**, maka:

        - Sales scenario: **{money(scenario_sales_20)}**
        - Profit scenario: **{money(scenario_profit_20)}**
        - Potential profit improvement: **{money(profit_change_20)}**
        - Scenario margin: **{pct(scenario_margin_20)}**

        ⚠️ Ini **bukan forecast**. Simulasi mengasumsikan volume penjualan,
        base price, dan cost structure tetap. Dalam kondisi nyata, penurunan
        discount dapat menyebabkan sebagian customer tidak melakukan pembelian.
        """
    )

    # --------------------------------------------------------
    # RETENTION SCENARIO
    # --------------------------------------------------------

    st.markdown("### Demand Retention Scenario")

    retention_values = [0.60, 0.70, 0.80, 0.90, 1.00]

    retention_results = []

    for retention in retention_values:

        retained_sales = scenario_sales_20 * retention

        # Cost is also assumed to follow retained volume
        retained_cost = hd_sim["Estimated_Cost"].sum() * retention

        retained_profit = (
            retained_sales - retained_cost
        )

        retention_results.append({
            "Customer Volume Retention": retention,
            "Scenario Sales": retained_sales,
            "Scenario Profit": retained_profit,
            "Profit Improvement": retained_profit - (
                hd_profit * retention
            )
        })

    retention_df = pd.DataFrame(retention_results)

    st.dataframe(
        retention_df.style.format({
            "Customer Volume Retention": "{:.0%}",
            "Scenario Sales": "${:,.0f}",
            "Scenario Profit": "${:,.0f}",
            "Profit Improvement": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        **Interpretation:** management tidak perlu langsung mengubah seluruh
        kebijakan discount. Jalankan pilot terlebih dahulu dan ukur apakah
        volume customer tetap bertahan.
        """
    )

    st.markdown(
        """
        **Owner:** Commercial / Pricing  
        **Timeline:** M1–M3 diagnosis → M4–M6 pilot → M7–M9 evaluation → M10–M12 scale  
        **KPI:** Profit Margin, Profit per Order, High-Discount Sales, Customer Retention
        """
    )

    st.divider()

    # ========================================================
    # PRIORITY 2
    # ========================================================
    st.markdown("## 2️⃣ Turn Around the Tables Sub-Category")

    tables_df = strategic_df[
        strategic_df["Sub-Category"] == "Tables"
    ].copy()

    tables_sales = tables_df["Sales"].sum()
    tables_profit = tables_df["Profit"].sum()
    tables_margin = (
        tables_profit / tables_sales
        if tables_sales
        else 0
    )

    tables_discount = tables_df["Discount"].mean()

    t1, t2, t3, t4 = st.columns(4)

    t1.metric(
        "Tables Sales",
        money(tables_sales)
    )

    t2.metric(
        "Tables Profit",
        money(tables_profit)
    )

    t3.metric(
        "Tables Margin",
        pct(tables_margin)
    )

    t4.metric(
        "Avg Discount",
        pct(tables_discount)
    )

    st.markdown(
        f"""
        ### Problem

        **Tables menghasilkan sales {money(tables_sales)}, tetapi profit
        {money(tables_profit)} dengan margin {pct(tables_margin)}.**

        Artinya masalahnya bukan tidak ada demand. Produk tetap menghasilkan
        sales, tetapi economics-nya tidak sehat.
        """
    )

    # --------------------------------------------------------
    # TABLES SCENARIO TO 20%
    # --------------------------------------------------------

    tables_sim = tables_df.copy()

    tables_sim["Base_Price"] = np.where(
        tables_sim["Discount"] < 1,
        tables_sim["Sales"] / (1 - tables_sim["Discount"]),
        tables_sim["Sales"]
    )

    tables_sim["Estimated_Cost"] = (
        tables_sim["Sales"] - tables_sim["Profit"]
    )

    tables_sim["Scenario_Sales_20"] = (
        tables_sim["Base_Price"] * 0.80
    )

    tables_sim["Scenario_Profit_20"] = (
        tables_sim["Scenario_Sales_20"]
        - tables_sim["Estimated_Cost"]
    )

    tables_profit_20 = tables_sim["Scenario_Profit_20"].sum()
    tables_profit_improvement = (
        tables_profit_20 - tables_profit
    )

    tables_margin_20 = (
        tables_profit_20 /
        tables_sim["Scenario_Sales_20"].sum()
        if tables_sim["Scenario_Sales_20"].sum()
        else 0
    )

    st.markdown(
        """
        ### Recommended Pilot: Review Tables Discount & Product Mix

        Karena average discount Tables berada di atas 20%, lakukan review
        terhadap produk Tables dengan negative profit dan uji discount
        maksimum 20%.
        """
    )

    t5, t6, t7 = st.columns(3)

    t5.metric(
        "Current Profit",
        money(tables_profit)
    )

    t6.metric(
        "Scenario Profit @20%",
        money(tables_profit_20),
        delta=money(tables_profit_improvement)
    )

    t7.metric(
        "Scenario Margin",
        pct(tables_margin_20)
    )

    st.markdown(
        f"""
        **Scenario result:** jika discount Tables diturunkan menuju 20% dan
        volume tetap, profit secara mekanis berubah dari **{money(tables_profit)}**
        menjadi sekitar **{money(tables_profit_20)}**.

        Potential improvement secara simulasi: **{money(tables_profit_improvement)}**.

        Angka ini digunakan sebagai **decision benchmark**, bukan forecast.
        """
    )

    # Top loss-making Tables products
    tables_product = (
        tables_df
        .groupby("Product Name")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean")
        )
        .reset_index()
    )

    tables_product["Margin"] = np.where(
        tables_product["Sales"] != 0,
        tables_product["Profit"] /
        tables_product["Sales"],
        0
    )

    st.markdown("### Products to Review First")

    st.dataframe(
        tables_product
        .sort_values("Profit")
        .head(10)
        .style.format({
            "Sales": "${:,.0f}",
            "Profit": "${:,.0f}",
            "Margin": "{:.1%}",
            "Avg_Discount": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        **Owner:** Product + Commercial + Finance  
        **Timeline:** M1–M3 product diagnosis → M4–M6 pricing pilot → M7–M9 evaluation → M10–M12 policy  
        **KPI:** Tables Profit, Tables Margin, Loss-Making Product Sales
        """
    )

    st.divider()

    # ========================================================
    # PRIORITY 3
    # ========================================================
    st.markdown("## 3️⃣ Regional Profitability Recovery")

    region_df = (
        strategic_df
        .groupby("Region")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean")
        )
        .reset_index()
    )

    region_df["Margin"] = np.where(
        region_df["Sales"] != 0,
        region_df["Profit"] / region_df["Sales"],
        0
    )

    # Southeast Asia
    sea = region_df[
        region_df["Region"] == "Southeast Asia"
    ]

    if not sea.empty:

        sea_sales = sea.iloc[0]["Sales"]
        sea_profit = sea.iloc[0]["Profit"]
        sea_margin = sea.iloc[0]["Margin"]
        sea_discount = sea.iloc[0]["Avg_Discount"]

        # Gap to global margin
        sea_profit_at_global_margin = (
            sea_sales * global_margin
        )

        sea_profit_gap = (
            sea_profit_at_global_margin - sea_profit
        )

    else:
        sea_sales = sea_profit = sea_margin = sea_discount = 0
        sea_profit_gap = 0

    # EMEA
    emea = region_df[
        region_df["Region"] == "EMEA"
    ]

    if not emea.empty:

        emea_sales = emea.iloc[0]["Sales"]
        emea_profit = emea.iloc[0]["Profit"]
        emea_margin = emea.iloc[0]["Margin"]

        emea_profit_at_global_margin = (
            emea_sales * global_margin
        )

        emea_profit_gap = (
            emea_profit_at_global_margin - emea_profit
        )

    else:
        emea_sales = emea_profit = emea_margin = 0
        emea_profit_gap = 0

    st.markdown("### Southeast Asia")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Sales",
        money(sea_sales)
    )

    r2.metric(
        "Current Profit",
        money(sea_profit)
    )

    r3.metric(
        "Current Margin",
        pct(sea_margin)
    )

    r4.metric(
        "Avg Discount",
        pct(sea_discount)
    )

    st.markdown(
        f"""
        Southeast Asia menghasilkan sales sebesar **{money(sea_sales)}**
        tetapi hanya menghasilkan margin **{pct(sea_margin)}**.

        Sebagai benchmark, jika Southeast Asia mencapai **global margin
        {pct(global_margin)}**, profit region tersebut secara matematis akan
        menjadi sekitar **{money(sea_sales * global_margin)}**.

        **Profit gap terhadap benchmark:** sekitar **{money(sea_profit_gap)}**.
        """
    )

    st.markdown(
        "### Countries Requiring Attention"
    )

    sea_country = (
        strategic_df[
            strategic_df["Region"] == "Southeast Asia"
        ]
        .groupby("Country")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean"),
            Shipping_Cost=("Shipping Cost", "sum")
        )
        .reset_index()
    )

    sea_country["Margin"] = np.where(
        sea_country["Sales"] != 0,
        sea_country["Profit"] /
        sea_country["Sales"],
        0
    )

    st.dataframe(
        sea_country
        .sort_values("Profit")
        .head(8)
        .style.format({
            "Sales": "${:,.0f}",
            "Profit": "${:,.0f}",
            "Margin": "{:.1%}",
            "Avg_Discount": "{:.1%}",
            "Shipping_Cost": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### EMEA")

    e1, e2, e3 = st.columns(3)

    e1.metric(
        "EMEA Sales",
        money(emea_sales)
    )

    e2.metric(
        "EMEA Margin",
        pct(emea_margin)
    )

    e3.metric(
        "Profit Gap vs Global Margin",
        money(emea_profit_gap)
    )

    st.markdown(
        f"""
        **Recommended action:** jangan langsung melakukan expansion.
        Fokus terlebih dahulu pada negara dengan sales material tetapi margin
        rendah/negatif.

        Target awal bukan "menyamakan semua negara dengan global margin",
        tetapi menutup sebagian **profit gap** melalui pricing, discount,
        product mix, dan shipping economics.
        """
    )

    st.markdown(
        """
        **Owner:** Regional Commercial + Finance + Operations  
        **Timeline:** M1–M3 country diagnosis → M4–M6 pilot → M7–M9 evaluation → M10–M12 scale  
        **KPI:** Regional Profit, Regional Margin, Profit per Country, Shipping Cost / Sales
        """
    )

    st.divider()

    # ========================================================
    # 12-MONTH ROADMAP
    # ========================================================
    st.markdown("## 🗺️ 12-Month Roadmap")

    roadmap = pd.DataFrame([
        {
            "Period": "M1–M3",
            "Focus": "Diagnose",
            "Main Action": (
                "Identify discount ≥21%, loss-making Tables products, "
                "and low-margin countries"
            ),
            "Output": "Priority list"
        },
        {
            "Period": "M4–M6",
            "Focus": "Pilot",
            "Main Action": (
                "Test 20% discount guardrail, Tables pricing review, "
                "and regional interventions"
            ),
            "Output": "Pilot results"
        },
        {
            "Period": "M7–M9",
            "Focus": "Evaluate",
            "Main Action": (
                "Measure profit, margin and customer retention"
            ),
            "Output": "Validated actions"
        },
        {
            "Period": "M10–M12",
            "Focus": "Scale",
            "Main Action": (
                "Institutionalize successful pricing, product and "
                "regional controls"
            ),
            "Output": "New management routine"
        }
    ])

    st.dataframe(
        roadmap,
        use_container_width=True,
        hide_index=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[1, 2, 3, 4],
            y=[1, 1, 1, 1],
            mode="lines+markers+text",
            line=dict(width=4),
            marker=dict(size=22),
            text=[
                "Diagnose",
                "Pilot",
                "Evaluate",
                "Scale"
            ],
            textposition="top center",
            showlegend=False
        )
    )

    fig.update_layout(
        title="Global Superstore — 12-Month Revival Roadmap",
        xaxis=dict(
            tickmode="array",
            tickvals=[1, 2, 3, 4],
            ticktext=[
                "M1–M3",
                "M4–M6",
                "M7–M9",
                "M10–M12"
            ],
            range=[0.5, 4.5]
        ),
        yaxis=dict(
            visible=False,
            range=[0.7, 1.3]
        ),
        height=300
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ========================================================
    # SUMMARY: PROS & CONS
    # ========================================================
    st.markdown("## ⚖️ Recommendation Trade-Off")

    comparison = pd.DataFrame([
        {
            "Recommendation":
                "1. Control High-Discount Transactions",
            "Kelebihan":
                "Potensi perbaikan profit terbesar; langsung menyasar "
                "kelompok discount ≥21% yang saat ini loss-making.",
            "Kekurangan":
                "Discount lebih rendah dapat menurunkan customer retention "
                "atau volume penjualan."
        },
        {
            "Recommendation":
                "2. Turn Around Tables",
            "Kelebihan":
                "Masalah sangat spesifik dan mudah ditelusuri sampai level "
                "product; sales sudah ada tetapi profit negatif.",
            "Kekurangan":
                "Perbaikan hanya berdampak pada satu sub-category dan "
                "beberapa produk mungkin memang sulit dibuat profitable."
        },
        {
            "Recommendation":
                "3. Regional Profitability Recovery",
            "Kelebihan":
                "Menangani profit leakage berdasarkan lokasi dan dapat "
                "dikombinasikan dengan pricing, product mix dan shipping.",
            "Kekurangan":
                "Lebih kompleks karena kondisi tiap negara berbeda dan "
                "perubahan strategi dapat memengaruhi sales."
        }
    ])

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        ### Management Decision Logic

        Ketiga rekomendasi sebaiknya **tidak dijalankan sekaligus secara
        agresif**.

        **M1–M3:** identifikasi area prioritas →  
        **M4–M6:** pilot →  
        **M7–M9:** ukur hasil →  
        **M10–M12:** scale hanya tindakan yang terbukti memperbaiki
        profit/margin tanpa penurunan volume yang tidak dapat diterima.

        """
    )
    
# ============================================================
# TAB 6 — AI Recommendation
# ============================================================
with tab6:
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
    api_key = st.secrets["GOOGLE_API_KEY"]

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
                model="gemini-3.7-flash",
                google_api_key=api_key,
                temperature=0.5,
            )
            chain = prompt | llm
            response = chain.invoke({"context": context})
            st.markdown(response.content)
        except Exception:
            st.warning(
                "🤖 **AI Recommendation sedang tidak tersedia.**\n\n"
                "Analisis dashboard tetap dapat digunakan. "
                "Silakan coba generate rekomendasi kembali beberapa saat lagi."
            )

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

# ============================================================
# TAB 7 — AI CHART MAKER
# ============================================================
with tab7:

    st.subheader("🤖 AI Chart Maker")

    st.caption(
        "Masukkan pertanyaan bisnis. AI akan menentukan variabel yang relevan, "
        "kemudian beberapa alternatif visualisasi akan dibuat agar Anda dapat "
        "memilih chart yang paling sesuai."
    )

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except (KeyError, FileNotFoundError):
            api_key = None

    # --------------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------------

    user_request = st.text_area(
        "Apa yang ingin Anda analisis?",
        placeholder=(
            "Contoh: "
            "Saya ingin melihat apakah discount tinggi "
            "berhubungan dengan profit rendah."
        ),
        height=100
    )

    # --------------------------------------------------------
    # EXAMPLE QUESTIONS
    # --------------------------------------------------------

    st.markdown("### 💡 Contoh Pertanyaan")

    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        st.info(
            "📉 **Profitability**\n\n"
            "Apa hubungan discount dengan profit?"
        )

    with ex2:
        st.info(
            "🌍 **Geography**\n\n"
            "Negara mana yang memiliki profit paling rendah?"
        )

    with ex3:
        st.info(
            "📦 **Product**\n\n"
            "Bagaimana sales dan profit tiap sub-category?"
        )

    # --------------------------------------------------------
    # DATA SCHEMA
    # --------------------------------------------------------

    data_schema = {
        "columns": list(strategic_df.columns),
        "numeric_columns": strategic_df.select_dtypes(
            include=np.number
        ).columns.tolist(),
        "categorical_columns": strategic_df.select_dtypes(
            exclude=np.number
        ).columns.tolist(),
    }

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if st.button(
        "🚀 Analyze with AI",
        type="primary",
        use_container_width=True
    ):

        if not user_request.strip():
            st.warning(
                "Masukkan pertanyaan bisnis terlebih dahulu."
            )
            st.stop()

        if not api_key:
            st.error(
                "GOOGLE_API_KEY belum dikonfigurasi."
            )
            st.stop()

        # ----------------------------------------------------
        # AI PROMPT
        # ----------------------------------------------------

        system_prompt = """
You are an expert business data analyst.

Your task is to understand the user's business question
and identify the most relevant columns from the dataset.

IMPORTANT:
You are NOT choosing the chart type.

You ONLY determine:
1. Main dimension / X variable
2. Main metric / Y variable
3. Optional grouping variable
4. Appropriate aggregation
5. Business interpretation

Use ONLY columns available in the dataset.

Available columns:
{columns}

Numeric columns:
{numeric_columns}

Categorical columns:
{categorical_columns}

Rules:

- Never invent column names.
- If the question asks "which", use a categorical dimension.
- If the question asks about sales/profit/margin, use the relevant metric.
- If the question asks about relationship/correlation between two numeric
  variables, identify both numeric variables.
- If the question asks about trends over time, use the appropriate date/year
  column.
- Prefer Profit, Sales, Quantity, Discount, Shipping Cost when relevant.
- Aggregation should normally be "sum" for Sales, Profit, Quantity,
  Shipping Cost and "mean" for Discount.
- Do not calculate results.
- Do not invent numbers.

Return ONLY valid JSON:

{
  "x": "column name",
  "y": "column name",
  "color": "column name or null",
  "aggregation": "sum or mean or count",
  "title": "short title",
  "business_question": "what the visualization is trying to answer",
  "interpretation": "why these variables are relevant"
}
"""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                system_prompt
            ),
            (
                "human",
                """
Dataset schema:

Columns:
{columns}

Numeric columns:
{numeric_columns}

Categorical columns:
{categorical_columns}

User question:
{user_request}
"""
            )
        ])

        try:

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.1,
            )

            chain = prompt | llm

            response = chain.invoke({
                "columns": ", ".join(
                    data_schema["columns"]
                ),
                "numeric_columns": ", ".join(
                    data_schema["numeric_columns"]
                ),
                "categorical_columns": ", ".join(
                    data_schema["categorical_columns"]
                ),
                "user_request": user_request
            })

            # ------------------------------------------------
            # EXTRACT TEXT
            # ------------------------------------------------

            content = response.content

            if isinstance(content, list):

                text_parts = []

                for item in content:

                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                    ):
                        text_parts.append(
                            item.get("text", "")
                        )

                    elif isinstance(item, str):
                        text_parts.append(item)

                content = "\n".join(text_parts)

            if not isinstance(content, str):
                content = str(content)

            # ------------------------------------------------
            # CLEAN JSON
            # ------------------------------------------------

            content = content.strip()

            if content.startswith("```"):
                content = (
                    content
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

            chart_spec = json.loads(content)

            # ------------------------------------------------
            # VALIDATE COLUMNS
            # ------------------------------------------------

            if chart_spec["x"] not in strategic_df.columns:
                raise ValueError(
                    f"Invalid X column: {chart_spec['x']}"
                )

            if (
                chart_spec.get("y")
                and chart_spec["y"] not in strategic_df.columns
            ):
                raise ValueError(
                    f"Invalid Y column: {chart_spec['y']}"
                )

            if (
                chart_spec.get("color")
                and chart_spec["color"] not in strategic_df.columns
            ):
                raise ValueError(
                    f"Invalid color column: {chart_spec['color']}"
                )

            # ------------------------------------------------
            # SHOW AI ANALYSIS
            # ------------------------------------------------

            st.success(
                "AI berhasil memahami pertanyaan Anda."
            )

            st.markdown("## 🧠 AI Analysis")

            st.markdown(
                f"""
                **Business Question**

                {chart_spec.get("business_question", "-")}

                **Recommended Variables**

                - **X / Dimension:** `{chart_spec["x"]}`
                - **Y / Metric:** `{chart_spec.get("y", "-")}`
                - **Grouping:** `{chart_spec.get("color") or "-"}`
                - **Aggregation:** `{chart_spec.get("aggregation", "sum")}`

                **Why these variables?**

                {chart_spec.get("interpretation", "-")}
                """
            )

            st.divider()

            # ------------------------------------------------
            # PREPARE DATA
            # ------------------------------------------------

            chart_df = strategic_df.copy()

            x_col = chart_spec["x"]
            y_col = chart_spec.get("y")
            color_col = chart_spec.get("color")
            aggregation = chart_spec.get(
                "aggregation",
                "sum"
            )

            # =================================================
            # AGGREGATED DATA
            # =================================================

            if y_col:

                if aggregation == "mean":

                    grouped_df = (
                        chart_df
                        .groupby(
                            x_col,
                            as_index=False
                        )[y_col]
                        .mean()
                    )

                elif aggregation == "count":

                    grouped_df = (
                        chart_df
                        .groupby(x_col)
                        .size()
                        .reset_index(
                            name="Count"
                        )
                    )

                    y_plot = "Count"

                else:

                    grouped_df = (
                        chart_df
                        .groupby(
                            x_col,
                            as_index=False
                        )[y_col]
                        .sum()
                    )

                if aggregation != "count":
                    y_plot = y_col

                # Top 20 for categorical charts
                if not pd.api.types.is_numeric_dtype(
                    grouped_df[x_col]
                ):
                    grouped_df = (
                        grouped_df
                        .sort_values(
                            y_plot,
                            ascending=False
                        )
                        .head(20)
                    )

            # =================================================
            # CHART OPTIONS
            # =================================================

            st.markdown("## 📊 Choose Your Visualization")

            chart_tabs = st.tabs([
                "📊 Bar",
                "📈 Line",
                "🔵 Scatter",
                "🥧 Pie",
                "🌊 Area",
                "📦 Box"
            ])

            # =================================================
            # 1. BAR
            # =================================================

            with chart_tabs[0]:

                if y_col:

                    bar_df = grouped_df.copy()

                    fig_bar = px.bar(
                        bar_df,
                        x=x_col,
                        y=y_plot,
                        color=color_col,
                        title=(
                            f"{chart_spec.get('title', 'Analysis')} "
                            f"— Bar Chart"
                        )
                    )

                    fig_bar.update_layout(
                        height=500
                    )

                    st.plotly_chart(
                        fig_bar,
                        use_container_width=True
                    )

                    st.caption(
                        "Cocok untuk membandingkan nilai antar kategori."
                    )

                else:

                    st.warning(
                        "Bar chart membutuhkan metric."
                    )

            # =================================================
            # 2. LINE
            # =================================================

            with chart_tabs[1]:

                if y_col:

                    line_df = grouped_df.copy()

                    fig_line = px.line(
                        line_df,
                        x=x_col,
                        y=y_plot,
                        color=color_col,
                        title=(
                            f"{chart_spec.get('title', 'Analysis')} "
                            f"— Line Chart"
                        ),
                        markers=True
                    )

                    fig_line.update_layout(
                        height=500
                    )

                    st.plotly_chart(
                        fig_line,
                        use_container_width=True
                    )

                    st.caption(
                        "Cocok untuk melihat pola atau perubahan sepanjang waktu."
                    )

                else:

                    st.warning(
                        "Line chart membutuhkan metric."
                    )

            # =================================================
            # 3. SCATTER
            # =================================================

            with chart_tabs[2]:

                if (
                    y_col
                    and pd.api.types.is_numeric_dtype(
                        chart_df[x_col]
                    )
                    and pd.api.types.is_numeric_dtype(
                        chart_df[y_col]
                    )
                ):

                    fig_scatter = px.scatter(
                        chart_df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=(
                            f"{chart_spec.get('title', 'Analysis')} "
                            f"— Scatter Plot"
                        ),
                        opacity=0.55
                    )

                    fig_scatter.update_layout(
                        height=500
                    )

                    st.plotly_chart(
                        fig_scatter,
                        use_container_width=True
                    )

                    st.caption(
                        "Cocok untuk melihat hubungan antara dua variabel numerik."
                    )

                else:

                    st.warning(
                        "Scatter plot membutuhkan dua variabel numerik."
                    )

            # =================================================
            # 4. PIE
            # =================================================

            with chart_tabs[3]:

                if y_col:

                    pie_df = grouped_df.copy()

                    pie_df = (
                        pie_df
                        .sort_values(
                            y_plot,
                            ascending=False
                        )
                        .head(10)
                    )

                    fig_pie = px.pie(
                        pie_df,
                        names=x_col,
                        values=y_plot,
                        title=(
                            f"{chart_spec.get('title', 'Analysis')} "
                            f"— Pie Chart"
                        )
                    )

                    fig_pie.update_layout(
                        height=500
                    )

                    st.plotly_chart(
                        fig_pie,
                        use_container_width=True
                    )

                    st.caption(
                        "Cocok untuk melihat komposisi dari beberapa kategori."
                    )

                else:

                    st.warning(
                        "Pie chart membutuhkan metric."
                    )

            # =================================================
            # 5. AREA
            # =================================================

            with chart_tabs[4]:

                if y_col:

                    area_df = grouped_df.copy()

                    fig_area = px.area(
                        area_df,
                        x=x_col,
                        y=y_plot,
                        color=color_col,
                        title=(
                            f"{chart_spec.get('title', 'Analysis')} "
                            f"— Area Chart"
                        )
                    )

                    fig_area.update_layout(
                        height=500
                    )

                    st.plotly_chart(
                        fig_area,
                        use_container_width=True
                    )

                    st.caption(
                        "Cocok untuk melihat perubahan nilai dan volume sepanjang waktu."
                    )

                else:

                    st.warning(
                        "Area chart membutuhkan metric."
                    )

            # =================================================
            # 6. BOX
            # =================================================

            with chart_tabs[5]:

                if (
                    y_col
                    and not pd.api.types.is_numeric_dtype(
                        chart_df[x_col]
                    )
                ):

                    # Limit categories for readability
                    top_categories = (
                        chart_df[x_col]
                        .value_counts()
                        .head(15)
                        .index
                    )

                    box_df = chart_df[
                        chart_df[x_col]
                        .isin(top_categories)
                    ]

                    fig_box = px.box(
                        box_df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=(
                            f"{chart_spec.get('title', 'Analysis')} "
                            f"— Box Plot"
                        )
                    )

                    fig_box.update_layout(
                        height=500
                    )

                    st.plotly_chart(
                        fig_box,
                        use_container_width=True
                    )

                    st.caption(
                        "Cocok untuk melihat distribusi, median, dan outlier."
                    )

                else:

                    st.warning(
                        "Box plot membutuhkan kategori dan metric numerik."
                    )

            # ------------------------------------------------
            # DATA PREVIEW
            # ------------------------------------------------

            st.divider()

            with st.expander(
                "🔎 View AI Specification"
            ):

                st.json(chart_spec)

            with st.expander(
                "📋 View Aggregated Data"
            ):

                if y_col:
                    st.dataframe(
                        grouped_df,
                        use_container_width=True,
                        hide_index=True
                    )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Global Superstore Revival Strategy Dashboard • Built with Streamlit + Plotly + LangChain + Gemini"
)
