import streamlit as st
import pandas as pd
import plotly.express as px
from db.database import ProcurementDB
from db.exporters import export_excel, export_html_report, export_json_payload
from analysis.analytics_engine import AnalyticsEngine


def analytics_tab():
    st.header("📊 Analytics Dashboard")

    db = ProcurementDB()
    runs = db.get_runs()

    if not runs:
        st.warning(
            "⚠️ No saved batches yet. Process supplier files in the **File Processing** tab first — "
            "data is saved automatically."
        )
        return

    engine = AnalyticsEngine(db)

    opp_engine = st.session_state.get("opportunity_engine")
    internal_data = opp_engine.internal_data if opp_engine and hasattr(opp_engine, "internal_data") else []

    st.caption(f"**{len(runs)} batch(es) stored** · Latest: {runs[0]['run_at'][:16]}")

    sub1, sub2, sub3, sub4 = st.tabs(
        ["📈 Price Trends", "⚠️ Stockout Risk", "🏆 Supplier Scorecard", "🏷️ Brand Health"]
    )

    with sub1:
        st.subheader("📈 Price Trends")

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            ean_filter = st.text_input("EAN search", placeholder="e.g. 3433...", key="pt_ean")
        with col_f2:
            all_brands = sorted({
                r.get("brand", "") for r in db.get_all_supplier_prices() if r.get("brand")
            })
            brand_filter = st.selectbox("Brand", ["All"] + all_brands, key="pt_brand")
        with col_f3:
            all_suppliers = sorted({
                r.get("supplier", "") for r in db.get_all_supplier_prices() if r.get("supplier")
            })
            supplier_filter = st.multiselect("Suppliers", all_suppliers, key="pt_supplier")
        with col_f4:
            days_options = {"30 days": 30, "90 days": 90, "All": 3650}
            days_label = st.selectbox("Time range", list(days_options.keys()), index=1, key="pt_days")
            days = days_options[days_label]

        threshold = st.slider("Price alert threshold (%)", 1, 20, 5, key="pt_threshold")

        df_trends = engine.compute_price_trends(
            ean=ean_filter or None,
            brand=brand_filter if brand_filter != "All" else None,
            supplier=None,
            days=days,
        )
        if supplier_filter:
            df_trends = df_trends[df_trends["supplier"].isin(supplier_filter)]

        if df_trends.empty:
            st.info("No price history found for the selected filters.")
        else:
            fig = px.line(
                df_trends,
                x="run_at",
                y="price_net",
                color="supplier",
                facet_col="ean" if not ean_filter else None,
                markers=True,
                labels={"run_at": "Batch date", "price_net": "Price (€)", "supplier": "Supplier"},
                title="Price over time by supplier",
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

            alerts = engine.compute_price_trend_alerts(threshold_pct=float(threshold))
            if alerts:
                st.warning(f"**{len(alerts)} price alert(s)** — movement ≥ {threshold}% between batches")
                alert_df = pd.DataFrame(alerts)[["ean", "supplier", "brand", "prev_price", "current_price", "pct_change", "direction"]]
                st.dataframe(alert_df, use_container_width=True)

    with sub2:
        st.subheader("⚠️ Stockout Risk")

        if not internal_data:
            st.info("Load internal data in the **Opportunities** tab to see stockout risk.")
        else:
            df_stock = engine.compute_stockout_risk(internal_data)

            critical = df_stock[df_stock["urgency"] == "Critical"]
            warning = df_stock[df_stock["urgency"] == "Warning"]
            ok = df_stock[df_stock["urgency"] == "OK"]

            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Critical (<7 days)", len(critical))
            c2.metric("🟠 Warning (7–30 days)", len(warning))
            c3.metric("🟢 OK (>30 days)", len(ok))

            def color_urgency(val):
                colors = {"Critical": "background-color:#3a0000;color:#ff6b6b",
                          "Warning": "background-color:#3a2000;color:#ffa040",
                          "OK": "background-color:#003a00;color:#7cf5a0"}
                return colors.get(val, "")

            styled = df_stock.style.map(color_urgency, subset=["urgency"])
            st.dataframe(styled, use_container_width=True, height=400)

            stockout_bytes = export_excel([], pd.DataFrame(), df_stock, pd.DataFrame(), pd.DataFrame())
            st.download_button(
                "📥 Download Stockout Report (Excel)",
                data=stockout_bytes,
                file_name=f"stockout_risk_{runs[0]['run_at'][:10]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with sub3:
        st.subheader("🏆 Supplier Scorecard")
        df_sup = engine.compute_supplier_win_rates()

        if df_sup.empty:
            st.info("Not enough batch history yet. Win rates are more meaningful after 2+ batches.")
        else:
            fig_win = px.bar(
                df_sup,
                x="win_rate_pct",
                y="supplier",
                orientation="h",
                color="win_rate_pct",
                color_continuous_scale="Greens",
                labels={"win_rate_pct": "Win rate %", "supplier": ""},
                title="Win rate — % of EANs where supplier was cheapest",
            )
            fig_win.update_layout(height=max(300, len(df_sup) * 40), showlegend=False)
            st.plotly_chart(fig_win, use_container_width=True)

            display_cols = ["supplier", "sku_count", "win_rate_pct", "avg_price_index", "price_stability"]
            st.dataframe(df_sup[display_cols], use_container_width=True)

            sup_bytes = export_excel([], pd.DataFrame(), pd.DataFrame(), df_sup, pd.DataFrame())
            st.download_button(
                "📥 Download Supplier Scorecard (Excel)",
                data=sup_bytes,
                file_name=f"supplier_scorecard_{runs[0]['run_at'][:10]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with sub4:
        st.subheader("🏷️ Brand Health")

        if not internal_data:
            st.info("Load internal data in the **Opportunities** tab to see brand health.")
        else:
            df_brand = engine.compute_brand_health(internal_data)

            if df_brand.empty:
                st.info("No brand data available.")
            else:
                def brand_trend_arrow(val):
                    if val > 1:
                        return f"↑ {val:.1f}%"
                    elif val < -1:
                        return f"↓ {abs(val):.1f}%"
                    return f"→ {abs(val):.1f}%"

                df_display = df_brand.copy()
                df_display["price_trend"] = df_display["price_trend_pct"].apply(brand_trend_arrow)

                st.dataframe(
                    df_display[["brand", "sku_count", "price_trend", "at_risk_skus", "coverage_pct", "top_opportunity"]],
                    use_container_width=True,
                    height=400,
                )

                brand_bytes = export_excel([], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), df_brand)
                st.download_button(
                    "📥 Download Brand Health (Excel)",
                    data=brand_bytes,
                    file_name=f"brand_health_{runs[0]['run_at'][:10]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    st.divider()
    st.subheader("📤 Full Exports")
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        if st.button("📊 Generate Full Excel Workbook"):
            opp_list = list(st.session_state.get("processed_data") or [])
            df_trends_full = engine.compute_price_trends(days=90)
            df_stock_full = engine.compute_stockout_risk(internal_data) if internal_data else pd.DataFrame()
            df_sup_full = engine.compute_supplier_win_rates()
            df_brand_full = engine.compute_brand_health(internal_data) if internal_data else pd.DataFrame()
            xlsx_bytes = export_excel(opp_list, df_trends_full, df_stock_full, df_sup_full, df_brand_full)
            st.download_button(
                "⬇️ Download Full Workbook",
                data=xlsx_bytes,
                file_name=f"procurement_report_{runs[0]['run_at'][:10]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with exp_col2:
        if st.button("📄 Generate HTML Report"):
            opp_list = list(st.session_state.get("processed_data") or [])
            df_stock_full = engine.compute_stockout_risk(internal_data).to_dict("records") if internal_data else []
            df_sup_full = engine.compute_supplier_win_rates().to_dict("records")
            df_brand_full = engine.compute_brand_health(internal_data).to_dict("records") if internal_data else []
            batch_id = st.session_state.get("current_run_id", "unknown")
            html_str = export_html_report(opp_list, df_stock_full, df_sup_full, df_brand_full, batch_id)
            st.download_button(
                "⬇️ Download HTML Report",
                data=html_str.encode("utf-8"),
                file_name=f"procurement_report_{runs[0]['run_at'][:10]}.html",
                mime="text/html",
            )

    with exp_col3:
        if st.button("🤖 Generate JSON Payload (LLM-ready)"):
            opp_list = list(st.session_state.get("processed_data") or [])
            df_stock_full = engine.compute_stockout_risk(internal_data).to_dict("records") if internal_data else []
            df_sup_full = engine.compute_supplier_win_rates().to_dict("records")
            alerts = engine.compute_price_trend_alerts()
            batch_id = st.session_state.get("current_run_id", "unknown")
            json_str = export_json_payload(opp_list, alerts, df_stock_full, df_sup_full, batch_id)
            st.download_button(
                "⬇️ Download JSON Payload",
                data=json_str.encode("utf-8"),
                file_name=f"procurement_payload_{runs[0]['run_at'][:10]}.json",
                mime="application/json",
            )
