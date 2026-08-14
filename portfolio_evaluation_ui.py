import pandas as pd
import streamlit as st

from portfolio_evaluation import build_all_portfolio_evaluations, choose_focus_portfolio


def _fmt_pct(x):
    try:
        return "N/A" if pd.isna(x) else f"{float(x) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(x)


def _fmt_num(x):
    try:
        return "N/A" if pd.isna(x) else f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def _display_summary(df):
    if df is None or df.empty:
        return
    out = df.copy()
    percent_cols = {
        "CAGR", "Maximum Drawdown", "OOS Sharpe trung vị", "VaR 95%", "CVaR 95%"
    }
    for col in percent_cols:
        if col in out.columns:
            out[col] = out[col].map(_fmt_pct)
    for col in {"Sharpe", "Information Ratio"}:
        if col in out.columns:
            out[col] = out[col].map(_fmt_num)
    st.dataframe(out, use_container_width=True, hide_index=True)


def render_portfolio_evaluation(results, target_return=0.15, risk_free_rate=0.0):
    analyses, summary = build_all_portfolio_evaluations(
        results,
        target_return=target_return,
        risk_free_rate=risk_free_rate,
    )
    if summary.empty:
        st.info("Không đủ dữ liệu hiện có để đánh giá từng danh mục.")
        return

    focus = choose_focus_portfolio(summary)
    st.subheader("12. ĐÁNH GIÁ CÁC DANH MỤC")
    st.markdown("**So sánh tổng thể các danh mục**")
    _display_summary(summary)

    if focus:
        st.markdown(f"**Danh mục trọng tâm: {focus}**")
        focus_row = summary.loc[summary["Danh mục"] == focus]
        if not focus_row.empty:
            row = focus_row.iloc[0]
            st.write(
                f"Đánh giá {row.get('Đánh giá', 'N/A')}. "
                f"Mục tiêu: {row.get('Mục tiêu', 'N/A')}. "
                f"Rủi ro điều chỉnh: {row.get('Rủi ro điều chỉnh', 'N/A')}. "
                f"Rủi ro giảm giá: {row.get('Rủi ro giảm giá', 'N/A')}. "
                f"So với VNINDEX: {row.get('So với VNINDEX', 'N/A')}. "
                f"Độ ổn định ngoài mẫu: {row.get('Độ ổn định OOS', 'N/A')}."
            )
            warning = row.get("Cảnh báo", "")
            if warning and warning != "Không có cảnh báo nổi bật":
                st.warning(warning)

    st.markdown("**Phân tích từng danh mục**")
    for name in summary["Danh mục"].tolist():
        with st.expander(name, expanded=(name == focus)):
            row = summary.loc[summary["Danh mục"] == name]
            if not row.empty:
                _display_summary(row)
            advanced = analyses.get(name, {})
            if isinstance(advanced, dict) and advanced.get("error"):
                st.warning(advanced["error"])
            warnings = ""
            if not row.empty:
                warnings = row.iloc[0].get("Cảnh báo", "")
            if warnings and warnings != "Không có cảnh báo nổi bật":
                st.warning(warnings)

    st.markdown("**Kết luận so sánh**")
    if focus:
        focus_row = summary.loc[summary["Danh mục"] == focus].iloc[0]
        overall = focus_row.get("Đánh giá", "Chưa đủ cơ sở")
        st.write(
            f"Danh mục trọng tâm là {focus}. Kết luận được xây dựng riêng cho danh mục này, "
            f"đồng thời đối chiếu với các nghiệm tối ưu còn lại. Đánh giá hiện tại: {overall}."
        )
