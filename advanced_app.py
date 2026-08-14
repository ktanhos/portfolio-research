import pandas as pd
import streamlit as st


def render_advanced_section(
    advanced_results,
    title="11. PHÂN TÍCH NÂNG CAO",
):
    st.subheader(title)
    st.caption(
        "Các phân tích dưới đây chỉ sử dụng dữ liệu đã có trong lần chạy hiện tại."
    )

    sections = [
        ("11.1. FACTOR ANALYSIS", "factor_proxy"),
        ("11.2. MULTIFACTOR REGRESSION", "multifactor_regression"),
        ("11.3. ACTIVE PORTFOLIO ANALYSIS", "active_analysis"),
        ("11.4. VAR VÀ CVAR", "var_analysis"),
        ("11.5. ROBUSTNESS CHECK", "robustness"),
        ("11.6. WALK FORWARD VÀ OUT OF SAMPLE", "walk_forward"),
    ]

    for heading, key in sections:
        st.markdown(f"**{heading}**")
        table = advanced_results.get(key, pd.DataFrame())
        if isinstance(table, pd.DataFrame) and not table.empty:
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=False,
            )
        else:
            st.info("Không đủ dữ liệu hiện có để thực hiện phân tích này.")
