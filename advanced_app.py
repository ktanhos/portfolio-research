import pandas as pd
import streamlit as st


def render_advanced_section(advanced_results, title=None):
    """Hiển thị trực tiếp kết quả advanced_portfolio_analysis từ run_research."""
    if not isinstance(advanced_results, dict) or not advanced_results:
        st.info("Không đủ dữ liệu hiện có để thực hiện phân tích nâng cao.")
        return

    sections = [
        ("11.1. PHÂN TÍCH YẾU TỐ", "factor_proxy"),
        ("11.2. HỒI QUY ĐA YẾU TỐ", "multifactor_regression"),
        ("11.3. PHÂN TÍCH DANH MỤC CHỦ ĐỘNG", "active_analysis"),
        ("11.4. VAR VÀ CVAR", "var_analysis"),
        ("11.5. KIỂM ĐỊNH ĐỘ BỀN", "robustness"),
        ("11.6. KIỂM ĐỊNH NGOÀI MẪU", "walk_forward"),
    ]

    for heading, key in sections:
        st.markdown(f"**{heading}**")
        table = advanced_results.get(key, pd.DataFrame())
        if isinstance(table, pd.DataFrame) and not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=False)
        else:
            error_key = f"{key}_error"
            error = advanced_results.get(error_key)
            if error:
                st.warning(f"Không thể thực hiện: {error}")
            else:
                st.info("Không đủ dữ liệu hiện có để thực hiện phân tích này.")
