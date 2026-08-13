import contextlib
import io
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from portfolio_engine import (
    configure_vnstock,
    run_research,
    clean_margin_table,
)

st.set_page_config(
    page_title="Portfolio Research",
    page_icon="📊",
    layout="wide",
)

st.title("PORTFOLIO RESEARCH")
st.caption("Phân tích và tối ưu danh mục cổ phiếu Việt Nam")

# ------------------------------------------------------------
# CẤU HÌNH
# ------------------------------------------------------------

st.subheader("1. CẤU HÌNH NGHIÊN CỨU")

left, right = st.columns(2)

with left:
    api_key = st.text_input(
        "API key Vnstock",
        type="password",
        placeholder="Nhập API key của bạn",
    )

    tickers_text = st.text_input(
        "Danh sách mã cổ phiếu",
        value="GMD, VCG, CTR, HAH, HPG, DGC",
    )

    start_date = st.date_input(
        "Từ ngày",
        value=date(2022, 1, 1),
    )

    end_date = st.date_input(
        "Đến ngày",
        value=date.today(),
    )

with right:
    risk_free_rate = st.number_input(
        "Lãi suất phi rủi ro",
        min_value=0.0,
        max_value=1.0,
        value=0.04,
        step=0.005,
        format="%.3f",
    )

    target_return = st.number_input(
        "Lợi nhuận mục tiêu",
        min_value=-1.0,
        max_value=3.0,
        value=0.15,
        step=0.01,
        format="%.2f",
    )

    risk_profile = st.selectbox(
        "Khẩu vị rủi ro",
        ["Thận trọng", "Cân bằng", "Tăng trưởng"],
        index=1,
    )

    benchmark = st.text_input(
        "Benchmark",
        value="VNINDEX",
    )

risk_aversion_map = {
    "Thận trọng": 5.0,
    "Cân bằng": 3.0,
    "Tăng trưởng": 1.0,
}

risk_aversion = risk_aversion_map[risk_profile]

# ------------------------------------------------------------
# ĐÒN BẨY
# ------------------------------------------------------------

st.subheader("2. ĐÒN BẨY")

use_leverage = st.checkbox(
    "Sử dụng đòn bẩy",
    value=False,
)

margin_table = None
max_leverage = 1.0

tickers = list(dict.fromkeys([
    x.strip().upper()
    for x in tickers_text.split(",")
    if x.strip()
]))

if use_leverage:

    max_leverage = st.number_input(
        "Tổng vị thế tối đa",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
    )

    st.caption(
        "1,0 lần = không vay | "
        "1,5 lần = vay 50% vốn tự có | "
        "2,0 lần = vay 100% vốn tự có"
    )

    default_margin = pd.DataFrame({
        "Mã": tickers,
        "Được cấp margin": ["Không"] * len(tickers),
        "Vốn vay / vốn tự có (%)": [0.0] * len(tickers),
        "Lãi suất vay (%)": [0.0] * len(tickers),
        "Ngày cập nhật": [""] * len(tickers),
    })

    edited_margin = st.data_editor(
        default_margin,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mã": st.column_config.TextColumn(
                "Mã",
                disabled=True,
            ),
            "Được cấp margin": st.column_config.SelectboxColumn(
                "Được cấp margin",
                options=["Không", "Có"],
            ),
            "Vốn vay / vốn tự có (%)": st.column_config.NumberColumn(
                "Vốn vay / vốn tự có (%)",
                min_value=0.0,
                max_value=500.0,
                step=5.0,
            ),
            "Lãi suất vay (%)": st.column_config.NumberColumn(
                "Lãi suất vay (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
            ),
            "Ngày cập nhật": st.column_config.TextColumn(
                "Ngày cập nhật",
                help="Ngày thông tin margin được xác nhận hoặc có hiệu lực.",
            ),
        },
        key="margin_editor",
    )

    margin_table = edited_margin.rename(columns={
        "Vốn vay / vốn tự có (%)": "Tỷ lệ cho vay",
        "Lãi suất vay (%)": "Lãi suất vay",
    }).copy()

    margin_table["Tỷ lệ cho vay"] = (
        pd.to_numeric(
            margin_table["Tỷ lệ cho vay"],
            errors="coerce",
        ).fillna(0) / 100
    )

    margin_table["Lãi suất vay"] = (
        pd.to_numeric(
            margin_table["Lãi suất vay"],
            errors="coerce",
        ).fillna(0) / 100
    )

# ------------------------------------------------------------
# CHẠY
# ------------------------------------------------------------

st.markdown("---")

run_analysis = st.button(
    "CHẠY PHÂN TÍCH",
    type="primary",
    use_container_width=True,
)

if run_analysis:

    if len(tickers) < 2:
        st.error("Cần ít nhất 2 mã cổ phiếu.")
        st.stop()

    if start_date >= end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()

    if not api_key.strip():
        st.error("Vui lòng nhập API key Vnstock.")
        st.stop()

    try:

        start_text = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end_text = pd.Timestamp(end_date).strftime("%Y-%m-%d")

        with st.status(
            "Đang thực hiện phân tích...",
            expanded=True,
        ) as status:

            st.write("Đang xác thực API Vnstock...")
            auth = configure_vnstock(api_key)

            st.write("Đang lấy dữ liệu và chạy mô hình...")

            # run_research hiện có các biểu đồ matplotlib.
            # Tạm chặn plt.show để thu toàn bộ figure rồi hiển thị
            # theo bố cục Streamlit.
            figures = []

            original_show = plt.show

            def capture_show(*args, **kwargs):
                figures.extend(
                    [
                        plt.figure(num)
                        for num in plt.get_fignums()
                    ]
                )
                plt.close("all")

            plt.show = capture_show

            log_buffer = io.StringIO()

            try:
                with contextlib.redirect_stdout(log_buffer):

                    results = run_research(
                        tickers=tickers,
                        start_date=start_text,
                        end_date=end_text,
                        risk_free_rate=float(risk_free_rate),
                        risk_aversion=float(risk_aversion),
                        benchmark=benchmark.strip().upper(),
                        leverage=use_leverage,
                        margin_table=margin_table,
                        max_leverage=float(max_leverage),
                        include_company=True,
                        include_income=True,
                        target_return=float(target_return),
                        api_authenticated=auth["authenticated"],
                    )
            finally:
                plt.show = original_show

            status.update(
                label="Phân tích hoàn tất",
                state="complete",
            )

        st.success("Đã hoàn tất phân tích.")

        # ----------------------------------------------------
        # 3. TỔNG QUAN DOANH NGHIỆP
        # ----------------------------------------------------

        company_table = results.get(
            "company_table",
            pd.DataFrame(),
        )

        st.subheader("3. TỔNG QUAN DOANH NGHIỆP")

        if not company_table.empty:
            st.dataframe(
                company_table,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # 4. VNINDEX
        # ----------------------------------------------------

        st.subheader("4. VNINDEX")

        benchmark_summary = results.get(
            "benchmark_summary",
            pd.DataFrame(),
        )

        if not benchmark_summary.empty:
            st.dataframe(
                benchmark_summary,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # 5. PHÂN TÍCH TỪNG CỔ PHIẾU
        # ----------------------------------------------------

        st.subheader("5. PHÂN TÍCH TỪNG CỔ PHIẾU")

        asset_summary = results.get(
            "asset_summary",
            pd.DataFrame(),
        )

        if not asset_summary.empty:
            st.dataframe(
                asset_summary,
                use_container_width=True,
            )

        st.markdown("**Ma trận tương quan**")

        correlation = results.get(
            "correlation",
            pd.DataFrame(),
        )

        if not correlation.empty:
            st.dataframe(
                correlation,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # 6. TỐI ƯU DANH MỤC
        # ----------------------------------------------------

        st.subheader("6. TỐI ƯU DANH MỤC")

        portfolio_table = results.get(
            "portfolio_table",
            pd.DataFrame(),
        )

        if not portfolio_table.empty:
            st.dataframe(
                portfolio_table,
                use_container_width=True,
                hide_index=True,
            )

        weights = results.get(
            "weights",
            pd.DataFrame(),
        )

        if not weights.empty:
            st.markdown("**Phân bổ tỷ trọng**")
            st.dataframe(
                weights,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # 6.1 / 6.2 ĐÒN BẨY
        # ----------------------------------------------------

        if use_leverage:

            st.subheader("6.1. PHÂN TÍCH CÓ ĐÒN BẨY")

            levered_table = results.get(
                "levered_table",
                pd.DataFrame(),
            )

            if not levered_table.empty:
                st.dataframe(
                    levered_table,
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("6.2. PHÂN BỔ DANH MỤC CÓ ĐÒN BẨY")

            levered_alloc = results.get(
                "levered_alloc",
                pd.DataFrame(),
            )

            if not levered_alloc.empty:
                st.dataframe(
                    levered_alloc,
                    use_container_width=True,
                    hide_index=True,
                )

        # ----------------------------------------------------
        # 6.3 TARGET RETURN
        # ----------------------------------------------------

        st.subheader("6.3. MỤC TIÊU LỢI NHUẬN")

        target_summary = results.get(
            "target_summary",
            None,
        )

        if isinstance(target_summary, pd.DataFrame) and not target_summary.empty:
            st.dataframe(
                target_summary,
                use_container_width=True,
                hide_index=True,
            )

        target_check = results.get(
            "target_check",
            pd.DataFrame(),
        )

        if not target_check.empty:
            st.dataframe(
                target_check,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # 7. SO SÁNH VỚI VNINDEX
        # ----------------------------------------------------

        st.subheader("7. SO SÁNH VỚI VNINDEX")

        comparison = results.get(
            "comparison",
            pd.DataFrame(),
        )

        if not comparison.empty:
            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # 8. CÁC CHỈ TIÊU RỦI RO
        # ----------------------------------------------------

        st.subheader("8. PHÂN TÍCH RỦI RO")

        attainment = results.get(
            "attainment",
            pd.DataFrame(),
        )

        if isinstance(attainment, pd.DataFrame) and not attainment.empty:
            st.markdown("**Khả năng đạt mục tiêu**")
            st.dataframe(
                attainment,
                use_container_width=True,
                hide_index=True,
            )

        concentration = results.get(
            "concentration",
            pd.DataFrame(),
        )

        if isinstance(concentration, pd.DataFrame) and not concentration.empty:
            st.markdown("**Mức độ tập trung danh mục**")
            st.dataframe(
                concentration,
                use_container_width=True,
                hide_index=True,
            )

        risk_diagnostics = results.get(
            "risk_diagnostics",
            pd.DataFrame(),
        )

        if isinstance(risk_diagnostics, pd.DataFrame) and not risk_diagnostics.empty:
            st.markdown("**Max Drawdown và thời gian phục hồi**")
            st.dataframe(
                risk_diagnostics,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # 9. KẾT LUẬN ĐỊNH LƯỢNG
        # ----------------------------------------------------

        st.subheader("9. ĐÁNH GIÁ THEO MỤC TIÊU NHÀ ĐẦU TƯ")

        conclusion = results.get(
            "conclusion",
            pd.DataFrame(),
        )

        if isinstance(conclusion, pd.DataFrame) and not conclusion.empty:
            st.dataframe(
                conclusion,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # 10. BIỂU ĐỒ
        # ----------------------------------------------------

        st.subheader("10. BIỂU ĐỒ")

        for i, fig in enumerate(figures, start=1):

            st.pyplot(
                fig,
                clear_figure=True,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        with st.expander("Thông tin xử lý"):

            log_text = log_buffer.getvalue().strip()

            if log_text:
                st.text(log_text)
            else:
                st.write("Không có thông báo bổ sung.")

    except Exception as e:

        st.error(
            f"{type(e).__name__}: {str(e)}"
        )

