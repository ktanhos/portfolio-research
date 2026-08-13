import contextlib
import io
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from portfolio_engine import (
    configure_vnstock,
    run_research,
)

st.set_page_config(
    page_title="Portfolio Research",
    page_icon="📊",
    layout="wide",
)

st.title("PORTFOLIO RESEARCH")
st.caption("Phân tích và tối ưu danh mục cổ phiếu Việt Nam")


# ============================================================
# ĐỊNH DẠNG HIỂN THỊ
# ============================================================

def fmt_number_vi(value, decimals=2, signed=False):
    if pd.isna(value):
        return "N/A"

    value = float(value)
    sign = "+" if signed and value > 0 else ""

    text = f"{abs(value):,.{decimals}f}"
    text = (
        text
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )

    if value < 0:
        return "-" + text

    return sign + text


def fmt_percent_vi(value, signed=False):
    if pd.isna(value):
        return "N/A"
    return fmt_number_vi(
        float(value) * 100,
        2,
        signed=signed,
    ) + "%"


def fmt_money_vnd(value):
    if pd.isna(value):
        return "N/A"

    value = float(value)
    absolute = abs(value)

    if absolute >= 1e12:
        return (
            fmt_number_vi(value / 1e12, 2)
            + " nghìn tỷ"
        )

    if absolute >= 1e9:
        return (
            fmt_number_vi(value / 1e9, 2)
            + " tỷ"
        )

    if absolute >= 1e6:
        return (
            fmt_number_vi(value / 1e6, 2)
            + " triệu"
        )

    return (
        fmt_number_vi(value, 0)
        + " đồng"
    )


PERCENT_KEYWORDS = [
    "lợi suất",
    "lợi nhuận",
    "rủi ro",
    "biến động",
    "drawdown",
    "alpha",
    "tracking error",
    "vượt vnindex",
    "mục tiêu",
    "chênh lệch",
    "tỷ trọng",
    "tỷ lệ",
    "downside deviation",
    "var",
    "cvar",
    "rolling return",
    "rolling volatility",
    "tổng vị thế",
    "vốn vay",
    "lãi suất vay",
    "chi phí vay",
    "tăng thêm",
    "thay đổi lợi nhuận",
    "thay đổi rủi ro",
    "rủi ro mới",
    "cú sốc",
    "tác động danh mục",
    "mức lỗ danh mục mục tiêu",
    "cú sốc riêng cần thiết",
]

MONEY_COLUMNS = {
    "Vốn hóa",
    "Doanh thu gần nhất",
    "LNST gần nhất",
}

INTEGER_COLUMNS = {
    "Số CP lưu hành",
    "Số tuần",
    "Số phiên",
    "Số quan sát đuôi",
    "Phục hồi (ngày)",
    "Thời gian phục hồi",
}

RATIO_COLUMNS = {
    "P/E",
    "P/B",
    "Sharpe",
    "Sortino",
    "Beta",
    "Tương quan",
    "Tương quan VNINDEX",
    "Information Ratio",
    "HHI",
    "Điểm tham khảo",
    "Số tài sản hiệu dụng",
    "Số tài sản hiệu dụng theo |w|",
    "Thay đổi Sharpe",
    "Rolling Sharpe thấp nhất",
    "Rolling Sharpe trung vị",
    "Rolling Sharpe cao nhất",
}


def is_percent_column(column):
    name = str(column).strip().lower()

    if column in RATIO_COLUMNS:
        return False

    if column in MONEY_COLUMNS or column in INTEGER_COLUMNS:
        return False

    return any(
        keyword in name
        for keyword in PERCENT_KEYWORDS
    )


def render_df(
    df,
    hide_index=True,
    percent_signed_columns=None,
):
    if df is None or df.empty:
        return

    percent_signed_columns = (
        percent_signed_columns or set()
    )

    out = df.copy()

    for column in out.columns:

        if column in MONEY_COLUMNS:
            out[column] = out[column].map(
                fmt_money_vnd
            )

        elif column in INTEGER_COLUMNS:
            out[column] = out[column].map(
                lambda x:
                "N/A"
                if pd.isna(x)
                else fmt_number_vi(x, 0)
            )

        elif column in RATIO_COLUMNS:
            decimals = 3

            if column in {
                "HHI",
                "Số tài sản hiệu dụng",
                "Số tài sản hiệu dụng theo |w|",
            }:
                decimals = 2

            if column == "Rủi ro biên":
                decimals = 4

            if column == "Thay đổi Sharpe":
                decimals = 4

            out[column] = out[column].map(
                lambda x:
                "N/A"
                if pd.isna(x)
                else fmt_number_vi(x, decimals)
            )

        elif is_percent_column(column):
            signed = column in percent_signed_columns
            out[column] = out[column].map(
                lambda x:
                fmt_percent_vi(
                    x,
                    signed=signed,
                )
            )

        elif pd.api.types.is_numeric_dtype(
            out[column]
        ):
            out[column] = out[column].map(
                lambda x:
                "N/A"
                if pd.isna(x)
                else fmt_number_vi(x, 2)
            )

    st.dataframe(
        out,
        use_container_width=True,
        hide_index=hide_index,
    )


def render_weight_table(
    df,
    hide_index=False,
):
    if df is None or df.empty:
        return

    out = df.copy()

    for column in out.columns:
        if column in {
            "Danh mục",
            "Mã",
        }:
            continue

        if pd.api.types.is_numeric_dtype(
            out[column]
        ):
            out[column] = out[column].map(
                lambda x:
                "N/A"
                if pd.isna(x)
                else fmt_percent_vi(x)
            )

    st.dataframe(
        out,
        use_container_width=True,
        hide_index=hide_index,
    )


# ============================================================
# 1. CẤU HÌNH NGHIÊN CỨU
# ============================================================

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

    risk_free_rate_input = st.number_input(
        "Lãi suất phi rủi ro (%)",
        min_value=0.0,
        max_value=100.0,
        value=4.0,
        step=0.25,
        format="%.2f",
        help="Nhập trực tiếp theo phần trăm. Ví dụ 4,00% nhập 4,00.",
    )

    target_return_input = st.number_input(
        "Lợi nhuận mục tiêu (%)",
        min_value=-100.0,
        max_value=300.0,
        value=15.0,
        step=1.0,
        format="%.2f",
        help="Nhập trực tiếp theo phần trăm. Ví dụ 15,00% nhập 15,00.",
    )

    risk_profile = st.selectbox(
        "Khẩu vị rủi ro",
        [
            "Thận trọng",
            "Cân bằng",
            "Tăng trưởng",
        ],
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

risk_aversion = risk_aversion_map[
    risk_profile
]

risk_free_rate = (
    float(risk_free_rate_input) / 100
)

target_return = (
    float(target_return_input) / 100
)

st.caption(
    f"Lãi suất phi rủi ro đang dùng: "
    f"{fmt_percent_vi(risk_free_rate)} | "
    f"Mục tiêu lợi nhuận đang dùng: "
    f"{fmt_percent_vi(target_return)}"
)

# ============================================================
# 2. ĐÒN BẨY
# ============================================================

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
        "Tổng vị thế tối đa (lần)",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
        format="%.1f",
    )

    st.caption(
        "1,0 lần = không vay | "
        "1,5 lần = vay 50% vốn tự có | "
        "2,0 lần = vay 100% vốn tự có"
    )

    if tickers:

        st.markdown("#### Thông số margin từng mã")

        margin_rows = []

        for ticker in tickers:

            c1, c2, c3, c4 = st.columns(
                [1.0, 1.2, 1.5, 1.5]
            )

            with c1:
                st.write(ticker)

            with c2:
                eligible = st.selectbox(
                    "Margin",
                    ["Không", "Có"],
                    key=f"margin_eligible_{ticker}",
                    label_visibility="collapsed",
                )

            with c3:
                loan_pct = st.number_input(
                    "Vay / vốn tự có (%)",
                    min_value=0.0,
                    max_value=500.0,
                    value=0.0,
                    step=5.0,
                    format="%.2f",
                    key=f"loan_{ticker}",
                    label_visibility="collapsed",
                )

            with c4:
                borrowing_rate = st.number_input(
                    "Lãi suất vay (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.25,
                    format="%.2f",
                    key=f"rate_{ticker}",
                    label_visibility="collapsed",
                )

            margin_rows.append({
                "Mã": ticker,
                "Được cấp margin": eligible,
                "Tỷ lệ cho vay": float(loan_pct),
                "Lãi suất vay": float(borrowing_rate),
                "Ngày cập nhật": "",
            })

        margin_table = pd.DataFrame(
            margin_rows
        )

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

# ============================================================
# CHẠY
# ============================================================

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
        st.error(
            "Ngày bắt đầu phải trước ngày kết thúc."
        )
        st.stop()

    if not api_key.strip():
        st.error(
            "Vui lòng nhập API key Vnstock."
        )
        st.stop()

    try:

        start_text = pd.Timestamp(
            start_date
        ).strftime("%Y-%m-%d")

        end_text = pd.Timestamp(
            end_date
        ).strftime("%Y-%m-%d")

        with st.status(
            "Đang thực hiện phân tích...",
            expanded=True,
        ) as status:

            st.write(
                "Đang xác thực API Vnstock..."
            )

            auth = configure_vnstock(
                api_key
            )

            st.write(
                "Đang lấy dữ liệu và chạy mô hình..."
            )

            figures = []

            original_show = plt.show

            def capture_show(
                *args,
                **kwargs
            ):
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

                with contextlib.redirect_stdout(
                    log_buffer
                ):

                    results = run_research(
                        tickers=tickers,
                        start_date=start_text,
                        end_date=end_text,
                        risk_free_rate=risk_free_rate,
                        risk_aversion=risk_aversion,
                        benchmark=benchmark.strip().upper(),
                        leverage=use_leverage,
                        margin_table=margin_table,
                        max_leverage=float(
                            max_leverage
                        ),
                        include_company=True,
                        include_income=True,
                        target_return=target_return,
                        api_authenticated=auth[
                            "authenticated"
                        ],
                    )

            finally:

                plt.show = original_show

            status.update(
                label="Phân tích hoàn tất",
                state="complete",
            )

        st.success(
            "Đã hoàn tất phân tích."
        )

        # ====================================================
        # 3. TỔNG QUAN DOANH NGHIỆP
        # ====================================================

        company_table = results.get(
            "company_table",
            pd.DataFrame(),
        )

        st.subheader(
            "3. TỔNG QUAN DOANH NGHIỆP"
        )

        if not company_table.empty:
            render_df(
                company_table,
                hide_index=True,
            )

        # ====================================================
        # 4. VNINDEX
        # ====================================================

        st.subheader("4. VNINDEX")

        benchmark_summary = results.get(
            "benchmark_summary",
            pd.DataFrame(),
        )

        if not benchmark_summary.empty:
            render_df(
                benchmark_summary,
                hide_index=True,
            )

        # ====================================================
        # 5. PHÂN TÍCH TỪNG CỔ PHIẾU
        # ====================================================

        st.subheader(
            "5. PHÂN TÍCH TỪNG CỔ PHIẾU"
        )

        asset_summary = results.get(
            "asset_summary",
            pd.DataFrame(),
        )

        if not asset_summary.empty:
            render_df(
                asset_summary,
                hide_index=False,
            )

        correlation = results.get(
            "correlation",
            pd.DataFrame(),
        )

        if not correlation.empty:
            st.markdown(
                "**Ma trận tương quan**"
            )
            render_df(
                correlation,
                hide_index=False,
            )

        # ====================================================
        # 6. TỐI ƯU DANH MỤC
        # ====================================================

        st.subheader(
            "6. TỐI ƯU DANH MỤC"
        )

        portfolio_table = results.get(
            "portfolio_table",
            pd.DataFrame(),
        )

        if not portfolio_table.empty:
            render_df(
                portfolio_table,
                hide_index=True,
            )

        weights = results.get(
            "weights",
            pd.DataFrame(),
        )

        if not weights.empty:
            st.markdown(
                "**Phân bổ tỷ trọng**"
            )
            render_weight_table(
                weights,
                hide_index=False,
            )

        # ----------------------------------------------------
        # 6.1 ĐÒN BẨY
        # ----------------------------------------------------

        if use_leverage:

            st.subheader(
                "6.1. PHÂN TÍCH CÓ ĐÒN BẨY"
            )

            levered_table = results.get(
                "levered_table",
                pd.DataFrame(),
            )

            if not levered_table.empty:
                render_df(
                    levered_table,
                    hide_index=True,
                )

            st.subheader(
                "6.2. PHÂN BỔ DANH MỤC CÓ ĐÒN BẨY"
            )

            levered_alloc = results.get(
                "levered_alloc",
                pd.DataFrame(),
            )

            if not levered_alloc.empty:
                render_weight_table(
                    levered_alloc,
                    hide_index=True,
                )

        # ----------------------------------------------------
        # 6.3 MỤC TIÊU LỢI NHUẬN
        # ----------------------------------------------------

        st.subheader(
            "6.3. MỤC TIÊU LỢI NHUẬN"
        )

        target_summary = results.get(
            "target_summary",
            None,
        )

        if (
            isinstance(
                target_summary,
                pd.DataFrame,
            )
            and not target_summary.empty
        ):
            render_df(
                target_summary,
                hide_index=True,
            )

        target_check = results.get(
            "target_check",
            pd.DataFrame(),
        )

        if not target_check.empty:
            render_df(
                target_check,
                hide_index=True,
                percent_signed_columns={
                    "So với mục tiêu",
                },
            )

        # ====================================================
        # 7. SO SÁNH VỚI VNINDEX
        # ====================================================

        st.subheader(
            "7. SO SÁNH VỚI VNINDEX"
        )

        comparison = results.get(
            "comparison",
            pd.DataFrame(),
        )

        if not comparison.empty:
            render_df(
                comparison,
                hide_index=True,
                percent_signed_columns={
                    "Alpha",
                    "Vượt VNINDEX",
                },
            )

        # ====================================================
        # 8. CÁC CHỈ TIÊU RỦI RO
        # ====================================================

        st.subheader(
            "8. PHÂN TÍCH RỦI RO"
        )

        attainment = results.get(
            "attainment",
            pd.DataFrame(),
        )

        if (
            isinstance(
                attainment,
                pd.DataFrame,
            )
            and not attainment.empty
        ):
            st.markdown(
                "**Khả năng đạt mục tiêu**"
            )
            render_df(
                attainment,
                hide_index=True,
            )

        concentration = results.get(
            "concentration",
            pd.DataFrame(),
        )

        if (
            isinstance(
                concentration,
                pd.DataFrame,
            )
            and not concentration.empty
        ):
            st.markdown(
                "**Mức độ tập trung danh mục**"
            )
            render_df(
                concentration,
                hide_index=True,
            )

        risk_diagnostics = results.get(
            "risk_diagnostics",
            pd.DataFrame(),
        )

        if (
            isinstance(
                risk_diagnostics,
                pd.DataFrame,
            )
            and not risk_diagnostics.empty
        ):
            st.markdown(
                "**Max Drawdown và thời gian phục hồi**"
            )
            render_df(
                risk_diagnostics,
                hide_index=True,
            )

        # ====================================================
        # 9. KẾT LUẬN ĐỊNH LƯỢNG
        # ====================================================

        st.subheader(
            "9. ĐÁNH GIÁ THEO MỤC TIÊU NHÀ ĐẦU TƯ"
        )

        conclusion = results.get(
            "conclusion",
            pd.DataFrame(),
        )

        if (
            isinstance(
                conclusion,
                pd.DataFrame,
            )
            and not conclusion.empty
        ):
            render_df(
                conclusion,
                hide_index=True,
                percent_signed_columns={
                    "Chênh lệch mục tiêu",
                },
            )

        # ====================================================
        # 10. PHÂN TÍCH NÂNG CAO
        # ====================================================

        st.subheader(
            "10. PHÂN TÍCH NÂNG CAO DANH MỤC"
        )

        advanced = results.get(
            "advanced_analysis",
            {},
        )

        advanced_risk = advanced.get(
            "advanced_risk",
            pd.DataFrame(),
        )

        if not advanced_risk.empty:
            st.markdown(
                "**10.1. Hồ sơ rủi ro nâng cao**"
            )
            render_df(
                advanced_risk,
                hide_index=True,
            )

        return_contributions = advanced.get(
            "return_contributions",
            {},
        )

        if return_contributions:
            st.markdown(
                "**10.2. Phân rã lợi nhuận**"
            )

            for name, table in (
                return_contributions.items()
            ):
                with st.expander(
                    f"Danh mục {name}",
                    expanded=False,
                ):
                    render_df(
                        table,
                        hide_index=False,
                    )

        risk_contributions = advanced.get(
            "risk_contributions",
            {},
        )

        if risk_contributions:
            st.markdown(
                "**10.3. Phân rã rủi ro**"
            )

            for name, table in (
                risk_contributions.items()
            ):
                with st.expander(
                    f"Danh mục {name}",
                    expanded=False,
                ):
                    render_df(
                        table,
                        hide_index=False,
                    )

        incremental_tables = advanced.get(
            "incremental_tables",
            {},
        )

        if incremental_tables:
            st.markdown(
                "**10.4. Rủi ro biên khi tăng tỷ trọng**"
            )

            for name, table in (
                incremental_tables.items()
            ):
                with st.expander(
                    f"Danh mục {name}",
                    expanded=False,
                ):
                    render_df(
                        table,
                        hide_index=False,
                        percent_signed_columns={
                            "Thay đổi lợi nhuận",
                            "Thay đổi rủi ro",
                        },
                    )

        stress_market = advanced.get(
            "stress_market",
            pd.DataFrame(),
        )

        if not stress_market.empty:
            st.markdown(
                "**10.5. Kiểm tra sức chịu đựng theo VNINDEX**"
            )
            render_df(
                stress_market,
                hide_index=True,
                percent_signed_columns={
                    "Cú sốc VNINDEX",
                    "Lợi suất danh mục ước tính",
                },
            )

        asset_stress_tables = advanced.get(
            "asset_stress_tables",
            {},
        )

        if asset_stress_tables:
            st.markdown(
                "**10.6. Kiểm tra sức chịu đựng theo từng cổ phiếu**"
            )

            for name, table in (
                asset_stress_tables.items()
            ):
                with st.expander(
                    f"Danh mục {name}",
                    expanded=False,
                ):
                    render_df(
                        table,
                        hide_index=True,
                        percent_signed_columns={
                            "Cú sốc",
                            "Tác động danh mục",
                        },
                    )

        reverse_stress_tables = advanced.get(
            "reverse_stress_tables",
            {},
        )

        if reverse_stress_tables:
            st.markdown(
                "**10.7. Kiểm tra sức chịu đựng ngược**"
            )

            for name, table in (
                reverse_stress_tables.items()
            ):
                with st.expander(
                    f"Danh mục {name}",
                    expanded=False,
                ):
                    render_df(
                        table,
                        hide_index=True,
                        percent_signed_columns={
                            "Mức lỗ danh mục mục tiêu",
                            "Cú sốc riêng cần thiết",
                        },
                    )

        stability = advanced.get(
            "stability",
            pd.DataFrame(),
        )

        if not stability.empty:
            st.markdown(
                "**10.8. Độ ổn định theo thời gian**"
            )
            render_df(
                stability,
                hide_index=True,
            )

        advanced_warnings = advanced.get(
            "advanced_warnings",
            pd.DataFrame(),
        )

        if not advanced_warnings.empty:
            st.markdown(
                "**10.9. Cảnh báo quản trị**"
            )
            render_df(
                advanced_warnings,
                hide_index=True,
            )

        # ====================================================
        # 11. BIỂU ĐỒ
        # ====================================================

        st.subheader(
            "11. BIỂU ĐỒ"
        )

        for fig in figures:
            st.pyplot(
                fig,
                clear_figure=True,
                use_container_width=True,
            )

        # ====================================================
        # THÔNG TIN XỬ LÝ
        # ====================================================

        with st.expander(
            "Thông tin xử lý"
        ):

            log_text = (
                log_buffer
                .getvalue()
                .strip()
            )

            if log_text:
                st.text(log_text)
            else:
                st.write(
                    "Không có thông báo bổ sung."
                )

    except Exception as e:

        st.error(
            f"{type(e).__name__}: {str(e)}"
        )
