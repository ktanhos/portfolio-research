import contextlib
import io
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
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
# ĐỊNH DẠNG HIỂN THỊ
# ------------------------------------------------------------

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

PERCENT_COLUMNS = {
    "ROA",
    "ROE",
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

TEXT_COLUMNS = {
    "Đạt mục tiêu",
    "Trạng thái",
    "Trạng thái mục tiêu",
    "Trạng thái phục hồi",
    "Được cấp margin",
    "Mã",
    "Danh mục",
    "Benchmark",
    "Năm doanh thu",
    "Năm LNST",
}

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


def _is_missing(value):
    try:
        return pd.isna(value)
    except Exception:
        return False


def fmt_number_vi(value, decimals=2, signed=False):
    if _is_missing(value):
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

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
    if _is_missing(value):
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    return fmt_number_vi(value * 100, 2, signed=signed) + "%"


def fmt_money_vnd(value):
    """Định dạng tiền từ giá trị nội bộ đã chuẩn hóa về VND."""
    if _is_missing(value):
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    absolute = abs(value)

    if absolute >= 1e12:
        return fmt_number_vi(value / 1e12, 2) + " nghìn tỷ"
    if absolute >= 1e9:
        return fmt_number_vi(value / 1e9, 2) + " tỷ"
    if absolute >= 1e6:
        return fmt_number_vi(value / 1e6, 2) + " triệu"

    return fmt_number_vi(value, 0) + " đồng"


def is_percent_column(column):
    name = str(column).strip().lower()

    if column in TEXT_COLUMNS:
        return False

    if column in PERCENT_COLUMNS:
        return True

    if column in RATIO_COLUMNS:
        return False

    if column in MONEY_COLUMNS or column in INTEGER_COLUMNS:
        return False

    return any(keyword in name for keyword in PERCENT_KEYWORDS)


def render_df(df, hide_index=True, percent_signed_columns=None):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return

    percent_signed_columns = percent_signed_columns or set()
    out = df.copy()

    for column in out.columns:

        if column in TEXT_COLUMNS:
            out[column] = out[column].map(
                lambda x: "N/A" if _is_missing(x) else str(x)
            )

        elif column in MONEY_COLUMNS:
            out[column] = out[column].map(fmt_money_vnd)

        elif column in PERCENT_COLUMNS:
            out[column] = out[column].map(fmt_percent_vi)

        elif column in INTEGER_COLUMNS:
            out[column] = out[column].map(
                lambda x: "N/A" if _is_missing(x) else fmt_number_vi(x, 0)
            )

        elif column in RATIO_COLUMNS:
            decimals = 3
            if column in {"P/E", "P/B"}:
                decimals = 2
            elif column in {
                "HHI",
                "Số tài sản hiệu dụng",
                "Số tài sản hiệu dụng theo |w|",
            }:
                decimals = 2
            elif column == "Thay đổi Sharpe":
                decimals = 4

            out[column] = out[column].map(
                lambda x: "N/A" if _is_missing(x) else fmt_number_vi(x, decimals)
            )

        elif is_percent_column(column):
            signed = column in percent_signed_columns
            out[column] = out[column].map(
                lambda x: fmt_percent_vi(x, signed=signed)
            )

        elif pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].map(
                lambda x: "N/A" if _is_missing(x) else fmt_number_vi(x, 2)
            )

    st.dataframe(
        out,
        use_container_width=True,
        hide_index=hide_index,
    )


def render_weight_table(df, hide_index=False):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return

    out = df.copy()

    for column in out.columns:
        if column in TEXT_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].map(fmt_percent_vi)

    st.dataframe(
        out,
        use_container_width=True,
        hide_index=hide_index,
    )


def prepare_asset_summary(df):
    """Đưa mã cổ phiếu từ index thành cột rõ ràng."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    if "Mã" not in out.columns:
        index_name = out.index.name
        out = out.reset_index()
        source_name = index_name if index_name in out.columns else "index"
        if source_name in out.columns:
            out = out.rename(columns={source_name: "Mã"})

    return out


def annotate_scatter_figure(fig, labels):
    """Gắn lại nhãn điểm trên các biểu đồ phân tán của danh mục."""
    if fig is None or not labels:
        return

    offsets = [
        (8, 10),
        (8, -18),
        (10, 22),
        (10, -30),
        (-105, 10),
        (-120, -22),
    ]

    for ax in fig.axes:
        title = ax.get_title().lower()
        if not any(
            key in title
            for key in [
                "rủi ro và lợi suất",
                "mục tiêu lợi nhuận",
                "markowitz",
            ]
        ):
            continue

        # Xóa nhãn cũ để tránh nhãn chồng lên nhau.
        for txt in list(ax.texts):
            try:
                txt.remove()
            except Exception:
                pass

        points = []
        for collection in ax.collections:
            try:
                arr = collection.get_offsets()
                if arr is None:
                    continue
                for point in np.asarray(arr):
                    if len(point) >= 2 and np.all(np.isfinite(point[:2])):
                        points.append((float(point[0]), float(point[1])))
            except Exception:
                continue

        if not points:
            continue

        # Với đường biên Markowitz có thể có nhiều điểm frontier.
        # Chỉ gắn nhãn số điểm tương ứng với danh mục chính.
        count = min(len(points), len(labels))
        for i in range(count):
            x, y = points[i]
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(
                str(labels[i]),
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                ),
            )


def build_rolling_target_figure(results, target_return):
    """Tạo biểu đồ Rolling Return 12 tháng so với mục tiêu."""
    returns = results.get("returns")
    weights = results.get("weights")

    if not isinstance(returns, pd.DataFrame) or returns.empty:
        return None
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        return None

    rolling_rows = {}

    for name in weights.columns:
        try:
            w = pd.to_numeric(weights[name], errors="coerce").fillna(0)
            p_returns = returns.reindex(columns=w.index).fillna(0).dot(w)
            rolling_rows[name] = (
                (1 + p_returns)
                .rolling(52)
                .apply(np.prod, raw=True)
                - 1
            )
        except Exception:
            continue

    if not rolling_rows:
        return None

    rolling_df = pd.DataFrame(rolling_rows).dropna(how="all")
    if rolling_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(13, 6))

    for col in rolling_df.columns:
        ax.plot(
            rolling_df.index,
            rolling_df[col] * 100,
            label=col,
            linewidth=1.5,
        )

    ax.axhline(
        target_return * 100,
        linestyle="--",
        linewidth=1.4,
        label=f"Mục tiêu {target_return:.1%}",
    )

    ax.set_title("Rolling Return 12 tháng so với mục tiêu")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Rolling Return 12 tháng (%)")
    ax.legend(ncol=2, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    return fig


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

use_leverage = st.checkbox("Sử dụng đòn bẩy", value=False)
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
            "Mã": st.column_config.TextColumn("Mã", disabled=True),
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

        with st.status("Đang thực hiện phân tích...", expanded=True) as status:
            st.write("Đang xác thực API Vnstock...")
            auth = configure_vnstock(api_key)
            st.write("Đang lấy dữ liệu và chạy mô hình...")

            figures = []
            original_show = plt.show

            def capture_show(*args, **kwargs):
                for num in plt.get_fignums():
                    fig = plt.figure(num)
                    if fig not in figures:
                        figures.append(fig)
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

            status.update(label="Phân tích hoàn tất", state="complete")

        st.success("Đã hoàn tất phân tích.")

        # ----------------------------------------------------
        # 3. TỔNG QUAN DOANH NGHIỆP
        # ----------------------------------------------------
        company_table = results.get("company_table", pd.DataFrame())
        st.subheader("3. TỔNG QUAN DOANH NGHIỆP")
        if not company_table.empty:
            render_df(company_table, hide_index=True)

        # ----------------------------------------------------
        # 4. VNINDEX
        # ----------------------------------------------------
        st.subheader("4. VNINDEX")
        benchmark_summary = results.get("benchmark_summary", pd.DataFrame())
        if not benchmark_summary.empty:
            render_df(benchmark_summary, hide_index=True)

        # ----------------------------------------------------
        # 5. PHÂN TÍCH TỪNG CỔ PHIẾU
        # ----------------------------------------------------
        st.subheader("5. PHÂN TÍCH TỪNG CỔ PHIẾU")
        asset_summary = prepare_asset_summary(
            results.get("asset_summary", pd.DataFrame())
        )
        if not asset_summary.empty:
            render_df(asset_summary, hide_index=True)

        st.markdown("**Ma trận tương quan**")
        correlation = results.get("correlation", pd.DataFrame())
        if not correlation.empty:
            render_df(correlation, hide_index=True)

        # ----------------------------------------------------
        # 6. TỐI ƯU DANH MỤC
        # ----------------------------------------------------
        st.subheader("6. TỐI ƯU DANH MỤC")
        portfolio_table = results.get("portfolio_table", pd.DataFrame())
        if not portfolio_table.empty:
            render_df(portfolio_table, hide_index=True)

        weights = results.get("weights", pd.DataFrame())
        if not weights.empty:
            st.markdown("**Phân bổ tỷ trọng**")
            render_weight_table(weights, hide_index=False)

        if use_leverage:
            st.subheader("6.1. PHÂN TÍCH CÓ ĐÒN BẨY")
            levered_table = results.get("levered_table", pd.DataFrame())
            if not levered_table.empty:
                render_df(levered_table, hide_index=True)

            st.subheader("6.2. PHÂN BỔ DANH MỤC CÓ ĐÒN BẨY")
            levered_alloc = results.get("levered_alloc", pd.DataFrame())
            if not levered_alloc.empty:
                render_df(levered_alloc, hide_index=True)

        st.subheader("6.3. MỤC TIÊU LỢI NHUẬN")
        target_summary = results.get("target_summary")
        if isinstance(target_summary, pd.DataFrame) and not target_summary.empty:
            target_summary = target_summary.copy()

            if "Mục tiêu" in target_summary.columns:
                target_summary = target_summary.rename(
                    columns={"Mục tiêu": "Lợi suất mục tiêu"}
                )

            if (
                "Lợi suất kỳ vọng" in target_summary.columns
                and "Lợi suất mục tiêu" in target_summary.columns
            ):
                target_summary["Chênh lệch"] = (
                    pd.to_numeric(
                        target_summary["Lợi suất kỳ vọng"],
                        errors="coerce",
                    )
                    - pd.to_numeric(
                        target_summary["Lợi suất mục tiêu"],
                        errors="coerce",
                    )
                )

            preferred = [
                "Lợi suất mục tiêu",
                "Lợi suất kỳ vọng",
                "Chênh lệch",
                "Rủi ro thấp nhất",
                "Sharpe",
                "Trạng thái",
            ]
            ordered = [
                column
                for column in preferred
                if column in target_summary.columns
            ]
            remaining = [
                column
                for column in target_summary.columns
                if column not in ordered
            ]
            target_summary = target_summary[ordered + remaining]

            render_df(target_summary, hide_index=True)

        target_check = results.get("target_check", pd.DataFrame())
        if not target_check.empty:
            render_df(target_check, hide_index=True)

        # ----------------------------------------------------
        # 7. SO SÁNH VỚI VNINDEX
        # ----------------------------------------------------
        st.subheader("7. SO SÁNH VỚI VNINDEX")
        comparison = results.get("comparison", pd.DataFrame())
        if not comparison.empty:
            render_df(comparison, hide_index=True)

        # ----------------------------------------------------
        # 8. PHÂN TÍCH RỦI RO
        # ----------------------------------------------------
        st.subheader("8. PHÂN TÍCH RỦI RO")

        attainment = results.get("attainment", pd.DataFrame())
        if isinstance(attainment, pd.DataFrame) and not attainment.empty:
            st.markdown("**Khả năng đạt mục tiêu**")
            render_df(attainment, hide_index=True)

        concentration = results.get("concentration", pd.DataFrame())
        if isinstance(concentration, pd.DataFrame) and not concentration.empty:
            st.markdown("**Mức độ tập trung danh mục**")
            render_df(concentration, hide_index=True)

        risk_diagnostics = results.get("risk_diagnostics", pd.DataFrame())
        if isinstance(risk_diagnostics, pd.DataFrame) and not risk_diagnostics.empty:
            st.markdown("**Max Drawdown và thời gian phục hồi**")
            render_df(risk_diagnostics, hide_index=True)

        # ----------------------------------------------------
        # 9. KẾT LUẬN ĐỊNH LƯỢNG
        # ----------------------------------------------------
        st.subheader("9. ĐÁNH GIÁ THEO MỤC TIÊU NHÀ ĐẦU TƯ")
        conclusion = results.get("conclusion", pd.DataFrame())
        if isinstance(conclusion, pd.DataFrame) and not conclusion.empty:
            render_df(conclusion, hide_index=True)

        # ----------------------------------------------------
        # 10. BIỂU ĐỒ
        # ----------------------------------------------------
        st.subheader("10. BIỂU ĐỒ")

        comparison_labels = (
            comparison["Danh mục"].astype(str).tolist()
            if "Danh mục" in comparison.columns
            else []
        )
        conclusion_labels = (
            conclusion["Danh mục"].astype(str).tolist()
            if isinstance(conclusion, pd.DataFrame)
            and "Danh mục" in conclusion.columns
            else []
        )

        # Thay biểu đồ scatter mục tiêu hiện tại bằng Rolling Return 12 tháng.
        rolling_target_fig = build_rolling_target_figure(
            results,
            float(target_return),
        )

        processed_figures = []
        replaced_target_chart = False

        for fig in figures:
            title = ""
            if fig.axes:
                title = fig.axes[0].get_title().lower()

            if "danh mục so với mục tiêu lợi nhuận" in title:
                if rolling_target_fig is not None and not replaced_target_chart:
                    processed_figures.append(rolling_target_fig)
                    replaced_target_chart = True
                continue

            if "so sánh rủi ro và lợi suất" in title:
                annotate_scatter_figure(fig, comparison_labels)
            elif "markowitz" in title:
                labels = [
                    x for x in [
                        "Naive",
                        "Minimum Variance",
                        "Optimal Risky",
                        "Maximum Return",
                        "Complete Portfolio",
                    ]
                    if x in comparison_labels
                ]
                annotate_scatter_figure(fig, labels)

            processed_figures.append(fig)

        if rolling_target_fig is not None and not replaced_target_chart:
            processed_figures.append(rolling_target_fig)

        for fig in processed_figures:
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
        st.error(f"{type(e).__name__}: {str(e)}")
