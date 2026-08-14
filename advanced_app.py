import pandas as pd
import numpy as np
import streamlit as st


def _fmt_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def _fmt_num(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.2f}"


def build_portfolio_evaluation(advanced_results, target_return=0.15):
    """Tạo đánh giá từ các kết quả đã tính, không gọi dữ liệu mới."""
    robustness = advanced_results.get("robustness", pd.DataFrame())
    active = advanced_results.get("active_analysis", pd.DataFrame())
    var = advanced_results.get("var_analysis", pd.DataFrame())
    walk = advanced_results.get("walk_forward", pd.DataFrame())
    factor = advanced_results.get("factor_proxy", pd.DataFrame())

    metrics = {}
    if isinstance(robustness, pd.DataFrame) and not robustness.empty:
        latest = robustness.iloc[-1]
        metrics["CAGR"] = latest.get("CAGR", np.nan)
        metrics["Sharpe"] = latest.get("Sharpe", np.nan)
        metrics["Max Drawdown"] = latest.get("Max Drawdown", np.nan)

    if isinstance(active, pd.DataFrame) and not active.empty:
        row = active.iloc[0]
        metrics["Active Return"] = row.get("Active Return", np.nan)
        metrics["Tracking Error"] = row.get("Tracking Error", np.nan)
        metrics["Information Ratio"] = row.get("Information Ratio", np.nan)
        metrics["Beta"] = row.get("Beta", np.nan)

    if isinstance(var, pd.DataFrame) and not var.empty:
        row = var.iloc[0]
        metrics["Historical VaR"] = row.get("Historical VaR", np.nan)
        metrics["CVaR"] = row.get("CVaR", np.nan)

    test_sharpes = []
    if isinstance(walk, pd.DataFrame) and not walk.empty and "Test Sharpe" in walk.columns:
        test_sharpes = pd.to_numeric(walk["Test Sharpe"], errors="coerce").dropna().tolist()
        if test_sharpes:
            metrics["OOS Sharpe trung vị"] = float(np.median(test_sharpes))
            metrics["OOS Sharpe thấp nhất"] = float(np.min(test_sharpes))

    rows = []
    cagr = metrics.get("CAGR", np.nan)
    sharpe = metrics.get("Sharpe", np.nan)
    max_dd = metrics.get("Max Drawdown", np.nan)
    ir = metrics.get("Information Ratio", np.nan)
    oos_sharpe = metrics.get("OOS Sharpe trung vị", np.nan)

    if pd.notna(cagr):
        target_status = "Đạt" if cagr >= target_return else "Không đạt"
        target_comment = f"CAGR {_fmt_pct(cagr)} so với mục tiêu {_fmt_pct(target_return)}"
    else:
        target_status = "Không đủ dữ liệu"
        target_comment = "Không xác định được CAGR"
    rows.append({"Tiêu chí": "Mục tiêu lợi suất", "Kết quả": target_status, "Đánh giá": target_comment})

    if pd.notna(sharpe):
        if sharpe >= 1.0:
            risk_status = "Tốt"
        elif sharpe >= 0.5:
            risk_status = "Khá"
        elif sharpe >= 0:
            risk_status = "Yếu"
        else:
            risk_status = "Không đạt"
        risk_comment = f"Sharpe {_fmt_num(sharpe)}"
    else:
        risk_status = "Không đủ dữ liệu"
        risk_comment = "Không xác định được Sharpe"
    rows.append({"Tiêu chí": "Hiệu quả điều chỉnh rủi ro", "Kết quả": risk_status, "Đánh giá": risk_comment})

    if pd.notna(max_dd):
        dd_abs = abs(max_dd)
        if dd_abs <= 0.15:
            dd_status = "Tốt"
        elif dd_abs <= 0.25:
            dd_status = "Chấp nhận được"
        elif dd_abs <= 0.35:
            dd_status = "Cần lưu ý"
        else:
            dd_status = "Rủi ro cao"
        dd_comment = f"Maximum Drawdown {_fmt_pct(max_dd)}"
    else:
        dd_status = "Không đủ dữ liệu"
        dd_comment = "Không xác định được Maximum Drawdown"
    rows.append({"Tiêu chí": "Rủi ro giảm giá", "Kết quả": dd_status, "Đánh giá": dd_comment})

    if pd.notna(ir):
        if ir >= 0.5:
            active_status = "Tích cực"
        elif ir >= 0:
            active_status = "Trung tính"
        else:
            active_status = "Kém"
        active_comment = f"Information Ratio {_fmt_num(ir)}"
    else:
        active_status = "Không đủ dữ liệu"
        active_comment = "Không xác định được Information Ratio"
    rows.append({"Tiêu chí": "So với VNINDEX", "Kết quả": active_status, "Đánh giá": active_comment})

    if pd.notna(oos_sharpe):
        if oos_sharpe >= 0.5:
            stability_status = "Ổn định"
        elif oos_sharpe >= 0:
            stability_status = "Trung bình"
        else:
            stability_status = "Không ổn định"
        stability_comment = f"Sharpe ngoài mẫu trung vị {_fmt_num(oos_sharpe)}"
    else:
        stability_status = "Không đủ dữ liệu"
        stability_comment = "Chưa đủ mẫu để đánh giá ngoài mẫu"
    rows.append({"Tiêu chí": "Độ ổn định ngoài mẫu", "Kết quả": stability_status, "Đánh giá": stability_comment})

    warnings = []
    if pd.notna(cagr) and cagr < target_return:
        warnings.append("Lợi suất thực tế chưa đạt mục tiêu.")
    if pd.notna(max_dd) and max_dd <= -0.35:
        warnings.append("Maximum Drawdown ở mức cao, cần đặc biệt lưu ý khả năng chịu lỗ.")
    if pd.notna(ir) and ir < 0:
        warnings.append("Danh mục đang kém hiệu quả so với VNINDEX sau khi điều chỉnh theo rủi ro chủ động.")
    if pd.notna(oos_sharpe) and oos_sharpe < 0:
        warnings.append("Hiệu quả ngoài mẫu có dấu hiệu không ổn định.")
    if pd.notna(metrics.get("CVaR", np.nan)) and pd.notna(metrics.get("Historical VaR", np.nan)):
        if abs(metrics["CVaR"]) > abs(metrics["Historical VaR"]) * 1.5:
            warnings.append("Rủi ro đuôi phân phối lớn hơn đáng kể so với VaR lịch sử.")

    positive = sum(x in {"Đạt", "Tốt", "Khá", "Chấp nhận được", "Tích cực", "Ổn định"} for x in [target_status, risk_status, dd_status, active_status, stability_status])
    negative = sum(x in {"Không đạt", "Rủi ro cao", "Kém", "Không ổn định"} for x in [target_status, risk_status, dd_status, active_status, stability_status])
    if negative >= 2:
        overall = "Cần xem xét"
    elif positive >= 4:
        overall = "Tích cực"
    elif positive >= 2:
        overall = "Trung tính"
    else:
        overall = "Chưa đủ cơ sở"

    conclusion_parts = []
    if pd.notna(cagr):
        conclusion_parts.append(f"CAGR {_fmt_pct(cagr)}")
    if pd.notna(target_return):
        conclusion_parts.append(f"mục tiêu {_fmt_pct(target_return)}")
    if pd.notna(sharpe):
        conclusion_parts.append(f"Sharpe {_fmt_num(sharpe)}")
    if pd.notna(max_dd):
        conclusion_parts.append(f"Maximum Drawdown {_fmt_pct(max_dd)}")
    if pd.notna(ir):
        conclusion_parts.append(f"Information Ratio {_fmt_num(ir)}")

    return {
        "summary": pd.DataFrame(rows),
        "overall": overall,
        "conclusion": ". ".join(conclusion_parts) + "." if conclusion_parts else "Chưa đủ dữ liệu để kết luận.",
        "warnings": warnings,
        "factor_count": len(factor) if isinstance(factor, pd.DataFrame) else 0,
    }


def render_advanced_section(advanced_results, title=None, target_return=0.15):
    """Hiển thị phân tích nâng cao từ đúng kết quả run_research, không lấy dữ liệu mới."""
    if not isinstance(advanced_results, dict) or not advanced_results:
        st.info("Không đủ dữ liệu hiện có để thực hiện phân tích nâng cao.")
        return

    all_portfolios = advanced_results.get("_all_portfolios")
    if not isinstance(all_portfolios, dict) or not all_portfolios:
        all_portfolios = {advanced_results.get("_portfolio_name", "Complete Portfolio"): advanced_results}

    portfolio_names = [name for name, value in all_portfolios.items() if isinstance(value, dict) and not value.get("error")]
    if not portfolio_names:
        st.warning("Không có danh mục hợp lệ để đánh giá.")
        return

    default_name = advanced_results.get("_portfolio_name", portfolio_names[0])
    if default_name not in portfolio_names:
        default_name = portfolio_names[0]

    st.markdown("**Danh mục được đánh giá**")
    selected_name = st.selectbox(
        "Chọn danh mục để phân tích sâu",
        portfolio_names,
        index=portfolio_names.index(default_name),
        key="advanced_portfolio_selector"
    )

    selected_results = dict(all_portfolios[selected_name])
    selected_results["_portfolio_name"] = selected_name
    selected_results["_target_return"] = advanced_results.get("_target_return", target_return)
    target_return = selected_results["_target_return"]

    st.markdown(f"**Danh mục được đánh giá: {selected_name}**")
    st.caption("Đây là danh mục được lựa chọn để đánh giá, không phải danh mục mặc định là tốt nhất.")

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
        table = selected_results.get(key, pd.DataFrame())
        if isinstance(table, pd.DataFrame) and not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=False)
        else:
            error_key = f"{key}_error"
            error = selected_results.get(error_key)
            if error:
                st.warning(f"Không thể thực hiện: {error}")
            else:
                st.info("Không đủ dữ liệu hiện có để thực hiện phân tích này.")

    evaluation = build_portfolio_evaluation(selected_results, target_return=float(target_return))

    st.markdown("---")
    st.markdown("**11.7. ĐÁNH GIÁ DANH MỤC ĐƯỢC LỰA CHỌN**")
    st.markdown(f"### Đánh giá tổng thể: {evaluation['overall']}")
    st.write(evaluation["conclusion"])

    if not evaluation["summary"].empty:
        st.dataframe(evaluation["summary"], use_container_width=True, hide_index=True)

    st.markdown("**Cảnh báo chính**")
    if evaluation["warnings"]:
        for warning in evaluation["warnings"]:
            st.warning(warning)
    else:
        st.success("Không phát hiện cảnh báo nổi bật từ các chỉ tiêu hiện có.")
