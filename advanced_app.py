import pandas as pd
import numpy as np
import streamlit as st

from advanced_portfolio import (
    factor_proxy_analysis,
    multifactor_regression,
    var_analysis,
    active_analysis,
    robustness_analysis,
    walk_forward_analysis,
)


def _fmt_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def _fmt_num(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.2f}"


def build_portfolio_evaluation(advanced_results, target_return=0.15):
    robustness = advanced_results.get("robustness", pd.DataFrame())
    active = advanced_results.get("active_analysis", pd.DataFrame())
    var = advanced_results.get("var_analysis", pd.DataFrame())
    walk = advanced_results.get("walk_forward", pd.DataFrame())

    metrics = {}
    if isinstance(robustness, pd.DataFrame) and not robustness.empty:
        latest = robustness.iloc[-1]
        metrics["CAGR"] = latest.get("CAGR", np.nan)
        metrics["Sharpe"] = latest.get("Sharpe", np.nan)
        metrics["Max Drawdown"] = latest.get("Max Drawdown", np.nan)
    if isinstance(active, pd.DataFrame) and not active.empty:
        row = active.iloc[0]
        metrics["Information Ratio"] = row.get("Information Ratio", np.nan)
        metrics["Beta"] = row.get("Beta", np.nan)
    if isinstance(var, pd.DataFrame) and not var.empty:
        row = var.iloc[0]
        metrics["Historical VaR"] = row.get("Historical VaR", np.nan)
        metrics["CVaR"] = row.get("CVaR", np.nan)

    if isinstance(walk, pd.DataFrame) and not walk.empty and "Test Sharpe" in walk.columns:
        values = pd.to_numeric(walk["Test Sharpe"], errors="coerce").dropna()
        if not values.empty:
            metrics["OOS Sharpe trung vị"] = float(values.median())
            metrics["OOS Sharpe thấp nhất"] = float(values.min())

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

    if pd.notna(sharpe):
        risk_status = "Tốt" if sharpe >= 1.0 else "Khá" if sharpe >= 0.5 else "Yếu" if sharpe >= 0 else "Không đạt"
        risk_comment = f"Sharpe {_fmt_num(sharpe)}"
    else:
        risk_status = "Không đủ dữ liệu"
        risk_comment = "Không xác định được Sharpe"

    if pd.notna(max_dd):
        dd_abs = abs(max_dd)
        dd_status = "Tốt" if dd_abs <= 0.15 else "Chấp nhận được" if dd_abs <= 0.25 else "Cần lưu ý" if dd_abs <= 0.35 else "Rủi ro cao"
        dd_comment = f"Maximum Drawdown {_fmt_pct(max_dd)}"
    else:
        dd_status = "Không đủ dữ liệu"
        dd_comment = "Không xác định được Maximum Drawdown"

    if pd.notna(ir):
        active_status = "Tích cực" if ir >= 0.5 else "Trung tính" if ir >= 0 else "Kém"
        active_comment = f"Information Ratio {_fmt_num(ir)}"
    else:
        active_status = "Không đủ dữ liệu"
        active_comment = "Không xác định được Information Ratio"

    if pd.notna(oos_sharpe):
        stability_status = "Ổn định" if oos_sharpe >= 0.5 else "Trung bình" if oos_sharpe >= 0 else "Không ổn định"
        stability_comment = f"Sharpe ngoài mẫu trung vị {_fmt_num(oos_sharpe)}"
    else:
        stability_status = "Không đủ dữ liệu"
        stability_comment = "Chưa đủ mẫu để đánh giá ngoài mẫu"

    statuses = [target_status, risk_status, dd_status, active_status, stability_status]
    positive = sum(x in {"Đạt", "Tốt", "Khá", "Chấp nhận được", "Tích cực", "Ổn định"} for x in statuses)
    negative = sum(x in {"Không đạt", "Rủi ro cao", "Kém", "Không ổn định"} for x in statuses)
    overall = "Cần xem xét" if negative >= 2 else "Tích cực" if positive >= 4 else "Trung tính" if positive >= 2 else "Chưa đủ cơ sở"

    warnings = []
    if pd.notna(cagr) and cagr < target_return:
        warnings.append("Lợi suất thực tế chưa đạt mục tiêu.")
    if pd.notna(max_dd) and max_dd <= -0.35:
        warnings.append("Maximum Drawdown ở mức cao, cần đặc biệt lưu ý khả năng chịu lỗ.")
    if pd.notna(ir) and ir < 0:
        warnings.append("Danh mục đang kém hiệu quả so với VNINDEX sau khi điều chỉnh theo rủi ro chủ động.")
    if pd.notna(oos_sharpe) and oos_sharpe < 0:
        warnings.append("Hiệu quả ngoài mẫu có dấu hiệu không ổn định.")
    if pd.notna(metrics.get("CVaR", np.nan)) and pd.notna(metrics.get("Historical VaR", np.nan)) and abs(metrics["CVaR"]) > abs(metrics["Historical VaR"]) * 1.5:
        warnings.append("Rủi ro đuôi phân phối lớn hơn đáng kể so với VaR lịch sử.")

    parts = []
    if pd.notna(cagr):
        parts.append(f"CAGR {_fmt_pct(cagr)}")
    parts.append(f"mục tiêu {_fmt_pct(target_return)}")
    if pd.notna(sharpe):
        parts.append(f"Sharpe {_fmt_num(sharpe)}")
    if pd.notna(max_dd):
        parts.append(f"Maximum Drawdown {_fmt_pct(max_dd)}")
    if pd.notna(ir):
        parts.append(f"Information Ratio {_fmt_num(ir)}")

    summary = pd.DataFrame([
        {"Tiêu chí": "Mục tiêu lợi suất", "Kết quả": target_status, "Đánh giá": target_comment},
        {"Tiêu chí": "Hiệu quả điều chỉnh rủi ro", "Kết quả": risk_status, "Đánh giá": risk_comment},
        {"Tiêu chí": "Rủi ro giảm giá", "Kết quả": dd_status, "Đánh giá": dd_comment},
        {"Tiêu chí": "So với VNINDEX", "Kết quả": active_status, "Đánh giá": active_comment},
        {"Tiêu chí": "Độ ổn định ngoài mẫu", "Kết quả": stability_status, "Đánh giá": stability_comment},
    ])

    return {
        "summary": summary,
        "overall": overall,
        "conclusion": ". ".join(parts) + ".",
        "warnings": warnings,
    }


def _recalculate_selected_analysis(selected_results, advanced_results, target_return):
    p_returns = selected_results.get("_portfolio_returns")
    if p_returns is None or not isinstance(p_returns, pd.Series) or p_returns.empty:
        return selected_results

    benchmark_returns = advanced_results.get("_advanced_source_benchmark_returns")
    source_returns = advanced_results.get("_advanced_source_returns")
    company_table = advanced_results.get("_advanced_source_company_table")
    rf = float(advanced_results.get("risk_free_rate", 0.0))
    fresh = dict(selected_results)

    try:
        if isinstance(source_returns, pd.DataFrame) and not source_returns.empty:
            fresh["factor_proxy"] = factor_proxy_analysis(source_returns, company_table=company_table)
    except Exception as exc:
        fresh["factor_proxy"] = pd.DataFrame()
        fresh["factor_proxy_error"] = str(exc)

    if benchmark_returns is not None:
        try:
            factor_returns = pd.DataFrame({"Market": pd.Series(benchmark_returns)})
            fresh["multifactor_regression"] = multifactor_regression(p_returns, factor_returns, rf=rf)
            fresh["active_analysis"] = active_analysis(p_returns, benchmark_returns, rf=rf)
        except Exception as exc:
            fresh["multifactor_regression"] = pd.DataFrame()
            fresh["active_analysis"] = pd.DataFrame()
            fresh["regression_error"] = str(exc)
    else:
        fresh["multifactor_regression"] = pd.DataFrame()
        fresh["active_analysis"] = pd.DataFrame()

    try:
        fresh["var_analysis"] = var_analysis(p_returns, level=0.95)
    except Exception as exc:
        fresh["var_analysis"] = pd.DataFrame()
        fresh["var_error"] = str(exc)

    try:
        fresh["robustness"] = robustness_analysis(p_returns, target_return=float(target_return))
    except Exception as exc:
        fresh["robustness"] = pd.DataFrame()
        fresh["robustness_error"] = str(exc)

    try:
        fresh["walk_forward"] = walk_forward_analysis(p_returns, rf=rf)
    except Exception as exc:
        fresh["walk_forward"] = pd.DataFrame()
        fresh["walk_forward_error"] = str(exc)

    fresh["_portfolio_name"] = selected_results.get("_portfolio_name")
    fresh["_portfolio_returns"] = p_returns
    fresh["_target_return"] = target_return
    return fresh


def _render_one_portfolio(name, raw_results, source_results, target_return):
    selected = dict(raw_results)
    selected["_portfolio_name"] = name
    selected = _recalculate_selected_analysis(selected, source_results, target_return)

    st.markdown(f"### Danh mục: {name}")
    st.caption("Các phân tích dưới đây được tính riêng từ chuỗi lợi suất ngày của danh mục này.")

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
        table = selected.get(key, pd.DataFrame())
        if isinstance(table, pd.DataFrame) and not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=False)
        else:
            error = selected.get(f"{key}_error")
            if error:
                st.warning(f"Không thể thực hiện: {error}")
            else:
                st.info("Không đủ dữ liệu hiện có để thực hiện phân tích này.")

    evaluation = build_portfolio_evaluation(selected, target_return=float(target_return))
    st.markdown("**11.7. ĐÁNH GIÁ DANH MỤC**")
    st.markdown(f"Đánh giá tổng thể: **{evaluation['overall']}**")
    st.write(evaluation["conclusion"])
    st.dataframe(evaluation["summary"], use_container_width=True, hide_index=True)

    if evaluation["warnings"]:
        st.markdown("**Cảnh báo chính**")
        for warning in evaluation["warnings"]:
            st.warning(warning)


def render_advanced_section(advanced_results, title=None, target_return=0.15):
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

    st.markdown("**Danh mục được đánh giá**")
    st.caption("Mục 11 hiển thị đồng thời toàn bộ danh mục. Không cần chọn danh mục hoặc nhấn Enter; mỗi danh mục được tính độc lập từ cùng một bộ dữ liệu nguồn.")

    # Tính trước từng danh mục để tạo bảng tổng hợp. Đây cũng là bằng chứng
    # rằng dữ liệu phía dưới thực sự được tính riêng theo từng danh mục.
    evaluations = []
    calculated = {}
    for name in portfolio_names:
        selected = _recalculate_selected_analysis(dict(all_portfolios[name]), advanced_results, target_return)
        calculated[name] = selected
        evaluation = build_portfolio_evaluation(selected, target_return=float(target_return))
        evaluations.append({
            "Danh mục": name,
            "Đánh giá tổng thể": evaluation["overall"],
            "Kết luận": evaluation["conclusion"],
        })

    st.markdown("### So sánh nhanh các danh mục")
    st.dataframe(pd.DataFrame(evaluations), use_container_width=True, hide_index=True)

    tabs = st.tabs(portfolio_names)
    for tab, name in zip(tabs, portfolio_names):
        with tab:
            _render_one_portfolio(name, calculated[name], advanced_results, target_return)
