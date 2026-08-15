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
    return f"{float(x) * 100:,.2f}%".replace(",", "_").replace(".", ",").replace("_", ".")


def _fmt_num(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"{float(x):,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _fmt_money_trillion(x):
    if pd.isna(x):
        return "N/A"
    value = float(x) / 1e12
    return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".") + " nghìn tỷ đồng"


def _format_advanced_table(df, section_key=None):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()

    if section_key == "factor_proxy":
        out = out.rename(columns={
            "Momentum 12T": "Momentum 12 tháng",
            "Volatility": "Biến động năm",
            "Size proxy vốn hóa": "Quy mô vốn hóa",
            "Value proxy P/B": "Định giá P/B",
            "Quality proxy ROE": "Chất lượng ROE",
        })
        for col in ["Momentum 12 tháng", "Biến động năm", "Chất lượng ROE"]:
            if col in out.columns:
                out[col] = out[col].map(_fmt_pct)
        if "Quy mô vốn hóa" in out.columns:
            out["Quy mô vốn hóa"] = out["Quy mô vốn hóa"].map(_fmt_money_trillion)
        if "Định giá P/B" in out.columns:
            out["Định giá P/B"] = out["Định giá P/B"].map(lambda x: _fmt_num(x, 2))
        return out

    if section_key == "multifactor_regression":
        if "Hệ số" in out.columns:
            for idx in out.index:
                value = pd.to_numeric(out.loc[idx, "Hệ số"], errors="coerce")
                if pd.notna(value):
                    out.loc[idx, "Hệ số"] = _fmt_pct(value) if idx in {"R²", "Specific Risk", "Alpha năm"} else _fmt_num(value, 4)
        return out.rename(index={"Specific Risk": "Rủi ro riêng"})

    if section_key == "active_analysis":
        for col in ["Active Return", "Tracking Error"]:
            if col in out.columns:
                out[col] = out[col].map(_fmt_pct)
        return out

    if section_key == "var_analysis":
        for col in ["Historical VaR", "Parametric VaR", "Monte Carlo VaR", "CVaR"]:
            if col in out.columns:
                out[col] = out[col].map(_fmt_pct)
        return out

    if section_key == "robustness":
        for col in ["CAGR", "Rủi ro", "Max Drawdown"]:
            if col in out.columns:
                out[col] = out[col].map(_fmt_pct)
        if "Cửa sổ" in out.columns:
            out["Cửa sổ"] = out["Cửa sổ"].map(lambda x: _fmt_num(x, 0))
        return out

    if section_key == "walk_forward":
        for col in ["Train CAGR", "Test CAGR", "Test Max Drawdown"]:
            if col in out.columns:
                out[col] = out[col].map(_fmt_pct)
        if "Fold" in out.columns:
            out["Fold"] = out["Fold"].map(lambda x: _fmt_num(x, 0))
        return out

    return out


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

    target_status = "Đạt" if pd.notna(cagr) and cagr >= target_return else "Không đạt" if pd.notna(cagr) else "Không đủ dữ liệu"
    target_comment = f"CAGR {_fmt_pct(cagr)} so với mục tiêu {_fmt_pct(target_return)}" if pd.notna(cagr) else "Không xác định được CAGR"
    risk_status = "Tốt" if pd.notna(sharpe) and sharpe >= 1.0 else "Khá" if pd.notna(sharpe) and sharpe >= 0.5 else "Yếu" if pd.notna(sharpe) and sharpe >= 0 else "Không đạt" if pd.notna(sharpe) else "Không đủ dữ liệu"
    risk_comment = f"Sharpe {_fmt_num(sharpe)}" if pd.notna(sharpe) else "Không xác định được Sharpe"
    dd_abs = abs(max_dd) if pd.notna(max_dd) else np.nan
    dd_status = "Tốt" if pd.notna(dd_abs) and dd_abs <= 0.15 else "Chấp nhận được" if pd.notna(dd_abs) and dd_abs <= 0.25 else "Cần lưu ý" if pd.notna(dd_abs) and dd_abs <= 0.35 else "Rủi ro cao" if pd.notna(dd_abs) else "Không đủ dữ liệu"
    dd_comment = f"Maximum Drawdown {_fmt_pct(max_dd)}" if pd.notna(max_dd) else "Không xác định được Maximum Drawdown"
    active_status = "Tích cực" if pd.notna(ir) and ir >= 0.5 else "Trung tính" if pd.notna(ir) and ir >= 0 else "Kém" if pd.notna(ir) else "Không đủ dữ liệu"
    active_comment = f"Information Ratio {_fmt_num(ir)}" if pd.notna(ir) else "Không xác định được Information Ratio"
    stability_status = "Ổn định" if pd.notna(oos_sharpe) and oos_sharpe >= 0.5 else "Trung bình" if pd.notna(oos_sharpe) and oos_sharpe >= 0 else "Không ổn định" if pd.notna(oos_sharpe) else "Không đủ dữ liệu"
    stability_comment = f"Sharpe ngoài mẫu trung vị {_fmt_num(oos_sharpe)}" if pd.notna(oos_sharpe) else "Chưa đủ mẫu để đánh giá ngoài mẫu"

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
    return {"summary": summary, "overall": overall, "conclusion": ". ".join(parts) + ".", "warnings": warnings}


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
            st.dataframe(_format_advanced_table(table, key), use_container_width=True, hide_index=key == "factor_proxy")
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

    # Target Return là mục tiêu lợi suất dùng để đánh giá, không phải một danh mục.
    # Không đưa nó vào so sánh hay các tab phân tích danh mục.
    excluded_portfolios = {
        "target return",
        "target return portfolio",
        "target",
    }
    portfolio_names = [
        name for name, value in all_portfolios.items()
        if isinstance(value, dict)
        and not value.get("error")
        and str(name).strip().lower() not in excluded_portfolios
    ]

    if not portfolio_names:
        st.warning("Không có danh mục hợp lệ để đánh giá.")
        return

    st.markdown("**Danh mục được đánh giá**")
    st.caption("Mục 11 chỉ phân tích các danh mục thực sự được tạo ra. Mục tiêu lợi suất chỉ được dùng làm ngưỡng đánh giá, không được coi là một danh mục.")

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
