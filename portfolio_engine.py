import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import portfolio_engine_core as _core
from margin_patch import install_margin_patch
from input_normalizer import (
    normalize_company_dataframe,
    normalize_income_dataframe,
)
from advanced_portfolio import (
    factor_proxy_analysis,
    multifactor_regression,
    var_analysis,
    active_analysis,
    robustness_analysis,
    walk_forward_analysis,
)

install_margin_patch(_core)
_original_get_company_info = _core.get_company_info
_LAST_RESULTS = None


def get_company_info(tickers, prices=None):
    df = _original_get_company_info(tickers, prices=prices)
    df = normalize_company_dataframe(df)
    if isinstance(df, pd.DataFrame) and "P/E" in df.columns:
        pe = pd.to_numeric(df["P/E"], errors="coerce")
        mask = pe.abs() > 1000
        df.loc[mask, "P/E"] = pe.loc[mask] / 1000.0
    return df


_core.get_company_info = get_company_info
_original_income_row_value = _core.income_row_value


def income_row_value(income, keywords, exclude_keywords=None):
    if income is None or not isinstance(income, pd.DataFrame) or income.empty:
        return _original_income_row_value(income, keywords, exclude_keywords=exclude_keywords)
    normalized = normalize_income_dataframe(income)
    has_unit_metadata = False
    for col in ("unit_multiplier", "unit"):
        if col in normalized.columns:
            values = normalized[col].dropna().astype(str).str.strip()
            if not values.empty and (values != "").any():
                has_unit_metadata = True
                break
    value, year, item = _original_income_row_value(normalized, keywords, exclude_keywords=exclude_keywords)
    if pd.notna(value) and not has_unit_metadata:
        value = float(value) * 1000.0
    return value, year, item


_core.income_row_value = income_row_value
_core_run_research = _core.run_research


def _portfolio_expected_points(portfolio_results):
    expected = []
    preferred_order = [
        "Naive",
        "Minimum Variance",
        "Optimal Risky",
        "Maximum Return",
        "Complete Portfolio",
    ]
    if not isinstance(portfolio_results, dict):
        return expected
    for name in preferred_order:
        item = portfolio_results.get(name)
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        stats = item[1]
        if not isinstance(stats, dict):
            continue
        try:
            x = float(stats.get("volatility", np.nan))
            y = float(stats.get("return", np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            expected.append((name, x, y))
    return expected


def _annotate_comparison_figures(portfolio_results=None):
    """Chuẩn hóa biểu đồ Risk Return theo đúng 5 danh mục nguồn.

    Không suy đoán tên theo thứ tự điểm Matplotlib. Toàn bộ tọa độ được
    lấy trực tiếp từ portfolio_results. Nếu biểu đồ gốc thiếu một điểm,
    điểm đó được bổ sung tại đúng tọa độ nguồn và được gắn nhãn trực tiếp.
    """
    expected = _portfolio_expected_points(portfolio_results)
    if not expected:
        return

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        for ax in fig.axes:
            title = str(ax.get_title()).strip().lower()
            if "so sánh rủi ro" not in title and "rủi ro và lợi suất" not in title:
                continue

            # Giữ các đường tham chiếu như Target Return, nhưng dựng lại
            # toàn bộ tập điểm danh mục từ dữ liệu nguồn để không bị thiếu
            # Optimal Risky hoặc sai nhãn do thứ tự collection.
            for collection in list(ax.collections):
                try:
                    collection.remove()
                except Exception:
                    pass
            for text in list(ax.texts):
                try:
                    text.remove()
                except Exception:
                    pass

            x_values = [x for _, x, _ in expected]
            y_values = [y for _, _, y in expected]
            ax.scatter(x_values, y_values, s=90, zorder=5)

            for name, x, y in expected:
                ax.annotate(
                    name,
                    (x, y),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", alpha=0.75),
                    zorder=10,
                )

            ax.set_title("So sánh rủi ro và lợi suất")
            ax.set_xlabel("Rủi ro năm (%)")
            ax.set_ylabel("Lợi suất kỳ vọng năm (%)")


def _portfolio_name_from_title(title, names):
    text = str(title or "").strip().lower()
    for name in names:
        if name.lower() in text:
            return name
    return None


def _clean_zero_weight_pies(portfolio_results, tickers=None):
    """Loại cổ phiếu có tỷ trọng 0% khỏi pie chart nhưng không sửa dữ liệu gốc."""
    if not isinstance(portfolio_results, dict):
        return
    preferred = [
        "Naive",
        "Minimum Variance",
        "Optimal Risky",
        "Maximum Return",
        "Complete Portfolio",
    ]
    names = [name for name in preferred if name in portfolio_results]
    tickers = list(tickers or [])

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        for ax in fig.axes:
            wedges = [patch for patch in ax.patches if patch.__class__.__name__ == "Wedge"]
            if not wedges:
                continue

            title = ax.get_title()
            portfolio_name = _portfolio_name_from_title(title, names)
            if portfolio_name is None and len(names) == 1:
                portfolio_name = names[0]
            if portfolio_name is None:
                continue

            item = portfolio_results.get(portfolio_name)
            if not isinstance(item, (tuple, list)) or len(item) < 1:
                continue
            try:
                weights = np.asarray(item[0], dtype=float).reshape(-1)
            except Exception:
                continue
            if len(weights) == 0:
                continue

            labels = tickers[:len(weights)] if tickers else []
            zero_labels = {
                str(labels[i]).strip()
                for i, weight in enumerate(weights)
                if i < len(labels) and np.isfinite(weight) and abs(weight) <= 1e-10
            }

            if not zero_labels:
                continue

            # Xóa lát cắt bằng 0 và nhãn tương ứng. Dữ liệu weights trong
            # portfolio_results hoàn toàn không bị thay đổi.
            for wedge in list(wedges):
                label = str(wedge.get_label()).strip()
                if label in zero_labels:
                    try:
                        wedge.remove()
                    except Exception:
                        pass

            for text in list(ax.texts):
                if str(text.get_text()).strip() in zero_labels:
                    try:
                        text.remove()
                    except Exception:
                        pass

            if portfolio_name:
                ax.set_title(f"Phân bổ danh mục — {portfolio_name}")


def _postprocess_portfolio_figures(portfolio_results, tickers=None):
    _annotate_comparison_figures(portfolio_results)
    _clean_zero_weight_pies(portfolio_results, tickers=tickers)


def build_advanced_portfolio_analysis(returns, portfolio_returns, benchmark_returns=None, company_table=None, risk_free_rate=0.0, target_return=0.15):
    result = {}
    try:
        result["factor_proxy"] = factor_proxy_analysis(returns, company_table=company_table)
    except Exception as exc:
        result["factor_proxy"] = pd.DataFrame()
        result["factor_proxy_error"] = str(exc)
    if benchmark_returns is not None:
        try:
            factor_returns = pd.DataFrame({"Market": pd.Series(benchmark_returns)})
            result["multifactor_regression"] = multifactor_regression(portfolio_returns, factor_returns, rf=risk_free_rate)
            result["active_analysis"] = active_analysis(portfolio_returns, benchmark_returns, rf=risk_free_rate)
        except Exception as exc:
            result["multifactor_regression"] = pd.DataFrame()
            result["active_analysis"] = pd.DataFrame()
            result["regression_error"] = str(exc)
    else:
        result["multifactor_regression"] = pd.DataFrame()
        result["active_analysis"] = pd.DataFrame()
    try:
        result["var_analysis"] = var_analysis(portfolio_returns, level=0.95)
    except Exception as exc:
        result["var_analysis"] = pd.DataFrame()
        result["var_error"] = str(exc)
    try:
        result["robustness"] = robustness_analysis(portfolio_returns, target_return=target_return)
    except Exception as exc:
        result["robustness"] = pd.DataFrame()
        result["robustness_error"] = str(exc)
    try:
        result["walk_forward"] = walk_forward_analysis(portfolio_returns, rf=risk_free_rate)
    except Exception as exc:
        result["walk_forward"] = pd.DataFrame()
        result["walk_forward_error"] = str(exc)
    return result


def _daily_risk_free_return(annual_rate):
    try:
        return (1.0 + float(annual_rate)) ** (1.0 / 252.0) - 1.0
    except Exception:
        return 0.0


def _portfolio_return_from_item(item, returns, risk_free_rate=0.0, name=None):
    if item is None or not isinstance(item, (tuple, list)) or len(item) < 1:
        return pd.Series(dtype=float)
    raw_weights = item[0]
    if raw_weights is None:
        return pd.Series(dtype=float)
    try:
        w = np.asarray(raw_weights, dtype=float).reshape(-1)
    except Exception:
        return pd.Series(dtype=float)
    if len(w) != returns.shape[1]:
        return pd.Series(dtype=float)
    asset_returns = returns.mul(w, axis=1).sum(axis=1)
    if name == "Complete Portfolio":
        risk_free_weight = max(1.0 - float(w.sum()), 0.0)
        asset_returns = asset_returns + risk_free_weight * _daily_risk_free_return(risk_free_rate)
    return pd.to_numeric(asset_returns, errors="coerce").dropna()


def _build_portfolio_returns(results):
    returns = results.get("returns")
    portfolio_results = results.get("portfolio_results", {})
    risk_free_rate = results.get("risk_free_rate", 0.0)
    portfolio_returns = {}
    if not isinstance(returns, pd.DataFrame) or returns.empty or not isinstance(portfolio_results, dict):
        return portfolio_returns
    for name, item in portfolio_results.items():
        try:
            p_returns = _portfolio_return_from_item(item, returns, risk_free_rate=risk_free_rate, name=name)
            if not p_returns.empty:
                portfolio_returns[name] = p_returns
        except Exception:
            continue
    return portfolio_returns


def _run_all_advanced(returns, portfolio_returns, benchmark_returns, company_table, risk_free_rate, target_return):
    analyses = {}
    for name, p_returns in portfolio_returns.items():
        try:
            analysis = build_advanced_portfolio_analysis(returns, p_returns, benchmark_returns, company_table, float(risk_free_rate), float(target_return))
            analysis["_portfolio_name"] = name
            analysis["_portfolio_returns"] = p_returns
            analysis["_data_frequency"] = "Ngày"
            analyses[name] = analysis
        except Exception as exc:
            analyses[name] = {"error": str(exc), "_portfolio_name": name}
    return analyses


def run_research(*args, **kwargs):
    global _LAST_RESULTS
    results = _core_run_research(*args, **kwargs)
    if not isinstance(results, dict):
        return results

    # Chuẩn hóa toàn bộ biểu đồ ngay sau khi engine tạo xong figures.
    # Không thay đổi dữ liệu hoặc công thức tính danh mục.
    _postprocess_portfolio_figures(
        results.get("portfolio_results"),
        tickers=list(results.get("returns").columns) if isinstance(results.get("returns"), pd.DataFrame) else kwargs.get("tickers", []),
    )

    returns = results.get("returns")
    rf = kwargs.get("risk_free_rate", results.get("risk_free_rate", 0.0))
    target = kwargs.get("target_return", results.get("target_return", 0.15))
    results["risk_free_rate"] = float(rf)
    portfolio_returns = _build_portfolio_returns(results)
    results["portfolio_returns"] = portfolio_returns

    results["_advanced_source_returns"] = returns
    results["_advanced_source_benchmark_returns"] = results.get("benchmark_returns")
    results["_advanced_source_company_table"] = results.get("company_table")

    all_advanced = _run_all_advanced(returns, portfolio_returns, results.get("benchmark_returns"), results.get("company_table"), float(rf), float(target))
    results["advanced_portfolio_analysis_by_portfolio"] = all_advanced

    requested_name = kwargs.get("advanced_portfolio_name")
    if requested_name in all_advanced and not all_advanced[requested_name].get("error"):
        selected_name = requested_name
    elif "Complete Portfolio" in all_advanced and not all_advanced["Complete Portfolio"].get("error"):
        selected_name = "Complete Portfolio"
    elif "Optimal Risky" in all_advanced and not all_advanced["Optimal Risky"].get("error"):
        selected_name = "Optimal Risky"
    elif "Minimum Variance" in all_advanced and not all_advanced["Minimum Variance"].get("error"):
        selected_name = "Minimum Variance"
    elif all_advanced:
        selected_name = next(iter(all_advanced))
    else:
        selected_name = None

    if selected_name is not None:
        selected_analysis = dict(all_advanced[selected_name])
        selected_analysis["_portfolio_name"] = selected_name
        selected_analysis["_target_return"] = float(target)
        selected_analysis["_risk_profile"] = kwargs.get("risk_profile")
        selected_analysis["_all_portfolios"] = all_advanced
        results["advanced_portfolio_analysis"] = selected_analysis
        results["advanced_portfolio_name"] = selected_name
    else:
        results["advanced_portfolio_analysis"] = {}
        results["advanced_portfolio_analysis_error"] = "Không có danh mục hợp lệ để đánh giá."
    _LAST_RESULTS = results
    return results


def get_last_results():
    return _LAST_RESULTS

run_advanced_portfolio_analysis = build_advanced_portfolio_analysis
configure_vnstock = _core.configure_vnstock
clean_margin_table = _core.clean_margin_table
