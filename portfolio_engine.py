import pandas as pd
import numpy as np

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
    value, year, item = _original_income_row_value(
        normalized, keywords, exclude_keywords=exclude_keywords
    )
    if pd.notna(value) and not has_unit_metadata:
        value = float(value) * 1000.0
    return value, year, item


_core.income_row_value = income_row_value
_core_run_research = _core.run_research


def build_advanced_portfolio_analysis(
    returns,
    portfolio_returns,
    benchmark_returns=None,
    company_table=None,
    risk_free_rate=0.0,
    target_return=0.15,
):
    result = {}
    try:
        result["factor_proxy"] = factor_proxy_analysis(returns, company_table=company_table)
    except Exception as exc:
        result["factor_proxy"] = pd.DataFrame()
        result["factor_proxy_error"] = str(exc)

    if benchmark_returns is not None:
        try:
            factor_returns = pd.DataFrame({"Market": pd.Series(benchmark_returns)})
            result["multifactor_regression"] = multifactor_regression(
                portfolio_returns, factor_returns, rf=risk_free_rate
            )
            result["active_analysis"] = active_analysis(
                portfolio_returns, benchmark_returns, rf=risk_free_rate
            )
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


def _build_portfolio_returns(results):
    returns = results.get("returns")
    portfolio_results = results.get("portfolio_results", {})
    portfolio_returns = {}
    if isinstance(returns, pd.DataFrame) and not returns.empty:
        for name, item in portfolio_results.items():
            try:
                if item is None or item[0] is None:
                    continue
                w = np.asarray(item[0], dtype=float)
                if len(w) != returns.shape[1]:
                    continue
                portfolio_returns[name] = returns.mul(w, axis=1).sum(axis=1).dropna()
            except Exception:
                continue
    return portfolio_returns


def _run_all_advanced(returns, portfolio_returns, benchmark_returns, company_table, risk_free_rate, target_return):
    analyses = {}
    for name, p_returns in portfolio_returns.items():
        try:
            analyses[name] = build_advanced_portfolio_analysis(
                returns=returns,
                portfolio_returns=p_returns,
                benchmark_returns=benchmark_returns,
                company_table=company_table,
                risk_free_rate=float(risk_free_rate),
                target_return=float(target_return),
            )
        except Exception as exc:
            analyses[name] = {"error": str(exc)}
    return analyses


def run_research(*args, **kwargs):
    results = _core_run_research(*args, **kwargs)
    if not isinstance(results, dict):
        return results

    returns = results.get("returns")
    portfolio_returns = _build_portfolio_returns(results)
    results["portfolio_returns"] = portfolio_returns

    rf = kwargs.get("risk_free_rate", 0.0)
    target = kwargs.get("target_return", results.get("target_return", 0.15))
    all_advanced = _run_all_advanced(
        returns=returns,
        portfolio_returns=portfolio_returns,
        benchmark_returns=results.get("benchmark_returns"),
        company_table=results.get("company_table"),
        risk_free_rate=float(rf),
        target_return=float(target),
    )
    results["advanced_portfolio_analysis_by_portfolio"] = all_advanced

    selected_name = None
    for candidate in ["Complete Portfolio", "Optimal Risky", "Minimum Variance"]:
        if candidate in all_advanced and not all_advanced[candidate].get("error"):
            selected_name = candidate
            break
    if selected_name is None and all_advanced:
        selected_name = next(iter(all_advanced))

    if selected_name is not None:
        results["advanced_portfolio_analysis"] = all_advanced[selected_name]
        results["advanced_portfolio_name"] = selected_name
    else:
        results["advanced_portfolio_analysis"] = {}
        results["advanced_portfolio_analysis_error"] = "Không có danh mục hợp lệ."
    return results


run_advanced_portfolio_analysis = build_advanced_portfolio_analysis
configure_vnstock = _core.configure_vnstock
clean_margin_table = _core.clean_margin_table
