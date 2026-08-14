import numpy as np
import pandas as pd

from portfolio_engine import run_advanced_portfolio_analysis


def _safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _status_target(cagr, target):
    if pd.isna(cagr):
        return "Không đủ dữ liệu"
    return "Đạt" if cagr >= target else "Không đạt"


def _status_sharpe(value):
    if pd.isna(value):
        return "Không đủ dữ liệu"
    if value >= 1.0:
        return "Tốt"
    if value >= 0.5:
        return "Khá"
    if value >= 0:
        return "Yếu"
    return "Không đạt"


def _status_drawdown(value):
    if pd.isna(value):
        return "Không đủ dữ liệu"
    x = abs(value)
    if x <= 0.15:
        return "Tốt"
    if x <= 0.25:
        return "Chấp nhận được"
    if x <= 0.35:
        return "Cần lưu ý"
    return "Rủi ro cao"


def _status_active(value):
    if pd.isna(value):
        return "Không đủ dữ liệu"
    if value >= 0.5:
        return "Tích cực"
    if value >= 0:
        return "Trung tính"
    return "Kém"


def _status_oos(value):
    if pd.isna(value):
        return "Không đủ dữ liệu"
    if value >= 0.5:
        return "Ổn định"
    if value >= 0:
        return "Trung bình"
    return "Không ổn định"


def summarize_portfolio(name, advanced, target_return):
    robustness = advanced.get("robustness", pd.DataFrame())
    active = advanced.get("active_analysis", pd.DataFrame())
    walk = advanced.get("walk_forward", pd.DataFrame())
    var = advanced.get("var_analysis", pd.DataFrame())

    cagr = sharpe = max_dd = ir = oos_sharpe = var95 = cvar = np.nan
    if isinstance(robustness, pd.DataFrame) and not robustness.empty:
        row = robustness.iloc[-1]
        cagr = _safe_float(row.get("CAGR"))
        sharpe = _safe_float(row.get("Sharpe"))
        max_dd = _safe_float(row.get("Max Drawdown"))
    if isinstance(active, pd.DataFrame) and not active.empty:
        row = active.iloc[0]
        ir = _safe_float(row.get("Information Ratio"))
    if isinstance(walk, pd.DataFrame) and not walk.empty and "Test Sharpe" in walk.columns:
        values = pd.to_numeric(walk["Test Sharpe"], errors="coerce").dropna()
        if not values.empty:
            oos_sharpe = float(values.median())
    if isinstance(var, pd.DataFrame) and not var.empty:
        row = var.iloc[0]
        var95 = _safe_float(row.get("Historical VaR"))
        cvar = _safe_float(row.get("CVaR"))

    statuses = [
        _status_target(cagr, target_return),
        _status_sharpe(sharpe),
        _status_drawdown(max_dd),
        _status_active(ir),
        _status_oos(oos_sharpe),
    ]
    positive = sum(x in {"Đạt", "Tốt", "Khá", "Chấp nhận được", "Tích cực", "Ổn định"} for x in statuses)
    negative = sum(x in {"Không đạt", "Rủi ro cao", "Kém", "Không ổn định"} for x in statuses)
    if negative >= 2:
        overall = "Cần xem xét"
    elif positive >= 4:
        overall = "Tích cực"
    elif positive >= 2:
        overall = "Trung tính"
    else:
        overall = "Chưa đủ cơ sở"

    warnings = []
    if pd.notna(cagr) and cagr < target_return:
        warnings.append("Lợi suất thực tế chưa đạt mục tiêu.")
    if pd.notna(max_dd) and max_dd <= -0.35:
        warnings.append("Maximum Drawdown ở mức cao.")
    if pd.notna(ir) and ir < 0:
        warnings.append("Hiệu quả tương đối so với VNINDEX là âm sau khi điều chỉnh theo active risk.")
    if pd.notna(oos_sharpe) and oos_sharpe < 0:
        warnings.append("Hiệu quả ngoài mẫu có dấu hiệu không ổn định.")
    if pd.notna(var95) and pd.notna(cvar) and abs(cvar) > abs(var95) * 1.5:
        warnings.append("Rủi ro đuôi phân phối lớn hơn đáng kể so với VaR lịch sử.")

    return {
        "Danh mục": name,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Maximum Drawdown": max_dd,
        "Information Ratio": ir,
        "OOS Sharpe trung vị": oos_sharpe,
        "VaR 95%": var95,
        "CVaR 95%": cvar,
        "Mục tiêu": _status_target(cagr, target_return),
        "Rủi ro điều chỉnh": _status_sharpe(sharpe),
        "Rủi ro giảm giá": _status_drawdown(max_dd),
        "So với VNINDEX": _status_active(ir),
        "Độ ổn định OOS": _status_oos(oos_sharpe),
        "Đánh giá": overall,
        "Cảnh báo": " ".join(warnings) if warnings else "Không có cảnh báo nổi bật",
    }


def build_all_portfolio_evaluations(results, target_return=0.15, risk_free_rate=0.0):
    """Đánh giá từng danh mục từ đúng dữ liệu run_research đã trả về.

    Không gọi dữ liệu bên ngoài và không truy vấn API mới.
    """
    portfolio_returns = results.get("portfolio_returns", {})
    returns = results.get("returns")
    benchmark_returns = results.get("benchmark_returns")
    company_table = results.get("company_table")
    analyses = {}
    summaries = []

    if not isinstance(portfolio_returns, dict):
        return {}, pd.DataFrame()

    for name, p_returns in portfolio_returns.items():
        try:
            advanced = run_advanced_portfolio_analysis(
                returns=returns,
                portfolio_returns=p_returns,
                benchmark_returns=benchmark_returns,
                company_table=company_table,
                risk_free_rate=float(risk_free_rate),
                target_return=float(target_return),
            )
            analyses[name] = advanced
            summaries.append(summarize_portfolio(name, advanced, target_return))
        except Exception as exc:
            analyses[name] = {"error": str(exc)}
            summaries.append({
                "Danh mục": name,
                "Đánh giá": "Không đủ cơ sở",
                "Cảnh báo": str(exc),
            })

    return analyses, pd.DataFrame(summaries)


def choose_focus_portfolio(summary, preferred="Complete Portfolio"):
    if summary is None or summary.empty:
        return None
    if "Danh mục" in summary.columns and preferred in summary["Danh mục"].values:
        return preferred
    if "Đánh giá" in summary.columns:
        order = {"Tích cực": 3, "Trung tính": 2, "Chưa đủ cơ sở": 1, "Cần xem xét": 0}
        ranked = summary.assign(_rank=summary["Đánh giá"].map(order).fillna(-1))
        return ranked.sort_values("_rank", ascending=False).iloc[0]["Danh mục"]
    return summary.iloc[0]["Danh mục"]
