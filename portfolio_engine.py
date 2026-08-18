import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import portfolio_engine_core as _core
from margin_patch import install_margin_patch
from input_normalizer import normalize_company_dataframe, normalize_income_dataframe
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
_original_income_row_value = _core.income_row_value
_core_run_research = _core.run_research
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
        normalized,
        keywords,
        exclude_keywords=exclude_keywords,
    )
    if pd.notna(value) and not has_unit_metadata:
        value = float(value) * 1000.0
    return value, year, item


_core.income_row_value = income_row_value


def _is_bank_industry(value):
    text = _core.strip_accents(str(value or "")).lower()
    return "ngan hang" in text


def _fresh_bank_revenue(ticker):
    """Lấy Tổng thu nhập hoạt động mới nhất của ngân hàng, bỏ qua cache cũ."""
    try:
        fun = _core.Fundamental()
        income = _core.normalize_columns(
            fun.equity(ticker).income_statement(period="year", orient="report")
        )
        if income is None or income.empty:
            return np.nan, None

        item_col = _core.find_col(income, ["item", "item_name", "name", "indicator"])
        if item_col is None:
            return np.nan, None

        year_cols = sorted(
            [c for c in income.columns if str(c).isdigit() and len(str(c)) == 4],
            key=lambda x: int(str(x)),
            reverse=True,
        )
        if not year_cols:
            return np.nan, None

        items = (
            income[item_col]
            .astype(str)
            .map(_core.strip_accents)
            .str.lower()
            .str.replace("_", " ", regex=False)
            .str.replace("-", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        keywords = [
            "tong thu nhap hoat dong",
            "total operating income",
            "total operating revenue",
            "operating income total",
        ]

        for keyword in keywords:
            mask = items.str.contains(
                _core.strip_accents(keyword).lower(),
                regex=False,
                na=False,
            )
            matches = income.loc[mask]
            if matches.empty:
                continue
            row = matches.iloc[0]
            for year in year_cols:
                value = _core.safe_float(row[year])
                if pd.notna(value):
                    has_unit_metadata = False
                    for col in ("unit_multiplier", "unit"):
                        if col in income.columns:
                            values = income[col].dropna().astype(str).str.strip()
                            if not values.empty and (values != "").any():
                                has_unit_metadata = True
                                break
                    if not has_unit_metadata:
                        value = float(value) * 1000.0
                    return value, str(year)
        return np.nan, None
    except Exception as exc:
        print(f"Không cập nhật được doanh thu ngân hàng {ticker}: {exc}")
        return np.nan, None


def _refresh_bank_revenues(results):
    if not isinstance(results, dict):
        return
    table = results.get("company_table")
    if not isinstance(table, pd.DataFrame) or table.empty:
        return
    if "Mã" not in table.columns or "Ngành" not in table.columns:
        return

    updated = table.copy()
    for idx, row in updated.iterrows():
        if not _is_bank_industry(row.get("Ngành")):
            continue
        ticker = str(row.get("Mã", "")).strip().upper()
        if not ticker:
            continue
        revenue, year = _fresh_bank_revenue(ticker)
        if pd.notna(revenue):
            updated.at[idx, "Doanh thu gần nhất"] = revenue
            if "Năm doanh thu" in updated.columns:
                updated.at[idx, "Năm doanh thu"] = year
    results["company_table"] = updated


# -----------------------------------------------------------------------------
# BIỂU ĐỒ
# -----------------------------------------------------------------------------

_PREFERRED_PORTFOLIOS = [
    "Naive",
    "Minimum Variance",
    "Optimal Risky",
    "Maximum Return",
    "Complete Portfolio",
]


def _is_filtered_chart(fig):
    if fig is None:
        return False
    titles = []
    for ax in fig.axes:
        titles.append(str(ax.get_title()).strip().lower())
    joined = " | ".join(titles)
    return any(
        marker in joined
        for marker in (
            "so sánh rủi ro và lợi suất",
            "danh mục so với mục tiêu lợi nhuận",
            "đường biên markowitz",
        )
    )


def _run_core_without_original_target_charts(*args, **kwargs):
    """Chặn các biểu đồ có nhãn chồng nhau của lõi và vẽ lại một bản duy nhất."""
    original_show = plt.show

    def filtered_show(*show_args, **show_kwargs):
        active = [plt.figure(num) for num in list(plt.get_fignums())]
        for fig in active:
            if _is_filtered_chart(fig):
                try:
                    plt.close(fig)
                except Exception:
                    pass
        remaining = [fig for fig in active if fig.number in plt.get_fignums()]
        if remaining:
            original_show(*show_args, **show_kwargs)

    plt.show = filtered_show
    try:
        return _core_run_research(*args, **kwargs)
    finally:
        plt.show = original_show


def _portfolio_points(portfolio_results, include_target=False):
    points = []
    if not isinstance(portfolio_results, dict):
        return points
    names = list(_PREFERRED_PORTFOLIOS)
    if include_target:
        names.append("Target Return")
    for name in names:
        item = portfolio_results.get(name)
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        weights, stats = item[0], item[1]
        if weights is None or not isinstance(stats, dict):
            continue
        try:
            x = float(stats.get("volatility", np.nan)) * 100.0
            y = float(stats.get("return", np.nan)) * 100.0
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            points.append((name, x, y))
    return points


def _annotation_candidates():
    return [
        (12, 14), (12, -26), (-12, 14), (-12, -26),
        (30, 20), (30, -22), (-30, 20), (-30, -22),
        (48, 0), (-48, 0), (0, 30), (0, -34),
    ]


def _bbox_overlap(a, b):
    x_overlap = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    y_overlap = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return x_overlap * y_overlap


def _annotate_clean(ax, points, fontsize=9, duplicate_tolerance=0.20):
    """Gắn nhãn có kiểm soát va chạm, đặc biệt với các điểm trùng nhau."""
    if not points:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    candidates = _annotation_candidates()

    for name, x, y in points:
        near_count = sum(
            abs(x - ox) <= duplicate_tolerance
            and abs(y - oy) <= duplicate_tolerance
            and other_name != name
            for other_name, ox, oy in points
        )

        local_candidates = candidates
        if near_count:
            local_candidates = [
                (30, 22), (-30, 22), (30, -28), (-30, -28),
                (52, 4), (-52, 4), (0, 34), (0, -38),
                (70, 20), (-70, 20),
            ]

        best = None
        for dx, dy in local_candidates:
            ann = ax.annotate(
                name,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                fontsize=fontsize,
                bbox=dict(boxstyle="round,pad=0.22", alpha=0.78),
                arrowprops=(
                    dict(arrowstyle="-", alpha=0.35, linewidth=0.8)
                    if abs(dx) + abs(dy) >= 45 else None
                ),
                annotation_clip=True,
                zorder=20,
            )
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer).expanded(1.08, 1.15)
            overlap = sum(_bbox_overlap(bbox, old) for old in placed)
            outside = (
                max(0.0, fig.bbox.x0 - bbox.x0)
                + max(0.0, bbox.x1 - fig.bbox.x1)
                + max(0.0, fig.bbox.y0 - bbox.y0)
                + max(0.0, bbox.y1 - fig.bbox.y1)
            )
            score = overlap * 1000.0 + outside * 1000.0 + 0.01 * (dx * dx + dy * dy)
            if best is None or score < best[0]:
                if best is not None:
                    best[2].remove()
                best = (score, bbox, ann)
            else:
                ann.remove()

        if best is not None:
            placed.append(best[1])


def _render_target_chart(portfolio_results, target_return):
    points = _portfolio_points(portfolio_results, include_target=False)
    if not points:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, x, y in points:
        ax.scatter(x, y, s=105, zorder=5)
    _annotate_clean(ax, points)
    ax.axhline(
        float(target_return) * 100.0,
        linestyle="--",
        linewidth=1.5,
        label=f"Mục tiêu {float(target_return):.1%}",
        zorder=1,
    )
    ax.set_xlabel("Rủi ro năm (%)")
    ax.set_ylabel("Lợi suất kỳ vọng năm (%)")
    ax.set_title("Danh mục so với mục tiêu lợi nhuận")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()


def _render_markowitz_chart(portfolio_results, frontier, target_return):
    if not isinstance(frontier, pd.DataFrame) or frontier.empty:
        return
    if not {"Risk", "Return"}.issubset(frontier.columns):
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        pd.to_numeric(frontier["Risk"], errors="coerce") * 100.0,
        pd.to_numeric(frontier["Return"], errors="coerce") * 100.0,
        linewidth=2.2,
        label="Đường biên Markowitz",
        zorder=2,
    )

    points = _portfolio_points(portfolio_results, include_target=False)
    for name, x, y in points:
        ax.scatter(x, y, s=90, zorder=5)

    target_points = _portfolio_points(portfolio_results, include_target=True)
    target_point = next((p for p in target_points if p[0] == "Target Return"), None)
    if target_point is not None:
        _, tx, ty = target_point
        ax.scatter(
            tx, ty, s=190, marker="*",
            label="Danh mục Markowitz theo mục tiêu",
            zorder=7,
        )

    labels = points.copy()
    if target_point is not None:
        labels.append(target_point)
    _annotate_clean(ax, labels, fontsize=9, duplicate_tolerance=0.25)

    ax.axhline(
        float(target_return) * 100.0,
        linestyle="--",
        linewidth=1.5,
        label=f"Mục tiêu {float(target_return):.1%}",
        zorder=1,
    )
    ax.set_xlabel("Rủi ro năm (%)")
    ax.set_ylabel("Lợi suất kỳ vọng năm (%)")
    ax.set_title("Đường biên Markowitz: lợi suất mục tiêu và rủi ro")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()


def _portfolio_name_from_title(title, names):
    text = str(title or "").strip().lower()
    for name in names:
        if name.lower() in text:
            return name
    return None


def _clean_zero_weight_pies(portfolio_results, tickers=None):
    if not isinstance(portfolio_results, dict):
        return
    names = [name for name in _PREFERRED_PORTFOLIOS if name in portfolio_results]
    tickers = list(tickers or [])
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        for ax in fig.axes:
            wedges = [p for p in ax.patches if p.__class__.__name__ == "Wedge"]
            if not wedges:
                continue
            portfolio_name = _portfolio_name_from_title(ax.get_title(), names)
            if portfolio_name is None:
                continue
            item = portfolio_results.get(portfolio_name)
            if not isinstance(item, (tuple, list)) or not item:
                continue
            try:
                weights = np.asarray(item[0], dtype=float).reshape(-1)
            except Exception:
                continue
            labels = tickers[:len(weights)]
            zero_labels = {
                str(labels[i]).strip()
                for i, weight in enumerate(weights)
                if i < len(labels) and np.isfinite(weight) and abs(weight) <= 1e-10
            }
            for wedge in list(wedges):
                if str(wedge.get_label()).strip() in zero_labels:
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
    rf = results.get("risk_free_rate", 0.0)
    output = {}
    if not isinstance(returns, pd.DataFrame) or returns.empty or not isinstance(portfolio_results, dict):
        return output
    for name, item in portfolio_results.items():
        try:
            series = _portfolio_return_from_item(item, returns, risk_free_rate=rf, name=name)
            if not series.empty:
                output[name] = series
        except Exception:
            continue
    return output


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


def _run_all_advanced(returns, portfolio_returns, benchmark_returns, company_table, risk_free_rate, target_return):
    analyses = {}
    for name, p_returns in portfolio_returns.items():
        try:
            analysis = build_advanced_portfolio_analysis(
                returns,
                p_returns,
                benchmark_returns,
                company_table,
                float(risk_free_rate),
                float(target_return),
            )
            analysis["_portfolio_name"] = name
            analysis["_portfolio_returns"] = p_returns
            analysis["_data_frequency"] = "Ngày"
            analyses[name] = analysis
        except Exception as exc:
            analyses[name] = {"error": str(exc), "_portfolio_name": name}
    return analyses


def run_research(*args, **kwargs):
    global _LAST_RESULTS
    results = _run_core_without_original_target_charts(*args, **kwargs)
    if not isinstance(results, dict):
        return results

    returns = results.get("returns")
    rf = float(kwargs.get("risk_free_rate", results.get("risk_free_rate", 0.0)))
    target = float(kwargs.get("target_return", results.get("target_return", 0.15)))

    _refresh_bank_revenues(results)

    _render_target_chart(results.get("portfolio_results"), target)
    _render_markowitz_chart(
        results.get("portfolio_results"),
        results.get("efficient_frontier", results.get("frontier")),
        target,
    )

    if isinstance(returns, pd.DataFrame):
        _clean_zero_weight_pies(
            results.get("portfolio_results"),
            tickers=list(returns.columns),
        )

    results["risk_free_rate"] = rf
    portfolio_returns = _build_portfolio_returns(results)
    results["portfolio_returns"] = portfolio_returns
    results["_advanced_source_returns"] = returns
    results["_advanced_source_benchmark_returns"] = results.get("benchmark_returns")
    results["_advanced_source_company_table"] = results.get("company_table")

    all_advanced = _run_all_advanced(
        returns,
        portfolio_returns,
        results.get("benchmark_returns"),
        results.get("company_table"),
        rf,
        target,
    ) if isinstance(returns, pd.DataFrame) else {}
    results["advanced_portfolio_analysis_by_portfolio"] = all_advanced

    requested_name = kwargs.get("advanced_portfolio_name")
    selected_name = None
    for candidate in [requested_name, "Complete Portfolio", "Optimal Risky", "Minimum Variance"]:
        if candidate in all_advanced and not all_advanced[candidate].get("error"):
            selected_name = candidate
            break
    if selected_name is None and all_advanced:
        selected_name = next(iter(all_advanced))

    if selected_name is not None:
        selected = dict(all_advanced[selected_name])
        selected["_portfolio_name"] = selected_name
        selected["_target_return"] = target
        selected["_risk_profile"] = kwargs.get("risk_profile")
        selected["_all_portfolios"] = all_advanced
        results["advanced_portfolio_analysis"] = selected
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
