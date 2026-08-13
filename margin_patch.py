import numpy as np
import pandas as pd


def _safe_margin_pct(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "không", "khong", "no", "none", "nan", "n/a", "na", "null"}:
            return 0.0
        text = text.replace("%", "").replace(",", ".")
        try:
            value = float(text)
        except (TypeError, ValueError):
            return 0.0
    else:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
    if not np.isfinite(value):
        return 0.0
    if value > 1:
        value /= 100.0
    return max(value, 0.0)


def _eligible(value):
    if pd.isna(value):
        return "Không"
    return "Có" if str(value).strip().lower() in {"có", "co", "yes", "true", "1", "y"} else "Không"


def clean_margin_table(df, tickers, engine):
    if df is None or df.empty:
        return engine.default_margin_table(tickers)

    out = df.copy()
    for col in ["Mã", "Được cấp margin", "Tỷ lệ cho vay", "Lãi suất vay", "Ngày cập nhật"]:
        if col not in out.columns:
            if col == "Mã":
                out[col] = tickers
            elif col == "Được cấp margin":
                out[col] = "Không"
            elif col in ["Tỷ lệ cho vay", "Lãi suất vay"]:
                out[col] = 0.0
            else:
                out[col] = ""

    out["Mã"] = out["Mã"].astype(str).str.strip().str.upper()
    out = out[out["Mã"].isin(tickers)].copy()
    out["Được cấp margin"] = out["Được cấp margin"].apply(_eligible)
    out["Tỷ lệ cho vay"] = out["Tỷ lệ cho vay"].apply(_safe_margin_pct)
    out["Lãi suất vay"] = out["Lãi suất vay"].apply(_safe_margin_pct)

    mask = out["Được cấp margin"] != "Có"
    out.loc[mask, ["Tỷ lệ cho vay", "Lãi suất vay"]] = 0.0

    missing = [t for t in tickers if t not in set(out["Mã"])]
    if missing:
        out = pd.concat([out, engine.default_margin_table(missing)], ignore_index=True)

    return out[["Mã", "Được cấp margin", "Tỷ lệ cho vay", "Lãi suất vay", "Ngày cập nhật"]].set_index("Mã").reindex(tickers).reset_index()


def margin_position_limits(margin_table, tickers, max_leverage, engine):
    margin = clean_margin_table(margin_table, tickers, engine)
    limits = {}
    for ticker in tickers:
        row = margin[margin["Mã"] == ticker]
        if row.empty or row.iloc[0]["Được cấp margin"] != "Có":
            limits[ticker] = 1.0
        else:
            loan = _safe_margin_pct(row.iloc[0]["Tỷ lệ cho vay"])
            limits[ticker] = min(1.0 + loan, max(float(max_leverage), 1.0))
    return np.array([limits[t] for t in tickers], dtype=float)


def apply_leverage_to_portfolio(base_weights, returns, margin_table, max_leverage, engine):
    base = np.clip(np.asarray(base_weights, dtype=float), 0, None)
    if base.sum() <= 0:
        return None
    base = base / base.sum()
    tickers = list(returns.columns)
    limits = margin_position_limits(margin_table, tickers, max_leverage, engine)
    scales = [limit / weight for weight, limit in zip(base, limits) if weight > 0]
    if not scales:
        return None

    factor = max(1.0, min(max(float(max_leverage), 1.0), min(scales)))
    levered = base * factor
    borrowed = max(float(levered.sum()) - 1.0, 0.0)

    margin = clean_margin_table(margin_table, tickers, engine)
    incremental = np.clip(levered - base, 0, None)
    total = float(incremental.sum())

    borrowing_rate = 0.0
    if total > 0:
        for i, ticker in enumerate(tickers):
            row = margin[margin["Mã"] == ticker]
            rate = _safe_margin_pct(row.iloc[0]["Lãi suất vay"]) if not row.empty else 0.0
            borrowing_rate += incremental[i] / total * rate

    borrowing_cost = borrowed * borrowing_rate
    net_returns = returns @ levered - borrowing_cost / engine.PERIODS_PER_YEAR

    return {
        "weights": levered,
        "leverage": float(levered.sum()),
        "borrowed": borrowed,
        "borrowing_rate": borrowing_rate,
        "borrowing_cost": borrowing_cost,
        "returns": net_returns,
    }


def install_margin_patch(engine):
    engine.clean_margin_table = lambda df, tickers: clean_margin_table(df, tickers, engine)
    engine.margin_position_limits = lambda margin_table, tickers, max_leverage: margin_position_limits(margin_table, tickers, max_leverage, engine)
    engine.apply_leverage_to_portfolio = lambda base_weights, returns, margin_table, max_leverage: apply_leverage_to_portfolio(base_weights, returns, margin_table, max_leverage, engine)
