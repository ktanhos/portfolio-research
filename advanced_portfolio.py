import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _annualized_return(r):
    r = pd.Series(r).dropna()
    if r.empty:
        return np.nan
    wealth = (1 + r).prod()
    years = len(r) / TRADING_DAYS
    return wealth ** (1 / years) - 1 if years > 0 and wealth > 0 else np.nan


def _volatility(r):
    r = pd.Series(r).dropna()
    return r.std() * np.sqrt(TRADING_DAYS) if len(r) > 1 else np.nan


def _sharpe(r, rf):
    vol = _volatility(r)
    ret = _annualized_return(r)
    return (ret - rf) / vol if pd.notna(vol) and vol > 0 else np.nan


def _max_drawdown(r):
    r = pd.Series(r).dropna()
    if r.empty:
        return np.nan
    wealth = (1 + r).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def rolling_return(r, window=252):
    r = pd.Series(r).dropna()
    return (1 + r).rolling(window).apply(np.prod, raw=True) - 1


def rolling_sharpe(r, rf=0.0, window=252):
    r = pd.Series(r)
    mean = r.rolling(window).mean() * TRADING_DAYS
    vol = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return (mean - rf) / vol.replace(0, np.nan)


def _patch_core_scatter_labels():
    try:
        import portfolio_engine_core as core

        def _annotate(ax, data, x_column, y_column, label_column):
            if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                return
            offsets = [(8, 8), (8, -14), (-8, 8), (-8, -14), (14, 18), (14, -22), (-14, 18), (-14, -22)]
            existing = {str(t.get_text()).strip() for t in ax.texts if str(t.get_text()).strip()}
            for i, (_, row) in enumerate(data.iterrows()):
                try:
                    x = float(row[x_column])
                    y = float(row[y_column])
                    label = str(row[label_column]).strip()
                except Exception:
                    continue
                if not np.isfinite(x) or not np.isfinite(y) or not label or label in existing:
                    continue
                dx, dy = offsets[i % len(offsets)]
                ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", alpha=0.7), zorder=10)
                existing.add(label)

        core.annotate_scatter_labels = _annotate
    except Exception:
        pass


_patch_core_scatter_labels()


def factor_proxy_analysis(returns, company_table=None):
    if returns is None or returns.empty:
        return pd.DataFrame()
    rows = []
    prices = (1 + returns.fillna(0)).cumprod()
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        if r.empty:
            continue
        trend = prices[ticker].iloc[-1] / prices[ticker].iloc[max(0, len(prices) - 252)] - 1 if len(prices) > 252 else np.nan
        vol = _volatility(r)
        liquidity = np.nan
        quality = np.nan
        value = np.nan
        if company_table is not None and not company_table.empty and "Mã" in company_table.columns:
            row = company_table.loc[company_table["Mã"] == ticker]
            if not row.empty:
                first = row.iloc[0]
                if "Vốn hóa" in row.columns:
                    liquidity = first.get("Vốn hóa", np.nan)
                if "P/B" in row.columns:
                    value = first.get("P/B", np.nan)
                if "ROE" in row.columns:
                    quality = first.get("ROE", np.nan)
        rows.append({"Mã": ticker, "Momentum 12T": trend, "Volatility": vol, "Size proxy vốn hóa": liquidity, "Value proxy P/B": value, "Quality proxy ROE": quality})
    return pd.DataFrame(rows)


def multifactor_regression(portfolio_returns, factor_returns, rf=0.0):
    y = pd.Series(portfolio_returns).dropna()
    if factor_returns is None or factor_returns.empty:
        return pd.DataFrame()
    x = factor_returns.reindex(y.index).copy()
    x["_intercept"] = 1.0
    data = pd.concat([y.rename("Portfolio"), x], axis=1).dropna()
    if len(data) < max(30, data.shape[1] * 5):
        return pd.DataFrame()
    yv = data["Portfolio"].values - rf / TRADING_DAYS
    cols = [c for c in data.columns if c != "Portfolio"]
    X = data[cols].values
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    fitted = X @ beta
    resid = yv - fitted
    ssr = float(np.sum((yv - yv.mean()) ** 2))
    sse = float(np.sum(resid ** 2))
    r2 = 1 - sse / ssr if ssr > 0 else np.nan
    out = pd.DataFrame({"Hệ số": beta}, index=cols)
    out.loc["R²", "Hệ số"] = r2
    out.loc["Specific Risk", "Hệ số"] = resid.std() * np.sqrt(TRADING_DAYS)
    out.loc["Alpha năm", "Hệ số"] = beta[cols.index("_intercept")] * TRADING_DAYS if "_intercept" in cols else np.nan
    return out


def var_analysis(r, level=0.95, simulations=10000, seed=42):
    r = pd.Series(r).dropna()
    if r.empty:
        return pd.DataFrame()
    loss = -r
    hist = float(loss.quantile(level))
    mu = float(r.mean())
    sigma = float(r.std())
    if sigma > 0:
        rng = np.random.default_rng(seed)
        sim = rng.normal(mu, sigma, simulations)
        mc = float(np.quantile(-sim, level))
        param = float(-(mu + sigma * __import__('scipy').stats.norm.ppf(1-level)))
    else:
        mc = param = np.nan
    tail = loss[loss >= hist]
    cvar = float(tail.mean()) if not tail.empty else np.nan
    return pd.DataFrame([{"Historical VaR": hist, "Parametric VaR": param, "Monte Carlo VaR": mc, "CVaR": cvar}])


def active_analysis(portfolio_returns, benchmark_returns, rf=0.0):
    p = pd.Series(portfolio_returns).dropna()
    b = pd.Series(benchmark_returns).reindex(p.index).dropna()
    p = p.reindex(b.index).dropna()
    b = b.reindex(p.index).dropna()
    if p.empty:
        return pd.DataFrame()
    active = p - b
    active_return = _annualized_return(p) - _annualized_return(b)
    te = active.std() * np.sqrt(TRADING_DAYS) if len(active) > 1 else np.nan
    ir = active_return / te if pd.notna(te) and te > 0 else np.nan
    return pd.DataFrame([{"Active Return": active_return, "Tracking Error": te, "Information Ratio": ir, "Beta": p.cov(b) / b.var() if b.var() > 0 else np.nan, "Tương quan": p.corr(b)}])


def robustness_analysis(r, target_return=0.15, windows=(126, 252, 504)):
    r = pd.Series(r).dropna()
    rows = []
    for w in windows:
        if len(r) < w:
            continue
        x = r.iloc[-w:]
        rows.append({"Rolling Window": w, "CAGR": _annualized_return(x), "Rủi ro": _volatility(x), "Sharpe": _sharpe(x, 0.0), "Max Drawdown": _max_drawdown(x), "Đạt mục tiêu": "Có" if _annualized_return(x) >= target_return else "Không"})
    return pd.DataFrame(rows)


def walk_forward_analysis(r, train_ratio=0.6, test_ratio=0.2, rf=0.0):
    r = pd.Series(r).dropna()
    n = len(r)
    if n < 252:
        return pd.DataFrame()
    train = max(60, int(n * train_ratio))
    test = max(30, int(n * test_ratio))
    rows = []
    start = 0
    fold = 1
    while start + train + test <= n:
        train_r = r.iloc[start:start+train]
        test_r = r.iloc[start+train:start+train+test]
        rows.append({"Fold": fold, "Train CAGR": _annualized_return(train_r), "Train Sharpe": _sharpe(train_r, rf), "Test CAGR": _annualized_return(test_r), "Test Sharpe": _sharpe(test_r, rf), "Test Max Drawdown": _max_drawdown(test_r)})
        start += test
        fold += 1
    return pd.DataFrame(rows)
