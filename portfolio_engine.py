
# ============================================================
# PORTFOLIO RESEARCH 
# VNSTOCK COMMUNITY / GOOGLE COLAB
# ============================================================


import warnings
warnings.filterwarnings("ignore")

import unicodedata

import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from vnstock import Market, Fundamental, Reference, register_user


# Tương thích khi chạy engine bên ngoài Google Colab.
def display(*args, **kwargs):
    return None

def clear_output(*args, **kwargs):
    return None

PERIODS_PER_YEAR = 52
RISK_FREE_RATE_RUNTIME = 0.04

# ============================================================
# XÁC THỰC VNSTOCK
# ============================================================

def configure_vnstock(api_key):
    """
    Đăng ký API key Vnstock nếu người dùng nhập key.
    Không in hoặc lưu API key vào kết quả hiển thị.
    """
    api_key = (api_key or "").strip()

    if not api_key:
        return {
            "authenticated": False,
            "message": "Đang dùng chế độ khách của Vnstock."
        }

    try:
        register_user(api_key=api_key)
        return {
            "authenticated": True,
            "message": "Đã xác thực API key Vnstock."
        }
    except Exception as e:
        raise ValueError(
            "API key Vnstock không hợp lệ hoặc đăng ký thất bại. "
            f"Chi tiết: {e}"
        )

# ============================================================
# CACHE
# ============================================================

CACHE_DIR = Path(".portfolio_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_PAUSE = 1.2

def cache_path(kind, key):
    safe = str(key).replace("/", "_").replace("\\", "_").replace(":", "_")
    return CACHE_DIR / f"{kind}_{safe}.csv"

def pause_api():
    time.sleep(REQUEST_PAUSE)

def normalize_columns(df):
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df

def find_col(df, candidates):
    if df is None or df.empty:
        return None
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.replace(",", "").replace("%", "").strip()
        return float(x)
    except Exception:
        return np.nan

def first_value(df, candidates, row=0):
    if df is None or df.empty:
        return np.nan
    df = normalize_columns(df)
    for c in candidates:
        if c in df.columns:
            try:
                return df.iloc[row][c]
            except Exception:
                pass
    return np.nan

def parse_date(value):
    return pd.to_datetime(value, dayfirst=True).strftime("%Y-%m-%d")

# ============================================================
# PRICE DATA
# ============================================================

def get_price_data(tickers, start_date, end_date):
    market = Market()
    output = {}

    for ticker in tickers:
        key = f"{ticker}_{start_date}_{end_date}"
        path = cache_path("price", key)

        if path.exists():
            try:
                cached = pd.read_csv(
                    path,
                    parse_dates=["Date"],
                    index_col="Date"
                )
                s = pd.to_numeric(cached["close"], errors="coerce")
                if (
                    len(s) > 100
                    and s.index.min() <= pd.Timestamp(start_date)
                    and s.index.max() >= pd.Timestamp(end_date)
                ):
                    output[ticker] = s
                    print(f"{ticker}: dùng dữ liệu giá từ cache")
                    continue
            except Exception:
                pass

        print(f"{ticker}: đang lấy dữ liệu giá...")

        try:
            # start/end are explicitly supplied.
            # count is included as a safety net for versions that
            # otherwise return only the default 100 rows.
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            calendar_days = max((end_ts - start_ts).days, 1)
            estimated_sessions = int(calendar_days * 0.72) + 50
            count = max(estimated_sessions, 300)

            # Vnstock documents count as the number of candles when
            # retrieving historical data. In some versions, passing
            # start/end together can still fall back to the 100-row default.
            # Therefore request the required number of sessions backwards
            # from the selected end date, then filter to start/end below.
            try:
                df = market.equity(ticker).ohlcv(
                    end=end_date,
                    interval="1D",
                    count=count
                )
            except TypeError:
                df = market.equity(ticker).ohlcv(
                    end=end_date,
                    interval="1D",
                    count=count
                )

            df = normalize_columns(df)

            if df.empty:
                raise ValueError("API trả về dữ liệu rỗng")

            date_col = "time" if "time" in df.columns else "date"
            if date_col not in df.columns:
                raise ValueError("Không tìm thấy cột thời gian")

            if "close" not in df.columns:
                raise ValueError("Không tìm thấy cột close")

            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()

            close = pd.to_numeric(
                df["close"], errors="coerce"
            ).dropna()

            close = close[
                (close.index >= pd.Timestamp(start_date))
                & (close.index <= pd.Timestamp(end_date))
            ]

            if len(close) < 150:
                raise ValueError(
                    f"Chỉ nhận được {len(close)} phiên, "
                    "không đủ cho khoảng thời gian yêu cầu."
                )

            tmp = close.to_frame("close")
            tmp.index.name = "Date"
            tmp.to_csv(path)

            output[ticker] = close

            pause_api()

        except Exception as e:
            print(f"Lỗi giá {ticker}: {e}")

    prices = pd.DataFrame(output).sort_index()
    prices.index = pd.to_datetime(prices.index).normalize()
    prices.index.name = "Date"

    return prices

# ============================================================
# BENCHMARK
# ============================================================

def get_benchmark_prices(benchmark, start_date, end_date):
    """
    Lấy benchmark theo từng đoạn thời gian ngắn.

    Vấn đề của phiên bản Community hiện tại:
    index.ohlcv có thể trả tối đa khoảng 100 phiên dù truyền
    start/end cho một khoảng rất dài.

    Vì vậy không gọi một lần cho cả giai đoạn 2022 đến 2026.
    Chia thành các đoạn tối đa 120 ngày, lấy từng đoạn rồi ghép lại.

    Cách này:
    1. Không mất dữ liệu do giới hạn 100 phiên.
    2. Không cần dùng length lớn mà nguồn có thể bỏ qua.
    3. Có cache riêng cho từng đoạn.
    4. Giữ nguyên giá trị dữ liệu, chỉ chia nhỏ truy vấn.
    """

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()

    if start_ts > end_ts:
        raise ValueError("Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc.")

    # Cache toàn bộ benchmark của đúng khoảng nghiên cứu.
    full_key = f"{benchmark}_{start_date}_{end_date}"
    full_path = cache_path("benchmark_full_v12", full_key)

    if full_path.exists():
        try:
            cached = pd.read_csv(
                full_path,
                parse_dates=["Date"],
                index_col="Date"
            )

            close = pd.to_numeric(
                cached["close"],
                errors="coerce"
            ).dropna()

            close.index = pd.to_datetime(
                close.index
            ).normalize()

            close = close[
                (close.index >= start_ts)
                & (close.index <= end_ts)
            ].sort_index()

            expected = max(
                int(((end_ts - start_ts).days + 1) * 0.68),
                10
            )

            # Chỉ dùng cache nếu nó đủ dài cho khoảng nghiên cứu.
            minimum_required = max(
                10,
                int(expected * 0.70)
            )

            if len(close) >= minimum_required:
                print(
                    f"{benchmark}: dùng benchmark từ cache "
                    f"({len(close):,} phiên)"
                )
                return close

        except Exception:
            pass

    market = Market()
    pieces = []

    # Mỗi đoạn 120 ngày thường chỉ khoảng 80 phiên giao dịch,
    # thấp hơn giới hạn 100 phiên của API Community.
    chunk_days = 120

    cursor = start_ts

    while cursor <= end_ts:

        chunk_end = min(
            cursor + pd.Timedelta(days=chunk_days - 1),
            end_ts
        )

        chunk_key = (
            f"{benchmark}_"
            f"{cursor.strftime('%Y%m%d')}_"
            f"{chunk_end.strftime('%Y%m%d')}"
        )

        chunk_path = cache_path(
            "benchmark_chunk_v12",
            chunk_key
        )

        close_chunk = None

        # ----------------------------------------------------
        # CACHE CỦA ĐOẠN
        # ----------------------------------------------------
        if chunk_path.exists():
            try:
                cached = pd.read_csv(
                    chunk_path,
                    parse_dates=["Date"],
                    index_col="Date"
                )

                close_chunk = pd.to_numeric(
                    cached["close"],
                    errors="coerce"
                ).dropna()

                close_chunk.index = pd.to_datetime(
                    close_chunk.index
                ).normalize()

                close_chunk = close_chunk[
                    (close_chunk.index >= cursor)
                    & (close_chunk.index <= chunk_end)
                ]

            except Exception:
                close_chunk = None

        # ----------------------------------------------------
        # API
        # ----------------------------------------------------
        if close_chunk is None or len(close_chunk) == 0:

            # Không in từng đoạn API để tránh log dài.

            try:
                df = market.index(benchmark).ohlcv(
                    start=cursor.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    interval="1D"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Không lấy được {benchmark} cho đoạn "
                    f"{cursor.date()} → {chunk_end.date()}: {e}"
                )

            df = normalize_columns(df)

            if df.empty or "close" not in df.columns:
                raise ValueError(
                    f"{benchmark}: API không trả dữ liệu "
                    f"cho đoạn {cursor.date()} → {chunk_end.date()}."
                )

            date_col = (
                "time"
                if "time" in df.columns
                else "date"
                if "date" in df.columns
                else None
            )

            if date_col is None:
                raise ValueError(
                    f"{benchmark}: không tìm thấy cột thời gian."
                )

            df[date_col] = pd.to_datetime(
                df[date_col]
            )

            df = df.set_index(
                date_col
            ).sort_index()

            close_chunk = pd.to_numeric(
                df["close"],
                errors="coerce"
            ).dropna()

            close_chunk.index = pd.to_datetime(
                close_chunk.index
            ).normalize()

            close_chunk = close_chunk[
                (close_chunk.index >= cursor)
                & (close_chunk.index <= chunk_end)
            ]

            if len(close_chunk) == 0:
                raise ValueError(
                    f"{benchmark}: đoạn "
                    f"{cursor.date()} → {chunk_end.date()} "
                    f"không có phiên giao dịch."
                )

            tmp = close_chunk.to_frame("close")
            tmp.index.name = "Date"
            tmp.to_csv(chunk_path)

            pause_api()

        pieces.append(close_chunk)

        cursor = chunk_end + pd.Timedelta(days=1)

    # --------------------------------------------------------
    # GHÉP TOÀN BỘ CÁC ĐOẠN
    # --------------------------------------------------------
    close = pd.concat(pieces)
    close = close[
        ~close.index.duplicated(keep="last")
    ].sort_index()

    close = close[
        (close.index >= start_ts)
        & (close.index <= end_ts)
    ]

    expected = max(
        int(((end_ts - start_ts).days + 1) * 0.68),
        10
    )

    minimum_required = max(
        10,
        int(expected * 0.70)
    )

    if len(close) < minimum_required:
        raise ValueError(
            f"{benchmark} chỉ có {len(close)} phiên "
            f"trong khoảng {start_date} đến {end_date}. "
            f"Ước tính tối thiểu cần khoảng {minimum_required} phiên."
        )

    full = close.to_frame("close")
    full.index.name = "Date"
    full.to_csv(full_path)

    print(f"{benchmark}: {len(close):,} phiên.")

    return close

# ============================================================
# KIỂM TRA CHẤT LƯỢNG BENCHMARK
# ============================================================

def align_analysis_period(
    prices,
    benchmark_prices,
    requested_start,
    requested_end
):
    """
    Chuẩn hóa khoảng thời gian nghiên cứu theo ngày giao dịch thực tế.

    Người dùng không cần nhập đúng ngày giao dịch.
    Ví dụ:
        01/01/2022 là ngày nghỉ
        12/08/2026 là ngày giao dịch

    Hệ thống sẽ tự hiểu:
        ngày bắt đầu = phiên đầu tiên có dữ liệu >= ngày yêu cầu
        ngày kết thúc = phiên cuối cùng có dữ liệu <= ngày yêu cầu

    Sau đó tất cả tài sản và benchmark được cắt theo cùng
    khoảng ngày giao dịch thực tế.
    """

    if prices is None or prices.empty:
        raise ValueError("Không có dữ liệu cổ phiếu.")

    if benchmark_prices is None or benchmark_prices.empty:
        raise ValueError("Không có dữ liệu benchmark.")

    requested_start = pd.Timestamp(requested_start).normalize()
    requested_end = pd.Timestamp(requested_end).normalize()

    if requested_start > requested_end:
        raise ValueError(
            "Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc."
        )

    asset_dates = pd.DatetimeIndex(prices.index).normalize()
    benchmark_dates = pd.DatetimeIndex(
        benchmark_prices.index
    ).normalize()

    common_dates = asset_dates.intersection(
        benchmark_dates
    ).sort_values()

    # Chỉ cần tìm phiên thực tế gần nhất trong khoảng yêu cầu.
    valid_dates = common_dates[
        (common_dates >= requested_start)
        & (common_dates <= requested_end)
    ]

    if len(valid_dates) < 100:
        raise ValueError(
            f"Chỉ có {len(valid_dates)} phiên giao dịch chung "
            f"trong khoảng yêu cầu. Không đủ dữ liệu để phân tích."
        )

    effective_start = valid_dates[0]
    effective_end = valid_dates[-1]

    if effective_start != requested_start or effective_end != requested_end:
        print(
            f"Khoảng nghiên cứu thực tế: "
            f"{effective_start.date()} → {effective_end.date()}"
        )

    return effective_start, effective_end


def validate_benchmark_alignment(
    benchmark_prices,
    asset_prices,
    start_date,
    end_date
):
    """
    Kiểm tra benchmark sau khi đã quy đổi sang ngày giao dịch thực tế.
    """

    if benchmark_prices is None or benchmark_prices.empty:
        raise ValueError("Benchmark không có dữ liệu.")

    common_dates = (
        pd.DatetimeIndex(asset_prices.index)
        .normalize()
        .intersection(
            pd.DatetimeIndex(benchmark_prices.index)
            .normalize()
        )
    )

    common_dates = common_dates.sort_values()

    valid_dates = common_dates[
        (common_dates >= pd.Timestamp(start_date))
        & (common_dates <= pd.Timestamp(end_date))
    ]

    if len(valid_dates) < 100:
        raise ValueError(
            f"Benchmark chỉ giao với dữ liệu cổ phiếu "
            f"{len(valid_dates)} phiên."
        )

    return True


# ============================================================
# TẦN SUẤT PHÂN TÍCH: GIÁ ĐÓNG CỬA HÀNG TUẦN
# ============================================================

ANALYSIS_FREQUENCY = "W-FRI"
PERIODS_PER_YEAR = 52

def to_weekly_close(prices):
    """
    Chuyển dữ liệu giá ngày thành giá đóng cửa tuần.

    Mỗi tuần chỉ giữ phiên cuối cùng có dữ liệu.
    Nếu thứ Sáu là ngày nghỉ, phiên gần nhất trước đó
    trong cùng tuần sẽ được dùng.

    Dữ liệu ngày vẫn có thể được lưu trong cache.
    Chỉ phần phân tích danh mục sử dụng dữ liệu tuần.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()

    x = prices.copy()
    x.index = pd.to_datetime(x.index).normalize()
    x = x.sort_index()

    weekly = x.resample(ANALYSIS_FREQUENCY).last()
    weekly = weekly.dropna(how="all")

    return weekly

def to_weekly_series(series):
    if series is None or series.empty:
        return pd.Series(dtype=float)

    x = series.copy()
    x.index = pd.to_datetime(x.index).normalize()
    x = x.sort_index()

    return x.resample(ANALYSIS_FREQUENCY).last().dropna()

# ============================================================
# RETURNS
# ============================================================

def calculate_returns(prices):
    return prices.pct_change().dropna(how="all")

# ============================================================
# ASSET STATISTICS
# ============================================================

def calculate_asset_statistics(
    returns,
    benchmark_returns,
    risk_free_rate
):
    annual_return = returns.mean() * PERIODS_PER_YEAR
    annual_volatility = returns.std() * np.sqrt(PERIODS_PER_YEAR)

    sharpe = (
        annual_return - risk_free_rate
    ) / annual_volatility.replace(0, np.nan)

    combined = pd.concat(
        [returns, benchmark_returns.rename("Benchmark")],
        axis=1
    ).dropna()

    market_var = combined["Benchmark"].var()

    beta = pd.Series(
        {
            ticker:
            combined[ticker].cov(
                combined["Benchmark"]
            ) / market_var
            for ticker in returns.columns
        },
        name="Beta"
    )

    correlation = pd.Series(
        {
            ticker:
            combined[ticker].corr(
                combined["Benchmark"]
            )
            for ticker in returns.columns
        },
        name="Correlation"
    )

    return pd.DataFrame({
        "Lợi suất năm": annual_return,
        "Biến động năm": annual_volatility,
        "Sharpe": sharpe,
        "Beta": beta,
        "Tương quan VNINDEX": correlation
    })

# ============================================================
# PORTFOLIO STATISTICS
# ============================================================

def portfolio_statistics(
    weights,
    returns,
    risk_free_rate
):
    weights = np.asarray(weights, dtype=float)

    mu = returns.mean() * PERIODS_PER_YEAR
    cov = returns.cov() * PERIODS_PER_YEAR

    p_return = weights @ mu
    p_var = weights @ cov @ weights
    p_vol = np.sqrt(max(p_var, 0))

    sharpe = (
        (p_return - risk_free_rate) / p_vol
        if p_vol > 0 else np.nan
    )

    return {
        "return": p_return,
        "volatility": p_vol,
        "variance": p_var,
        "sharpe": sharpe
    }

def portfolio_drawdown(weights, returns):
    weights = np.asarray(weights, dtype=float)
    daily_portfolio = returns @ weights
    wealth = (1 + daily_portfolio).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    return drawdown

# ============================================================
# OPTIMIZATION
# ============================================================

def target_return_min_variance(
    returns,
    target_return,
    leverage,
    risk_free_rate=0.0
):
    """
    Danh mục phương sai thấp nhất với:
    tổng tỷ trọng = 100%
    không bán khống
    lợi suất kỳ vọng >= mục tiêu

    Đòn bẩy chỉ áp dụng ở Complete Portfolio, không cho phép
    short ở các danh mục cổ phiếu cơ sở.
    """

    global _N_ASSETS
    _N_ASSETS = returns.shape[1]

    n = _N_ASSETS
    cov = returns.cov() * PERIODS_PER_YEAR
    mu = returns.mean() * PERIODS_PER_YEAR

    initial = np.ones(n) / n
    bounds = bounds_for(False)

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        },
        {
            "type": "ineq",
            "fun": lambda w: w @ mu - target_return
        }
    ]

    result = minimize(
        lambda w: w @ cov @ w,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-10
        }
    )

    if not result.success:
        raise ValueError(
            "Không tìm được danh mục đạt mục tiêu "
            f"{target_return:.2%}: {result.message}"
        )

    weights = result.x

    stats = portfolio_statistics(
        weights,
        returns,
        risk_free_rate
    )

    return weights, stats


def feasible_return_range(returns, leverage):
    """
    Tìm biên lợi nhuận kỳ vọng khả thi dưới cùng ràng buộc
    tỷ trọng và đòn bẩy.
    """
    global _N_ASSETS
    _N_ASSETS = returns.shape[1]

    n = _N_ASSETS
    mu = returns.mean() * PERIODS_PER_YEAR
    initial = np.ones(n) / n
    bounds = bounds_for(leverage)

    cons = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }

    min_r = minimize(
        lambda w: w @ mu,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000}
    )

    max_r = minimize(
        lambda w: -(w @ mu),
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000}
    )

    if not min_r.success or not max_r.success:
        return np.nan, np.nan

    return min_r.x @ mu, max_r.x @ mu


def build_efficient_frontier(
    returns,
    risk_free_rate,
    leverage,
    points=21
):
    """
    Xây đường biên hiệu quả bằng cách tối thiểu hóa phương sai
    tại nhiều mức lợi nhuận kỳ vọng mục tiêu.
    Không gọi API.
    """
    global _N_ASSETS, RISK_FREE_RATE_RUNTIME
    _N_ASSETS = returns.shape[1]
    RISK_FREE_RATE_RUNTIME = risk_free_rate

    min_return, max_return = feasible_return_range(
        returns,
        leverage
    )

    if pd.isna(min_return) or pd.isna(max_return):
        return pd.DataFrame()

    # Không dùng hai đầu biên tuyệt đối vì nghiệm tại biên
    # đôi khi nhạy với sai số tối ưu.
    targets = np.linspace(
        min_return,
        max_return,
        points
    )

    rows = []

    for target in targets:
        try:
            w, stats = target_return_min_variance(
                returns,
                float(target),
                leverage=False,
                risk_free_rate=risk_free_rate
            )

            rows.append({
                "Target Return": target,
                "Return": stats["return"],
                "Risk": stats["volatility"],
                "Sharpe": stats["sharpe"]
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def bounds_for(leverage=False):
    """
    Luôn khống chế short:
    0 <= w_i <= 1

    Đòn bẩy không được thực hiện bằng short.
    Đòn bẩy chỉ được áp dụng ở Complete Portfolio thông qua
    tỷ trọng danh mục rủi ro lớn hơn 100%.
    """
    return tuple(
        (0, 1)
        for _ in range(_N_ASSETS)
    )

def optimize_portfolios(
    returns,
    risk_free_rate,
    risk_aversion,
    leverage,
    target_return
):
    global _N_ASSETS
    _N_ASSETS = returns.shape[1]
    n = _N_ASSETS
    cov = returns.cov() * PERIODS_PER_YEAR
    mu = returns.mean() * PERIODS_PER_YEAR

    initial = np.ones(n) / n

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }

    bounds = bounds_for(leverage)

    # Naive
    naive_w = initial.copy()

    # Minimum variance
    def minvar_obj(w):
        return w @ cov @ w

    minvar = minimize(
        minvar_obj,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not minvar.success:
        raise ValueError(
            "Minimum Variance thất bại: "
            + minvar.message
        )

    minvar_w = minvar.x

    # Optimal risky
    def neg_sharpe(w):
        r = w @ mu
        v = np.sqrt(max(w @ cov @ w, 1e-12))
        return -(r - risk_free_rate) / v

    opt = minimize(
        neg_sharpe,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000}
    )

    if not opt.success:
        raise ValueError(
            "Optimal Risky thất bại: "
            + opt.message
        )

    optimal_w = opt.x

    # Maximum return
    def neg_return(w):
        return -(w @ mu)

    maxret = minimize(
        neg_return,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not maxret.success:
        raise ValueError(
            "Maximum Return thất bại: "
            + maxret.message
        )

    maxret_w = maxret.x

    results = {}

    results["Naive"] = (
        naive_w,
        portfolio_statistics(
            naive_w, returns, risk_free_rate
        )
    )

    results["Minimum Variance"] = (
        minvar_w,
        portfolio_statistics(
            minvar_w, returns, risk_free_rate
        )
    )

    results["Optimal Risky"] = (
        optimal_w,
        portfolio_statistics(
            optimal_w, returns, risk_free_rate
        )
    )

    results["Maximum Return"] = (
        maxret_w,
        portfolio_statistics(
            maxret_w, returns, risk_free_rate
        )
    )

    # Complete portfolio
    opt_stats = results["Optimal Risky"][1]
    theoretical = (
        opt_stats["return"] - risk_free_rate
    ) / (
        risk_aversion * opt_stats["volatility"] ** 2
    )

    risky_weight = (
        max(theoretical, 0.0) if leverage
        else np.clip(theoretical, 0, 1)
    )

    complete_w = optimal_w * risky_weight

    complete_return = (
        risky_weight * opt_stats["return"]
        + (1 - risky_weight) * risk_free_rate
    )

    complete_vol = (
        abs(risky_weight) * opt_stats["volatility"]
    )

    complete_sharpe = (
        (complete_return - risk_free_rate)
        / complete_vol
        if complete_vol > 0 else np.nan
    )

    complete_stats = {
        "return": complete_return,
        "volatility": complete_vol,
        "variance": complete_vol ** 2,
        "sharpe": complete_sharpe,
        "theoretical_risky_weight": theoretical,
        "risky_weight": risky_weight,
        "risk_free_weight": 1 - risky_weight
    }

    results["Complete Portfolio"] = (
        complete_w,
        complete_stats
    )

    # Target Return Portfolio
    # Tìm danh mục có rủi ro thấp nhất nhưng đạt mục tiêu lợi nhuận
    if target_return is not None:
        try:
            target_w, target_stats = target_return_min_variance(
                returns,
                target_return,
                leverage=False,
                risk_free_rate=risk_free_rate
            )
            target_stats["target_return"] = target_return
            target_stats["return_gap"] = (
                target_stats["return"] - target_return
            )

            results["Target Return"] = (
                target_w,
                target_stats
            )

        except Exception as e:
            # Lưu trạng thái lỗi thay vì làm toàn bộ ứng dụng dừng.
            results["Target Return"] = (
                None,
                {
                    "return": np.nan,
                    "volatility": np.nan,
                    "variance": np.nan,
                    "sharpe": np.nan,
                    "target_return": target_return,
                    "return_gap": np.nan,
                    "error": str(e)
                }
            )

    return results

# ============================================================
# BENCHMARK COMPARISON
# ============================================================

def benchmark_comparison(
    portfolio_results,
    returns,
    benchmark_returns,
    risk_free_rate
):
    rows = []

    benchmark_annual_return = (
        benchmark_returns.mean() * PERIODS_PER_YEAR
    )

    benchmark_vol = (
        benchmark_returns.std()
        * np.sqrt(PERIODS_PER_YEAR)
    )

    benchmark_sharpe = (
        benchmark_annual_return
        - risk_free_rate
    ) / benchmark_vol

    benchmark_wealth = (
        1 + benchmark_returns
    ).cumprod()

    benchmark_dd = (
        benchmark_wealth
        / benchmark_wealth.cummax()
        - 1
    )

    benchmark_mdd = benchmark_dd.min()

    for name, item in portfolio_results.items():

        weights = item[0]
        stats = item[1]

        if weights is None:
            continue

        p_returns = returns @ weights
        wealth = (1 + p_returns).cumprod()

        aligned = pd.concat(
            [
                p_returns.rename("Portfolio"),
                benchmark_returns.rename("Benchmark")
            ],
            axis=1
        ).dropna()

        active = (
            aligned["Portfolio"]
            - aligned["Benchmark"]
        )

        tracking_error = (
            active.std()
            * np.sqrt(PERIODS_PER_YEAR)
        )

        information_ratio = (
            active.mean() * PERIODS_PER_YEAR
            / tracking_error
            if tracking_error > 0 else np.nan
        )

        beta = (
            aligned["Portfolio"].cov(
                aligned["Benchmark"]
            )
            / aligned["Benchmark"].var()
        )

        correlation = (
            aligned["Portfolio"]
            .corr(aligned["Benchmark"])
        )

        alpha = (
            p_returns.mean() * PERIODS_PER_YEAR
            - risk_free_rate
            - beta * (
                benchmark_annual_return
                - risk_free_rate
            )
        )

        peak = wealth.cummax()
        drawdown = wealth / peak - 1
        mdd = drawdown.min()

        rows.append({
            "Danh mục": name,
            "Lợi suất": stats["return"],
            "Rủi ro": stats["volatility"],
            "Sharpe": stats["sharpe"],
            "Max Drawdown": mdd,
            "Alpha": alpha,
            "Beta": beta,
            "Tương quan": correlation,
            "Tracking Error": tracking_error,
            "Information Ratio": information_ratio,
            "Vượt VNINDEX": (
                stats["return"]
                - benchmark_annual_return
            )
        })

    benchmark_row = {
        "Danh mục": "VNINDEX",
        "Lợi suất": benchmark_annual_return,
        "Rủi ro": benchmark_vol,
        "Sharpe": benchmark_sharpe,
        "Max Drawdown": benchmark_mdd,
        "Alpha": 0,
        "Beta": 1,
        "Tương quan": 1,
        "Tracking Error": 0,
        "Information Ratio": np.nan,
        "Vượt VNINDEX": 0
    }

    return (
        pd.DataFrame(rows),
        pd.DataFrame([benchmark_row]),
        benchmark_wealth,
        benchmark_dd
    )

# ============================================================
# COMPANY INFO
# ============================================================

def get_company_info(tickers, prices=None):
    """
    Lấy thông tin công ty và các chỉ tiêu cơ bản.

    Chỉ tiêu:
    Ngành, số CP lưu hành, vốn hóa, P/E, P/B, EPS, ROA, ROE.

    Các dữ liệu được cache riêng để tránh gọi API lặp lại.
    """

    ref = Reference()
    rows = []

    industry_path = cache_path("industry_sectors", "all")

    try:
        if industry_path.exists():
            sectors = pd.read_csv(industry_path)
        else:
            sectors = normalize_columns(
                ref.industry.sectors()
            )

            if sectors.empty:
                raise ValueError("Không lấy được bảng phân ngành")

            sectors.to_csv(industry_path, index=False)
            pause_api()

    except Exception as e:
        print(f"Không lấy được bảng phân ngành: {e}")
        sectors = pd.DataFrame()

    sector_symbol_col = find_col(
        sectors,
        ["symbol", "ticker", "stock_code", "code"]
    )

    sector_name_col = find_col(
        sectors,
        [
            "icb_name_vi",
            "icb_name",
            "industry_name_vi",
            "industry_name",
            "industry"
        ]
    )

    # --------------------------------------------------------
    # Hàm lấy ratio từ Fundamental
    # --------------------------------------------------------
    def get_ratio_cached(ticker):
        """
        Lấy ratio theo dạng time_series trước.

        Vnstock hiện công bố các cột chuẩn:
        pe_ratio, pb_ratio, eps, roa, roe...
        """

        path = cache_path("ratio", ticker)

        if path.exists():
            try:
                cached = pd.read_csv(path)

                if not cached.empty:

                    _cols = {
                        str(c).strip().lower()
                        .replace(" ", "_")
                        .replace("-", "_")
                        .replace("/", "_")
                        for c in cached.columns
                    }

                    _has_pe = any(
                        x in _cols
                        for x in [
                            "pe",
                            "p_e",
                            "pe_ratio",
                            "pricetoearning",
                            "pricetoearning"
                        ]
                    )

                    _has_pb = any(
                        x in _cols
                        for x in [
                            "pb",
                            "p_b",
                            "pb_ratio",
                            "pricetobook"
                        ]
                    )

                    # Chỉ dùng cache cũ nếu có cấu trúc ratio phù hợp.
                    if _has_pe or _has_pb:
                        return cached

            except Exception:
                pass

        try:
            fun = Fundamental()

            # Ưu tiên time_series vì P/E và P/B nằm trực tiếp
            # trong tên cột chuẩn.
            try:
                ratio = normalize_columns(
                    fun.equity(ticker).ratio(
                        orient="time_series"
                    )
                )
            except TypeError:
                ratio = normalize_columns(
                    fun.equity(ticker).ratio()
                )

            if ratio is not None and not ratio.empty:

                ratio.to_csv(
                    path,
                    index=False
                )

                pause_api()

                return ratio

        except Exception as e:
            print(
                f"Không lấy được chỉ số định giá {ticker}: {e}"
            )

        return pd.DataFrame()


    def extract_ratio_value(ratio, names):

        if ratio is None or ratio.empty:
            return np.nan

        # Chuẩn hóa tên cột.
        normalized_cols = {}

        for col in ratio.columns:

            key = (
                str(col)
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("/", "_")
            )

            normalized_cols[key] = col

        # ----------------------------------------------------
        # 1. Tìm trực tiếp theo tên cột chuẩn
        # ----------------------------------------------------
        aliases = []

        for name in names:

            key = (
                str(name)
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("/", "_")
            )

            aliases.append(key)

        for alias in aliases:

            for normalized, original in normalized_cols.items():

                if (
                    normalized == alias
                    or alias in normalized
                ):

                    values = pd.to_numeric(
                        ratio[original],
                        errors="coerce"
                    ).dropna()

                    if not values.empty:
                        return float(values.iloc[-1])

        # ----------------------------------------------------
        # 2. Nếu là dạng report: tìm dòng item
        # ----------------------------------------------------
        item_col = find_col(
            ratio,
            [
                "item",
                "item_en",
                "metric",
                "indicator"
            ]
        )

        if item_col:

            item_text = (
                ratio[item_col]
                .astype(str)
                .str.lower()
            )

            for alias in aliases:

                mask = item_text.str.contains(
                    alias,
                    regex=False,
                    na=False
                )

                if mask.any():

                    row = ratio.loc[
                        mask
                    ].iloc[-1]

                    numeric = pd.to_numeric(
                        row.drop(
                            labels=[item_col],
                            errors="ignore"
                        ),
                        errors="coerce"
                    ).dropna()

                    if not numeric.empty:
                        return float(
                            numeric.iloc[-1]
                        )

        return np.nan


    for ticker in tickers:

        path = cache_path("company", ticker)

        try:
            if path.exists():
                info = pd.read_csv(path)
            else:
                info = normalize_columns(
                    ref.company(ticker).info()
                )

                if info.empty:
                    raise ValueError("Không có dữ liệu")

                info.to_csv(path, index=False)
                pause_api()

            shares = safe_float(
                first_value(
                    info,
                    [
                        "issue_share",
                        "shares_outstanding",
                        "outstanding_shares",
                        "listed_shares",
                        "listed_share"
                    ]
                )
            )

            industry = np.nan

            if (
                not sectors.empty
                and sector_symbol_col
                and sector_name_col
            ):
                match = sectors[
                    sectors[sector_symbol_col]
                    .astype(str)
                    .str.upper()
                    == ticker
                ]

                if not match.empty:
                    industry = match.iloc[0][sector_name_col]

            if pd.isna(industry):
                industry = first_value(
                    info,
                    [
                        "industry",
                        "industry_name",
                        "icb_name",
                        "icb_name_vi"
                    ]
                )

            # Vốn hóa tại phiên cuối kỳ nghiên cứu.
            market_cap = np.nan

            if (
                prices is not None
                and ticker in prices.columns
                and not prices[ticker].dropna().empty
                and not pd.isna(shares)
            ):
                last_price = float(
                    prices[ticker].dropna().iloc[-1]
                )

                market_cap = (
                    shares
                    * last_price
                    * 1000
                )

            if pd.isna(market_cap):
                market_cap = safe_float(
                    first_value(
                        info,
                        [
                            "market_cap",
                            "market_capitalization"
                        ]
                    )
                )

            # ------------------------------------------------
            # PE / PB / EPS / ROA / ROE
            # ------------------------------------------------
            ratio = get_ratio_cached(ticker)

            pe = extract_ratio_value(
                ratio,
                [
                    "pe",
                    "p_e",
                    "pe_ratio",
                    "price_to_earning",
                    "price_to_earnings",
                    "pricetoearning",
                    "priceToEarning"
                ]
            )

            pb = extract_ratio_value(
                ratio,
                [
                    "pb",
                    "p_b",
                    "pb_ratio",
                    "price_to_book",
                    "pricetobook",
                    "priceToBook"
                ]
            )

            eps = extract_ratio_value(
                ratio,
                [
                    "eps",
                    "earning_per_share",
                    "earnings_per_share"
                ]
            )

            # Fallback P/E = giá cuối kỳ / EPS.
            # Chỉ dùng khi P/E không được API trả trực tiếp.
            if pd.isna(pe) and pd.notna(eps):

                try:
                    if (
                        prices is not None
                        and ticker in prices.columns
                    ):

                        _last_price = (
                            prices[ticker]
                            .dropna()
                            .iloc[-1]
                        )

                        _eps_value = float(eps)

                        if _eps_value > 0:
                            pe = (
                                float(_last_price)
                                / _eps_value
                            )

                except Exception:
                    pass

            # Fallback P/B = vốn hóa / vốn chủ sở hữu.
            if pd.isna(pb):

                try:

                    _bs_path = cache_path(
                        "balance_sheet",
                        ticker
                    )

                    if _bs_path.exists():

                        _bs = pd.read_csv(
                            _bs_path
                        )

                    else:

                        _bs = normalize_columns(
                            Fundamental()
                            .equity(ticker)
                            .balance_sheet(
                                period="year",
                                orient="time_series"
                            )
                        )

                        if (
                            _bs is not None
                            and not _bs.empty
                        ):

                            _bs.to_csv(
                                _bs_path,
                                index=False
                            )

                            pause_api()

                    if (
                        _bs is not None
                        and not _bs.empty
                    ):

                        _equity_col = find_col(
                            _bs,
                            [
                                "equity",
                                "total_equity",
                                "owners_equity",
                                "shareholders_equity"
                            ]
                        )

                        if _equity_col:

                            _equity = pd.to_numeric(
                                _bs[_equity_col],
                                errors="coerce"
                            ).dropna()

                            if (
                                not _equity.empty
                                and _equity.iloc[-1] > 0
                                and pd.notna(market_cap)
                            ):

                                pb = (
                                    float(market_cap)
                                    / float(_equity.iloc[-1])
                                )

                except Exception:
                    pass

            roa = extract_ratio_value(
                ratio,
                [
                    "roa",
                    "return_on_assets"
                ]
            )

            roe = extract_ratio_value(
                ratio,
                [
                    "roe",
                    "return_on_equity"
                ]
            )

            rows.append({
                "Mã": ticker,
                "Ngành": industry,
                "Số CP lưu hành": shares,
                "Vốn hóa": market_cap,
                "P/E": pe,
                "P/B": pb,
                "EPS": eps,
                "ROA": roa,
                "ROE": roe
            })

        except Exception as e:
            print(
                f"Không lấy được thông tin {ticker}: {e}"
            )

            rows.append({
                "Mã": ticker,
                "Ngành": np.nan,
                "Số CP lưu hành": np.nan,
                "Vốn hóa": np.nan,
                "P/E": np.nan,
                "P/B": np.nan,
                "EPS": np.nan,
                "ROA": np.nan,
                "ROE": np.nan
            })

    return pd.DataFrame(rows)

# ============================================================
# INCOME STATEMENT
# ============================================================

# ============================================================

def strip_accents(text):
    text = str(text)
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def income_row_value(
    income,
    keywords,
    exclude_keywords=None
):
    """
    BCTC Community trả dạng:
    item | item_id | 2025 | 2024 | 2023 | 2022

    Vì vậy không thể tìm doanh thu/lợi nhuận bằng tên cột.
    Phải tìm dòng item tương ứng rồi lấy cột năm mới nhất.
    """
    if income is None or income.empty:
        return np.nan, None, None

    income = normalize_columns(income)

    item_col = find_col(
        income,
        ["item", "item_name", "name", "indicator"]
    )

    if item_col is None:
        return np.nan, None, None

    year_cols = []
    for col in income.columns:
        if str(col).isdigit() and len(str(col)) == 4:
            year_cols.append(col)

    if not year_cols:
        return np.nan, None, None

    year_cols = sorted(
        year_cols,
        key=lambda x: int(str(x)),
        reverse=True
    )

    exclude_keywords = exclude_keywords or []

    normalized_items = (
        income[item_col]
        .astype(str)
        .map(strip_accents)
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    for keyword_group in keywords:

        mask = normalized_items.str.contains(
            strip_accents(keyword_group).lower(),
            regex=False,
            na=False
        )

        for exclude in exclude_keywords:
            mask &= ~normalized_items.str.contains(
                strip_accents(exclude).lower(),
                regex=False,
                na=False
            )

        matches = income.loc[mask]

        if not matches.empty:
            row = matches.iloc[0]

            for year in year_cols:
                value = safe_float(row[year])

                if pd.notna(value):
                    return value, str(year), str(
                        row[item_col]
                    )

    return np.nan, None, None


def get_income_summary(tickers):
    """
    Đọc BCTC theo đúng schema report của vnstock Community.

    Schema thường gặp:
    item | item_id | 2025 | 2024 | 2023 | 2022

    Do đó doanh thu và LNST phải được lấy theo dòng item,
    không phải tìm trực tiếp trong tên cột.
    """

    fun = Fundamental()
    rows = []

    for ticker in tickers:

        path = cache_path("income", ticker)

        try:
            income = pd.DataFrame()

            if path.exists():
                try:
                    income = normalize_columns(
                        pd.read_csv(path)
                    )
                except Exception:
                    income = pd.DataFrame()

            if income.empty:
                income = normalize_columns(
                    fun.equity(ticker).income_statement(
                        period="year",
                        orient="report"
                    )
                )

                if income.empty:
                    raise ValueError(
                        "BCTC trả về rỗng"
                    )

                income.to_csv(
                    path,
                    index=False
                )

                pause_api()

            # Doanh thu thuần ưu tiên trước
            revenue, revenue_year, revenue_item = (
                income_row_value(
                    income,
                    [
                        "doanh thu thuan",
                        "net revenue",
                        "revenue from sales and services",
                        "doanh thu ban hang va cung cap dich vu"
                    ]
                )
            )

            # LNST ưu tiên lợi nhuận sau thuế của công ty mẹ.
            net_profit, profit_year, profit_item = (
                income_row_value(
                    income,
                    [
                        "loi nhuan sau thue cua co dong cong ty me",
                        "loi nhuan sau thue",
                        "profit after tax",
                        "net profit",
                        "profit attributable to parent"
                    ]
                )
            )

            if pd.isna(revenue) and pd.isna(net_profit):
                raise ValueError(
                    "Không nhận diện được dòng doanh thu/LNST. "
                    f"Các cột: {list(income.columns)}"
                )

            rows.append({
                "Mã": ticker,
                "Doanh thu gần nhất": revenue,
                "Năm doanh thu": revenue_year,
                "LNST gần nhất": net_profit,
                "Năm LNST": profit_year
            })

        except Exception as e:
            print(
                f"Không lấy được KQKD {ticker}: {e}"
            )

            rows.append({
                "Mã": ticker,
                "Doanh thu gần nhất": np.nan,
                "Năm doanh thu": np.nan,
                "LNST gần nhất": np.nan,
                "Năm LNST": np.nan
            })

    return pd.DataFrame(rows)

# ============================================================
# MERGE COMPANY TABLE
# ============================================================

def build_company_table(tickers, prices, include_company, include_income):

    base = pd.DataFrame({"Mã": tickers})

    if include_company:
        company = get_company_info(
            tickers,
            prices=prices
        )
        base = base.merge(
            company,
            on="Mã",
            how="left"
        )

    if include_income:
        income = get_income_summary(tickers)
        base = base.merge(
            income,
            on="Mã",
            how="left"
        )

    for col in [
        "Ngành",
        "Số CP lưu hành",
        "Vốn hóa",
        "P/E",
        "P/B",
        "EPS",
        "ROA",
        "ROE",
        "Doanh thu gần nhất",
        "LNST gần nhất"
    ]:
        if col not in base.columns:
            base[col] = np.nan

    return base[
        [
            "Mã",
            "Ngành",
            "Số CP lưu hành",
            "Vốn hóa",
            "P/E",
            "P/B",
            "EPS",
            "ROA",
            "ROE",
            "Doanh thu gần nhất",
            "LNST gần nhất"
        ]
    ]

# ============================================================
# FORMATTING
# ============================================================

# ============================================================

def format_money_vnd(x):
    if pd.isna(x):
        return "N/A"

    x = float(x)

    if abs(x) >= 1e12:
        return f"{x / 1e12:,.2f} nghìn tỷ"

    if abs(x) >= 1e9:
        return f"{x / 1e9:,.2f} tỷ"

    if abs(x) >= 1e6:
        return f"{x / 1e6:,.0f} triệu"

    return f"{x:,.0f} đồng"


def format_company_table(df):
    styled = (
        df.style
        .hide(axis="index")
        .set_properties(**{
            "text-align": "center",
            "padding": "7px 10px"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("padding", "8px 10px"),
                    ("font-weight", "bold")
                ]
            }
        ])
    )

    def fmt_num(x):
        return f"{x:,.0f}" if pd.notna(x) else "N/A"

    def fmt_ratio(x):
        return f"{x:.2f}" if pd.notna(x) else "N/A"

    def fmt_percent(x):
        if pd.isna(x):
            return "N/A"

        # Một số nguồn trả 0.15, một số trả 15.
        value = float(x)
        if abs(value) <= 1:
            value *= 100

        return f"{value:.2f}%"

    return styled.format({
        "Số CP lưu hành": fmt_num,
        "Vốn hóa": format_money_vnd,
        "P/E": fmt_ratio,
        "P/B": fmt_ratio,
        "EPS": fmt_num,
        "ROA": fmt_percent,
        "ROE": fmt_percent,
        "Doanh thu gần nhất": format_money_vnd,
        "LNST gần nhất": format_money_vnd
    })


# ============================================================
# MAIN ENGINE
# ============================================================


# ============================================================
# PHÂN TÍCH MỞ RỘNG V10
# ============================================================

def optimize_target_at_return(
    returns,
    target_return,
    leverage=False
):
    """
    Tìm danh mục có phương sai thấp nhất với lợi suất kỳ vọng
    tối thiểu bằng target_return.
    """
    n = returns.shape[1]
    expected_returns = returns.mean() * PERIODS_PER_YEAR
    covariance_matrix = returns.cov() * PERIODS_PER_YEAR

    def objective(w):
        return w @ covariance_matrix @ w

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        },
        {
            "type": "ineq",
            "fun": lambda w: w @ expected_returns - target_return
        }
    ]

    if leverage:
        bounds = tuple((0, 1) for _ in range(n))
    else:
        bounds = tuple((0, 1) for _ in range(n))

    x0 = np.ones(n) / n

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        return None, None

    stats = portfolio_statistics(
        result.x,
        returns,
        0
    )

    return result.x, stats


def build_target_sensitivity(
    returns,
    risk_free_rate,
    leverage=False,
    target_grid=None
):
    """
    Phân tích độ nhạy: mục tiêu lợi nhuận thay đổi thì
    rủi ro tối thiểu cần thiết thay đổi thế nào.
    """
    if target_grid is None:
        target_grid = [
            0.05, 0.08, 0.10, 0.12, 0.15,
            0.18, 0.20, 0.25, 0.30
        ]

    rows = []

    for target in target_grid:
        weights, stats = optimize_target_at_return(
            returns,
            target,
            leverage
        )

        if weights is None:
            rows.append({
                "Mục tiêu": target,
                "Khả thi": "Không",
                "Lợi suất": np.nan,
                "Rủi ro tối thiểu": np.nan,
                "Sharpe": np.nan
            })
        else:
            sharpe = (
                stats["return"] - risk_free_rate
            ) / stats["volatility"] if stats["volatility"] > 0 else np.nan

            rows.append({
                "Mục tiêu": target,
                "Khả thi": "Có",
                "Lợi suất": stats["return"],
                "Rủi ro tối thiểu": stats["volatility"],
                "Sharpe": sharpe
            })

    return pd.DataFrame(rows)


def portfolio_rolling_return(
    portfolio_returns,
    window=52
):
    return (
        (1 + portfolio_returns)
        .rolling(window)
        .apply(np.prod, raw=True)
        - 1
    )


def target_attainment_analysis(
    portfolio_returns,
    target_return,
    window=52
):
    """
    Đo mức độ lịch sử đạt mục tiêu lợi nhuận:
    tỷ lệ số kỳ cuộn đạt hoặc vượt mục tiêu.
    """
    rolling = portfolio_rolling_return(
        portfolio_returns,
        window
    ).dropna()

    if rolling.empty:
        return {
            "Tỷ lệ đạt mục tiêu": np.nan,
            "Lợi suất cuộn thấp nhất": np.nan,
            "Lợi suất cuộn trung vị": np.nan,
            "Lợi suất cuộn cao nhất": np.nan
        }

    return {
        "Tỷ lệ đạt mục tiêu":
            (rolling >= target_return).mean(),
        "Lợi suất cuộn thấp nhất":
            rolling.min(),
        "Lợi suất cuộn trung vị":
            rolling.median(),
        "Lợi suất cuộn cao nhất":
            rolling.max()
    }


def portfolio_concentration(weights):
    """
    Đo mức tập trung bằng:
    HHI = tổng bình phương tỷ trọng.
    Effective Number of Assets = 1 / HHI.
    """
    w = np.asarray(weights, dtype=float)

    hhi = np.sum(w ** 2)

    if hhi > 0:
        effective_n = 1 / hhi
    else:
        effective_n = np.nan

    abs_weights = np.abs(w)
    total_abs = abs_weights.sum()

    if total_abs > 0:
        normalized_abs = abs_weights / total_abs
        effective_n_abs = 1 / np.sum(normalized_abs ** 2)
    else:
        effective_n_abs = np.nan

    return {
        "HHI": hhi,
        "Số tài sản hiệu dụng": effective_n,
        "Số tài sản hiệu dụng theo |w|": effective_n_abs,
        "Tỷ trọng lớn nhất": np.max(abs_weights) if len(w) else np.nan
    }


def concentration_table(weights):
    rows = []

    for name in weights.columns:
        metrics = portfolio_concentration(
            weights[name].values
        )

        rows.append({
            "Danh mục": name,
            **metrics
        })

    return pd.DataFrame(rows)


def max_drawdown_stats(portfolio_returns):
    """
    Max Drawdown và thời gian phục hồi.

    Đã phục hồi: số ngày từ đáy đến ngày phục hồi.
    Chưa phục hồi: 0 ngày và trạng thái Chưa phục hồi.
    """

    portfolio_returns = pd.Series(
        portfolio_returns
    ).dropna()

    if portfolio_returns.empty:
        return {
            "Max Drawdown": np.nan,
            "Thời gian phục hồi": 0,
            "Trạng thái phục hồi": "Không có dữ liệu",
            "Đáy Drawdown": pd.NaT,
            "Ngày phục hồi": pd.NaT
        }

    wealth = (1 + portfolio_returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1

    max_dd = drawdown.min()

    if pd.isna(max_dd):
        return {
            "Max Drawdown": np.nan,
            "Thời gian phục hồi": 0,
            "Trạng thái phục hồi": "Không có dữ liệu",
            "Đáy Drawdown": pd.NaT,
            "Ngày phục hồi": pd.NaT
        }

    trough_date = drawdown.idxmin()
    peak_before_trough = peak.loc[:trough_date].iloc[-1]

    recovery_dates = wealth.loc[trough_date:][
        wealth.loc[trough_date:] >= peak_before_trough
    ].index

    if len(recovery_dates) > 0:
        recovery_date = recovery_dates[0]
        recovery_days = (recovery_date - trough_date).days
        recovery_status = "Đã phục hồi"
    else:
        recovery_date = pd.NaT
        recovery_days = 0
        recovery_status = "Chưa phục hồi"

    return {
        "Max Drawdown": max_dd,
        "Thời gian phục hồi": recovery_days,
        "Trạng thái phục hồi": recovery_status,
        "Đáy Drawdown": trough_date,
        "Ngày phục hồi": recovery_date
    }


def build_risk_diagnostics(
    portfolio_results,
    returns,
    benchmark_returns,
    target_return
):
    rows = []

    for name, item in portfolio_results.items():

        # Target Return là nghiệm Markowitz riêng,
        # không đưa vào bảng rủi ro của các danh mục chính.
        if name == "Target Return":
            continue

        if item[0] is None:
            continue

        portfolio_returns = returns @ item[0]

        diagnostics = max_drawdown_stats(portfolio_returns)
        attainment = target_attainment_analysis(
            portfolio_returns,
            target_return
        )

        rows.append({
            "Danh mục": name,
            "Max Drawdown": diagnostics["Max Drawdown"],
            "Phục hồi (ngày)": diagnostics["Thời gian phục hồi"],
            "Trạng thái phục hồi": diagnostics["Trạng thái phục hồi"],
            "Tỷ lệ đạt mục tiêu": attainment["Tỷ lệ đạt mục tiêu"],
            "Rolling 12T thấp nhất": attainment["Lợi suất cuộn thấp nhất"],
            "Rolling 12T trung vị": attainment["Lợi suất cuộn trung vị"],
            "Rolling 12T cao nhất": attainment["Lợi suất cuộn cao nhất"]
        })

    return pd.DataFrame(rows)


def investment_conclusion(
    portfolio_table,
    comparison,
    target_return,
    risk_aversion
):
    """
    Đánh giá các danh mục chính theo mục tiêu lợi nhuận.
    Tương thích với cả cột Lợi suất và Lợi suất kỳ vọng.
    """

    rows = []

    for _, row in portfolio_table.iterrows():

        if "Lợi suất kỳ vọng" in row.index:
            expected_return = row["Lợi suất kỳ vọng"]
        elif "Lợi suất" in row.index:
            expected_return = row["Lợi suất"]
        else:
            continue

        risk = row["Rủi ro"]
        sharpe = row["Sharpe"]

        gap = expected_return - target_return

        target_status = (
            "Đạt mục tiêu"
            if gap >= 0
            else "Chưa đạt mục tiêu"
        )

        rows.append({
            "Danh mục": row["Danh mục"],
            "Mục tiêu": target_return,
            "Lợi suất kỳ vọng": expected_return,
            "Chênh lệch mục tiêu": gap,
            "Rủi ro": risk,
            "Sharpe": sharpe,
            "Trạng thái mục tiêu": target_status
        })

    result = pd.DataFrame(rows)

    if not result.empty:

        risk_median = result["Rủi ro"].median()

        result["Điểm tham khảo"] = (
            result["Sharpe"].fillna(-999)
            + np.where(
                result["Chênh lệch mục tiêu"] >= 0,
                1.0,
                -1.0
            )
            - np.maximum(
                result["Rủi ro"] - risk_median,
                0
            )
        )

        result = result.sort_values(
            "Điểm tham khảo",
            ascending=False
        )

    return result




# ============================================================
# MARGIN
# ============================================================

def default_margin_table(tickers):
    return pd.DataFrame({
        "Mã": tickers,
        "Được cấp margin": ["Không"] * len(tickers),
        "Tỷ lệ cho vay": [0.0] * len(tickers),
        "Lãi suất vay": [0.0] * len(tickers),
        "Ngày cập nhật": [""] * len(tickers)
    })


def clean_margin_table(df, tickers):
    if df is None or df.empty:
        return default_margin_table(tickers)

    out = df.copy()

    for col in [
        "Mã",
        "Được cấp margin",
        "Tỷ lệ cho vay",
        "Lãi suất vay",
        "Ngày cập nhật"
    ]:
        if col not in out.columns:
            if col == "Mã":
                out[col] = tickers
            elif col == "Được cấp margin":
                out[col] = "Không"
            elif col in ["Tỷ lệ cho vay", "Lãi suất vay"]:
                out[col] = 0.0
            else:
                out[col] = ""

    out["Mã"] = (
        out["Mã"].astype(str).str.strip().str.upper()
    )

    out = out[out["Mã"].isin(tickers)].copy()

    def pct(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, str):
            x = x.strip().replace("%", "").replace(",", ".")
        x = float(x)
        return x / 100 if x > 1 else max(x, 0.0)

    out["Được cấp margin"] = (
        out["Được cấp margin"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "có": "Có",
            "co": "Có",
            "yes": "Có",
            "true": "Có",
            "1": "Có"
        })
        .fillna("Không")
    )

    out["Tỷ lệ cho vay"] = out["Tỷ lệ cho vay"].apply(pct)
    out["Lãi suất vay"] = out["Lãi suất vay"].apply(pct)

    out.loc[
        out["Được cấp margin"] != "Có",
        ["Tỷ lệ cho vay", "Lãi suất vay"]
    ] = 0.0

    # Mã thiếu trong bảng được xem là không có margin.
    existing = set(out["Mã"])
    missing = [x for x in tickers if x not in existing]

    if missing:
        out = pd.concat(
            [out, default_margin_table(missing)],
            ignore_index=True
        )

    return out[
        [
            "Mã",
            "Được cấp margin",
            "Tỷ lệ cho vay",
            "Lãi suất vay",
            "Ngày cập nhật"
        ]
    ].reset_index(drop=True)


def margin_position_limits(margin_table, tickers, max_leverage):
    """
    Tạo giới hạn vị thế cho từng mã.

    Không bán khống.
    Mã không có margin: tối đa 1,0 lần vốn tự có.
    Mã có margin: giới hạn theo vốn vay / vốn tự có
    và giới hạn tổng vị thế của danh mục.
    """

    margin = clean_margin_table(
        margin_table,
        tickers
    )

    limits = {}

    for ticker in tickers:

        row = margin[
            margin["Mã"] == ticker
        ]

        if row.empty:
            limits[ticker] = 1.0
            continue

        row = row.iloc[0]

        if row["Được cấp margin"] != "Có":
            limits[ticker] = 1.0
            continue

        loan_to_equity = float(
            row["Tỷ lệ cho vay"]
        )

        # Ví dụ:
        # 100% vốn vay / vốn tự có
        # => 1 đồng vốn tự có + 1 đồng vốn vay
        # => tổng vị thế tối đa 2,0 lần vốn tự có.
        limits[ticker] = min(
            1.0 + max(loan_to_equity, 0.0),
            max_leverage
        )

    return np.array(
        [limits[t] for t in tickers],
        dtype=float
    )


def apply_leverage_to_portfolio(
    base_weights,
    returns,
    margin_table,
    max_leverage
):
    """
    Tăng quy mô danh mục cơ sở trong giới hạn margin.

    Thành phần danh mục không đổi.
    Chỉ thay đổi tổng mức đầu tư.
    """

    base = np.asarray(
        base_weights,
        dtype=float
    )

    base = np.clip(base, 0, None)

    if base.sum() <= 0:
        return None

    base = base / base.sum()

    tickers = list(returns.columns)

    limits = margin_position_limits(
        margin_table,
        tickers,
        max_leverage
    )

    # Hệ số đòn bẩy tối đa mà danh mục cơ sở có thể chịu.
    # Không mã nào được vượt giới hạn vị thế.
    stock_scales = []

    for w, limit in zip(base, limits):
        if w > 0:
            stock_scales.append(
                limit / w
            )

    if not stock_scales:
        return None

    leverage_factor = min(
        max_leverage,
        min(stock_scales)
    )

    leverage_factor = max(
        1.0,
        leverage_factor
    )

    levered_weights = base * leverage_factor

    gross_exposure = levered_weights.sum()
    borrowed = max(
        gross_exposure - 1.0,
        0.0
    )

    margin = clean_margin_table(
        margin_table,
        tickers
    )

    incremental = (
        levered_weights - base
    ).clip(min=0)

    incremental_total = incremental.sum()

    borrowing_rate = 0.0

    if incremental_total > 0:

        for i, ticker in enumerate(tickers):

            row = margin[
                margin["Mã"] == ticker
            ]

            if row.empty:
                rate = 0.0
            else:
                rate = float(
                    row.iloc[0]["Lãi suất vay"]
                )

            borrowing_rate += (
                incremental[i]
                / incremental_total
                * rate
            )

    asset_returns = returns @ levered_weights

    annual_borrowing_cost = (
        borrowed * borrowing_rate
    )

    daily_borrowing_cost = (
        annual_borrowing_cost
        / PERIODS_PER_YEAR
    )

    net_returns = (
        asset_returns
        - daily_borrowing_cost
    )

    return {
        "weights": levered_weights,
        "leverage": gross_exposure,
        "borrowed": borrowed,
        "borrowing_rate": borrowing_rate,
        "borrowing_cost": annual_borrowing_cost,
        "returns": net_returns
    }


def build_margin_widgets(tickers):
    """
    Tạo bảng nhập margin.
    """

    widgets_by_ticker = {}

    rows = []

    for ticker in tickers:

        eligible = widgets.Dropdown(
            options=["Không", "Có"],
            value="Không",
            description="",
            layout=widgets.Layout(width="110px")
        )

        loan = widgets.FloatText(
            value=0.0,
            description="",
            layout=widgets.Layout(width="100px")
        )

        rate = widgets.FloatText(
            value=0.0,
            description="",
            layout=widgets.Layout(width="100px")
        )

        updated = widgets.Text(
            value="",
            description="",
            placeholder="YYYY-MM-DD",
            layout=widgets.Layout(width="125px")
        )

        widgets_by_ticker[ticker] = {
            "eligible": eligible,
            "loan": loan,
            "rate": rate,
            "updated": updated
        }

        rows.append(
            widgets.HBox([
                widgets.Label(
                    ticker,
                    layout=widgets.Layout(width="70px")
                ),
                eligible,
                loan,
                rate,
                updated
            ])
        )

    header = widgets.HBox([
        widgets.Label(
            "Mã",
            layout=widgets.Layout(width="70px")
        ),
        widgets.Label(
            "Được cấp margin",
            layout=widgets.Layout(width="110px")
        ),
        widgets.Label(
            "Vốn vay / vốn tự có (%)",
            layout=widgets.Layout(width="100px")
        ),
        widgets.Label(
            "Lãi suất vay (%)",
            layout=widgets.Layout(width="100px")
        ),
        widgets.Label(
            "Ngày cập nhật",
            layout=widgets.Layout(width="125px")
        )
    ])

    return widgets.VBox(
        [header] + rows
    ), widgets_by_ticker


def read_margin_widgets(tickers, widgets_by_ticker):
    rows = []

    for ticker in tickers:

        w = widgets_by_ticker[ticker]

        rows.append({
            "Mã": ticker,
            "Được cấp margin":
                w["eligible"].value,
            "Tỷ lệ cho vay":
                float(w["loan"].value) / 100,
            "Lãi suất vay":
                float(w["rate"].value) / 100,
            "Ngày cập nhật":
                w["updated"].value
        })

    return pd.DataFrame(rows)


def run_research(
    tickers,
    start_date,
    end_date,
    risk_free_rate,
    risk_aversion,
    benchmark,
    leverage,
    include_company,
    include_income,
    target_return,
    api_authenticated=False,
    margin_table=None,
    max_leverage=2.0):

    clear_output(wait=True)

    print("=" * 78)
    print("PORTFOLIO RESEARCH APP V21.2")
    print("=" * 78)
    print(f"Mã: {', '.join(tickers)}")
    print(f"Thời gian: {start_date} → {end_date}")
    print(f"Benchmark: {benchmark}")
    print(f"Lãi suất phi rủi ro: {risk_free_rate:.2%}")
    print(
        f"Đòn bẩy: {'Có' if leverage else 'Không'}"
    )
    print(
        f"Xác thực Vnstock: "
        f"{'API key Community' if api_authenticated else 'Chế độ khách'}"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # 1. DỮ LIỆU GIÁ
    # --------------------------------------------------------
    print("\n1. ĐANG CHUẨN BỊ DỮ LIỆU GIÁ")


    prices = get_price_data(
        tickers,
        start_date,
        end_date
    )

    if prices.empty or len(prices) < 150:
        raise ValueError(
            "Không đủ dữ liệu giá. "
            "Hãy kiểm tra khoảng thời gian hoặc API."
        )

    print(
        f"Đã lấy {len(prices):,} phiên dữ liệu ngày "
        f"cho {len(prices.columns)} mã."
    )

    # --------------------------------------------------------
    # 2. TỔNG QUAN DOANH NGHIỆP
    # --------------------------------------------------------
    print("\n2. TỔNG QUAN DOANH NGHIỆP")
    print("Vốn hóa được tính tại phiên cuối của khoảng thời gian nghiên cứu.")

    try:
        company_table = build_company_table(
            tickers,
            prices,
            include_company,
            include_income
        )
        display(format_company_table(company_table))
    except Exception as e:
        print(f"Bỏ qua phần thông tin doanh nghiệp: {e}")
        company_table = pd.DataFrame()

    # --------------------------------------------------------
    # 3. BENCHMARK
    # --------------------------------------------------------
    print("\n3. VNINDEX")

    benchmark_prices = get_benchmark_prices(
        benchmark,
        start_date,
        end_date
    )

    # Không yêu cầu ngày nhập phải là ngày giao dịch.
    # Tự động quy đổi sang phiên thực tế đầu tiên và cuối cùng.
    prices = prices.dropna(how="any")

    effective_start, effective_end = align_analysis_period(
        prices,
        benchmark_prices,
        start_date,
        end_date
    )

    # Cắt dữ liệu ngày theo khoảng thực tế.
    daily_prices = prices[
        (prices.index >= effective_start)
        & (prices.index <= effective_end)
    ].copy()

    daily_benchmark_prices = benchmark_prices[
        (benchmark_prices.index >= effective_start)
        & (benchmark_prices.index <= effective_end)
    ].copy()

    validate_benchmark_alignment(
        daily_benchmark_prices,
        daily_prices,
        effective_start,
        effective_end
    )

    # --------------------------------------------------------
    # CHUYỂN SANG GIÁ ĐÓNG CỬA TUẦN
    # --------------------------------------------------------
    prices = to_weekly_close(daily_prices)
    benchmark_prices = to_weekly_series(
        daily_benchmark_prices
    )

    # Sau khi chuyển tuần, chỉ giữ những tuần có đủ toàn bộ
    # cổ phiếu và benchmark.
    common_weekly = prices.index.intersection(
        benchmark_prices.index
    ).sort_values()

    prices = prices.loc[common_weekly].dropna(how="any")
    benchmark_prices = benchmark_prices.loc[
        common_weekly
    ].dropna()

    if len(common_weekly) < 52:
        raise ValueError(
            "Không đủ dữ liệu tuần để phân tích. "
            f"Chỉ có {len(common_weekly)} tuần giao nhau."
        )

    effective_weekly_start = common_weekly[0]
    effective_weekly_end = common_weekly[-1]

    print(
        f"Dữ liệu phân tích: giá tuần, {len(common_weekly):,} tuần."
    )

    returns = calculate_returns(prices)

    benchmark_returns = (
        benchmark_prices.pct_change()
        .dropna()
    )

    benchmark_summary = pd.DataFrame([{
        "Benchmark": benchmark,
        "Số tuần": len(benchmark_prices),
        "Lợi suất năm": benchmark_returns.mean() * PERIODS_PER_YEAR,
        "Biến động năm": benchmark_returns.std() * np.sqrt(PERIODS_PER_YEAR),
        "Sharpe": (
            benchmark_returns.mean() * PERIODS_PER_YEAR
            - risk_free_rate
        ) / (
            benchmark_returns.std() * np.sqrt(PERIODS_PER_YEAR)
        )
        if benchmark_returns.std() > 0 else np.nan
    }])

    display(
        benchmark_summary.style.format({
            "Lợi suất năm": "{:.2%}",
            "Biến động năm": "{:.2%}",
            "Sharpe": "{:.3f}"
        })
    )

    # --------------------------------------------------------
    # 4. ASSET ANALYSIS
    # --------------------------------------------------------
    print("\n4. PHÂN TÍCH TỪNG CỔ PHIẾU")

    asset_summary = calculate_asset_statistics(
        returns,
        benchmark_returns,
        risk_free_rate
    )

    display(
        asset_summary.style.format({
            "Lợi suất năm": "{:.2%}",
            "Biến động năm": "{:.2%}",
            "Sharpe": "{:.3f}",
            "Beta": "{:.3f}",
            "Tương quan VNINDEX": "{:.3f}"
        })
    )

    print("\nMA TRẬN TƯƠNG QUAN")

    display(
        returns.corr().style.format("{:.3f}")
    )

    # --------------------------------------------------------
    # 5. PORTFOLIOS
    # --------------------------------------------------------
    print("\n5. TỐI ƯU DANH MỤC")

    portfolio_results = optimize_portfolios(
        returns,
        risk_free_rate,
        risk_aversion,
        False,
        target_return
    )

    # --------------------------------------------------------
    # ĐÒN BẨY
    # --------------------------------------------------------
    leveraged_results = {}
    _levered_table = pd.DataFrame()
    _levered_alloc = pd.DataFrame()

    if leverage and margin_table is not None:

        _margin = clean_margin_table(
            margin_table,
            list(returns.columns)
        )

        for _name, _item in portfolio_results.items():

            if _name == "Target Return":
                continue

            if _item[0] is None:
                continue

            _levered = apply_leverage_to_portfolio(
                _item[0],
                returns,
                _margin,
                max_leverage
            )

            if _levered is None:
                continue

            _stats = portfolio_statistics(
                _levered["weights"],
                returns,
                risk_free_rate
            )

            # Điều chỉnh lợi suất theo chi phí vay.
            _stats["return"] -= _levered["borrowing_cost"]

            _stats["sharpe"] = (
                (_stats["return"] - risk_free_rate)
                / _stats["volatility"]
                if _stats["volatility"] > 0
                else np.nan
            )

            leveraged_results[_name] = {
                **_levered,
                "stats": _stats
            }


    # Tách nghiệm Markowitz theo mục tiêu khỏi các danh mục chiến lược chính.
    target_item = portfolio_results.get("Target Return")

    main_portfolio_results = {
        name: result
        for name, result in portfolio_results.items()
        if name != "Target Return"
    }

    # Tỷ trọng các danh mục chính, không bao gồm Target Return.
    valid_weights = {
        name: result[0]
        for name, result in main_portfolio_results.items()
        if result[0] is not None
    }

    weights = pd.DataFrame(
        valid_weights,
        index=returns.columns
    ).clip(lower=0)

    for _col in weights.columns:
        _total = weights[_col].sum()
        if _total > 0:
            weights[_col] = weights[_col] / _total

    portfolio_table = pd.DataFrame([
        {
            "Danh mục": name,
            "Lợi suất kỳ vọng": result[1]["return"],
            "Rủi ro": result[1]["volatility"],
            "Sharpe": result[1]["sharpe"]
        }
        for name, result in main_portfolio_results.items()
        if result[0] is not None
    ])

    portfolio_table["So với mục tiêu"] = (
        portfolio_table["Lợi suất kỳ vọng"]
        - target_return
    )

    # Target Return là mục tiêu của nhà đầu tư, không phải
    # một danh mục trong bảng so sánh chính.
    # --------------------------------------------------------
    # 5.3. TARGET RETURN — MỤC TIÊU VÀ NGHIỆM MARKOWITZ RIÊNG
    # --------------------------------------------------------

    if leverage and leveraged_results:

        # ----------------------------------------------------
        # 5.1. PHÂN TÍCH CÓ ĐÒN BẨY
        # ----------------------------------------------------
        print("\n5.1. PHÂN TÍCH CÓ ĐÒN BẨY")

        _levered_rows = []

        for _name, _item in leveraged_results.items():

            _s = _item["stats"]

            _levered_rows.append({
                "Danh mục": _name,
                "Tổng vị thế": _item["leverage"],
                "Vốn vay": _item["borrowed"],
                "Lãi suất vay bình quân":
                    _item["borrowing_rate"],
                "Chi phí vay": _item["borrowing_cost"],
                "Lợi suất": _s["return"],
                "Rủi ro": _s["volatility"],
                "Sharpe": _s["sharpe"]
            })

        _levered_table = pd.DataFrame(
            _levered_rows
        )

        if not _levered_table.empty:

            display(
                _levered_table.style.format({
                    "Tổng vị thế": "{:.2%}",
                    "Vốn vay": "{:.2%}",
                    "Lãi suất vay bình quân": "{:.2%}",
                    "Chi phí vay": "{:.2%}",
                    "Lợi suất": "{:.2%}",
                    "Rủi ro": "{:.2%}",
                    "Sharpe": "{:.3f}"
                })
            )

        # ----------------------------------------------------
        # 5.2. PHÂN BỔ DANH MỤC CÓ ĐÒN BẨY
        # ----------------------------------------------------
        print("\n5.2. PHÂN BỔ DANH MỤC CÓ ĐÒN BẨY")

        _levered_alloc_rows = []

        for _name, _item in leveraged_results.items():

            _w = pd.Series(
                _item["weights"],
                index=returns.columns,
                dtype=float
            )

            _row = {
                "Danh mục": _name
            }

            for _ticker in returns.columns:
                _row[_ticker] = _w.get(
                    _ticker,
                    0.0
                )

            _row["Tổng vị thế"] = _item["leverage"]
            _row["Vốn vay"] = _item["borrowed"]

            _levered_alloc_rows.append(_row)

        _levered_alloc = pd.DataFrame(
            _levered_alloc_rows
        )

        if not _levered_alloc.empty:

            _fmt = {
                _ticker: "{:.2%}"
                for _ticker in returns.columns
            }

            _fmt.update({
                "Tổng vị thế": "{:.2%}",
                "Vốn vay": "{:.2%}"
            })

            display(
                _levered_alloc.style.format(_fmt)
            )

    # --------------------------------------------------------
    # 5.3. MỤC TIÊU LỢI NHUẬN
    # --------------------------------------------------------
    print("\n5.3. MỤC TIÊU LỢI NHUẬN")

    print(
        f"Mục tiêu của nhà đầu tư: {target_return:.2%}/năm."
    )

    target_summary = None

    if target_item is not None and target_item[0] is not None:

        target_stats = target_item[1]

        target_summary = pd.DataFrame([{
            "Mục tiêu": target_return,
            "Lợi suất kỳ vọng": target_stats["return"],
            "Rủi ro thấp nhất": target_stats["volatility"],
            "Sharpe": target_stats["sharpe"],
            "Trạng thái": (
                "Khả thi"
                if target_stats["return"] >= target_return
                else "Không đạt"
            )
        }])

        display(
            target_summary.style.format({
                "Mục tiêu": "{:.2%}",
                "Lợi suất kỳ vọng": "{:.2%}",
                "Rủi ro thấp nhất": "{:.2%}",
                "Sharpe": "{:.3f}"
            })
        )

    else:
        print(
            "Không tìm được danh mục Markowitz khả thi "
            "với mục tiêu hiện tại."
        )

    # So sánh các danh mục chính với ngưỡng mục tiêu.
    target_check = portfolio_table[
        [
            "Danh mục",
            "Lợi suất kỳ vọng",
            "Rủi ro",
            "Sharpe",
            "So với mục tiêu"
        ]
    ].copy()

    target_check["Đạt mục tiêu"] = np.where(
        target_check["Lợi suất kỳ vọng"] >= target_return,
        "Có",
        "Không"
    )

    display(
        target_check.style.format({
            "Lợi suất kỳ vọng": "{:.2%}",
            "Rủi ro": "{:.2%}",
            "Sharpe": "{:.3f}",
            "So với mục tiêu": "{:+.2%}"
        })
    )

    print("\nTỶ TRỌNG")

    display(
        weights.style.format("{:.2%}")
    )

    # --------------------------------------------------------
    # 6. BENCHMARK COMPARISON
    # --------------------------------------------------------
    print("\n6. SO SÁNH VỚI VNINDEX")

    comparison, benchmark_row, benchmark_wealth, benchmark_dd = (
        benchmark_comparison(
            main_portfolio_results,
            returns,
            benchmark_returns,
            risk_free_rate
        )
    )

    display(
        comparison.style.format({
            "Lợi suất": "{:.2%}",
            "Rủi ro": "{:.2%}",
            "Sharpe": "{:.3f}",
            "Max Drawdown": "{:.2%}",
            "Alpha": "{:.2%}",
            "Beta": "{:.3f}",
            "Tương quan": "{:.3f}",
            "Tracking Error": "{:.2%}",
            "Information Ratio": "{:.3f}",
            "Vượt VNINDEX": "{:.2%}"
        })
    )

    # --------------------------------------------------------
    # 7. TĂNG TRƯỞNG TÍCH LŨY
    # --------------------------------------------------------
    print("\n7. TĂNG TRƯỞNG TÍCH LŨY")

    plt.figure(figsize=(13, 6))

    for name, result in main_portfolio_results.items():
        if result[0] is None:
            continue

        p_returns = returns @ result[0]
        wealth = 100 * (1 + p_returns).cumprod()

        plt.plot(
            wealth.index,
            wealth.values,
            label=name,
            linewidth=1.6
        )

    aligned_bench = benchmark_returns.reindex(
        returns.index
    ).dropna()

    bench_wealth = (
        100 * (1 + aligned_bench).cumprod()
    )

    plt.plot(
        bench_wealth.index,
        bench_wealth.values,
        label=benchmark,
        linewidth=2.4
    )

    plt.title(
        "Tăng trưởng danh mục với 100 triệu đồng vốn ban đầu"
    )
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị danh mục (triệu đồng)")
    plt.legend(
        ncol=2,
        loc="upper left"
    )
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 8. LỢI SUẤT CUỘN 12 THÁNG
    # --------------------------------------------------------
    print("\n8. LỢI SUẤT CUỘN 12 THÁNG")

    rolling_rows = {}

    for name, result in main_portfolio_results.items():
        if result[0] is None:
            continue

        p_returns = returns @ result[0]

        rolling_rows[name] = (
            1 + p_returns
        ).rolling(52).apply(np.prod, raw=True) - 1

    aligned_bench = benchmark_returns.reindex(
        returns.index
    ).dropna()

    rolling_rows[benchmark] = (
        1 + aligned_bench
    ).rolling(52).apply(np.prod, raw=True) - 1

    rolling_df = pd.DataFrame(
        rolling_rows
    ).dropna(how="all")

    plt.figure(figsize=(13, 6))

    for col in rolling_df.columns:
        plt.plot(
            rolling_df.index,
            rolling_df[col] * 100,
            label=col,
            linewidth=1.4
        )

    plt.axhline(
        target_return * 100,
        linestyle="--",
        linewidth=1.5,
        label=f"Mục tiêu {target_return:.1%}"
    )

    plt.title(
        "Lợi suất thực tế trong từng giai đoạn 12 tháng"
    )
    plt.xlabel("Thời gian")
    plt.ylabel("Lợi suất 12 tháng (%)")
    plt.legend(
        ncol=2,
        loc="best"
    )
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 9. RỦI RO VÀ LỢI SUẤT
    # --------------------------------------------------------
    print("\n9. RỦI RO VÀ LỢI SUẤT")

    plot_df = comparison.copy()

    plt.figure(figsize=(11, 6))

    plt.scatter(
        plot_df["Rủi ro"] * 100,
        plot_df["Lợi suất"] * 100,
        s=110
    )

    for _, row in plot_df.iterrows():
        plt.annotate(
            row["Danh mục"],
            (
                row["Rủi ro"] * 100,
                row["Lợi suất"] * 100
            ),
            xytext=(8, 6),
            textcoords="offset points"
        )

    plt.axhline(
        target_return * 100,
        linestyle="--",
        linewidth=1.3,
        label=f"Mục tiêu {target_return:.1%}"
    )

    plt.xlabel("Rủi ro năm (%)")
    plt.ylabel("Lợi suất kỳ vọng năm (%)")
    plt.title("So sánh rủi ro và lợi suất")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 10. DRAWDOWN
    # --------------------------------------------------------
    print("\n10. DRAWDOWN")

    plt.figure(figsize=(13, 6))

    for name, result in main_portfolio_results.items():
        if result[0] is None:
            continue

        dd = portfolio_drawdown(
            result[0],
            returns
        )

        plt.plot(
            dd.index,
            dd.values * 100,
            label=name,
            linewidth=1.4
        )

    aligned_bench = benchmark_returns.reindex(
        returns.index
    ).dropna()

    bench_wealth = (
        1 + aligned_bench
    ).cumprod()

    bench_dd = (
        bench_wealth
        / bench_wealth.cummax()
        - 1
    )

    plt.plot(
        bench_dd.index,
        bench_dd.values * 100,
        label=benchmark,
        linewidth=2.4
    )

    plt.axhline(
        0,
        linewidth=0.8
    )

    plt.title(
        "Mức sụt giảm từ đỉnh của từng danh mục"
    )
    plt.xlabel("Thời gian")
    plt.ylabel("Sụt giảm từ đỉnh (%)")
    plt.legend(
        ncol=2,
        loc="lower left"
    )
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 10. WEIGHTS
    # --------------------------------------------------------
    print("\n14. CƠ CẤU TỶ TRỌNG")

    valid_weights = {
        name: result[0]
        for name, result in main_portfolio_results.items()
        if result[0] is not None
    }

    weights = pd.DataFrame(
        valid_weights,
        index=returns.columns
    ).clip(lower=0)

    # Chuẩn hóa để mỗi danh mục cổ phiếu có tổng tỷ trọng 100%.
    for _col in weights.columns:
        _total = weights[_col].sum()
        if _total > 0:
            weights[_col] = weights[_col] / _total

    main_weight_columns = [
        name for name in [
            "Naive",
            "Minimum Variance",
            "Optimal Risky",
            "Maximum Return",
            "Complete Portfolio"
        ]
        if name in weights.columns
    ]

    main_weights = weights[main_weight_columns].copy()

    # Mỗi mã cổ phiếu giữ nguyên một màu ở mọi biểu đồ.
    stock_colors = {}
    color_cycle = plt.cm.tab10.colors

    for i, ticker in enumerate(main_weights.index):
        stock_colors[ticker] = color_cycle[i % len(color_cycle)]

    # Năm biểu đồ tròn trên cùng một vùng trắng.
    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 9),
        facecolor="white"
    )
    axes = axes.flatten()

    for ax, portfolio_name in zip(axes, main_weight_columns):

        values = main_weights[portfolio_name].astype(float)
        values = values[values > 0]

        if values.empty:
            ax.text(
                0.5, 0.5,
                "Không có tỷ trọng",
                ha="center",
                va="center"
            )
            ax.set_title(portfolio_name)
            ax.axis("off")
            continue

        pie_colors = [
            stock_colors[ticker]
            for ticker in values.index
        ]

        wedges, _, _ = ax.pie(
            values.values,
            labels=None,
            colors=pie_colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
            startangle=90,
            counterclock=False,
            wedgeprops={
                "edgecolor": "white",
                "linewidth": 1
            }
        )

        ax.set_title(
            portfolio_name,
            fontsize=12,
            fontweight="bold"
        )

        ax.legend(
            wedges,
            [
                f"{ticker}: {values[ticker]:.1%}"
                for ticker in values.index
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
            fontsize=9
        )

        ax.set_aspect("equal")

    # Ô cuối dùng làm chú giải màu chung.
    axes[5].axis("off")

    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=stock_colors[ticker],
            markeredgecolor="white",
            markersize=9,
            label=ticker
        )
        for ticker in main_weights.index
    ]

    axes[5].legend(
        handles=legend_handles,
        title="Mã cổ phiếu",
        loc="center",
        frameon=False
    )

    fig.suptitle(
        "Cơ cấu tỷ trọng các danh mục",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()

    # Danh mục Markowitz theo mục tiêu được trình bày riêng.
    if target_item is not None and target_item[0] is not None:

        target_w = pd.Series(
            target_item[0],
            index=returns.columns
        ).clip(lower=0)

        target_w = target_w[target_w > 0]

        if not target_w.empty:

            fig, ax = plt.subplots(
                figsize=(7, 7),
                facecolor="white"
            )

            target_colors = [
                stock_colors.get(
                    ticker,
                    color_cycle[i % len(color_cycle)]
                )
                for i, ticker in enumerate(target_w.index)
            ]

            wedges, _, _ = ax.pie(
                target_w.values,
                labels=None,
                colors=target_colors,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
                startangle=90,
                counterclock=False,
                wedgeprops={
                    "edgecolor": "white",
                    "linewidth": 1
                }
            )

            ax.set_title(
                f"Markowitz theo mục tiêu {target_return:.1%}",
                fontsize=14,
                fontweight="bold"
            )

            ax.legend(
                wedges,
                [
                    f"{ticker}: {target_w[ticker]:.1%}"
                    for ticker in target_w.index
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=2,
                frameon=False
            )

            ax.set_aspect("equal")
            plt.tight_layout()
            plt.show()

    # 11. MARKOWITZ: ĐƯỜNG BIÊN HIỆU QUẢ
    # --------------------------------------------------------
    print("\n11. MARKOWITZ: ĐƯỜNG BIÊN HIỆU QUẢ")

    print(
        "Mỗi điểm trên đường biên là một danh mục có "
        "rủi ro thấp nhất cho một mức lợi suất kỳ vọng."
    )
    print(
        f"Điểm mục tiêu hiện tại: {target_return:.2%} mỗi năm."
    )

    frontier = build_efficient_frontier(
        returns,
        risk_free_rate,
        leverage,
        points=31
    )

    if not frontier.empty:
        plt.figure(figsize=(12, 7))

        plt.plot(
            frontier["Risk"] * 100,
            frontier["Return"] * 100,
            linewidth=2.2,
            label="Đường biên Markowitz"
        )

        # Các danh mục đang nghiên cứu
        for name, item in main_portfolio_results.items():
            if item[0] is None:
                continue

            stats = item[1]

            plt.scatter(
                stats["volatility"] * 100,
                stats["return"] * 100,
                s=85
            )

            plt.annotate(
                name,
                (
                    stats["volatility"] * 100,
                    stats["return"] * 100
                ),
                xytext=(7, 5),
                textcoords="offset points"
            )

        # Điểm Target Return được đánh dấu riêng.
        if target_item is not None and target_item[0] is not None:
            target_stats = target_item[1]

            plt.scatter(
                target_stats["volatility"] * 100,
                target_stats["return"] * 100,
                s=180,
                marker="*",
                label="Danh mục Markowitz theo mục tiêu"
            )

        plt.axhline(
            target_return * 100,
            linestyle="--",
            linewidth=1.5,
            label=f"Mục tiêu {target_return:.1%}"
        )

        plt.xlabel("Rủi ro năm (%)")
        plt.ylabel("Lợi suất kỳ vọng năm (%)")
        plt.title(
            "Đường biên Markowitz: lợi suất mục tiêu và rủi ro"
        )
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 12. SO SÁNH VỚI MỤC TIÊU CỦA NHÀ ĐẦU TƯ
    # --------------------------------------------------------
    print("\n12. MỤC TIÊU LỢI NHUẬN VÀ KHẢ NĂNG ĐẠT MỤC TIÊU")

    target_analysis_rows = []

    for name, item in main_portfolio_results.items():
        if item[0] is None:
            continue

        stats = item[1]

        target_analysis_rows.append({
            "Danh mục": name,
            "Lợi suất kỳ vọng": stats["return"],
            "Mục tiêu": target_return,
            "Chênh lệch": stats["return"] - target_return,
            "Rủi ro": stats["volatility"],
            "Sharpe": stats["sharpe"]
        })

    target_analysis = pd.DataFrame(target_analysis_rows)

    display(
        target_analysis.style.format({
            "Lợi suất kỳ vọng": "{:.2%}",
            "Mục tiêu": "{:.2%}",
            "Chênh lệch": "{:+.2%}",
            "Rủi ro": "{:.2%}",
            "Sharpe": "{:.3f}"
        })
    )

    # --------------------------------------------------------
    # 13. CƠ CẤU TỶ TRỌNG
    # --------------------------------------------------------
    # --------------------------------------------------------
    # 14. COMPLETE PORTFOLIO DETAIL
    # --------------------------------------------------------
    complete_stats = portfolio_results[
        "Complete Portfolio"
    ][1]

    print("\n15. COMPLETE PORTFOLIO")

    print(
        f"Tỷ trọng lý thuyết vào danh mục rủi ro: "
        f"{complete_stats['theoretical_risky_weight']:.2%}"
    )

    print(
        f"Tỷ trọng thực tế vào danh mục rủi ro: "
        f"{complete_stats['risky_weight']:.2%}"
    )

    print(
        f"Tỷ trọng tài sản phi rủi ro: "
        f"{complete_stats['risk_free_weight']:.2%}"
    )


    # --------------------------------------------------------
    # 13. ĐỘ NHẠY MỤC TIÊU LỢI NHUẬN
    # --------------------------------------------------------
    print("\n13. ĐỘ NHẠY: MỤC TIÊU LỢI NHUẬN VÀ RỦI RO")

    sensitivity_targets = [
        0.05, 0.08, 0.10, 0.12, 0.15,
        0.18, 0.20, 0.25, 0.30
    ]

    sensitivity = build_target_sensitivity(
        returns,
        risk_free_rate,
        leverage,
        sensitivity_targets
    )

    display(
        sensitivity.style.format({
            "Mục tiêu": "{:.2%}",
            "Lợi suất": "{:.2%}",
            "Rủi ro tối thiểu": "{:.2%}",
            "Sharpe": "{:.3f}"
        })
    )

    valid_sensitivity = sensitivity.dropna(
        subset=["Rủi ro tối thiểu"]
    )

    if not valid_sensitivity.empty:
        plt.figure(figsize=(11, 6))

        plt.plot(
            valid_sensitivity["Mục tiêu"] * 100,
            valid_sensitivity["Rủi ro tối thiểu"] * 100,
            marker="o"
        )

        plt.axvline(
            target_return * 100,
            linestyle="--",
            linewidth=1.3,
            label=f"Mục tiêu hiện tại {target_return:.1%}"
        )

        plt.xlabel("Mục tiêu lợi nhuận (%)")
        plt.ylabel("Rủi ro tối thiểu (%)")
        plt.title("Đánh đổi giữa mục tiêu lợi nhuận và rủi ro")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 14. KHẢ NĂNG ĐẠT MỤC TIÊU TRONG LỊCH SỬ
    # --------------------------------------------------------
    print("\n14. KHẢ NĂNG ĐẠT MỤC TIÊU TRONG LỊCH SỬ")

    attainment_rows = []

    for name, item in main_portfolio_results.items():

        if item[0] is None:
            continue

        p_returns = returns @ item[0]

        attainment = target_attainment_analysis(
            p_returns,
            target_return
        )

        attainment_rows.append({
            "Danh mục": name,
            **attainment
        })

    attainment_table = pd.DataFrame(
        attainment_rows
    )

    display(
        attainment_table.style.format({
            "Tỷ lệ đạt mục tiêu": "{:.2%}",
            "Lợi suất cuộn thấp nhất": "{:.2%}",
            "Lợi suất cuộn trung vị": "{:.2%}",
            "Lợi suất cuộn cao nhất": "{:.2%}"
        })
    )

    # --------------------------------------------------------
    # 15. PHÂN TÍCH TẬP TRUNG
    # --------------------------------------------------------
    print("\n15. PHÂN TÍCH TẬP TRUNG DANH MỤC")

    concentration = concentration_table(
        weights
    )

    display(
        concentration.style.format({
            "HHI": "{:.3f}",
            "Số tài sản hiệu dụng": "{:.2f}",
            "Số tài sản hiệu dụng theo |w|": "{:.2f}",
            "Tỷ trọng lớn nhất": "{:.2%}"
        })
    )

    # --------------------------------------------------------
    # 16. MAX DRAWDOWN VÀ THỜI GIAN PHỤC HỒI
    # --------------------------------------------------------
    print("\n16. MAX DRAWDOWN VÀ THỜI GIAN PHỤC HỒI")
    print(
        "Nếu chưa quay lại đỉnh cũ, hệ thống ghi Chưa phục hồi "
        "và tính số ngày từ đáy đến cuối kỳ."
    )

    risk_diagnostics = build_risk_diagnostics(
        main_portfolio_results,
        returns,
        benchmark_returns,
        target_return
    )

    display(
        risk_diagnostics.style.format({
            "Max Drawdown": "{:.2%}",
            "Tỷ lệ đạt mục tiêu": "{:.2%}",
            "Rolling 12T thấp nhất": "{:.2%}",
            "Rolling 12T trung vị": "{:.2%}",
            "Rolling 12T cao nhất": "{:.2%}"
        })
    )

    # --------------------------------------------------------
    # 17. BẢNG KẾT LUẬN ĐỊNH LƯỢNG
    # --------------------------------------------------------
    print("\n17. BẢNG ĐÁNH GIÁ THEO MỤC TIÊU NHÀ ĐẦU TƯ")

    conclusion = investment_conclusion(
        portfolio_table,
        comparison,
        target_return,
        risk_aversion
    )

    display(
        conclusion.style.format({
            "Mục tiêu": "{:.2%}",
            "Lợi suất kỳ vọng": "{:.2%}",
            "Chênh lệch mục tiêu": "{:+.2%}",
            "Rủi ro": "{:.2%}",
            "Sharpe": "{:.3f}",
            "Điểm tham khảo": "{:.3f}"
        })
    )

    # --------------------------------------------------------
    # 18. BIỂU ĐỒ SO SÁNH RỦI RO VÀ MỤC TIÊU
    # --------------------------------------------------------
    print("\n18. VỊ TRÍ DANH MỤC SO VỚI MỤC TIÊU")

    plt.figure(figsize=(12, 7))

    for _, row in conclusion.iterrows():

        plt.scatter(
            row["Rủi ro"] * 100,
            row["Lợi suất kỳ vọng"] * 100,
            s=110
        )

        plt.annotate(
            row["Danh mục"],
            (
                row["Rủi ro"] * 100,
                row["Lợi suất kỳ vọng"] * 100
            ),
            xytext=(7, 5),
            textcoords="offset points"
        )

    plt.axhline(
        target_return * 100,
        linestyle="--",
        linewidth=1.5,
        label=f"Mục tiêu {target_return:.1%}"
    )

    plt.xlabel("Rủi ro năm (%)")
    plt.ylabel("Lợi suất kỳ vọng năm (%)")
    plt.title("Danh mục so với mục tiêu lợi nhuận")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    print("\nHOÀN TẤT PHÂN TÍCH.")

    return {
        "company_table": company_table,
        "prices": prices,
        "returns": returns,
        "benchmark_prices": benchmark_prices,
        "benchmark_returns": benchmark_returns,
        "requested_period": (
            pd.Timestamp(start_date),
            pd.Timestamp(end_date)
        ),
        "effective_period": (
            effective_start,
            effective_end
        ),
        "weekly_period": (
            effective_weekly_start,
            effective_weekly_end
        ),
        "analysis_frequency": "Tuần, giá đóng cửa cuối tuần",
        "asset_summary": asset_summary,
        "correlation": returns.corr(),
        "portfolio_results": portfolio_results,
        "weights": weights,
        "comparison": comparison,
        "benchmark": benchmark_row,
        "complete_stats": complete_stats,
        "target_return": target_return,
        "target_analysis": target_analysis,
        "efficient_frontier": frontier,
        "sensitivity": sensitivity,
        "attainment": attainment_table,
        "concentration": concentration,
        "risk_diagnostics": risk_diagnostics,
        "conclusion": conclusion,
        "frontier": frontier,
        "benchmark_summary": benchmark_summary,
        "portfolio_table": portfolio_table,
        "target_summary": target_summary,
        "target_check": target_check,
        "leveraged_results": leveraged_results,
        "levered_table": _levered_table,
        "levered_alloc": _levered_alloc
    }

# ============================================================

