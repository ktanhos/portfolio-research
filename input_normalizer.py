"""
Chuẩn hóa dữ liệu đầu vào trước khi đưa vào mô hình.

Quy ước nội bộ:
    Tiền BCTC: VND
    Vốn hóa: VND
    Số cổ phiếu: cổ phiếu
    EPS: VND/cổ phiếu
    P/E, P/B: số lần
    ROA, ROE: tỷ lệ thập phân

Lớp này không sửa dữ liệu giá, lợi suất hoặc Markowitz.
"""

import numpy as np
import pandas as pd

FINANCIAL_MONEY_COLUMNS = {
    "Doanh thu gần nhất",
    "LNST gần nhất",
    "Doanh thu",
    "LNST",
    "Lợi nhuận sau thuế",
    "Net Profit",
    "Revenue",
}

MONEY_COLUMNS = {
    "Vốn hóa",
    "Market Cap",
    "market_cap",
}

RATIO_COLUMNS = {
    "P/E",
    "P/B",
    "Sharpe",
    "Sortino",
    "Beta",
    "Tương quan",
    "Tương quan VNINDEX",
}

PERCENT_COLUMNS = {"ROA", "ROE", "roa", "roe"}

TEXT_VALUES = {
    "", "không", "khong", "n/a", "na", "none", "null", "nan"
}


def safe_numeric(value):
    """Chuyển số an toàn; giá trị văn bản không biến thành lỗi."""
    if value is None:
        return np.nan
    if isinstance(value, str):
        text = value.strip().lower()
        if text in TEXT_VALUES:
            return np.nan
        text = text.replace("%", "").replace(",", "")
        try:
            return float(text)
        except (TypeError, ValueError):
            return np.nan
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def normalize_financial_money(value):
    """
    Chuẩn hóa tiền BCTC về VND.

    Engine hiện nhận các giá trị BCTC theo đơn vị nghìn VND.
    Vì vậy 5.960.000.000 trong nguồn tương ứng 5.960.000.000.000 VND.

    Nếu dữ liệu đã ở VND với quy mô từ 1 nghìn tỷ trở lên thì giữ nguyên.
    """
    x = safe_numeric(value)
    if pd.isna(x):
        return np.nan

    # Nguồn hiện tại đang trả BCTC ở nghìn VND.
    # 5,96 tỷ trong bảng cũ thực chất là 5,96 nghìn tỷ.
    if abs(x) < 1e12:
        return x * 1000.0

    return x


def normalize_market_cap(value):
    x = safe_numeric(value)
    if pd.isna(x):
        return np.nan
    return x


def normalize_percent(value):
    x = safe_numeric(value)
    if pd.isna(x):
        return np.nan
    if abs(x) > 1:
        return x / 100.0
    return x


def normalize_company_dataframe(df):
    """Chuẩn hóa DataFrame tổng quan doanh nghiệp trước khi phân tích."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    for col in FINANCIAL_MONEY_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(normalize_financial_money)

    for col in MONEY_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(normalize_market_cap)

    for col in PERCENT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(normalize_percent)

    for col in ["Số CP lưu hành", "EPS"]:
        if col in out.columns:
            out[col] = out[col].map(safe_numeric)

    for col in RATIO_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(safe_numeric)

    return out


def normalize_result_tables(result):
    """Chuẩn hóa các bảng doanh nghiệp trong kết quả nghiên cứu."""
    if not isinstance(result, dict):
        return result

    for key in ["company_info", "company_summary", "company_table", "asset_summary"]:
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            result[key] = normalize_company_dataframe(value)

    return result
