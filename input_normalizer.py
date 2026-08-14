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

PERCENT_COLUMNS = {
    "ROA",
    "ROE",
    "roa",
    "roe",
}

TEXT_VALUES = {
    "",
    "không",
    "khong",
    "n/a",
    "na",
    "none",
    "null",
    "nan",
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

    Vnstock Community ở nguồn hiện tại trả các chỉ tiêu BCTC
    theo đơn vị tỷ đồng cho phần dữ liệu mà engine đang đọc.
    Vì vậy 5.96 phải được hiểu là 5.96 nghìn tỷ, tức 5.96e12 VND.

    Hàm này được thiết kế bảo thủ:
    nếu giá trị đã có quy mô VND thì không nhân thêm.
    """
    x = safe_numeric(value)
    if pd.isna(x):
        return np.nan

    # Giá trị BCTC đang được engine nhận thường ở đơn vị tỷ VND.
    # Mốc 1e10 giúp tránh nhân lại các giá trị vốn đã ở VND.
    if abs(x) < 1e10:
        return x * 1e9

    return x


def normalize_market_cap(value):
    x = safe_numeric(value)
    if pd.isna(x):
        return np.nan

    # Vốn hóa của engine hiện đã ở VND. Chỉ quy đổi nếu nguồn
    # trả về đơn vị tỷ hoặc triệu với quy mô rõ ràng.
    if abs(x) < 1e8:
        return x * 1e9
    return x


def normalize_percent(value):
    x = safe_numeric(value)
    if pd.isna(x):
        return np.nan

    # Nguồn có thể trả 0.15 hoặc 15 cho 15%.
    if abs(x) > 1:
        return x / 100.0
    return x


def normalize_company_dataframe(df):
    """Chuẩn hóa DataFrame tổng quan doanh nghiệp."""
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

    for key in [
        "company_info",
        "company_summary",
        "asset_summary",
    ]:
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            result[key] = normalize_company_dataframe(value)

    return result
