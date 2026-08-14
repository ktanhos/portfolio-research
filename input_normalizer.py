"""
Lớp chuẩn hóa dữ liệu đầu vào.

Quy ước nội bộ:
    Tiền BCTC: VND
    Vốn hóa: VND
    Số cổ phiếu: cổ phiếu
    EPS: VND/cổ phiếu
    P/E, P/B: số lần
    ROA, ROE: tỷ lệ thập phân

Ưu tiên sử dụng unit_multiplier hoặc unit do nguồn dữ liệu cung cấp.
Chỉ dùng quy tắc dự phòng khi nguồn không trả metadata đơn vị.
"""

import re
import numpy as np
import pandas as pd

FINANCIAL_MONEY_COLUMNS = {
    "Doanh thu gần nhất", "LNST gần nhất", "Doanh thu", "LNST",
    "Lợi nhuận sau thuế", "Net Profit", "Revenue"
}

MONEY_COLUMNS = {"Vốn hóa", "Market Cap", "market_cap"}
RATIO_COLUMNS = {"P/E", "P/B", "Sharpe", "Sortino", "Beta", "Tương quan", "Tương quan VNINDEX"}
PERCENT_COLUMNS = {"ROA", "ROE", "roa", "roe"}
TEXT_VALUES = {"", "không", "khong", "n/a", "na", "none", "null", "nan"}


def safe_numeric(value):
    """Chuyển số an toàn, hỗ trợ cả dấu thập phân Việt Nam."""
    if value is None:
        return np.nan
    if isinstance(value, str):
        text = value.strip().lower()
        if text in TEXT_VALUES:
            return np.nan
        text = text.replace("%", "").replace(" ", "")
        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif "," in text:
            parts = text.split(",")
            if len(parts) == 2 and len(parts[1]) != 3:
                text = text.replace(",", ".")
            else:
                text = text.replace(",", "")
        try:
            return float(text)
        except (TypeError, ValueError):
            return np.nan
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _unit_multiplier_from_text(unit):
    """Đọc hệ số từ mô tả đơn vị của nguồn dữ liệu."""
    if unit is None or pd.isna(unit):
        return None
    text = str(unit).strip().lower()
    text = text.replace("đ", "d").replace("₫", "d")

    if "nghìn tỷ" in text or "trillion" in text or "tn" == text:
        return 1e12
    if "tỷ" in text or "billion" in text or "bn" == text:
        return 1e9
    if "triệu" in text or "million" in text or "mn" == text:
        return 1e6
    if "nghìn" in text or "thousand" in text or "k vnd" in text:
        return 1e3
    if re.search(r"\bvnd\b|\bdong\b", text):
        return 1.0
    return None


def get_unit_multiplier(row=None, unit=None, unit_multiplier=None):
    """Ưu tiên hệ số nhân chính thức, sau đó mới đọc cột unit."""
    for value in (unit_multiplier, unit):
        if value is None:
            continue
        numeric = safe_numeric(value)
        if pd.notna(numeric) and numeric > 0:
            return float(numeric)
        parsed = _unit_multiplier_from_text(value)
        if parsed is not None:
            return parsed

    if row is not None and hasattr(row, "index"):
        for col in ("unit_multiplier", "unit"):
            if col in row.index:
                result = get_unit_multiplier(unit=row[col]) if col == "unit" else get_unit_multiplier(unit_multiplier=row[col])
                if result is not None:
                    return result
    return None


def normalize_financial_money(value, unit=None, unit_multiplier=None):
    """Đưa giá trị BCTC về VND."""
    x = safe_numeric(value)
    if pd.isna(x):
        return np.nan

    multiplier = get_unit_multiplier(unit=unit, unit_multiplier=unit_multiplier)
    if multiplier is not None:
        return x * multiplier

    # Dự phòng cho dữ liệu cũ không có metadata đơn vị.
    # Engine V21 hiện nhận số tiền BCTC thấp hơn VND đúng một bậc 1.000.
    if abs(x) < 1e12:
        return x * 1000.0
    return x


def normalize_market_cap(value, unit=None, unit_multiplier=None):
    x = safe_numeric(value)
    if pd.isna(x):
        return np.nan
    multiplier = get_unit_multiplier(unit=unit, unit_multiplier=unit_multiplier)
    return x * multiplier if multiplier is not None else x


def normalize_percent(value):
    x = safe_numeric(value)
    if pd.isna(x):
        return np.nan
    return x / 100.0 if abs(x) > 1 else x


def normalize_income_dataframe(df):
    """Chuẩn hóa bảng BCTC dạng report nếu nguồn cung cấp unit metadata."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in out.columns]

    unit_col = "unit" if "unit" in out.columns else None
    multiplier_col = "unit_multiplier" if "unit_multiplier" in out.columns else None
    item_col = next((c for c in ("item", "item_name", "name", "indicator") if c in out.columns), None)

    if item_col is None:
        return out

    money_terms = ("doanh thu", "revenue", "lợi nhuận", "profit", "chi phí", "expense", "tài sản", "asset", "nợ", "liabilities", "vốn chủ", "equity", "cash", "tiền")

    for idx in out.index:
        item = str(out.at[idx, item_col]).lower()
        if not any(term in item for term in money_terms):
            continue
        multiplier = get_unit_multiplier(
            unit=out.at[idx, unit_col] if unit_col else None,
            unit_multiplier=out.at[idx, multiplier_col] if multiplier_col else None,
        )
        if multiplier is None:
            continue
        for col in out.columns:
            if str(col).isdigit() and len(str(col)) == 4:
                x = safe_numeric(out.at[idx, col])
                if pd.notna(x):
                    out.at[idx, col] = x * multiplier
    return out


def normalize_company_dataframe(df):
    """Chuẩn hóa bảng tổng quan doanh nghiệp về đơn vị nội bộ."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()

    unit_col = next((c for c in ("unit", "đơn vị") if c in out.columns), None)
    multiplier_col = "unit_multiplier" if "unit_multiplier" in out.columns else None

    for col in FINANCIAL_MONEY_COLUMNS:
        if col in out.columns:
            out[col] = [
                normalize_financial_money(
                    value,
                    unit=out.iloc[i][unit_col] if unit_col else None,
                    unit_multiplier=out.iloc[i][multiplier_col] if multiplier_col else None,
                )
                for i, value in enumerate(out[col])
            ]

    for col in MONEY_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(safe_numeric)

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
    if not isinstance(result, dict):
        return result
    for key in ["company_info", "company_summary", "company_table", "asset_summary"]:
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            result[key] = normalize_company_dataframe(value)
    return result
