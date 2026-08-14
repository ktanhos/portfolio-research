import pandas as pd

import portfolio_engine_core as _core
from margin_patch import install_margin_patch
from input_normalizer import (
    normalize_company_dataframe,
    normalize_income_dataframe,
)

# Patch margin an toàn trước khi xuất API của engine.
install_margin_patch(_core)

# ------------------------------------------------------------
# CHUẨN HÓA THÔNG TIN DOANH NGHIỆP
# ------------------------------------------------------------
_original_get_company_info = _core.get_company_info


def get_company_info(tickers, prices=None):
    df = _original_get_company_info(tickers, prices=prices)
    return normalize_company_dataframe(df)

_core.get_company_info = get_company_info

# ------------------------------------------------------------
# CHUẨN HÓA BCTC THEO UNIT METADATA
# ------------------------------------------------------------
_original_income_row_value = _core.income_row_value


def income_row_value(income, keywords, exclude_keywords=None):
    """
    Đọc doanh thu/LNST sau khi chuẩn hóa đơn vị tiền.

    Thứ tự ưu tiên:
    1. unit_multiplier do Vnstock cung cấp.
    2. unit do Vnstock cung cấp.
    3. Quy tắc dự phòng của phiên bản V21 đối với dữ liệu cũ
       không có metadata đơn vị.
    """
    if income is None or not isinstance(income, pd.DataFrame) or income.empty:
        return _original_income_row_value(
            income,
            keywords,
            exclude_keywords=exclude_keywords,
        )

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

    if pd.notna(value):
        # Với dữ liệu V21 cũ không có metadata, nguồn hiện tại đang
        # thấp hơn đơn vị nội bộ VND một hệ số 1.000.
        if not has_unit_metadata:
            value = float(value) * 1000.0

    return value, year, item

_core.income_row_value = income_row_value

# Xuất toàn bộ API của engine gốc.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

# Giữ các hàm đã patch sau vòng export.
globals()["get_company_info"] = get_company_info
globals()["income_row_value"] = income_row_value
