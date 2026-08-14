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
    df = normalize_company_dataframe(df)

    # Một số phiên bản nguồn trả P/E theo đơn vị phần nghìn.
    # Ví dụ 33766 thực chất là 33,766 lần.
    if isinstance(df, pd.DataFrame) and "P/E" in df.columns:
        pe = pd.to_numeric(df["P/E"], errors="coerce")
        mask = pe.abs() > 1000
        df.loc[mask, "P/E"] = pe.loc[mask] / 1000.0

    return df


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

    if pd.notna(value) and not has_unit_metadata:
        value = float(value) * 1000.0

    return value, year, item


_core.income_row_value = income_row_value

# ------------------------------------------------------------
# BẢNG DOANH NGHIỆP: GIỮ RÕ NĂM CỦA BCTC
# ------------------------------------------------------------
_original_build_company_table = _core.build_company_table


def build_company_table(tickers, prices, include_company, include_income):
    table = _original_build_company_table(
        tickers,
        prices,
        include_company,
        include_income,
    )

    if not include_income or not isinstance(table, pd.DataFrame) or table.empty:
        return table

    try:
        income = _core.get_income_summary(tickers)
        year_cols = [c for c in ["Mã", "Năm doanh thu", "Năm LNST"] if c in income.columns]

        if len(year_cols) == 3:
            years = income[year_cols].copy()
            table = table.merge(years, on="Mã", how="left")

            table["Doanh thu gần nhất"] = pd.to_numeric(
                table["Doanh thu gần nhất"], errors="coerce"
            )
            table["LNST gần nhất"] = pd.to_numeric(
                table["LNST gần nhất"], errors="coerce"
            )

            table["Doanh thu gần nhất"] = table.apply(
                lambda r: r["Doanh thu gần nhất"], axis=1
            )

            ordered = [
                "Mã", "Ngành", "Số CP lưu hành", "Vốn hóa",
                "P/E", "P/B", "EPS", "ROA", "ROE",
                "Doanh thu gần nhất", "Năm doanh thu",
                "LNST gần nhất", "Năm LNST",
            ]
            table = table[[c for c in ordered if c in table.columns]]

    except Exception:
        pass

    return table


_core.build_company_table = build_company_table

# Xuất toàn bộ API của engine gốc.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

# Giữ các hàm đã patch sau vòng export.
globals()["get_company_info"] = get_company_info
globals()["income_row_value"] = income_row_value
globals()["build_company_table"] = build_company_table
