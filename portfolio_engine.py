import pandas as pd

import portfolio_engine_core as _core
from margin_patch import install_margin_patch
from input_normalizer import (
    normalize_company_dataframe,
    normalize_income_dataframe,
)
from advanced_portfolio import (
    factor_proxy_analysis,
    multifactor_regression,
    var_analysis,
    active_analysis,
    robustness_analysis,
    walk_forward_analysis,
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
    """Đọc doanh thu và LNST sau khi chuẩn hóa đơn vị tiền."""
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
# NÂNG CẤP ANALYTICS DÙNG DỮ LIỆU SẴN CÓ
# ------------------------------------------------------------

def build_advanced_portfolio_analysis(
    returns,
    portfolio_returns,
    benchmark_returns=None,
    company_table=None,
    risk_free_rate=0.0,
    target_return=0.15,
):
    """Chỉ dùng dữ liệu đã có trong lần chạy hiện tại.

    Không gọi thêm API, không truy vấn nguồn dữ liệu bên ngoài.
    """
    result = {}

    try:
        result["factor_proxy"] = factor_proxy_analysis(
            returns,
            company_table=company_table,
        )
    except Exception:
        result["factor_proxy"] = pd.DataFrame()

    if benchmark_returns is not None:
        try:
            factor_returns = pd.DataFrame({
                "Market": pd.Series(benchmark_returns),
            })
            result["multifactor_regression"] = multifactor_regression(
                portfolio_returns,
                factor_returns,
                rf=risk_free_rate,
            )
            result["active_analysis"] = active_analysis(
                portfolio_returns,
                benchmark_returns,
                rf=risk_free_rate,
            )
        except Exception:
            result["multifactor_regression"] = pd.DataFrame()
            result["active_analysis"] = pd.DataFrame()
    else:
        result["multifactor_regression"] = pd.DataFrame()
        result["active_analysis"] = pd.DataFrame()

    try:
        result["var_analysis"] = var_analysis(
            portfolio_returns,
            level=0.95,
        )
    except Exception:
        result["var_analysis"] = pd.DataFrame()

    try:
        result["robustness"] = robustness_analysis(
            portfolio_returns,
            target_return=target_return,
        )
    except Exception:
        result["robustness"] = pd.DataFrame()

    try:
        result["walk_forward"] = walk_forward_analysis(
            portfolio_returns,
            rf=risk_free_rate,
        )
    except Exception:
        result["walk_forward"] = pd.DataFrame()

    return result


# Gắn API mới vào module để app.py có thể sử dụng.
def run_advanced_portfolio_analysis(*args, **kwargs):
    return build_advanced_portfolio_analysis(*args, **kwargs)
