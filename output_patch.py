import numpy as np
import pandas as pd
import unicodedata


def _norm(text):
    text = str(text)
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    ).lower().replace("_", " ").replace("-", " ")


def _safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
        return float(value)
    except Exception:
        return np.nan


def _income_value(income, keywords):
    if income is None or income.empty:
        return np.nan

    item_col = next(
        (
            c for c in income.columns
            if str(c).strip().lower()
            in {"item", "item_name", "name", "indicator"}
        ),
        None
    )

    if item_col is None:
        return np.nan

    years = sorted(
        [
            c for c in income.columns
            if str(c).isdigit() and len(str(c)) == 4
        ],
        key=lambda x: int(str(x)),
        reverse=True
    )

    if not years:
        return np.nan

    items = (
        income[item_col]
        .astype(str)
        .map(_norm)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    for keyword in keywords:
        key = _norm(keyword)
        mask = items.str.contains(
            key,
            regex=False,
            na=False
        )
        matches = income.loc[mask]

        if matches.empty:
            continue

        row = matches.iloc[0]

        for year in years:
            value = _safe_float(row[year])
            if pd.notna(value):
                return value

    return np.nan


def install_output_patch(engine):
    original_income = engine.get_income_summary
    original_run = engine.run_research

    def patched_income_summary(tickers):
        base = original_income(tickers)
        base = (
            base.copy()
            if isinstance(base, pd.DataFrame)
            else pd.DataFrame()
        )

        if base.empty:
            base = pd.DataFrame({"Mã": tickers})

        try:
            from vnstock import Fundamental
            fun = Fundamental()
        except Exception:
            return base

        for ticker in tickers:
            if "Mã" not in base.columns:
                continue

            row_mask = (
                base["Mã"].astype(str).str.upper()
                == str(ticker).upper()
            )

            if not row_mask.any():
                continue

            try:
                income = pd.DataFrame()
                path = engine.cache_path(
                    "income",
                    ticker
                )

                if path.exists():
                    income = engine.normalize_columns(
                        pd.read_csv(path)
                    )

                if income.empty:
                    income = engine.normalize_columns(
                        fun.equity(ticker).income_statement(
                            period="year",
                            orient="report"
                        )
                    )

                    if not income.empty:
                        income.to_csv(
                            path,
                            index=False
                        )
                        engine.pause_api()

                if income.empty:
                    continue

                revenue = _income_value(
                    income,
                    [
                        "doanh thu thuan",
                        "net revenue",
                        "revenue from sales and services",
                        "doanh thu ban hang va cung cap dich vu",
                        "tong thu nhap hoat dong",
                        "total operating income",
                        "operating income"
                    ]
                )

                profit = _income_value(
                    income,
                    [
                        "loi nhuan sau thue cua co dong cong ty me",
                        "loi nhuan sau thue",
                        "profit after tax",
                        "net profit",
                        "profit attributable to parent"
                    ]
                )

                idx = base.index[row_mask][0]

                if (
                    pd.isna(base.at[idx, "Doanh thu gần nhất"])
                    and pd.notna(revenue)
                ):
                    base.at[idx, "Doanh thu gần nhất"] = revenue

                if (
                    pd.isna(base.at[idx, "LNST gần nhất"])
                    and pd.notna(profit)
                ):
                    base.at[idx, "LNST gần nhất"] = profit

            except Exception:
                continue

        return base

    def patched_run_research(*args, **kwargs):
        engine.get_income_summary = patched_income_summary
        result = original_run(*args, **kwargs)

        company = result.get("company_table")

        if isinstance(company, pd.DataFrame) and not company.empty:
            for col in [
                "Doanh thu gần nhất",
                "LNST gần nhất"
            ]:
                if col in company.columns:
                    company[col] = (
                        pd.to_numeric(
                            company[col],
                            errors="coerce"
                        ) * 1000
                    )

            result["company_table"] = company

        for table_key in [
            "target_check",
            "conclusion"
        ]:
            table = result.get(table_key)

            if isinstance(table, pd.DataFrame) and not table.empty:
                rename = {}

                if "Đạt mục tiêu" in table.columns:
                    rename["Đạt mục tiêu"] = "Đạt"

                if "Trạng thái mục tiêu" in table.columns:
                    rename["Trạng thái mục tiêu"] = "Trạng thái"

                if rename:
                    result[table_key] = table.rename(
                        columns=rename
                    )

        return result

    engine.get_income_summary = patched_income_summary
    engine.run_research = patched_run_research
