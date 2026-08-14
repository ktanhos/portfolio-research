import pandas as pd
import numpy as np
import streamlit as st


def _fmt_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def _fmt_num(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.2f}"


def build_portfolio_evaluation(advanced_results, target_return=0.15):
    """Tao danh gia tu cac ket qua da tinh, khong goi du lieu moi."""
    robustness = advanced_results.get("robustness", pd.DataFrame())
    active = advanced_results.get("active_analysis", pd.DataFrame())
    var = advanced_results.get("var_analysis", pd.DataFrame())
    walk = advanced_results.get("walk_forward", pd.DataFrame())
    factor = advanced_results.get("factor_proxy", pd.DataFrame())

    metrics = {}
    if isinstance(robustness, pd.DataFrame) and not robustness.empty:
        latest = robustness.iloc[-1]
        metrics["CAGR"] = latest.get("CAGR", np.nan)
        metrics["Sharpe"] = latest.get("Sharpe", np.nan)
        metrics["Max Drawdown"] = latest.get("Max Drawdown", np.nan)

    if isinstance(active, pd.DataFrame) and not active.empty:
        row = active.iloc[0]
        metrics["Active Return"] = row.get("Active Return", np.nan)
        metrics["Tracking Error"] = row.get("Tracking Error", np.nan)
        metrics["Information Ratio"] = row.get("Information Ratio", np.nan)
        metrics["Beta"] = row.get("Beta", np.nan)

    if isinstance(var, pd.DataFrame) and not var.empty:
        row = var.iloc[0]
        metrics["Historical VaR"] = row.get("Historical VaR", np.nan)
        metrics["CVaR"] = row.get("CVaR", np.nan)

    test_sharpes = []
    if isinstance(walk, pd.DataFrame) and not walk.empty and "Test Sharpe" in walk.columns:
        test_sharpes = pd.to_numeric(walk["Test Sharpe"], errors="coerce").dropna().tolist()
        if test_sharpes:
            metrics["OOS Sharpe trung vi"] = float(np.median(test_sharpes))
            metrics["OOS Sharpe thap nhat"] = float(np.min(test_sharpes))

    rows = []
    cagr = metrics.get("CAGR", np.nan)
    sharpe = metrics.get("Sharpe", np.nan)
    max_dd = metrics.get("Max Drawdown", np.nan)
    ir = metrics.get("Information Ratio", np.nan)
    oos_sharpe = metrics.get("OOS Sharpe trung vi", np.nan)

    if pd.notna(cagr):
        target_status = "Dat" if cagr >= target_return else "Khong dat"
        target_comment = f"CAGR {_fmt_pct(cagr)} so voi muc tieu {_fmt_pct(target_return)}"
    else:
        target_status = "Khong du du lieu"
        target_comment = "Khong xac dinh duoc CAGR"
    rows.append({"Tieu chi": "Muc tieu loi suat", "Ket qua": target_status, "Danh gia": target_comment})

    if pd.notna(sharpe):
        if sharpe >= 1.0:
            risk_status = "Tot"
        elif sharpe >= 0.5:
            risk_status = "Kha"
        elif sharpe >= 0:
            risk_status = "Yeu"
        else:
            risk_status = "Khong dat"
        risk_comment = f"Sharpe {_fmt_num(sharpe)}"
    else:
        risk_status = "Khong du du lieu"
        risk_comment = "Khong xac dinh duoc Sharpe"
    rows.append({"Tieu chi": "Hieu qua dieu chinh rui ro", "Ket qua": risk_status, "Danh gia": risk_comment})

    if pd.notna(max_dd):
        dd_abs = abs(max_dd)
        if dd_abs <= 0.15:
            dd_status = "Tot"
        elif dd_abs <= 0.25:
            dd_status = "Chap nhan duoc"
        elif dd_abs <= 0.35:
            dd_status = "Can luu y"
        else:
            dd_status = "Rui ro cao"
        dd_comment = f"Maximum Drawdown {_fmt_pct(max_dd)}"
    else:
        dd_status = "Khong du du lieu"
        dd_comment = "Khong xac dinh duoc Maximum Drawdown"
    rows.append({"Tieu chi": "Rui ro giam gia", "Ket qua": dd_status, "Danh gia": dd_comment})

    if pd.notna(ir):
        if ir >= 0.5:
            active_status = "Tich cuc"
        elif ir >= 0:
            active_status = "Trung tinh"
        else:
            active_status = "Kem"
        active_comment = f"Information Ratio {_fmt_num(ir)}"
    else:
        active_status = "Khong du du lieu"
        active_comment = "Khong xac dinh duoc Information Ratio"
    rows.append({"Tieu chi": "So voi VNINDEX", "Ket qua": active_status, "Danh gia": active_comment})

    if pd.notna(oos_sharpe):
        if oos_sharpe >= 0.5:
            stability_status = "On dinh"
        elif oos_sharpe >= 0:
            stability_status = "Trung binh"
        else:
            stability_status = "Khong on dinh"
        stability_comment = f"OOS Sharpe trung vi {_fmt_num(oos_sharpe)}"
    else:
        stability_status = "Khong du du lieu"
        stability_comment = "Chua du mau de danh gia ngoai mau"
    rows.append({"Tieu chi": "Do on dinh ngoai mau", "Ket qua": stability_status, "Danh gia": stability_comment})

    warnings = []
    if pd.notna(cagr) and cagr < target_return:
        warnings.append("Loi suat thuc te chua dat muc tieu.")
    if pd.notna(max_dd) and max_dd <= -0.35:
        warnings.append("Maximum Drawdown o muc cao, can dac biet luu y kha nang chiu lo.")
    if pd.notna(ir) and ir < 0:
        warnings.append("Danh muc dang kem hieu qua so voi VNINDEX sau khi dieu chinh theo active risk.")
    if pd.notna(oos_sharpe) and oos_sharpe < 0:
        warnings.append("Hieu qua ngoai mau co dau hieu khong on dinh.")
    if pd.notna(metrics.get("CVaR", np.nan)) and pd.notna(metrics.get("Historical VaR", np.nan)):
        if abs(metrics["CVaR"]) > abs(metrics["Historical VaR"]) * 1.5:
            warnings.append("Rui ro duoi phan phoi lon hon dang ke so voi VaR lich su.")

    positive = sum(x in {"Dat", "Tot", "Kha", "Chap nhan duoc", "Tich cuc", "On dinh"} for x in [target_status, risk_status, dd_status, active_status, stability_status])
    negative = sum(x in {"Khong dat", "Rui ro cao", "Kem", "Khong on dinh"} for x in [target_status, risk_status, dd_status, active_status, stability_status])
    if negative >= 2:
        overall = "Can xem xet"
    elif positive >= 4:
        overall = "Tich cuc"
    elif positive >= 2:
        overall = "Trung tinh"
    else:
        overall = "Chua du co so"

    conclusion_parts = []
    if pd.notna(cagr):
        conclusion_parts.append(f"CAGR {_fmt_pct(cagr)}")
    if pd.notna(target_return):
        conclusion_parts.append(f"muc tieu {_fmt_pct(target_return)}")
    if pd.notna(sharpe):
        conclusion_parts.append(f"Sharpe {_fmt_num(sharpe)}")
    if pd.notna(max_dd):
        conclusion_parts.append(f"Maximum Drawdown {_fmt_pct(max_dd)}")
    if pd.notna(ir):
        conclusion_parts.append(f"Information Ratio {_fmt_num(ir)}")

    return {
        "summary": pd.DataFrame(rows),
        "overall": overall,
        "conclusion": ". ".join(conclusion_parts) + "." if conclusion_parts else "Chua du du lieu de ket luan.",
        "warnings": warnings,
        "factor_count": len(factor) if isinstance(factor, pd.DataFrame) else 0,
    }


def render_advanced_section(advanced_results, title=None, target_return=0.15):
    """Hien thi phan tich nang cao tu dung ket qua run_research."""
    if not isinstance(advanced_results, dict) or not advanced_results:
        st.info("Khong du du lieu hien co de thuc hien phan tich nang cao.")
        return

    portfolio_name = advanced_results.get("_portfolio_name", "Complete Portfolio")
    target_return = advanced_results.get("_target_return", target_return)

    st.markdown(f"**Danh muc duoc danh gia: {portfolio_name}**")
    st.caption("Day la danh muc duoc lua chon de danh gia, khong phai danh muc mac dinh la tot nhat.")

    sections = [
        ("11.1. PHAN TICH YEU TO", "factor_proxy"),
        ("11.2. HOI QUY DA YEU TO", "multifactor_regression"),
        ("11.3. PHAN TICH DANH MUC CHU DONG", "active_analysis"),
        ("11.4. VAR VA CVAR", "var_analysis"),
        ("11.5. KIEM DINH DO BEN", "robustness"),
        ("11.6. KIEM DINH NGOAI MAU", "walk_forward"),
    ]

    for heading, key in sections:
        st.markdown(f"**{heading}**")
        table = advanced_results.get(key, pd.DataFrame())
        if isinstance(table, pd.DataFrame) and not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=False)
        else:
            error_key = f"{key}_error"
            error = advanced_results.get(error_key)
            if error:
                st.warning(f"Khong the thuc hien: {error}")
            else:
                st.info("Khong du du lieu hien co de thuc hien phan tich nay.")

    evaluation = build_portfolio_evaluation(advanced_results, target_return=float(target_return))

    st.markdown("---")
    st.markdown("**11.7. DANH GIA DANH MUC DUOC LUA CHON**")
    st.markdown(f"### Danh gia tong the: {evaluation['overall']}")
    st.write(evaluation["conclusion"])

    if not evaluation["summary"].empty:
        st.dataframe(evaluation["summary"], use_container_width=True, hide_index=True)

    st.markdown("**Canh bao chinh**")
    if evaluation["warnings"]:
        for warning in evaluation["warnings"]:
            st.warning(warning)
    else:
        st.success("Khong phat hien canh bao noi bat tu cac chi tieu hien co.")
