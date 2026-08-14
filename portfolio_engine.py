import portfolio_engine_core as _core
from margin_patch import install_margin_patch
from input_normalizer import normalize_company_dataframe

# Patch margin an toàn trước khi xuất API của engine.
install_margin_patch(_core)

# Chuẩn hóa dữ liệu doanh nghiệp ngay sau khi lấy từ nguồn,
# trước khi phần còn lại của engine sử dụng kết quả.
_original_get_company_info = _core.get_company_info


def get_company_info(tickers, prices=None):
    df = _original_get_company_info(tickers, prices=prices)
    return normalize_company_dataframe(df)

_core.get_company_info = get_company_info

# Xuất toàn bộ API của engine gốc.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

# Giữ hàm đã chuẩn hóa sau vòng export ở trên.
globals()["get_company_info"] = get_company_info
