import portfolio_engine_core as _core
from margin_patch import install_margin_patch

install_margin_patch(_core)

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)
