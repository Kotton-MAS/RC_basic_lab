"""案A の変異版 (遅延 __init__)。"""

from __future__ import annotations

import importlib
from types import ModuleType

_SOURCE = {
    "plot_comparison": "figures", "plot_state_space": "figures",
    "plot_ipc_conservation": "figures_capacity", "plot_ipc_profile": "figures_capacity",
    "plot_mc_sweep": "figures_capacity", "plot_memory_nonlinearity": "figures_capacity",
    "plot_narma10_control": "figures_capacity",
    "plot_esp_decay": "figures_esp", "plot_esp_map": "figures_esp",
    "plot_leak_timescale": "figures_esp", "plot_washout_sensitivity": "figures_esp",
    "label": "labels",
    "StyleContext": "style", "find_cjk_font": "style", "rc_params_for": "style",
    "setup_style": "style",
}


def __getattr__(name: str) -> object:
    if name in _SOURCE:
        module: ModuleType = importlib.import_module(
            f"rc_basics_lab.plotting.{_SOURCE[name]}"
        )
        return getattr(module, name)
    raise AttributeError(name)


__all__ = sorted(_SOURCE)
