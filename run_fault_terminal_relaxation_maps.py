from __future__ import annotations

import numpy as np

from fault_reachability_core import (
    RESULTS_DIR,
    S0,
    TERMINAL_L0,
    TERMINAL_L1,
    TERMINAL_L2,
    TERMINAL_L3,
    ReachabilityConfig,
    save_scan_npz,
    scan_terminal_relaxation_grid,
)
from fault_reachability_plots import plot_terminal_relaxation


# =============================================================================
# Direct-run settings
# =============================================================================
# This script studies fixed nominal timing under progressively relaxed terminal
# constraints. It corresponds to the "reachable-domain collapse" part.
RUN_SCAN = True
SHOW_FIGURES = True

RESULT_FILE = RESULTS_DIR / "fault_terminal_relaxation_reachability.npz"

# Use a broad grid for the global figure. Add a second local-zoom run by editing
# these values, for example te=190..199.9 and kappa=0.001..0.02.
TE_VALUES = np.array([50, 70, 90, 110, 130, 150, 170, 185, 190, 195, 198, 199], dtype=float)
KAPPA_VALUES = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20], dtype=float)

LEVELS = [TERMINAL_L0, TERMINAL_L1, TERMINAL_L2, TERMINAL_L3]

# Fixed timing by default.
STRATEGY = S0

# Keep None to use EarthEnv.m_gan, or set a stricter terminal mass lower bound.
FINAL_MASS_MIN = None

W_SMOOTHNESS = 0.0


def build_config() -> ReachabilityConfig:
    return ReachabilityConfig(
        result_file=RESULT_FILE,
        final_mass_min=FINAL_MASS_MIN,
        w_smoothness=W_SMOOTHNESS,
    )


def main() -> None:
    cfg = build_config()
    if RUN_SCAN:
        scan = scan_terminal_relaxation_grid(TE_VALUES, KAPPA_VALUES, LEVELS, cfg, strategy_name=STRATEGY)
        result_path = save_scan_npz(RESULT_FILE, scan)
    else:
        result_path = RESULT_FILE

    plot_terminal_relaxation(result_path, show=SHOW_FIGURES)


if __name__ == "__main__":
    main()
