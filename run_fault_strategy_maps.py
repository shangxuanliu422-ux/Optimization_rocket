from __future__ import annotations

import numpy as np

from fault_reachability_core import (
    RESULTS_DIR,
    S0,
    S1,
    S2,
    S3,
    ReachabilityConfig,
    save_scan_npz,
    scan_strategy_grid,
)
from fault_reachability_plots import (
    plot_stage1_compensation,
    plot_stage1_delta,
    plot_strategy_reachability,
    plot_t4_compensation,
    plot_t4_save,
)


# =============================================================================
# Direct-run settings
# =============================================================================
# RUN_SCAN=True: solve all grid cases, save npz, then plot.
# RUN_SCAN=False: skip solving and redraw figures from RESULT_FILE.
RUN_SCAN = True
SHOW_FIGURES = True

RESULT_FILE = RESULTS_DIR / "fault_strategy_reachability.npz"

# Start with a modest grid. For paper-quality figures, densify these lists.
TE_VALUES = np.array([50, 70, 90, 110, 130, 150, 170, 185, 195], dtype=float)
KAPPA_VALUES = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=float)

# S0: fixed stage-1 + fixed T4
# S1: stage-1 cutoff adjustable from failure time + fixed T4
# S2: fixed stage-1 + adjustable T4
# S3: stage-1 cutoff adjustable from failure time + adjustable T4
STRATEGIES = [S0, S1, S2, S3]

# Keep None to use EarthEnv.m_gan. If you want a stricter second-stage dry-mass
# lower bound, set for example FINAL_MASS_MIN = 100000.0.
FINAL_MASS_MIN = None

# T4 variable bounds for S2/S3.
T4_GUESS = 239.0
T4_MIN = 180.0
T4_MAX = 450.0
T4_FIXED = None  # None means use nominal T4 duration from biaozhundandao.npz.

# Hard-boundary scans usually should keep smoothness at zero.
W_SMOOTHNESS = 0.0


def build_config() -> ReachabilityConfig:
    return ReachabilityConfig(
        result_file=RESULT_FILE,
        T4_guess=T4_GUESS,
        T4_min=T4_MIN,
        T4_max=T4_MAX,
        T4_fixed=T4_FIXED,
        final_mass_min=FINAL_MASS_MIN,
        w_smoothness=W_SMOOTHNESS,
    )


def main() -> None:
    cfg = build_config()
    if RUN_SCAN:
        scan = scan_strategy_grid(TE_VALUES, KAPPA_VALUES, STRATEGIES, cfg)
        result_path = save_scan_npz(RESULT_FILE, scan)
    else:
        result_path = RESULT_FILE

    plot_strategy_reachability(result_path, show=SHOW_FIGURES)
    plot_stage1_compensation(result_path, show=SHOW_FIGURES)
    plot_stage1_delta(result_path, show=SHOW_FIGURES)
    plot_t4_compensation(result_path, show=SHOW_FIGURES)
    plot_t4_save(result_path, show=SHOW_FIGURES)


if __name__ == "__main__":
    main()
