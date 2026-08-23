from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import matplotlib.pyplot as plt

from core.env_models import EarthEnv
from fault_reachability_core import (
    RESULTS_DIR,
    S0,
    S1,
    S2,
    S3,
    ReachabilityConfig,
    load_scan_npz,
    nominal_t4_duration,
    resolve_input_path,
    solve_reachability_case,
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
# RUN_SCAN=True: solve requested grid points, save one npz per strategy, then plot.
# RUN_SCAN=False: skip solving and redraw figures from existing per-strategy npz.
RUN_SCAN = False
SHOW_FIGURES = True
USE_MULTIPROCESSING = True
N_WORKERS = 8

# None means auto-name by strategy, for example:
#   S0 -> results/fault_strategy_S0.npz
#   S2 -> results/fault_strategy_S2.npz
# If you set a concrete Path and run multiple strategies, the strategy name is
# appended to the stem so files still remain separate.
RESULT_FILE = None

# Resume mode:
# - Existing scanned points in each strategy npz are skipped.
# - New points are saved immediately after each grid point finishes.
# - Set FORCE_RERUN_EXISTING=True when you changed constraints or want to
#   overwrite old values for the same strategy/te/kappa.
FORCE_RERUN_EXISTING = False
SAVE_AFTER_EACH_POINT = True
# True means "a point exists, do not run it again", even if it was infeasible.
# Set False if you want failed/infeasible points to be retried on the next run.
SKIP_FAILED_EXISTING = True

# Boundary prefilter:
# If kappa is above the strategy boundary kappa_max(te) plus this margin, the
# point is known to be outside the reachable domain and will not be optimized.
ENABLE_BOUNDARY_PREFILTER = True
BOUNDARY_MARGIN = 0.04
BOUNDARY_FILE_BY_STRATEGY = {
    S0: RESULTS_DIR / "fault_boundary_S0.npz",
    S1: RESULTS_DIR / "fault_boundary_S1.npz",
    S2: RESULTS_DIR / "fault_boundary_S2.npz",
    S3: RESULTS_DIR / "fault_boundary_S3.npz",
}

# Start with a modest grid. For paper-quality figures, densify these lists.
TE_VALUES = np.array([168], dtype=float)
KAPPA_VALUES = np.array([0.86], dtype=float)

""" TE_VALUES = np.r_[np.arange(16.0, 200.0, 2.0), 199.99]
KAPPA_VALUES = np.r_[np.arange(0.0, 1.0, 0.02), 0.999] """

# S0: fixed stage-1 + fixed T4
# S1: stage-1 cutoff adjustable from failure time + fixed T4
# S2: fixed stage-1 + adjustable T4
# S3: stage-1 cutoff adjustable from failure time + adjustable T4
# You may run one or more strategies; each strategy is saved to its own npz.
STRATEGIES = [S1,S2,S3]

# Match fault_opt_pro / boundary-map settings where useful.
DT = 1.0
ENABLE_DPHI_LIMIT = True
ENABLE_SMOOTHNESS = True
ENABLE_ALPHA_LIMIT = True
ALPHA_MIN_DEG = -60.0
ALPHA_MAX_DEG = 30.0
ENABLE_QALPHA_LIMIT = True
QALPHA_LIMIT = 5000.0  # Pa rad = N rad / m^2

# Keep None to use EarthEnv.m_gan. If you want a stricter second-stage dry-mass
# lower bound, set for example FINAL_MASS_MIN = 100000.0.
FINAL_MASS_MIN = None

# T4 variable bounds for S2/S3.
T4_GUESS = 239.0
T4_MIN = 180.0
T4_MAX = 340.0
T4_FIXED = None  # None means use nominal T4 duration from biaozhundandao.npz.

# Smoothness improves some hard cases but also changes the tie-break preference.
W_SMOOTHNESS = 1000.0
W_STAGE1_TIME = 0.0
STAGE1_FREE_MIN_AFTER_FAULT = 1e-3


GRID_FIELDS = [
    "scanned",
    "reachable",
    "delta_T1",
    "eta1",
    "delta_T4",
    "T1_sep",
    "T1_dep",
    "T4_duration",
    "leftover_stage1_propellant",
    "final_mass",
    "messages",
]


def build_config() -> ReachabilityConfig:
    return ReachabilityConfig(
        dt=DT,
        T4_guess=T4_GUESS,
        T4_min=T4_MIN,
        T4_max=T4_MAX,
        T4_fixed=T4_FIXED,
        final_mass_min=FINAL_MASS_MIN,
        stage1_free_min_after_fault=STAGE1_FREE_MIN_AFTER_FAULT,
        enable_dphi_limit=ENABLE_DPHI_LIMIT,
        enable_smoothness=ENABLE_SMOOTHNESS,
        w_smoothness=W_SMOOTHNESS,
        w_stage1_time=W_STAGE1_TIME,
        enable_alpha_limit=ENABLE_ALPHA_LIMIT,
        alpha_min_deg=ALPHA_MIN_DEG,
        alpha_max_deg=ALPHA_MAX_DEG,
        enable_qalpha_limit=ENABLE_QALPHA_LIMIT,
        qalpha_limit=QALPHA_LIMIT,
    )


def strategy_result_path(strategy: str) -> Path:
    if RESULT_FILE is None:
        return RESULTS_DIR / f"fault_strategy_{strategy}.npz"

    path = Path(RESULT_FILE)
    if not path.is_absolute():
        path = RESULTS_DIR / path.name
    if len(STRATEGIES) <= 1:
        return path
    return path.with_name(f"{path.stem}_{strategy}{path.suffix}")


def output_stem() -> str:
    if RESULT_FILE is None:
        joined = "_".join(str(item) for item in STRATEGIES)
        return f"fault_strategy_{joined}"
    return Path(RESULT_FILE).stem


def _sorted_union(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return np.array(sorted(set(a.tolist()) | set(b.tolist())), dtype=float)


def _grid_indices(scan: dict) -> tuple[dict[float, int], dict[float, int]]:
    kappa_to_idx = {round(float(v), 12): idx for idx, v in enumerate(np.asarray(scan["kappa"], dtype=float))}
    te_to_idx = {round(float(v), 12): idx for idx, v in enumerate(np.asarray(scan["te"], dtype=float))}
    return kappa_to_idx, te_to_idx


def load_boundary_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curves = {}
    if not ENABLE_BOUNDARY_PREFILTER:
        return curves

    for strategy, path in BOUNDARY_FILE_BY_STRATEGY.items():
        path = Path(path)
        if not path.exists():
            print(f"Boundary prefilter disabled for {strategy}: missing {path}")
            continue
        with np.load(path, allow_pickle=True) as data:
            te = np.asarray(data["te"], dtype=float).reshape(-1)
            kappa_max = np.asarray(data["kappa_max"], dtype=float).reshape(-1)
            mask = np.isfinite(te) & np.isfinite(kappa_max)
            if "scanned" in data:
                mask &= np.asarray(data["scanned"]).reshape(-1).astype(bool)
            if "reachable" in data:
                mask &= np.asarray(data["reachable"]).reshape(-1).astype(bool)
        te = te[mask]
        kappa_max = kappa_max[mask]
        if len(te) < 2:
            print(f"Boundary prefilter disabled for {strategy}: not enough boundary points")
            continue
        order = np.argsort(te)
        te = te[order]
        kappa_max = kappa_max[order]

        # Keep the last value for duplicate te entries after sorting.
        unique_te = []
        unique_kappa = []
        for value_te, value_kappa in zip(te, kappa_max):
            if unique_te and np.isclose(value_te, unique_te[-1], rtol=0.0, atol=1e-9):
                unique_kappa[-1] = value_kappa
            else:
                unique_te.append(value_te)
                unique_kappa.append(value_kappa)
        curves[strategy] = (np.asarray(unique_te, dtype=float), np.asarray(unique_kappa, dtype=float))
        print(
            f"Boundary prefilter loaded {strategy}: "
            f"te=[{curves[strategy][0][0]:.3f}, {curves[strategy][0][-1]:.3f}], "
            f"{len(curves[strategy][0])} points"
        )
    return curves


def boundary_kappa_at(curves: dict[str, tuple[np.ndarray, np.ndarray]], strategy: str, te: float) -> float | None:
    curve = curves.get(strategy)
    if curve is None:
        return None
    te_grid, kappa_grid = curve
    if te < te_grid[0] or te > te_grid[-1]:
        return None
    return float(np.interp(float(te), te_grid, kappa_grid))


def is_outside_boundary(curves: dict[str, tuple[np.ndarray, np.ndarray]], strategy: str, te: float, kappa: float) -> tuple[bool, float | None]:
    kappa_max = boundary_kappa_at(curves, strategy, te)
    if kappa_max is None:
        return False, None
    return float(kappa) > kappa_max + BOUNDARY_MARGIN, kappa_max


def _blank_strategy_scan(strategy: str, te_values, kappa_values, cfg: ReachabilityConfig) -> dict:
    te_values = np.asarray(te_values, dtype=float)
    kappa_values = np.asarray(kappa_values, dtype=float)
    shape = (1, len(kappa_values), len(te_values))

    data = np.load(resolve_input_path(cfg.init_guess_file))
    env = EarthEnv(target=cfg.target)
    t4_nominal = nominal_t4_duration(data, env)
    data.close()

    return {
        "te": te_values,
        "kappa": kappa_values,
        "strategies": np.asarray([strategy]),
        "scanned": np.zeros(shape, dtype=bool),
        "reachable": np.zeros(shape, dtype=bool),
        "delta_T1": np.full(shape, np.nan),
        "eta1": np.full(shape, np.nan),
        "delta_T4": np.full(shape, np.nan),
        "T1_sep": np.full(shape, np.nan),
        "T1_dep": np.full(shape, np.nan),
        "T4_duration": np.full(shape, np.nan),
        "leftover_stage1_propellant": np.full(shape, np.nan),
        "final_mass": np.full(shape, np.nan),
        "messages": np.full(shape, "", dtype=object),
        "terminal_active": np.asarray(cfg.terminal_active),
        "T4_nominal": np.array(t4_nominal),
        "final_mass_min": np.array(env.m_gan if cfg.final_mass_min is None else cfg.final_mass_min),
    }


def _copy_strategy_data(dst: dict, src: dict, strategy: str) -> None:
    src_strategies = [str(item) for item in np.asarray(src["strategies"]).ravel()]
    if strategy not in src_strategies:
        return

    src_s_idx = src_strategies.index(strategy)
    dst_kappa_to_idx, dst_te_to_idx = _grid_indices(dst)
    src_kappa = np.asarray(src["kappa"], dtype=float)
    src_te = np.asarray(src["te"], dtype=float)

    for old_k_idx, kappa in enumerate(src_kappa):
        new_k_idx = dst_kappa_to_idx.get(round(float(kappa), 12))
        if new_k_idx is None:
            continue
        for old_t_idx, te in enumerate(src_te):
            new_t_idx = dst_te_to_idx.get(round(float(te), 12))
            if new_t_idx is None:
                continue
            for field in GRID_FIELDS:
                if field == "scanned" and field not in src:
                    messages = src.get("messages")
                    if messages is None:
                        dst[field][0, new_k_idx, new_t_idx] = True
                    else:
                        dst[field][0, new_k_idx, new_t_idx] = str(messages[src_s_idx, old_k_idx, old_t_idx]) != ""
                elif field in src:
                    dst[field][0, new_k_idx, new_t_idx] = src[field][src_s_idx, old_k_idx, old_t_idx]


def load_or_create_strategy_scan(strategy: str, te_values, kappa_values, cfg: ReachabilityConfig) -> dict:
    path = strategy_result_path(strategy)
    if path.exists():
        old = load_scan_npz(path)
        te_values = _sorted_union(old["te"], te_values)
        kappa_values = _sorted_union(old["kappa"], kappa_values)
        scan = _blank_strategy_scan(strategy, te_values, kappa_values, cfg)
        _copy_strategy_data(scan, old, strategy)
        print(f"Resume {strategy} from existing scan: {path}")
        return scan
    return _blank_strategy_scan(strategy, te_values, kappa_values, cfg)


def save_strategy_scan(strategy: str, scan: dict, quiet: bool = False) -> Path:
    path = strategy_result_path(strategy)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez(f, **scan)
    for attempt in range(6):
        try:
            tmp_path.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            sleep(0.2)
    if not quiet:
        print(f"Saved {strategy} scan: {path}")
    return path


def _store_case(scan: dict, k_idx: int, t_idx: int, res) -> None:
    scan["scanned"][0, k_idx, t_idx] = True
    scan["reachable"][0, k_idx, t_idx] = res.reachable
    scan["delta_T1"][0, k_idx, t_idx] = res.delta_T1
    scan["eta1"][0, k_idx, t_idx] = res.eta1
    scan["delta_T4"][0, k_idx, t_idx] = res.delta_T4
    scan["T1_sep"][0, k_idx, t_idx] = res.T1_sep
    scan["T1_dep"][0, k_idx, t_idx] = res.T1_dep
    scan["T4_duration"][0, k_idx, t_idx] = res.T4_duration
    scan["leftover_stage1_propellant"][0, k_idx, t_idx] = res.leftover_stage1_propellant
    scan["final_mass"][0, k_idx, t_idx] = res.final_mass
    scan["messages"][0, k_idx, t_idx] = res.message


def _store_outside_boundary(scan: dict, k_idx: int, t_idx: int, te: float, kappa: float, kappa_max: float) -> None:
    scan["scanned"][0, k_idx, t_idx] = True
    scan["reachable"][0, k_idx, t_idx] = False
    scan["delta_T1"][0, k_idx, t_idx] = np.nan
    scan["eta1"][0, k_idx, t_idx] = np.nan
    scan["delta_T4"][0, k_idx, t_idx] = np.nan
    scan["T1_sep"][0, k_idx, t_idx] = np.nan
    scan["T1_dep"][0, k_idx, t_idx] = np.nan
    scan["T4_duration"][0, k_idx, t_idx] = np.nan
    scan["leftover_stage1_propellant"][0, k_idx, t_idx] = np.nan
    scan["final_mass"][0, k_idx, t_idx] = np.nan
    scan["messages"][0, k_idx, t_idx] = (
        f"outside boundary: kappa={kappa:.6g} > "
        f"kappa_max(te={te:.6g})+margin={kappa_max:.6g}+{BOUNDARY_MARGIN:.6g}"
    )


def should_skip_point(scan: dict, k_idx: int, t_idx: int) -> bool:
    if FORCE_RERUN_EXISTING:
        return False
    if not bool(scan["scanned"][0, k_idx, t_idx]):
        return False
    if SKIP_FAILED_EXISTING:
        return True
    return bool(scan["reachable"][0, k_idx, t_idx])


def solve_strategy_grid_job(job):
    strategy, k_idx, t_idx, te, kappa, cfg = job
    start = perf_counter()
    res = solve_reachability_case(te, kappa, strategy, cfg)
    return strategy, k_idx, t_idx, te, kappa, res, perf_counter() - start


def scan_strategy_maps(cfg: ReachabilityConfig) -> dict:
    strategy_scans = {
        strategy: load_or_create_strategy_scan(strategy, TE_VALUES, KAPPA_VALUES, cfg)
        for strategy in STRATEGIES
    }
    boundary_curves = load_boundary_curves()

    jobs = []
    outside_count = 0
    for strategy, scan in strategy_scans.items():
        for k_idx, kappa in enumerate(scan["kappa"]):
            for t_idx, te in enumerate(scan["te"]):
                requested_te = np.any(np.isclose(TE_VALUES, te, rtol=0.0, atol=1e-9))
                requested_kappa = np.any(np.isclose(KAPPA_VALUES, kappa, rtol=0.0, atol=1e-9))
                if not (requested_te and requested_kappa):
                    continue
                if should_skip_point(scan, k_idx, t_idx):
                    print(f"SKIP existing {strategy}: te={te:.3f}, kappa={kappa:.5f}")
                    continue
                outside, kappa_max = is_outside_boundary(boundary_curves, strategy, float(te), float(kappa))
                if outside:
                    _store_outside_boundary(scan, k_idx, t_idx, float(te), float(kappa), float(kappa_max))
                    outside_count += 1
                    print(
                        f"SKIP outside boundary {strategy}: te={te:.3f}, "
                        f"kappa={kappa:.5f}, kmax={kappa_max:.5f}, margin={BOUNDARY_MARGIN:.5f}"
                    )
                    if SAVE_AFTER_EACH_POINT:
                        save_strategy_scan(strategy, scan, quiet=True)
                    continue
                jobs.append((strategy, k_idx, t_idx, float(te), float(kappa)))

    total = len(jobs)
    print(
        f"Running {total} strategy-grid job(s). "
        f"Prefiltered {outside_count} outside-boundary point(s). "
        f"Existing points stay in their own strategy npz files."
    )
    if USE_MULTIPROCESSING and N_WORKERS > 1 and total > 1:
        print(f"Using {N_WORKERS} worker processes.", flush=True)
        worker_jobs = [(*job, cfg) for job in jobs]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(solve_strategy_grid_job, job) for job in worker_jobs]
            for count, future in enumerate(as_completed(futures), start=1):
                strategy, k_idx, t_idx, te, kappa, res, elapsed = future.result()
                _store_case(strategy_scans[strategy], k_idx, t_idx, res)
                status = "reachable" if res.reachable else f"failed: {res.message[:100]}"
                print(
                    f"[{count}/{total}] DONE {strategy}: te={te:.3f}, "
                    f"kappa={kappa:.5f}, {status}, time={elapsed:.1f}s",
                    flush=True,
                )
                if SAVE_AFTER_EACH_POINT:
                    save_strategy_scan(strategy, strategy_scans[strategy], quiet=True)
    else:
        env = EarthEnv(target=cfg.target)
        data = np.load(resolve_input_path(cfg.init_guess_file))
        for count, (strategy, k_idx, t_idx, te, kappa) in enumerate(jobs, start=1):
            print(f"[{count}/{total}] {strategy}: te={te:.3f}, kappa={kappa:.5f}", flush=True)
            start = perf_counter()
            res = solve_reachability_case(te, kappa, strategy, cfg, data=data, env=env)
            _store_case(strategy_scans[strategy], k_idx, t_idx, res)
            elapsed = perf_counter() - start
            status = "reachable" if res.reachable else f"failed: {res.message[:100]}"
            print(f"  {status}, time={elapsed:.1f}s", flush=True)
            if SAVE_AFTER_EACH_POINT:
                save_strategy_scan(strategy, strategy_scans[strategy], quiet=True)
        data.close()

    if not SAVE_AFTER_EACH_POINT:
        for strategy, scan in strategy_scans.items():
            save_strategy_scan(strategy, scan, quiet=True)
    for strategy, scan in strategy_scans.items():
        save_strategy_scan(strategy, scan)
    return combine_strategy_scans(strategy_scans, STRATEGIES, cfg)


def combine_strategy_scans(strategy_scans: dict[str, dict], strategies, cfg: ReachabilityConfig) -> dict:
    strategies = list(strategies)
    te_values = np.array([], dtype=float)
    kappa_values = np.array([], dtype=float)
    for scan in strategy_scans.values():
        te_values = _sorted_union(te_values, scan["te"])
        kappa_values = _sorted_union(kappa_values, scan["kappa"])

    combined = _blank_strategy_scan(strategies[0] if strategies else S0, te_values, kappa_values, cfg)
    combined["strategies"] = np.asarray(strategies)
    shape = (len(strategies), len(kappa_values), len(te_values))
    for field in GRID_FIELDS:
        if field in {"scanned", "reachable"}:
            combined[field] = np.zeros(shape, dtype=bool)
        elif field == "messages":
            combined[field] = np.full(shape, "", dtype=object)
        else:
            combined[field] = np.full(shape, np.nan)

    combined_kappa_to_idx, combined_te_to_idx = _grid_indices(combined)
    for s_idx, strategy in enumerate(strategies):
        scan = strategy_scans[strategy]
        scan_kappa = np.asarray(scan["kappa"], dtype=float)
        scan_te = np.asarray(scan["te"], dtype=float)
        for old_k_idx, kappa in enumerate(scan_kappa):
            new_k_idx = combined_kappa_to_idx[round(float(kappa), 12)]
            for old_t_idx, te in enumerate(scan_te):
                new_t_idx = combined_te_to_idx[round(float(te), 12)]
                for field in GRID_FIELDS:
                    combined[field][s_idx, new_k_idx, new_t_idx] = scan[field][0, old_k_idx, old_t_idx]
    return combined


def load_selected_strategy_scans(cfg: ReachabilityConfig) -> dict:
    strategy_scans = {}
    for strategy in STRATEGIES:
        path = strategy_result_path(strategy)
        if not path.exists():
            raise FileNotFoundError(f"Missing strategy scan npz for {strategy}: {path}")
        strategy_scans[strategy] = load_or_create_strategy_scan(strategy, TE_VALUES, KAPPA_VALUES, cfg)
    return combine_strategy_scans(strategy_scans, STRATEGIES, cfg)


def try_plot(description: str, func, scan: dict, output: str) -> None:
    try:
        func(scan, output=output, show=SHOW_FIGURES)
    except ValueError as exc:
        print(f"Skip {description}: {exc}")


def main() -> None:
    cfg = build_config()
    if RUN_SCAN:
        scan = scan_strategy_maps(cfg)
    else:
        scan = load_selected_strategy_scans(cfg)

    stem = output_stem()
    plot_strategy_reachability(scan, output=f"{stem}_reachability.png", show=SHOW_FIGURES)
    try_plot("stage-1 eta heatmap", plot_stage1_compensation, scan, f"{stem}_stage1_eta.png")
    try_plot("stage-1 delta heatmap", plot_stage1_delta, scan, f"{stem}_stage1_deltaT1.png")
    try_plot("T4 delta heatmap", plot_t4_compensation, scan, f"{stem}_t4_deltaT4.png")
    if S2 in STRATEGIES and S3 in STRATEGIES:
        try_plot("T4 saved heatmap", plot_t4_save, scan, f"{stem}_t4_save_S2_minus_S3.png")
    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
