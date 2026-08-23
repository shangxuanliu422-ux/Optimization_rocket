from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from core.env_models import EarthEnv
from fault_reachability_core import (
    FIGURES_DIR,
    INIT_GUESS_FILE,
    RESULTS_DIR,
    S0,
    S1,
    S2,
    S3,
    ReachabilityConfig,
    resolve_input_path,
    solve_reachability_case,
)


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["lines.linewidth"] = 2.2

PLOT_LABEL_FONTSIZE = 20
PLOT_TICK_FONTSIZE = 17
PLOT_CAPTION_FONTSIZE = 20
PLOT_LEGEND_FONTSIZE = 17
PLOT_MARKER_SIZE = 1.0
PLOT_MARKER_EDGE_WIDTH = 1


# =============================================================================
# Direct-run settings
# =============================================================================
# RUN_SCAN=True: solve boundary curves, save npz, then plot.
# RUN_SCAN=False: only redraw from RESULT_FILE.
RUN_SCAN = False
SHOW_FIGURES = True
USE_MULTIPROCESSING = True
N_WORKERS = 8
SHOW_BISECTION_LOG = True

# None means auto-name by strategy, for example:
#   S0 -> results/fault_boundary_S0.npz
#   S2 -> results/fault_boundary_S2.npz
# If you set a concrete Path and run multiple strategies, the strategy name is
# appended to the stem so the files still remain separate.
RESULT_FILE = None

# Resume mode:
# - Existing points in each strategy npz are skipped.
# - New points are saved immediately after each boundary point finishes.
# - Set FORCE_RERUN_EXISTING=True only when you changed constraints and want to
#   overwrite old points for the same strategy/te.
FORCE_RERUN_EXISTING = False
SAVE_AFTER_EACH_POINT = True
# False means failed/NaN points will be retried on the next run.
SKIP_FAILED_EXISTING = False

# Match fault_opt_pro direct-run settings. AUTO_EMERGENCY_CUTOFF is intentionally
# not used here; boundary scans keep T2 initial guess at 200 s - te.
DT = 1.0
ENABLE_DPHI_LIMIT = True
ENABLE_SMOOTHNESS = True
ENABLE_ALPHA_LIMIT = True
ALPHA_MIN_DEG = -60.0
ALPHA_MAX_DEG = 30.0
ENABLE_QALPHA_LIMIT = True
QALPHA_LIMIT = 5000.0  # Pa rad = N rad / m^2

# Plot only: missing points from another strategy are ignored. Keep real large
# jumps connected; explain discontinuities in text if needed.
BREAK_LARGE_KAPPA_JUMPS = False
KAPPA_JUMP_BREAK = 0.25

# Boundary curve x-axis. For paper-quality curves, densify this list.
TE_VALUES = np.array([16,20,25,30,40,50], dtype=float)

# Optional per-strategy x-axis. Use this when one strategy has a tiny reachable
# region, such as S0 only being meaningful near 195-200 s.
# None means "use TE_VALUES".
TE_VALUES_BY_STRATEGY = {
    S0: np.array([16], dtype=float),
    S1: None,
    S2: None,
    S3: None,
}

# S0: fixed stage-1 + fixed T4
# S1: stage-1 cutoff adjustable from failure time + fixed T4
# S2: fixed stage-1 + adjustable T4
# S3: stage-1 cutoff adjustable from failure time + adjustable T4
# You may run one or more strategies; each strategy is saved to its own npz.
STRATEGIES = [S0]

# Search interval for kappa_max.
KAPPA_LOW = 0.0
# Do not use exactly 1.0 here: the stage-1 depletion formula contains
# 1 / (1 - kappa), so kappa=1 is a singular endpoint.
KAPPA_HIGH = 0.02
# Each boundary point solves about 2 + BISECTION_ITERS nonlinear programs.
# Use 4-5 while debugging; increase to 8-10 for a paper-quality boundary.
BISECTION_ITERS = 10

# Keep None to use EarthEnv.m_gan. Set for example 100000.0 if you want a
# stricter terminal mass / second-stage dry-mass lower bound.
FINAL_MASS_MIN = None

# T4 variable bounds for S2/S3.
T4_GUESS = 239.0
T4_MIN = 180.0
T4_MAX = 340.0
T4_FIXED_DURATION = 239.0262

# Match fault_opt_pro. This objective term usually improves convergence; it does
# not relax any terminal constraint.
W_SMOOTHNESS = 1000.0
W_STAGE1_TIME = 0.0
STAGE1_FREE_MIN_AFTER_FAULT = 1e-3


@dataclass
class BoundaryPoint:
    reachable: bool
    kappa_max: float = np.nan
    delta_T1: float = np.nan
    eta1: float = np.nan
    delta_T4: float = np.nan
    T1_sep: float = np.nan
    T1_dep: float = np.nan
    T4_duration: float = np.nan
    message: str = ""


def current_result_file() -> Path:
    if RESULT_FILE is not None and len(STRATEGIES) == 1:
        return Path(RESULT_FILE)
    suffix = "_".join(str(strategy) for strategy in STRATEGIES)
    return RESULTS_DIR / f"fault_boundary_{suffix}.npz"


def strategy_result_file(strategy: str) -> Path:
    if RESULT_FILE is None:
        return RESULTS_DIR / f"fault_boundary_{strategy}.npz"

    path = Path(RESULT_FILE)
    if len(STRATEGIES) == 1:
        return path
    return path.with_name(f"{path.stem}_{strategy}{path.suffix}")


def build_config() -> ReachabilityConfig:
    return ReachabilityConfig(
        dt=DT,
        init_guess_file=INIT_GUESS_FILE,
        result_file=current_result_file(),
        T4_guess=T4_GUESS,
        T4_min=T4_MIN,
        T4_max=T4_MAX,
        T4_fixed=T4_FIXED_DURATION,
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


def solve_kappa_boundary_at_te(
    te: float,
    strategy: str,
    cfg: ReachabilityConfig,
    data,
    env: EarthEnv,
    verbose: bool = True,
) -> BoundaryPoint:
    """Find max reachable kappa for one te by feasibility bisection."""
    lo = float(KAPPA_LOW)
    hi = float(KAPPA_HIGH)

    def solve_trial(label: str, kappa: float):
        if verbose:
            print(f"  {label:<12} kappa={kappa:.5f} start", flush=True)
        trial_start = perf_counter()
        result = solve_reachability_case(te, kappa, strategy, cfg, data=data, env=env)
        if verbose:
            status = "reachable" if result.reachable else "infeasible"
            print(
                f"  {label:<12} kappa={kappa:.5f} {status}, "
                f"time={perf_counter() - trial_start:.1f}s",
                flush=True,
            )
        return result

    lo_res = solve_trial("check low", lo)
    if not lo_res.reachable:
        return BoundaryPoint(False, message=f"kappa_low infeasible: {lo_res.message}")

    best = lo_res

    hi_res = solve_trial("check high", hi)
    if hi_res.reachable:
        return BoundaryPoint(
            True,
            kappa_max=hi,
            delta_T1=hi_res.delta_T1,
            eta1=hi_res.eta1,
            delta_T4=hi_res.delta_T4,
            T1_sep=hi_res.T1_sep,
            T1_dep=hi_res.T1_dep,
            T4_duration=hi_res.T4_duration,
            message="hit KAPPA_HIGH",
        )

    for iter_idx in range(BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        if verbose:
            print(f"  bisect {iter_idx + 1}/{BISECTION_ITERS}: interval=[{lo:.5f}, {hi:.5f}]", flush=True)
        res = solve_trial(f"trial {iter_idx + 1}", mid)
        if res.reachable:
            lo = mid
            best = res
            if verbose:
                print(f"    reachable, new lower={lo:.5f}", flush=True)
        else:
            hi = mid
            if verbose:
                print(f"    infeasible, new upper={hi:.5f}", flush=True)

    return BoundaryPoint(
        True,
        kappa_max=lo,
        delta_T1=best.delta_T1,
        eta1=best.eta1,
        delta_T4=best.delta_T4,
        T1_sep=best.T1_sep,
        T1_dep=best.T1_dep,
        T4_duration=best.T4_duration,
        message=best.message,
    )


def _strategy_te_values(strategy: str) -> np.ndarray:
    values = TE_VALUES_BY_STRATEGY.get(strategy)
    if values is None:
        return np.asarray(TE_VALUES, dtype=float)
    return np.asarray(values, dtype=float)


def _all_te_values_for_scan(strategies: list[str]) -> np.ndarray:
    values = []
    for strategy in strategies:
        values.extend(_strategy_te_values(strategy).tolist())
    return np.asarray(sorted(set(float(v) for v in values)), dtype=float)


def _make_empty_scan(strategies: list[str], te_values: np.ndarray, cfg: ReachabilityConfig) -> dict:
    shape = (len(strategies), len(te_values))
    return {
        "te": te_values,
        "strategies": np.asarray(strategies),
        "scanned": np.zeros(shape, dtype=bool),
        "reachable": np.zeros(shape, dtype=bool),
        "kappa_max": np.full(shape, np.nan),
        "delta_T1": np.full(shape, np.nan),
        "eta1": np.full(shape, np.nan),
        "delta_T4": np.full(shape, np.nan),
        "T1_sep": np.full(shape, np.nan),
        "T1_dep": np.full(shape, np.nan),
        "T4_duration": np.full(shape, np.nan),
        "messages": np.full(shape, "", dtype=object),
    }


def _merge_existing_scan(
    existing: dict | None,
    requested_strategies: list[str],
    requested_te_values: np.ndarray,
    cfg: ReachabilityConfig,
) -> dict:
    if existing is None:
        return _make_empty_scan(requested_strategies, requested_te_values, cfg)

    old_strategies = [str(item) for item in existing.get("strategies", [])]
    strategies = list(old_strategies)
    for strategy in requested_strategies:
        if strategy not in strategies:
            strategies.append(strategy)

    old_te = np.asarray(existing.get("te", []), dtype=float)
    te_values = np.asarray(sorted(set(old_te.tolist() + requested_te_values.tolist())), dtype=float)
    scan = _make_empty_scan(strategies, te_values, cfg)

    new_strategy_to_idx = {strategy: idx for idx, strategy in enumerate(strategies)}
    new_te_to_idx = {float(te): idx for idx, te in enumerate(te_values)}

    keys = [
        "scanned",
        "reachable",
        "kappa_max",
        "delta_T1",
        "eta1",
        "delta_T4",
        "T1_sep",
        "T1_dep",
        "T4_duration",
        "messages",
    ]
    for old_s_idx, strategy in enumerate(old_strategies):
        if strategy not in new_strategy_to_idx:
            continue
        new_s_idx = new_strategy_to_idx[strategy]
        for old_t_idx, te in enumerate(old_te):
            new_t_idx = new_te_to_idx.get(float(te))
            if new_t_idx is None:
                continue
            for key in keys:
                if key in existing:
                    scan[key][new_s_idx, new_t_idx] = existing[key][old_s_idx, old_t_idx]

    return scan


def _load_existing_scan(path: Path) -> dict | None:
    if not path.exists():
        return None
    print(f"Resume from existing boundary data: {path}", flush=True)
    return load_boundary_npz(path)


def _scan_indices(scan: dict) -> tuple[dict[str, int], dict[float, int]]:
    strategies = [str(item) for item in scan["strategies"]]
    te_values = np.asarray(scan["te"], dtype=float)
    return (
        {strategy: idx for idx, strategy in enumerate(strategies)},
        {float(te): idx for idx, te in enumerate(te_values)},
    )


def _store_point_in_scan(scan: dict, strategy: str, te: float, point: BoundaryPoint) -> None:
    strategy_to_idx, te_to_idx = _scan_indices(scan)
    s_idx = strategy_to_idx[strategy]
    t_idx = te_to_idx[float(te)]
    scan["scanned"][s_idx, t_idx] = True
    scan["reachable"][s_idx, t_idx] = point.reachable
    scan["kappa_max"][s_idx, t_idx] = point.kappa_max
    scan["delta_T1"][s_idx, t_idx] = point.delta_T1
    scan["eta1"][s_idx, t_idx] = point.eta1
    scan["delta_T4"][s_idx, t_idx] = point.delta_T4
    scan["T1_sep"][s_idx, t_idx] = point.T1_sep
    scan["T1_dep"][s_idx, t_idx] = point.T1_dep
    scan["T4_duration"][s_idx, t_idx] = point.T4_duration
    scan["messages"][s_idx, t_idx] = point.message


def _point_is_valid(scan: dict, strategy: str, te: float) -> bool:
    strategy_to_idx, te_to_idx = _scan_indices(scan)
    s_idx = strategy_to_idx[strategy]
    t_idx = te_to_idx[float(te)]
    return bool(scan["reachable"][s_idx, t_idx]) and np.isfinite(scan["kappa_max"][s_idx, t_idx])


def _point_is_scanned(scan: dict, strategy: str, te: float) -> bool:
    strategy_to_idx, te_to_idx = _scan_indices(scan)
    return bool(scan["scanned"][strategy_to_idx[strategy], te_to_idx[float(te)]])


def _combine_strategy_scans(strategy_scans: dict[str, dict], requested_strategies: list[str], cfg: ReachabilityConfig) -> dict:
    te_all = []
    for strategy in requested_strategies:
        te_all.extend(np.asarray(strategy_scans[strategy]["te"], dtype=float).tolist())
    te_values = np.asarray(sorted(set(float(te) for te in te_all)), dtype=float)
    combined = _make_empty_scan(requested_strategies, te_values, cfg)

    keys = [
        "scanned",
        "reachable",
        "kappa_max",
        "delta_T1",
        "eta1",
        "delta_T4",
        "T1_sep",
        "T1_dep",
        "T4_duration",
        "messages",
    ]
    combined_strategy_to_idx, combined_te_to_idx = _scan_indices(combined)

    for strategy, source in strategy_scans.items():
        source_strategy_to_idx, source_te_to_idx = _scan_indices(source)
        if strategy not in source_strategy_to_idx:
            continue
        source_s_idx = source_strategy_to_idx[strategy]
        combined_s_idx = combined_strategy_to_idx[strategy]
        for te, source_t_idx in source_te_to_idx.items():
            combined_t_idx = combined_te_to_idx[float(te)]
            for key in keys:
                combined[key][combined_s_idx, combined_t_idx] = source[key][source_s_idx, source_t_idx]

    return combined


def load_selected_strategy_scans(cfg: ReachabilityConfig) -> dict:
    strategy_scans = {}
    for strategy in STRATEGIES:
        path = strategy_result_file(strategy)
        existing = _load_existing_scan(path)
        if existing is None:
            raise FileNotFoundError(f"No boundary data for {strategy}: {path}")
        strategy_scans[strategy] = _merge_existing_scan(existing, [strategy], _strategy_te_values(strategy), cfg)
    return _combine_strategy_scans(strategy_scans, list(STRATEGIES), cfg)


def _boundary_worker(strategy: str, te: float, cfg: ReachabilityConfig) -> tuple[str, float, BoundaryPoint, float]:
    start = perf_counter()
    env = EarthEnv(target=cfg.target)
    data = np.load(resolve_input_path(cfg.init_guess_file))
    point = solve_kappa_boundary_at_te(te, strategy, cfg, data=data, env=env, verbose=False)
    elapsed = perf_counter() - start
    return strategy, float(te), point, elapsed


def scan_boundary_curves(cfg: ReachabilityConfig) -> dict:
    requested_strategies = list(STRATEGIES)
    strategy_paths = {strategy: strategy_result_file(strategy) for strategy in requested_strategies}
    strategy_scans = {}
    for strategy in requested_strategies:
        strategy_scans[strategy] = _merge_existing_scan(
            _load_existing_scan(strategy_paths[strategy]),
            [strategy],
            _strategy_te_values(strategy),
            cfg,
        )

    jobs = []
    skipped = 0
    for strategy in requested_strategies:
        strategy_scan = strategy_scans[strategy]
        for te in _strategy_te_values(strategy):
            point_is_valid = _point_is_valid(strategy_scan, strategy, float(te))
            point_can_skip = _point_is_scanned(strategy_scan, strategy, float(te)) and (
                point_is_valid or SKIP_FAILED_EXISTING
            )
            if point_can_skip and not FORCE_RERUN_EXISTING:
                skipped += 1
                print(f"SKIP existing {strategy}: te={float(te):.3f}", flush=True)
                continue
            if _point_is_scanned(strategy_scan, strategy, float(te)) and not point_is_valid and not FORCE_RERUN_EXISTING:
                print(f"RETRY failed/invalid {strategy}: te={float(te):.3f}", flush=True)
            jobs.append((strategy, float(te)))

    total = len(jobs)
    result_files_text = ", ".join(f"{strategy}={path.name}" for strategy, path in strategy_paths.items())
    print(
        "Boundary config: "
        f"result_files=({result_files_text}), "
        f"dt={cfg.dt}, T4=[{cfg.T4_min}, {cfg.T4_max}], T4_fixed={cfg.T4_fixed}, "
        f"smoothness={'on' if cfg.enable_smoothness else 'off'}({cfg.w_smoothness}), "
        f"alpha_limit={'on' if cfg.enable_alpha_limit else 'off'}"
        f"([{cfg.alpha_min_deg}, {cfg.alpha_max_deg}] deg), "
        f"qalpha_limit={'on' if cfg.enable_qalpha_limit else 'off'}({cfg.qalpha_limit} Pa rad), "
        "T2_guess=200s-te",
        flush=True,
    )

    def store_point(strategy: str, te: float, point: BoundaryPoint, count: int, elapsed: float | None = None) -> None:
        strategy_scan = strategy_scans[strategy]
        _store_point_in_scan(strategy_scan, strategy, te, point)
        elapsed_text = f", time={elapsed:.1f}s" if elapsed is not None else ""
        prefix = f"[{count}/{total}] DONE {strategy}: te={te:.3f}{elapsed_text}"
        if point.reachable:
            print(
                f"{prefix}, kappa_max={point.kappa_max:.5f}, "
                f"dT1={point.delta_T1:.3f}s, eta1={point.eta1:.3f}, dT4={point.delta_T4:.3f}s"
            )
        else:
            print(f"{prefix} failed: {point.message[:120]}")
        if SAVE_AFTER_EACH_POINT:
            save_boundary_npz(strategy_paths[strategy], strategy_scan, quiet=True)

    if total == 0:
        print(f"No new boundary jobs. Reused {skipped} existing point(s).", flush=True)
        return _combine_strategy_scans(strategy_scans, requested_strategies, cfg)

    if USE_MULTIPROCESSING and N_WORKERS > 1 and total > 1:
        print(f"Running {total} boundary jobs with {N_WORKERS} worker processes.")
        for job_idx, (strategy, te) in enumerate(jobs, start=1):
            print(f"[{job_idx}/{total}] QUEUED {strategy}: te={te:.3f}", flush=True)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            future_to_job = {
                executor.submit(_boundary_worker, strategy, te, cfg): (strategy, te)
                for strategy, te in jobs
            }
            for count, future in enumerate(as_completed(future_to_job), start=1):
                strategy, te = future_to_job[future]
                try:
                    result_strategy, result_te, point, elapsed = future.result()
                    store_point(result_strategy, result_te, point, count, elapsed=elapsed)
                except Exception as exc:
                    store_point(strategy, te, BoundaryPoint(False, message=str(exc)), count)
    else:
        env = EarthEnv(target=cfg.target)
        data = np.load(resolve_input_path(cfg.init_guess_file))
        print(
            f"Running {total} boundary job(s) serially. "
            f"Set N_WORKERS > 1 and use more than one job for multiprocessing.",
            flush=True,
        )
        for count, (strategy, te) in enumerate(jobs, start=1):
            print(f"[{count}/{total}] Boundary {strategy}: te={te:.3f}", flush=True)
            start = perf_counter()
            point = solve_kappa_boundary_at_te(
                te,
                strategy,
                cfg,
                data=data,
                env=env,
                verbose=SHOW_BISECTION_LOG,
            )
            store_point(strategy, te, point, count, elapsed=perf_counter() - start)

    if not SAVE_AFTER_EACH_POINT:
        for strategy, strategy_scan in strategy_scans.items():
            save_boundary_npz(strategy_paths[strategy], strategy_scan, quiet=True)

    return _combine_strategy_scans(strategy_scans, requested_strategies, cfg)


def save_boundary_npz(path: Path, scan: dict, quiet: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez(f, **scan)
    tmp_path.replace(path)
    if not quiet:
        print(f"Saved boundary data: {path}")
    return path


def load_boundary_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _boundary_segments(x, y, jump_break: float | None = None):
    x, y = _finite_xy(x, y)
    if len(x) == 0:
        return []

    segments = []
    start = 0
    if jump_break is not None:
        for idx in range(1, len(y)):
            if abs(float(y[idx]) - float(y[idx - 1])) > jump_break:
                segments.append((x[start:idx], y[start:idx]))
                start = idx
    segments.append((x[start:], y[start:]))
    return [(seg_x, seg_y) for seg_x, seg_y in segments if len(seg_x) > 0]


def _caption_axis(ax, caption: str, y: float = -0.18) -> None:
    ax.text(
        0.5,
        y,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PLOT_CAPTION_FONTSIZE,
    )


def _style_boundary_axis(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    ax.grid(True, linestyle="--", alpha=0.35)


def _show_legend_if_any(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=PLOT_LEGEND_FONTSIZE)


def plot_boundary_curves(scan: dict, output_name: str = "fault_boundary_kappa_max.png") -> Path:
    te = scan["te"]
    strategies = [str(item) for item in scan["strategies"]]
    kappa_max = scan["kappa_max"]

    labels = {
        S0: "S0: Fixed timing",
        S1: "S1: Stage-1 adjustable",
        S2: "S2: Stage-2 adjustable",
        S3: "S3: Joint adjustment",
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, strategy in enumerate(strategies):
        y = kappa_max[idx]
        jump_break = KAPPA_JUMP_BREAK if BREAK_LARGE_KAPPA_JUMPS else None
        segments = _boundary_segments(te, y, jump_break=jump_break)
        for seg_idx, (seg_te, seg_y) in enumerate(segments):
            label = labels.get(strategy, strategy) if seg_idx == 0 else "_nolegend_"
            ax.plot(
                seg_te,
                seg_y,
                marker="o",
                markersize=PLOT_MARKER_SIZE,
                markeredgewidth=PLOT_MARKER_EDGE_WIDTH,
                linewidth=2.2,
                label=None,
            )
            ax.fill_between(seg_te, 0.0, seg_y, alpha=0.08)

    _style_boundary_axis(
        ax,
        r"Failure time $t_f$ (s)",
        r"$\kappa_{\max}$",
    )
    """ _caption_axis(ax, r"Reachable-domain boundary under fixed timing") """

    ax.set_ylim(bottom=0.0)
    _show_legend_if_any(ax)

    out = FIGURES_DIR / output_name
    fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure: {out}")
    if not SHOW_FIGURES:
        plt.close(fig)
    return out


def main() -> None:
    cfg = build_config()
    if RUN_SCAN:
        scan = scan_boundary_curves(cfg)
    else:
        scan = load_selected_strategy_scans(cfg)

    plot_boundary_curves(scan, output_name=f"{cfg.result_file.stem}_kappa_max.png")
    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
