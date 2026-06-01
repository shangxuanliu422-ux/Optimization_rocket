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


# =============================================================================
# Direct-run settings
# =============================================================================
# RUN_SCAN=True: solve boundary curves, save npz, then plot.
# RUN_SCAN=False: only redraw from RESULT_FILE.
RUN_SCAN = True
SHOW_FIGURES = True
USE_MULTIPROCESSING = True
N_WORKERS = 8
SHOW_BISECTION_LOG = True

RESULT_FILE = RESULTS_DIR / "fault_boundary_kappa_max.npz"

# Match fault_opt_pro direct-run settings. AUTO_EMERGENCY_CUTOFF is intentionally
# not used here; boundary scans keep T2 initial guess at 200 s - te.
DT = 1.0
ENABLE_DPHI_LIMIT = True
ENABLE_SMOOTHNESS = True
ENABLE_ALPHA_LIMIT = True
ALPHA_LIMIT_DEG = 60.0
ENABLE_QALPHA_LIMIT = True
QALPHA_LIMIT = 5000.0  # Pa rad = N rad / m^2

# Boundary curve x-axis. For paper-quality curves, densify this list.
TE_VALUES = np.array([175], dtype=float)

# Optional per-strategy x-axis. Use this when one strategy has a tiny reachable
# region, such as S0 only being meaningful near 195-200 s.
# None means "use TE_VALUES".
TE_VALUES_BY_STRATEGY = {
    S0: np.array([199.99], dtype=float),
    S1: None,
    S2: None,
    S3: None,
}

# S0: fixed stage-1 + fixed T4
# S1: stage-1 cutoff adjustable from failure time + fixed T4
# S2: fixed stage-1 + adjustable T4
# S3: stage-1 cutoff adjustable from failure time + adjustable T4
STRATEGIES = [S3]

# Search interval for kappa_max.
KAPPA_LOW = 0.0
# Do not use exactly 1.0 here: the stage-1 depletion formula contains
# 1 / (1 - kappa), so kappa=1 is a singular endpoint.
KAPPA_HIGH = 0.999
# Each boundary point solves about 2 + BISECTION_ITERS nonlinear programs.
# Use 4-5 while debugging; increase to 8-10 for a paper-quality boundary.
BISECTION_ITERS = 3

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


def build_config() -> ReachabilityConfig:
    return ReachabilityConfig(
        dt=DT,
        init_guess_file=INIT_GUESS_FILE,
        result_file=RESULT_FILE,
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
        alpha_limit_deg=ALPHA_LIMIT_DEG,
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


def _boundary_worker(strategy: str, te: float, cfg: ReachabilityConfig) -> tuple[str, float, BoundaryPoint, float]:
    start = perf_counter()
    env = EarthEnv(target=cfg.target)
    data = np.load(resolve_input_path(cfg.init_guess_file))
    point = solve_kappa_boundary_at_te(te, strategy, cfg, data=data, env=env, verbose=False)
    elapsed = perf_counter() - start
    return strategy, float(te), point, elapsed


def scan_boundary_curves(cfg: ReachabilityConfig) -> dict:
    strategies = list(STRATEGIES)
    te_values = _all_te_values_for_scan(strategies)
    te_to_idx = {float(te): idx for idx, te in enumerate(te_values)}
    strategy_to_idx = {strategy: idx for idx, strategy in enumerate(strategies)}

    shape = (len(strategies), len(te_values))

    kappa_max = np.full(shape, np.nan)
    reachable = np.zeros(shape, dtype=bool)
    delta_T1 = np.full(shape, np.nan)
    eta1 = np.full(shape, np.nan)
    delta_T4 = np.full(shape, np.nan)
    T1_sep = np.full(shape, np.nan)
    T1_dep = np.full(shape, np.nan)
    T4_duration = np.full(shape, np.nan)
    messages = np.empty(shape, dtype=object)
    scanned = np.zeros(shape, dtype=bool)

    jobs = []
    for s_idx, strategy in enumerate(strategies):
        for te in _strategy_te_values(strategy):
            jobs.append((strategy, float(te)))

    total = len(jobs)
    print(
        "Boundary config: "
        f"dt={cfg.dt}, T4=[{cfg.T4_min}, {cfg.T4_max}], T4_fixed={cfg.T4_fixed}, "
        f"smoothness={'on' if cfg.enable_smoothness else 'off'}({cfg.w_smoothness}), "
        f"alpha_limit={'on' if cfg.enable_alpha_limit else 'off'}({cfg.alpha_limit_deg} deg), "
        f"qalpha_limit={'on' if cfg.enable_qalpha_limit else 'off'}({cfg.qalpha_limit} Pa rad), "
        "T2_guess=200s-te",
        flush=True,
    )

    def store_point(strategy: str, te: float, point: BoundaryPoint, count: int, elapsed: float | None = None) -> None:
        s_idx = strategy_to_idx[strategy]
        t_idx = te_to_idx[float(te)]
        scanned[s_idx, t_idx] = True
        reachable[s_idx, t_idx] = point.reachable
        kappa_max[s_idx, t_idx] = point.kappa_max
        delta_T1[s_idx, t_idx] = point.delta_T1
        eta1[s_idx, t_idx] = point.eta1
        delta_T4[s_idx, t_idx] = point.delta_T4
        T1_sep[s_idx, t_idx] = point.T1_sep
        T1_dep[s_idx, t_idx] = point.T1_dep
        T4_duration[s_idx, t_idx] = point.T4_duration
        messages[s_idx, t_idx] = point.message
        elapsed_text = f", time={elapsed:.1f}s" if elapsed is not None else ""
        prefix = f"[{count}/{total}] DONE {strategy}: te={te:.3f}{elapsed_text}"
        if point.reachable:
            print(
                f"{prefix}, kappa_max={point.kappa_max:.5f}, "
                f"dT1={point.delta_T1:.3f}s, eta1={point.eta1:.3f}, dT4={point.delta_T4:.3f}s"
            )
        else:
            print(f"{prefix} failed: {point.message[:120]}")

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

    return {
        "te": te_values,
        "strategies": np.asarray(strategies),
        "scanned": scanned,
        "reachable": reachable,
        "kappa_max": kappa_max,
        "delta_T1": delta_T1,
        "eta1": eta1,
        "delta_T4": delta_T4,
        "T1_sep": T1_sep,
        "T1_dep": T1_dep,
        "T4_duration": T4_duration,
        "messages": messages,
        "kappa_low": np.array(KAPPA_LOW),
        "kappa_high": np.array(KAPPA_HIGH),
        "bisection_iters": np.array(BISECTION_ITERS),
        "dt": np.array(cfg.dt),
        "T4_guess": np.array(cfg.T4_guess),
        "T4_min": np.array(cfg.T4_min),
        "T4_max": np.array(cfg.T4_max),
        "T4_fixed": np.array(cfg.T4_fixed),
        "stage1_free_min_after_fault": np.array(cfg.stage1_free_min_after_fault),
        "enable_smoothness": np.array(cfg.enable_smoothness),
        "w_smoothness": np.array(cfg.w_smoothness),
        "w_stage1_time": np.array(cfg.w_stage1_time),
        "enable_alpha_limit": np.array(cfg.enable_alpha_limit),
        "alpha_limit_deg": np.array(cfg.alpha_limit_deg),
        "enable_qalpha_limit": np.array(cfg.enable_qalpha_limit),
        "qalpha_limit": np.array(cfg.qalpha_limit),
    }


def save_boundary_npz(path: Path, scan: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **scan)
    print(f"Saved boundary data: {path}")
    return path


def load_boundary_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def plot_boundary_curves(scan: dict, output_name: str = "fault_boundary_kappa_max.png") -> Path:
    te = scan["te"]
    strategies = [str(item) for item in scan["strategies"]]
    kappa_max = scan["kappa_max"]

    labels = {
        S0: "S0 fixed timing",
        S1: "S1 stage-1 adjustable",
        S2: "S2 T4 adjustable",
        S3: "S3 joint compensation",
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, strategy in enumerate(strategies):
        y = kappa_max[idx]
        ax.plot(te, y, marker="o", linewidth=2.2, label=labels.get(strategy, strategy))
        ax.fill_between(te, 0.0, y, alpha=0.08)

    ax.set_xlabel(r"Failure time $t_e$ (s)")
    ax.set_ylabel(r"Maximum reachable fault ratio $\kappa_{\max}$")
    ax.set_title(r"Reachable-domain boundary $\kappa_{\max}(t_e)$")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    out = FIGURES_DIR / output_name
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)
    return out


def plot_boundary_compensation(scan: dict, output_name: str = "fault_boundary_compensation.png") -> Path:
    te = scan["te"]
    strategies = [str(item) for item in scan["strategies"]]
    delta_T1 = scan["delta_T1"]
    delta_T4 = scan["delta_T4"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, strategy in enumerate(strategies):
        label = strategy
        if strategy in {S1, S3}:
            axes[0].plot(te, delta_T1[idx], marker="o", linewidth=2.0, label=label)
        if strategy in {S2, S3}:
            axes[1].plot(te, delta_T4[idx], marker="o", linewidth=2.0, label=label)

    axes[0].set_title(r"Stage-1 compensation on boundary")
    axes[0].set_xlabel(r"Failure time $t_e$ (s)")
    axes[0].set_ylabel(r"$\Delta T_1$ (s)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].set_title(r"Second-stage compensation on boundary")
    axes[1].set_xlabel(r"Failure time $t_e$ (s)")
    axes[1].set_ylabel(r"$\Delta T_4$ (s)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    out = FIGURES_DIR / output_name
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)
    return out


def main() -> None:
    cfg = build_config()
    if RUN_SCAN:
        scan = scan_boundary_curves(cfg)
        save_boundary_npz(RESULT_FILE, scan)
    else:
        scan = load_boundary_npz(RESULT_FILE)

    plot_boundary_curves(scan)
    plot_boundary_compensation(scan)


if __name__ == "__main__":
    main()
