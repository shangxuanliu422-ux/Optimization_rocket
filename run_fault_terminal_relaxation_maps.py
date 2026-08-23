from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import run_fault_boundary_maps as boundary_maps
from core.env_models import EarthEnv
from fault_reachability_core import (
    FIGURES_DIR,
    INIT_GUESS_FILE,
    RESULTS_DIR,
    S0,
    ReachabilityConfig,
    resolve_input_path,
)


# =============================================================================
# Direct-run settings
# =============================================================================
# This script compares fixed-timing S0 reachable-boundary curves under:
#   1) strict terminal constraints, loaded from the existing S0 boundary file;
#   2) true anomaly f relaxed;
#   3) engineering terminal tolerances with f relaxed.
RUN_SCAN = False
SHOW_FIGURES = True
USE_MULTIPROCESSING = True
N_WORKERS = 8
# Options: ["relax_f"], ["engineering"], or ["relax_f", "engineering"].
CASES_TO_RUN = ["relax_f", "engineering"]

STRICT_BOUNDARY_FILE = RESULTS_DIR / "fault_boundary_S0.npz"
RELAX_F_RESULT_FILE = RESULTS_DIR / "fault_terminal_boundary_relax_f.npz"
ENGINEERING_RESULT_FILE = RESULTS_DIR / "fault_terminal_boundary_engineering.npz"
FIGURE_FILE = FIGURES_DIR / "fault_terminal_tolerance_boundary_compare.png"
PLOT_TE_MIN = 185.0

# Only the late-failure region is meaningful for fixed nominal timing.
TE_VALUES = np.array(
    [
        185.0,
        190.0,
        195.0,
        196.0,
        197.0,
        198.0,
        198.5,
        199.0,
        199.2,
        199.4,
        199.6,
        199.8,
        199.9,
        199.95,
        199.99,
    ],
    dtype=float,
)

# Resume mode for the two new curves.
FORCE_RERUN_EXISTING = False
SAVE_AFTER_EACH_POINT = True

# Boundary search settings. Each point solves about 2 + BISECTION_ITERS NLPs.
KAPPA_LOW = 0.0
KAPPA_HIGH = 0.999
BISECTION_ITERS = 10
SHOW_BISECTION_LOG = True

# Fixed-timing S0 model settings.
DT = 1.0
ENABLE_DPHI_LIMIT = True
ENABLE_SMOOTHNESS = True
ENABLE_ALPHA_LIMIT = True
ALPHA_MIN_DEG = -60.0
ALPHA_MAX_DEG = 30.0
ENABLE_QALPHA_LIMIT = True
QALPHA_LIMIT = 5000.0
FINAL_MASS_MIN = None
T4_GUESS = 239.0
T4_MIN = 180.0
T4_MAX = 340.0
T4_FIXED_DURATION = 239.0262
W_SMOOTHNESS = 1000.0

# Engineering terminal envelope. a is stored in meters in the optimizer.
ENGINEERING_TOLERANCES = {
    "a": 10_000.0,  # +/- 10 km
    "i": 0.1,  # deg
    "Omega": 0.15,  # deg
    "omega": 0.3,  # deg
    # e is intentionally kept at the default strict tolerance.
}

STRICT_LABEL = "Strict constraints"
RELAX_F_LABEL = r"Relaxed $f$"
ENGINEERING_LABEL = "Engineering tolerances"


@dataclass(frozen=True)
class TerminalCase:
    name: str
    label: str
    result_file: Path
    terminal_active: tuple[str, ...]
    tolerance_updates: dict[str, float]


TERMINAL_CASES = [
    TerminalCase(
        name="relax_f",
        label=RELAX_F_LABEL,
        result_file=RELAX_F_RESULT_FILE,
        terminal_active=("a", "e", "i", "Omega", "omega"),
        tolerance_updates={},
    ),
    TerminalCase(
        name="engineering",
        label=ENGINEERING_LABEL,
        result_file=ENGINEERING_RESULT_FILE,
        terminal_active=("a", "e", "i", "Omega", "omega"),
        tolerance_updates=ENGINEERING_TOLERANCES,
    ),
]

CASE_BY_NAME = {case.name: case for case in TERMINAL_CASES}


def selected_cases() -> list[TerminalCase]:
    unknown = [name for name in CASES_TO_RUN if name not in CASE_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown CASES_TO_RUN entries: {unknown}; choose from {list(CASE_BY_NAME)}")
    return [CASE_BY_NAME[name] for name in CASES_TO_RUN]


def _patch_boundary_solver_settings() -> None:
    boundary_maps.KAPPA_LOW = KAPPA_LOW
    boundary_maps.KAPPA_HIGH = KAPPA_HIGH
    boundary_maps.BISECTION_ITERS = BISECTION_ITERS


def build_config(case: TerminalCase) -> ReachabilityConfig:
    cfg = ReachabilityConfig(
        dt=DT,
        init_guess_file=INIT_GUESS_FILE,
        result_file=case.result_file,
        T4_guess=T4_GUESS,
        T4_min=T4_MIN,
        T4_max=T4_MAX,
        T4_fixed=T4_FIXED_DURATION,
        final_mass_min=FINAL_MASS_MIN,
        terminal_active=case.terminal_active,
        enable_dphi_limit=ENABLE_DPHI_LIMIT,
        enable_smoothness=ENABLE_SMOOTHNESS,
        w_smoothness=W_SMOOTHNESS,
        enable_alpha_limit=ENABLE_ALPHA_LIMIT,
        alpha_min_deg=ALPHA_MIN_DEG,
        alpha_max_deg=ALPHA_MAX_DEG,
        enable_qalpha_limit=ENABLE_QALPHA_LIMIT,
        qalpha_limit=QALPHA_LIMIT,
    )
    cfg.tolerances = dict(cfg.tolerances)
    cfg.tolerances.update(case.tolerance_updates)
    return cfg


def _blank_scan(case: TerminalCase, te_values=None) -> dict:
    te_values = np.asarray(TE_VALUES if te_values is None else te_values, dtype=float)
    shape = (1, len(te_values))
    return {
        "te": te_values,
        "case": np.asarray(case.name),
        "label": np.asarray(case.label),
        "strategies": np.asarray([S0]),
        "terminal_active": np.asarray(case.terminal_active),
        "tolerances": np.asarray(dict(ReachabilityConfig().tolerances | case.tolerance_updates), dtype=object),
        "scanned": np.zeros(shape, dtype=bool),
        "reachable": np.zeros(shape, dtype=bool),
        "kappa_max": np.full(shape, np.nan),
        "messages": np.full(shape, "", dtype=object),
    }


def load_case_scan(case: TerminalCase, align_to_current_grid: bool = True) -> dict:
    if not case.result_file.exists():
        return _blank_scan(case)

    with np.load(case.result_file, allow_pickle=True) as data:
        old = {key: np.array(data[key]) for key in data.files}

    if not align_to_current_grid:
        print(f"Load existing {case.label}: {case.result_file}", flush=True)
        return old

    old_te = np.asarray(old.get("te", []), dtype=float)
    te_values = np.asarray(sorted(set(old_te.tolist() + np.asarray(TE_VALUES, dtype=float).tolist())), dtype=float)
    scan = _blank_scan(case, te_values=te_values)
    old_te_to_idx = {round(float(te), 12): idx for idx, te in enumerate(old_te)}
    new_te_to_idx = {round(float(te), 12): idx for idx, te in enumerate(te_values)}
    for te in old_te:
        old_idx = old_te_to_idx.get(round(float(te), 12))
        new_idx = new_te_to_idx.get(round(float(te), 12))
        if old_idx is None or new_idx is None:
            continue
        for field in ["scanned", "reachable", "kappa_max", "messages"]:
            if field in old:
                scan[field][0, new_idx] = old[field][0, old_idx]
    print(f"Resume {case.label}: {case.result_file}", flush=True)
    return scan


def save_case_scan(case: TerminalCase, scan: dict, quiet: bool = False) -> None:
    case.result_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = case.result_file.with_name(case.result_file.name + ".tmp")
    backup_path = case.result_file.with_suffix(case.result_file.suffix + ".bak")
    if case.result_file.exists():
        backup_path.write_bytes(case.result_file.read_bytes())
    with tmp_path.open("wb") as f:
        np.savez(f, **scan)
    tmp_path.replace(case.result_file)
    if not quiet:
        print(f"Saved {case.label}: {case.result_file}", flush=True)


def _store_boundary_point(scan: dict, t_idx: int, point: boundary_maps.BoundaryPoint) -> None:
    scan["scanned"][0, t_idx] = True
    scan["reachable"][0, t_idx] = point.reachable
    scan["kappa_max"][0, t_idx] = point.kappa_max
    scan["messages"][0, t_idx] = point.message


def should_skip(scan: dict, t_idx: int) -> bool:
    return (not FORCE_RERUN_EXISTING) and bool(scan["scanned"][0, t_idx])


def solve_case_boundary_job(job):
    case, t_idx, te, cfg = job
    _patch_boundary_solver_settings()
    print(f"  START {case.label}: te={te:.3f}", flush=True)
    env = EarthEnv(target=cfg.target)
    data = np.load(resolve_input_path(cfg.init_guess_file))
    try:
        start = perf_counter()
        point = boundary_maps.solve_kappa_boundary_at_te(
            float(te),
            S0,
            cfg,
            data=data,
            env=env,
            verbose=False,
        )
        elapsed = perf_counter() - start
        status = f"kappa_max={point.kappa_max:.5f}" if point.reachable else f"failed: {point.message[:100]}"
        print(f"  WORKER DONE {case.label}: te={te:.3f}, {status}, time={elapsed:.1f}s", flush=True)
        return case, t_idx, float(te), point, elapsed
    finally:
        data.close()


def scan_case_boundary(case: TerminalCase) -> dict:
    _patch_boundary_solver_settings()
    cfg = build_config(case)
    scan = load_case_scan(case)

    jobs = []
    scan_te_to_idx = {round(float(te), 12): idx for idx, te in enumerate(np.asarray(scan["te"], dtype=float))}
    for te in TE_VALUES:
        t_idx = scan_te_to_idx[round(float(te), 12)]
        if should_skip(scan, t_idx):
            print(f"SKIP existing {case.label}: te={te:.3f}", flush=True)
            continue
        jobs.append((case, t_idx, float(te), cfg))

    if USE_MULTIPROCESSING and N_WORKERS > 1 and len(jobs) > 1:
        print(f"Running {len(jobs)} {case.label} boundary job(s) with {N_WORKERS} workers.", flush=True)
        for job_idx, (_, _, te, _) in enumerate(jobs, start=1):
            print(f"[{job_idx}/{len(jobs)}] QUEUED {case.label}: te={te:.3f}", flush=True)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(solve_case_boundary_job, job) for job in jobs]
            for count, future in enumerate(as_completed(futures), start=1):
                result_case, t_idx, te, point, elapsed = future.result()
                _store_boundary_point(scan, t_idx, point)
                status = f"kappa_max={point.kappa_max:.5f}" if point.reachable else f"failed: {point.message[:100]}"
                print(
                    f"[{count}/{len(jobs)}] DONE {result_case.label}: "
                    f"te={te:.3f}, {status}, time={elapsed:.1f}s",
                    flush=True,
                )
                if SAVE_AFTER_EACH_POINT:
                    save_case_scan(case, scan, quiet=True)
        save_case_scan(case, scan)
        return scan

    env = EarthEnv(target=cfg.target)
    data = np.load(resolve_input_path(cfg.init_guess_file))
    try:
        for _, t_idx, te, _ in jobs:
            print(f"{case.label}: boundary te={te:.3f}", flush=True)
            start = perf_counter()
            point = boundary_maps.solve_kappa_boundary_at_te(
                float(te),
                S0,
                cfg,
                data=data,
                env=env,
                verbose=SHOW_BISECTION_LOG,
            )
            _store_boundary_point(scan, t_idx, point)
            status = f"kappa_max={point.kappa_max:.5f}" if point.reachable else f"failed: {point.message[:100]}"
            print(f"  DONE {case.label}: te={te:.3f}, {status}, time={perf_counter() - start:.1f}s", flush=True)
            if SAVE_AFTER_EACH_POINT:
                save_case_scan(case, scan, quiet=True)
    finally:
        data.close()

    save_case_scan(case, scan)
    return scan


def _load_strict_s0_boundary() -> tuple[np.ndarray, np.ndarray]:
    if not STRICT_BOUNDARY_FILE.exists():
        raise FileNotFoundError(f"Missing strict S0 boundary data: {STRICT_BOUNDARY_FILE}")

    with np.load(STRICT_BOUNDARY_FILE, allow_pickle=True) as data:
        te = np.asarray(data["te"], dtype=float).reshape(-1)
        kappa_max = np.asarray(data["kappa_max"], dtype=float)
        if kappa_max.ndim > 1:
            kappa_max = kappa_max[0]
        reachable = np.asarray(data["reachable"])
        if reachable.ndim > 1:
            reachable = reachable[0]
        mask = np.isfinite(te) & np.isfinite(kappa_max) & reachable.astype(bool)
    mask &= te >= PLOT_TE_MIN
    order = np.argsort(te[mask])
    return te[mask][order], kappa_max[mask][order]


def _case_xy(scan: dict) -> tuple[np.ndarray, np.ndarray]:
    te = np.asarray(scan["te"], dtype=float)
    kappa = np.asarray(scan["kappa_max"], dtype=float)[0]
    reachable = np.asarray(scan["reachable"], dtype=bool)[0]
    mask = np.isfinite(te) & np.isfinite(kappa) & reachable
    order = np.argsort(te[mask])
    return te[mask][order], kappa[mask][order]


def plot_terminal_boundary_comparison(case_scans: dict[str, dict]) -> Path:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["lines.linewidth"] = 2.2

    fig, ax = plt.subplots(figsize=(7, 5))
    strict_te, strict_kappa = _load_strict_s0_boundary()
    ax.plot(strict_te, strict_kappa, "k--", linewidth=2.2, label=STRICT_LABEL)

    styles = {
        "relax_f": {"color": "#2f78bd", "marker": "o"},
        "engineering": {"color": "#d62728", "marker": "s"},
    }
    for case in TERMINAL_CASES:
        scan = case_scans[case.name]
        te, kappa = _case_xy(scan)
        style = styles.get(case.name, {})
        ax.plot(
            te,
            kappa,
            label=case.label,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            markersize=0.7,
            linewidth=1.8,
        )

    ax.set_xlim(PLOT_TE_MIN, 200.05)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"Failure time $t_f$ (s)", fontsize=20)
    ax.set_ylabel(r"$\kappa_{\max}$", fontsize=25)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=17, loc="upper left")
    """ ax.text(
        0.5,
        -0.2,
        r"Fixed-timing S0 boundary under terminal constraint relaxation",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=18,
    ) """
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure: {FIGURE_FILE}", flush=True)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)
    return FIGURE_FILE


def main() -> None:
    case_scans = {}
    if RUN_SCAN:
        for case in selected_cases():
            case_scans[case.name] = scan_case_boundary(case)

    for case in TERMINAL_CASES:
        if case.name not in case_scans:
            case_scans[case.name] = load_case_scan(case, align_to_current_grid=False)

    plot_terminal_boundary_comparison(case_scans)


if __name__ == "__main__":
    main()
