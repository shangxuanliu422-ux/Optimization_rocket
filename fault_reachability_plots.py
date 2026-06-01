from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fault_reachability_core import FIGURES_DIR, load_scan_npz


REACHABLE_COLORS = ["#f2b6b6", "#9bd3a9"]


def _strategy_index(scan: dict, strategy_name: str) -> int:
    strategies = [str(item) for item in scan["strategies"]]
    if strategy_name not in strategies:
        raise ValueError(f"{strategy_name!r} not found in strategies: {strategies}")
    return strategies.index(strategy_name)


def _level_index(scan: dict, level_name: str) -> int:
    levels = [str(item) for item in scan["levels"]]
    if level_name not in levels:
        raise ValueError(f"{level_name!r} not found in levels: {levels}")
    return levels.index(level_name)


def _finish_figure(fig, output_path: Path, show: bool) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _plot_binary(ax, te, kappa, values, title):
    TE, K = np.meshgrid(te, kappa)
    ax.contourf(TE, K, values.astype(float), levels=[-0.5, 0.5, 1.5], colors=REACHABLE_COLORS)
    ax.contour(TE, K, values.astype(float), levels=[0.5], colors="black", linewidths=0.8)
    ax.set_title(title)
    ax.set_xlabel(r"Failure time $t_e$ (s)")
    ax.set_ylabel(r"Fault ratio $\kappa$")
    ax.grid(True, linestyle="--", alpha=0.35)


def _plot_heat(ax, te, kappa, values, title, colorbar_label, cmap="viridis"):
    values = np.asarray(values, dtype=float)
    TE, K = np.meshgrid(te, kappa)
    if not np.any(np.isfinite(values)):
        ax.text(0.5, 0.5, "No reachable cases", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(r"Failure time $t_e$ (s)")
        ax.set_ylabel(r"Fault ratio $\kappa$")
        return None
    contour = ax.contourf(TE, K, values, levels=30, cmap=cmap)
    ax.contour(TE, K, values, levels=10, colors="black", linewidths=0.35, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(r"Failure time $t_e$ (s)")
    ax.set_ylabel(r"Fault ratio $\kappa$")
    ax.grid(True, linestyle="--", alpha=0.25)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(colorbar_label)
    return contour


def plot_strategy_reachability(scan_or_path, output="strategy_reachability.png", show=True):
    scan = load_scan_npz(scan_or_path) if not isinstance(scan_or_path, dict) else scan_or_path
    te = scan["te"]
    kappa = scan["kappa"]
    strategies = [str(item) for item in scan["strategies"]]

    n = len(strategies)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.8 * nrows), squeeze=False)
    axes = axes.ravel()

    titles = {
        "S0": "S0 fixed stage-1, fixed T4",
        "S1": "S1 stage-1 adjustable, fixed T4",
        "S2": "S2 fixed stage-1, adjustable T4",
        "S3": "S3 joint compensation",
    }

    for idx, strategy_name in enumerate(strategies):
        _plot_binary(
            axes[idx],
            te,
            kappa,
            scan["reachable"][idx],
            titles.get(strategy_name, strategy_name),
        )
    for ax in axes[n:]:
        ax.axis("off")

    return _finish_figure(fig, FIGURES_DIR / output, show)


def plot_stage1_compensation(scan_or_path, output="stage1_compensation_eta.png", show=True):
    scan = load_scan_npz(scan_or_path) if not isinstance(scan_or_path, dict) else scan_or_path
    te = scan["te"]
    kappa = scan["kappa"]
    available = [str(item) for item in scan["strategies"]]
    names = [name for name in ["S1", "S3"] if name in available]
    if not names:
        raise ValueError("Scan does not contain S1 or S3.")

    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 4.8), squeeze=False)
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        idx = _strategy_index(scan, name)
        values = np.where(scan["reachable"][idx], scan["eta1"][idx], np.nan)
        _plot_heat(
            ax,
            te,
            kappa,
            values,
            rf"{name}: stage-1 utilization $\eta_1$",
            r"$\eta_1=(T_{1,sep}-t_e)/(T_{1,dep}-t_e)$",
            cmap="plasma",
        )
    return _finish_figure(fig, FIGURES_DIR / output, show)


def plot_stage1_delta(scan_or_path, output="stage1_compensation_deltaT1.png", show=True):
    scan = load_scan_npz(scan_or_path) if not isinstance(scan_or_path, dict) else scan_or_path
    te = scan["te"]
    kappa = scan["kappa"]
    available = [str(item) for item in scan["strategies"]]
    names = [name for name in ["S1", "S3"] if name in available]
    if not names:
        raise ValueError("Scan does not contain S1 or S3.")

    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 4.8), squeeze=False)
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        idx = _strategy_index(scan, name)
        values = np.where(scan["reachable"][idx], scan["delta_T1"][idx], np.nan)
        _plot_heat(ax, te, kappa, values, rf"{name}: stage-1 extension", r"$\Delta T_1$ (s)", cmap="magma")
    return _finish_figure(fig, FIGURES_DIR / output, show)


def plot_t4_compensation(scan_or_path, output="t4_compensation_delta.png", show=True):
    scan = load_scan_npz(scan_or_path) if not isinstance(scan_or_path, dict) else scan_or_path
    te = scan["te"]
    kappa = scan["kappa"]
    available = [str(item) for item in scan["strategies"]]
    names = [name for name in ["S2", "S3"] if name in available]
    if not names:
        raise ValueError("Scan does not contain S2 or S3.")

    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 4.8), squeeze=False)
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        idx = _strategy_index(scan, name)
        values = np.where(scan["reachable"][idx], scan["delta_T4"][idx], np.nan)
        _plot_heat(ax, te, kappa, values, rf"{name}: second-stage compensation", r"$\Delta T_4$ (s)")
    return _finish_figure(fig, FIGURES_DIR / output, show)


def plot_t4_save(scan_or_path, output="t4_save_S2_minus_S3.png", show=True):
    scan = load_scan_npz(scan_or_path) if not isinstance(scan_or_path, dict) else scan_or_path
    te = scan["te"]
    kappa = scan["kappa"]
    s2 = _strategy_index(scan, "S2")
    s3 = _strategy_index(scan, "S3")
    both_reachable = scan["reachable"][s2] & scan["reachable"][s3]
    values = np.where(both_reachable, scan["delta_T4"][s2] - scan["delta_T4"][s3], np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_heat(
        ax,
        te,
        kappa,
        values,
        r"Second-stage compensation saved by joint optimization",
        r"$\Delta T_4^{S2}-\Delta T_4^{S3}$ (s)",
        cmap="coolwarm",
    )
    return _finish_figure(fig, FIGURES_DIR / output, show)


def plot_terminal_relaxation(scan_or_path, output="terminal_relaxation_reachability.png", show=True):
    scan = load_scan_npz(scan_or_path) if not isinstance(scan_or_path, dict) else scan_or_path
    te = scan["te"]
    kappa = scan["kappa"]
    levels = [str(item) for item in scan["levels"]]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), squeeze=False)
    axes = axes.ravel()
    titles = {
        "L0": "L0 strict six orbital elements",
        "L1": "L1 relaxed true anomaly f",
        "L2": "L2 relaxed f, e, i",
        "L3": "L3 relaxed f, e, i, a",
    }
    for idx, level in enumerate(levels[:4]):
        _plot_binary(axes[idx], te, kappa, scan["reachable"][idx], titles.get(level, level))
    for ax in axes[len(levels):]:
        ax.axis("off")
    return _finish_figure(fig, FIGURES_DIR / output, show)
