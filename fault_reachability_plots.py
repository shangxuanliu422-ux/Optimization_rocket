from __future__ import annotations

from pathlib import Path

from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

from fault_reachability_core import FIGURES_DIR, RESULTS_DIR, load_scan_npz


REACHABLE_COLORS = ["#f2b6b6", "#9bd3a9"]

# 全局字体/线宽设置：只影响图片外观，不影响任何 npz 结果。
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["lines.linewidth"] = 2.2

# 字号参数：坐标轴标签、刻度、图下注释、色条标签。
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 17
CAPTION_FONTSIZE = 20
COLORBAR_FONTSIZE = 17
CONTOUR_LINEWIDTH = 0.55

# 热图色标参数：
# HEAT_CLIP_PERCENTILE 用 95 分位作为色标上限，避免少数极大值把整张图压暗。
# HEAT_POWER_GAMMA < 1 会增强低值区颜色差异；设为 1.0 就是普通线性色标。
# HEATMAP_COLOR_LEVELS 是 contourf 的色阶数量，越大颜色过渡越顺。
# HEATMAP_CONTOUR_LEVELS 是黑色等值线数量。
HEAT_CLIP_PERCENTILE = 95.0
HEAT_POWER_GAMMA = 0.55
HEATMAP_COLOR_LEVELS = 72
HEATMAP_CONTOUR_LEVELS = 12
HEAT_ZERO_TOL = 1e-8

# 原始网格点显示参数：
# 图里的灰色小点就是 npz 中真实跑过/有效的原始采样点，不是插值点。
# 这里用黑色低透明度绘制，视觉上会变成灰色，用来提醒读者原始采样分辨率。
GRID_POINT_SIZE = 6
GRID_POINT_ALPHA = 0.1

# boundary scan 边界曲线参数：
# 边界线从 results/fault_boundary_*.npz 读取，只作为叠加曲线，不改策略扫描数据。
# BOUNDARY_INTERP_POINTS 是边界曲线显示插值后的横向点数。
# BOUNDARY_SMOOTH_WINDOW / BOUNDARY_SMOOTH_SIGMA 控制边界线的一维高斯平滑强度。
BOUNDARY_LINEWIDTH = 1.25
SMOOTH_BOUNDARY_LINES = True
BOUNDARY_INTERP_POINTS = 360
BOUNDARY_SMOOTH_WINDOW = 9
BOUNDARY_SMOOTH_SIGMA = 2.0

# 热图显示插值参数：
# 这些只改变画图时的显示网格，不改变 npz 原始数据。
# HEATMAP_INTERP_TE_POINTS / HEATMAP_INTERP_KAPPA_POINTS 是显示用 dense grid 的分辨率。
# SMOOTH_HEATMAP_VALUES_FOR_DISPLAY 会对显示网格上的热图值轻微平滑，减少 0.02 kappa 采样造成的阶梯感。
INTERPOLATE_HEATMAP_FOR_DISPLAY = True
HEATMAP_INTERP_TE_POINTS = 360
HEATMAP_INTERP_KAPPA_POINTS = 260
SMOOTH_HEATMAP_VALUES_FOR_DISPLAY = True
HEATMAP_SMOOTH_SIGMA = 2.0

# 不可达/无数据区域显示参数：
# 底色和热图 0 附近颜色容易混淆，所以这里额外加非常淡的斜线纹理。
UNREACHABLE_FACE_COLOR = "#f0f0f0"
UNREACHABLE_HATCH = "///"
UNREACHABLE_HATCH_COLOR = "#d2d2d2"
UNREACHABLE_HATCH_ALPHA = 1.0

plt.rcParams["hatch.color"] = UNREACHABLE_HATCH_COLOR
plt.rcParams["hatch.linewidth"] = 0.35

_BOUNDARY_CACHE: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}


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


def _finish_figure(
    fig,
    output_path: Path,
    show: bool,
    *,
    w_pad: float | None = None,
    wspace: float | None = None,
) -> Path:
    """保存/显示图片。

    fig: matplotlib 生成的整张图。
    output_path: 图片保存路径。
    show: True 时弹出/显示图片窗口，False 时只保存文件并关闭 figure。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tight_kwargs = {"rect": [0, 0.05, 1, 0.98]}
    if w_pad is not None:
        tight_kwargs["w_pad"] = w_pad
    fig.tight_layout(**tight_kwargs)
    if wspace is not None:
        fig.subplots_adjust(wspace=wspace)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure: {output_path}")
    if show:
        fig.show()
    else:
        plt.close(fig)
    return output_path


def _caption_axis(ax, caption, y=-0.24, fontsize=CAPTION_FONTSIZE):
    """在子图下方放标题，避免 matplotlib 默认 title 挤占图内空间。"""
    ax.text(
        0.5,
        y,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
    )


def _style_axis(
    ax,
    xlabel,
    ylabel,
    grid_alpha=0.3,
    *,
    label_fontsize=LABEL_FONTSIZE,
    tick_fontsize=TICK_FONTSIZE,
):
    """统一坐标轴标签、刻度字号和网格透明度。"""
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, linestyle="--", alpha=grid_alpha)


def _dedupe_curve(te: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(te)
    te = np.asarray(te, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]

    unique_te = []
    unique_y = []
    for te_value, y_value in zip(te, y):
        if unique_te and np.isclose(te_value, unique_te[-1], rtol=0.0, atol=1e-9):
            unique_y[-1] = y_value
        else:
            unique_te.append(te_value)
            unique_y.append(y_value)
    return np.asarray(unique_te, dtype=float), np.asarray(unique_y, dtype=float)


def _load_boundary_curve(strategy_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """读取某个策略的 boundary scan 结果，返回 te-kappa_max 边界曲线。"""
    if strategy_name in _BOUNDARY_CACHE:
        return _BOUNDARY_CACHE[strategy_name]

    path = RESULTS_DIR / f"fault_boundary_{strategy_name}.npz"
    if not path.exists():
        _BOUNDARY_CACHE[strategy_name] = None
        return None

    with np.load(path, allow_pickle=True) as data:
        strategies = [str(item) for item in np.asarray(data["strategies"]).ravel()]
        strategy_idx = strategies.index(strategy_name) if strategy_name in strategies else 0
        te = np.asarray(data["te"], dtype=float).reshape(-1)
        kappa_max = np.asarray(data["kappa_max"], dtype=float)
        kappa_max = kappa_max[strategy_idx].reshape(-1) if kappa_max.ndim > 1 else kappa_max.reshape(-1)

        mask = np.isfinite(te) & np.isfinite(kappa_max)
        if "scanned" in data:
            scanned = np.asarray(data["scanned"])
            scanned = scanned[strategy_idx].reshape(-1) if scanned.ndim > 1 else scanned.reshape(-1)
            mask &= scanned.astype(bool)
        if "reachable" in data:
            reachable = np.asarray(data["reachable"])
            reachable = reachable[strategy_idx].reshape(-1) if reachable.ndim > 1 else reachable.reshape(-1)
            mask &= reachable.astype(bool)

    curve = _dedupe_curve(te[mask], kappa_max[mask]) if np.any(mask) else None
    if curve is not None and len(curve[0]) < 2:
        curve = None
    _BOUNDARY_CACHE[strategy_name] = curve
    return curve


def _normalize_boundary_strategies(boundary_strategy) -> list[str]:
    """把 None / 单个策略名 / 多个策略名统一成 list，方便后面循环处理。"""
    if boundary_strategy is None:
        return []
    if isinstance(boundary_strategy, str):
        return [boundary_strategy]
    return [str(item) for item in boundary_strategy]


def _gaussian_smooth(y: np.ndarray, window: int, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window < 3 or len(y) < window:
        return y
    if window % 2 == 0:
        window += 1
    radius = window // 2
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(y, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _interpolate_boundary_curve(te: np.ndarray, kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把粗边界曲线插值并轻微平滑，只用于显示，不回写 npz。"""
    te = np.asarray(te, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    if not SMOOTH_BOUNDARY_LINES or len(te) < 4:
        return te, kappa

    dense_count = max(int(BOUNDARY_INTERP_POINTS), len(te))
    te_dense = np.linspace(float(te[0]), float(te[-1]), dense_count)
    try:
        from scipy.interpolate import PchipInterpolator

        kappa_dense = PchipInterpolator(te, kappa)(te_dense)
    except Exception:
        kappa_dense = np.interp(te_dense, te, kappa)

    kappa_smooth = _gaussian_smooth(
        kappa_dense,
        BOUNDARY_SMOOTH_WINDOW,
        BOUNDARY_SMOOTH_SIGMA,
    )
    kappa_smooth = np.clip(kappa_smooth, 0.0, 0.999)
    return te_dense, kappa_smooth


def _overlay_boundary_lines(ax, boundary_strategy) -> None:
    """叠加 boundary scan 曲线；多策略时用不同线型画在同一张图上。"""
    line_styles = ["-", "--", ":"]
    for idx, strategy_name in enumerate(_normalize_boundary_strategies(boundary_strategy)):
        curve = _load_boundary_curve(strategy_name)
        if curve is None:
            continue
        te_boundary, kappa_boundary = curve
        te_plot, kappa_plot = _interpolate_boundary_curve(te_boundary, kappa_boundary)
        linestyle = line_styles[idx % len(line_styles)]

        # A light underlay keeps the boundary legible on both dark and bright heatmap regions.
        ax.plot(
            te_plot,
            kappa_plot,
            color="white",
            linewidth=BOUNDARY_LINEWIDTH + 1.8,
            linestyle=linestyle,
            alpha=0.9,
            zorder=7,
        )
        ax.plot(
            te_plot,
            kappa_plot,
            color="black",
            linewidth=BOUNDARY_LINEWIDTH,
            linestyle=linestyle,
            alpha=0.92,
            zorder=8,
        )


def _overlay_grid_points(ax, te, kappa, values=None, alpha: float = GRID_POINT_ALPHA) -> None:
    """叠加原始采样点。

    te/kappa: 原始扫描网格坐标。
    values: 若为 None，显示整个 te-kappa 网格；若给定数组，只显示 values 有限的点。
    alpha: 点的透明度。因为用黑色低透明度绘制，图上看起来就是灰色小点。
    """
    TE, K = np.meshgrid(te, kappa)
    if values is None:
        mask = np.ones_like(TE, dtype=bool)
    else:
        mask = np.isfinite(np.asarray(values, dtype=float))
    if not np.any(mask):
        return
    ax.scatter(
        TE[mask],
        K[mask],
        s=GRID_POINT_SIZE,
        c="black",
        alpha=alpha,
        linewidths=0,
        zorder=6,
    )


def _heat_limits(values: np.ndarray, clip_percentile: float) -> tuple[float, float]:
    """计算热图色标范围；上限可用分位数截断，避免极端值主导色条。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan

    vmin = float(np.nanmin(finite))
    if -HEAT_ZERO_TOL < vmin < 0.0:
        vmin = 0.0
    vmax_full = float(np.nanmax(finite))
    vmax_clip = float(np.nanpercentile(finite, clip_percentile))
    vmax = vmax_clip if vmax_clip > vmin else vmax_full
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _heat_levels(vmin: float, vmax: float, use_power_norm: bool) -> np.ndarray:
    """生成 contourf 的色阶；PowerNorm 时让色阶和非线性色标匹配。"""
    if not use_power_norm:
        return np.linspace(vmin, vmax, HEATMAP_COLOR_LEVELS)
    normalized = np.linspace(0.0, 1.0, HEATMAP_COLOR_LEVELS)
    return vmin + (vmax - vmin) * normalized ** (1.0 / HEAT_POWER_GAMMA)


def _heat_norm(values: np.ndarray, vmin: float, vmax: float, cmap: str, use_power_norm: bool):
    """选择色标归一化方式；发散色图跨过 0 时把 0 放在色条中心。"""
    finite = values[np.isfinite(values)]
    spans_zero = finite.size > 0 and vmin < 0.0 < vmax
    if cmap in {"coolwarm", "RdBu", "RdBu_r", "RdYlBu", "RdYlBu_r", "seismic"} and spans_zero:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    if use_power_norm:
        return PowerNorm(gamma=HEAT_POWER_GAMMA, vmin=vmin, vmax=vmax)
    return None


def _overlay_unreachable_hatch(ax, TE, K, display_values) -> None:
    """给不可达/无数据区域叠加淡斜线，避免和热图低值或 0 值颜色混淆。"""
    invalid = ~np.isfinite(display_values)
    if not np.any(invalid):
        return

    hatch_values = np.where(invalid, 1.0, np.nan)
    hatch = ax.contourf(
        TE,
        K,
        hatch_values,
        levels=[0.5, 1.5],
        colors=[UNREACHABLE_FACE_COLOR],
        hatches=[UNREACHABLE_HATCH],
        alpha=UNREACHABLE_HATCH_ALPHA,
        zorder=2,
    )
    for collection in getattr(hatch, "collections", []):
        collection.set_edgecolor(UNREACHABLE_HATCH_COLOR)
        collection.set_alpha(UNREACHABLE_HATCH_ALPHA)


def _display_boundary_limit(te_dense: np.ndarray, boundary_strategy) -> np.ndarray | None:
    """计算显示插值网格的可达上边界。

    单策略图：使用该策略自己的 boundary 曲线。
    多策略图：取多条 boundary 曲线的最小 kappa，表示公共可达域边界。
    """
    names = _normalize_boundary_strategies(boundary_strategy)
    if not names:
        return None

    limits = []
    for strategy_name in names:
        curve = _load_boundary_curve(strategy_name)
        if curve is None:
            return None
        boundary_te, boundary_kappa = _interpolate_boundary_curve(*curve)
        limits.append(np.interp(te_dense, boundary_te, boundary_kappa, left=np.nan, right=np.nan))

    stacked = np.vstack(limits)
    valid = np.all(np.isfinite(stacked), axis=0)
    limit = np.min(np.where(np.isfinite(stacked), stacked, np.inf), axis=0)
    limit[~valid] = np.nan
    return limit


def _smooth_display_values(values: np.ndarray) -> np.ndarray:
    """对显示网格上的热图值做轻微平滑；NaN 区域仍保持 NaN。"""
    values = np.asarray(values, dtype=float)
    if not SMOOTH_HEATMAP_VALUES_FOR_DISPLAY or HEATMAP_SMOOTH_SIGMA <= 0:
        return values

    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 4:
        return values

    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return values

    weights = finite.astype(float)
    filled = np.where(finite, values, 0.0)
    smooth_values = gaussian_filter(filled, sigma=HEATMAP_SMOOTH_SIGMA, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=HEATMAP_SMOOTH_SIGMA, mode="nearest")

    smoothed = np.full_like(values, np.nan, dtype=float)
    valid = finite & (smooth_weights > 1e-8)
    smoothed[valid] = smooth_values[valid] / smooth_weights[valid]
    return smoothed


def _interpolated_heatmap_display_grid(te, kappa, values, boundary_strategy):
    """生成热图显示用 dense grid。

    te/kappa/values: 来自 npz 的原始扫描网格和原始物理量。
    boundary_strategy: 用哪条边界曲线裁剪显示区域；多策略时裁剪公共可达域。
    返回值: TE, K, display_values。它们只用于画图，不会覆盖原始数据。
    """
    te = np.asarray(te, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    values = np.asarray(values, dtype=float)
    TE, K = np.meshgrid(te, kappa)

    if not INTERPOLATE_HEATMAP_FOR_DISPLAY or not _normalize_boundary_strategies(boundary_strategy):
        return TE, K, values

    te_dense = np.linspace(float(np.nanmin(te)), float(np.nanmax(te)), max(HEATMAP_INTERP_TE_POINTS, len(te)))
    kappa_dense = np.linspace(
        float(np.nanmin(kappa)),
        float(np.nanmax(kappa)),
        max(HEATMAP_INTERP_KAPPA_POINTS, len(kappa)),
    )
    TE_dense, K_dense = np.meshgrid(te_dense, kappa_dense)
    kappa_max = _display_boundary_limit(te_dense, boundary_strategy)
    if kappa_max is None or np.count_nonzero(np.isfinite(values)) < 4:
        return TE, K, values

    finite = np.isfinite(values)
    points = np.column_stack([TE[finite], K[finite]])
    point_values = values[finite]

    try:
        from scipy.interpolate import griddata

        display_values = griddata(points, point_values, (TE_dense, K_dense), method="linear")
        fill_values = griddata(points, point_values, (TE_dense, K_dense), method="nearest")
        display_values = np.where(np.isfinite(display_values), display_values, fill_values)
    except Exception:
        return TE, K, values

    outside = (~np.isfinite(kappa_max))[None, :] | (K_dense > kappa_max[None, :])
    display_values[outside] = np.nan
    display_values = _smooth_display_values(display_values)
    return TE_dense, K_dense, display_values


def _plot_binary(ax, te, kappa, values, title, boundary_strategy=None):
    """画可达/不可达二值图。

    ax: 当前子图对象。
    te/kappa: 横轴故障时间和纵轴故障比例的原始扫描网格。
    values: bool 数组，True 表示该网格点可达，False 表示不可达。
    title: 子图下方的说明文字。
    boundary_strategy: 叠加哪种策略的 boundary scan 曲线，例如 "S1" 或 "S3"。
    """
    TE, K = np.meshgrid(te, kappa)
    ax.contourf(TE, K, values.astype(float), levels=[-0.5, 0.5, 1.5], colors=REACHABLE_COLORS)
    ax.contour(TE, K, values.astype(float), levels=[0.5], colors="black", linewidths=0.25, alpha=0.25)
    _overlay_grid_points(ax, te, kappa, alpha=0.09)
    _overlay_boundary_lines(ax, boundary_strategy)
    _style_axis(ax, r"Failure time $t_f$ (s)", r"$\kappa$", grid_alpha=0.35)
    _caption_axis(ax, title)


def _plot_heat(
    ax,
    te,
    kappa,
    values,
    title,
    colorbar_label,
    cmap="viridis",
    boundary_strategy=None,
    clip_percentile=HEAT_CLIP_PERCENTILE,
    use_power_norm=True,
    colorbar_tick_format=None,
    colorbar_pad=0.035,
    label_fontsize=LABEL_FONTSIZE,
    tick_fontsize=TICK_FONTSIZE,
    caption_fontsize=CAPTION_FONTSIZE,
    colorbar_fontsize=COLORBAR_FONTSIZE,
):
    """画连续物理量热图，比如 delta_T1、delta_T4、eta1。

    ax: 当前子图对象。
    te/kappa: 横轴故障时间和纵轴故障比例的原始扫描网格。
    values: 要画成颜色的二维数组；不可达或不适用的位置应为 NaN。
    title: 子图下方的说明文字。
    colorbar_label: 色条标签，说明颜色代表的物理量和单位。
    cmap: matplotlib 色图名称，例如 "magma"、"plasma"、"RdYlBu_r"。
    boundary_strategy: 用哪条 boundary scan 曲线裁剪/叠加；多策略时表示公共可达域。
    clip_percentile: 色条上限分位数截断，默认 95 分位，减少极端值影响。
    use_power_norm: True 时使用非线性色标，增强低值区颜色差异。
    """
    values = np.asarray(values, dtype=float)
    ax.set_facecolor(UNREACHABLE_FACE_COLOR)
    if not np.any(np.isfinite(values)):
        ax.text(
            0.5,
            0.5,
            "No reachable cases",
            ha="center",
            va="center",
            fontsize=label_fontsize,
            transform=ax.transAxes,
        )
        _style_axis(
            ax,
            r"Failure time $t_f$ (s)",
            r"$\kappa$",
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
        )
        _caption_axis(ax, title, fontsize=caption_fontsize)
        return None

    vmin, vmax = _heat_limits(values, clip_percentile)
    norm = _heat_norm(values, vmin, vmax, cmap, use_power_norm)
    levels = _heat_levels(vmin, vmax, isinstance(norm, PowerNorm))
    TE, K, display_values = _interpolated_heatmap_display_grid(te, kappa, values, boundary_strategy)
    near_zero = np.isfinite(display_values) & (np.abs(display_values) < HEAT_ZERO_TOL)
    display_values = np.where(near_zero, 0.0, display_values)
    contour = ax.contourf(TE, K, display_values, levels=levels, cmap=cmap, norm=norm, extend="max")
    _overlay_unreachable_hatch(ax, TE, K, display_values)
    ax.contour(TE, K, display_values, levels=HEATMAP_CONTOUR_LEVELS, colors="black", linewidths=0.28, alpha=0.45)
    _overlay_grid_points(ax, te, kappa, values)
    _overlay_boundary_lines(ax, boundary_strategy)
    _style_axis(
        ax,
        r"Failure time $t_f$ (s)",
        r"$\kappa$",
        grid_alpha=0.25,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
    )
    _caption_axis(ax, title, fontsize=caption_fontsize)
    cbar = plt.colorbar(contour, ax=ax, pad=colorbar_pad)
    cbar.set_label(colorbar_label, fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    if colorbar_tick_format is not None:
        cbar.formatter = FormatStrFormatter(colorbar_tick_format)
        cbar.update_ticks()
    return contour


def plot_strategy_reachability(scan_or_path, output="strategy_reachability.png", show=True):
    """画 S0/S1/S2/S3 的可达域二值对比。

    scan_or_path: 可以是已经加载的 scan dict，也可以是 npz 文件路径。
    output: 输出到 figures 目录下的图片文件名。
    show: True 时保存后显示窗口，False 时只保存。
    """
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
            boundary_strategy=strategy_name,
        )
    for ax in axes[n:]:
        ax.axis("off")

    return _finish_figure(fig, FIGURES_DIR / output, show)


def plot_stage1_compensation(scan_or_path, output="stage1_compensation_eta.png", show=True):
    """画一级补偿利用率 eta1，主要用于比较 S1 和 S3 怎样使用一级剩余燃烧时间。"""
    scan = load_scan_npz(scan_or_path) if not isinstance(scan_or_path, dict) else scan_or_path
    te = scan["te"]
    kappa = scan["kappa"]
    available = [str(item) for item in scan["strategies"]]
    names = [name for name in ["S1", "S3"] if name in available]
    if not names:
        raise ValueError("Scan does not contain S1 or S3.")

    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 4.8), squeeze=False)
    axes = axes.ravel()
    panel_labels = {"S1": "(a)", "S3": "(b)"}
    for ax, name in zip(axes, names):
        idx = _strategy_index(scan, name)
        values = np.where(scan["reachable"][idx], scan["eta1"][idx], np.nan)
        _plot_heat(
            ax,
            te,
            kappa,
            values,
            rf"{panel_labels.get(name, '')} {name}: Stage-1 utilization $\eta_1$".strip(),
            r"$\eta_1$",
            cmap="plasma",
            boundary_strategy=name,
            colorbar_tick_format="%.2f",
        )
    return _finish_figure(fig, FIGURES_DIR / output, show, w_pad=0.35)


def plot_stage1_delta(scan_or_path, output="stage1_compensation_deltaT1.png", show=True):
    """画一级关机/分离时间变化量 Delta T1，主要用于解释 S3 的一级时序调整。"""
    scan = load_scan_npz(scan_or_path) if not isinstance(scan_or_path, dict) else scan_or_path
    te = scan["te"]
    kappa = scan["kappa"]
    available = [str(item) for item in scan["strategies"]]
    names = [name for name in ["S1", "S3"] if name in available]
    if not names:
        raise ValueError("Scan does not contain S1 or S3.")

    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 4.8), squeeze=False)
    axes = axes.ravel()
    panel_labels = {"S1": "(a)", "S3": "(b)"}
    for ax, name in zip(axes, names):
        idx = _strategy_index(scan, name)
        values = np.where(scan["reachable"][idx], scan["delta_T1"][idx], np.nan)
        has_negative_adjustment = np.nanmin(values) < -HEAT_ZERO_TOL if np.any(np.isfinite(values)) else False
        _plot_heat(
            ax,
            te,
            kappa,
            values,
            rf"{panel_labels.get(name, '')} {name}: Stage-1 timing adjustment".strip(),
            r"$\Delta T_1$ (s)",
            cmap="RdYlBu_r" if has_negative_adjustment else "YlOrRd",
            boundary_strategy=name,
            use_power_norm=False,
            colorbar_tick_format="%.2f",
        )
    return _finish_figure(fig, FIGURES_DIR / output, show, w_pad=0.35)


def plot_t4_compensation(scan_or_path, output="t4_compensation_delta.png", show=True):
    """画二级 T4 补偿量 Delta T4，主要用于比较 S2 和 S3 的二级补偿需求。"""
    scan = load_scan_npz(scan_or_path) if not isinstance(scan_or_path, dict) else scan_or_path
    te = scan["te"]
    kappa = scan["kappa"]
    available = [str(item) for item in scan["strategies"]]
    names = [name for name in ["S2", "S3"] if name in available]
    if not names:
        raise ValueError("Scan does not contain S2 or S3.")

    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 4.8), squeeze=False)
    axes = axes.ravel()
    panel_labels = {"S2": "(a)", "S3": "(b)"}
    for ax, name in zip(axes, names):
        idx = _strategy_index(scan, name)
        values = np.where(scan["reachable"][idx], scan["delta_T4"][idx], np.nan)
        _plot_heat(
            ax,
            te,
            kappa,
            values,
            rf"{panel_labels.get(name, '')} {name}: Stage-2 timing adjustment".strip(),
            r"$\Delta T_4$ (s)",
            boundary_strategy=name,
            colorbar_tick_format="%.2f",
        )
    return _finish_figure(fig, FIGURES_DIR / output, show, w_pad=0.0)


def plot_t4_save(scan_or_path, output="t4_save_S2_minus_S3.png", show=True):
    """画 S2 与 S3 的二级补偿差值 Delta T4(S2)-Delta T4(S3)。

    只在 S2 和 S3 都可达的公共区域内计算；正值表示 S3 比 S2 节省二级补偿时间。
    """
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
        r"S3 vs. S2: Stage-2 time saving",
        r"$\Delta T_4^{\mathrm{save}}$ (s)",
        cmap="coolwarm",
        boundary_strategy=["S2", "S3"],
        colorbar_tick_format="%.2f",
        label_fontsize=LABEL_FONTSIZE - 5,
        tick_fontsize=TICK_FONTSIZE - 5,
        caption_fontsize=CAPTION_FONTSIZE - 5,
        colorbar_fontsize=COLORBAR_FONTSIZE - 5,
    )
    return _finish_figure(fig, FIGURES_DIR / output, show)


def plot_terminal_relaxation(scan_or_path, output="terminal_relaxation_reachability.png", show=True):
    """画终端约束放松层级 L0-L3 的可达域二值对比。"""
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
