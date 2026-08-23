from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

from core.env_models import EarthEnv
from core.ocp_blocks import rk4_step
from core.rocket_stage import Rocket
from core.visual import plot_from_npz


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class Phase(IntEnum):
    """飞行阶段。"""

    STAGE1 = 1
    STAGE1_FAULT = 2
    STAGE2 = 3


@dataclass
class ReplayConfig:
    """步长验证配置。"""

    # 初猜文件，既支持绝对路径，也支持工程根目录或 results 目录下的文件名。
    input_npz: str = "biaozhundandao.npz"
    # 回放积分步长，默认设为 0.1，用来验证更细步长下的结果。
    dt: float = 1

    # 故障设置：0 表示无故障，1 表示推力下降但一级时长不变，2 表示推力和秒耗都下降且一级时长延长。
    fault_mode: int = 0
    te: float = 197.0
    kappa: float = 0.01

    # 回放结果文件。
    replay_result_npz: str = "fault_replay_case.npz"
    # 对比文件，可为空。
    compare_npz: str | None = None


def _resolve_input_npz(npz_name_or_path: str) -> Path:
    """按常见位置解析初猜文件路径。"""
    raw = Path(npz_name_or_path)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates = [BASE_DIR / raw, RESULTS_DIR / raw.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"初猜文件不存在: {npz_name_or_path}\n已尝试:\n{checked}")


def _build_fault_stage1_params(env: EarthEnv, fault_mode: int, te: float, kappa: float):
    """根据故障模式计算一级推力、秒耗和一级结束时间。"""
    if fault_mode == 1:
        p1_fault = env.P1 * (1.0 - kappa)
        mdot1_fault = env.mdot1
        t_yiji_used = env.t_yiji
    elif fault_mode == 2:
        p1_fault = env.P1 * (1.0 - kappa)
        mdot1_fault = env.mdot1 * (1.0 - kappa)
        t_yiji_used = float(np.round(te + (env.t_yiji - te) / (1.0 - kappa), 1))
    else:
        p1_fault = env.P1
        mdot1_fault = env.mdot1
        t_yiji_used = env.t_yiji
    return p1_fault, mdot1_fault, t_yiji_used


def _build_control_lookup(npz_data: dict, cfg: ReplayConfig, t_yiji_used: float, env: EarthEnv):
    """把优化结果里的离散控制量整理成细步长回放可直接调用的控制函数。"""
    u1_saved = npz_data["U1"]
    u3_saved = npz_data["U3"]
    u4_saved = npz_data["U4"]
    t1_saved = np.asarray(npz_data["t1"], dtype=float)
    t3_saved = np.asarray(npz_data["t3"], dtype=float)
    t4_saved = np.asarray(npz_data["t4"], dtype=float)

    t_full = np.concatenate([t1_saved, t3_saved[1:], t4_saved[1:]])
    u_full = np.hstack([u1_saved, u3_saved[:, 1:], u4_saved[:, 1:]])

    def control_at(t):
        """返回时刻 t 的程序角。"""
        t_arr = np.asarray(t, dtype=float)
        phi = np.interp(t_arr, t_full, u_full[0])
        psi = np.interp(t_arr, t_full, u_full[1])
        if t_arr.ndim == 0:
            return np.array([phi, psi], dtype=float)
        return np.vstack([phi, psi])

    return control_at, (t1_saved, t3_saved, t4_saved), float(t4_saved[-1])


def _integrate_replay(
    env: EarthEnv,
    cfg: ReplayConfig,
    stage1: Rocket,
    stage1_fault: Rocket,
    stage2: Rocket,
    t_yiji_used: float,
    t_end: float,
    t_grids,
    control_at,
):
    """按故障事件回放动力学，并记录 history。"""

    def get_control(t: float):
        u = control_at(t)
        u = np.asarray(u, dtype=float).reshape(-1)
        return float(u[0]), float(u[1])

    def rhs(t: float, y: np.ndarray, phase: Phase):
        if phase == Phase.STAGE1:
            stage = stage1
        elif phase == Phase.STAGE1_FAULT:
            stage = stage1_fault
        elif phase == Phase.STAGE2:
            stage = stage2
        else:
            raise ValueError("未知飞行阶段")

        phi, psi = get_control(t)
        u = np.array([phi, psi], dtype=float)
        return stage.dynamics(t, y, u, env)

    def rk4_with_phase(t0: float, y0: np.ndarray, dt_step: float, phase_now: Phase):
        # 这里复用 core.ocp_blocks 的 RK4，只是把阶段固定住。
        ode = lambda tk, xk, _uk: rhs(tk, xk, phase_now)
        zero_u = np.zeros(2, dtype=float)
        return rk4_step(ode, t0, y0, zero_u, zero_u, dt_step)

    t1_grid, t3_grid, t4_grid = t_grids

    t_current = float(t1_grid[0])
    y_current = env.y0.copy()

    t_history = [t_current]
    y_history = [y_current.copy()]

    phase = Phase.STAGE1
    print("开始 RK4 回放积分...")

    def integrate_grid(t_grid: np.ndarray, phase_now: Phase):
        nonlocal t_current, y_current, phase
        for k in range(len(t_grid) - 1):
            if phase_now == Phase.STAGE1 and t_current >= cfg.te:
                phase_now = Phase.STAGE1_FAULT

            next_t = float(t_grid[k + 1])
            step_dt = next_t - t_current
            if step_dt <= 0.0:
                continue
            y_current = rk4_with_phase(t_current, y_current, step_dt, phase_now)
            t_current = next_t
            t_history.append(t_current)
            y_history.append(y_current.copy())
        return phase_now

    phase = integrate_grid(t1_grid, phase)

    # 一级分离，质量跳变。
    y_current[6] -= env.m_pao
    phase = Phase.STAGE2
    t_history.append(t_current)
    y_history.append(y_current.copy())

    phase = integrate_grid(t3_grid, phase)

    # 整流罩分离，质量跳变。
    y_current[6] -= env.m_zhengliu
    t_history.append(t_current)
    y_history.append(y_current.copy())

    phase = integrate_grid(t4_grid, phase)

    return np.asarray(t_history), np.asarray(y_history)


def _find_time_indices(t_history: np.ndarray, target_time: float, tol: float = 1e-9):
    """找出某个时刻在 history 里的所有索引。"""
    indices = np.where(np.isclose(t_history, target_time, atol=tol, rtol=0.0))[0]
    if len(indices) == 0:
        raise ValueError(f"时间点 {target_time} 没有落在 history 网格上，请调整 dt。")
    return indices


def _export_replay_npz(save_path: Path, t_history: np.ndarray, y_history: np.ndarray, control_at, t_grids, t_yiji_used: float, env: EarthEnv):
    """把回放结果按时间段直接切开，导出成 visual 能读取的 npz。"""
    t_1, t_3, t_4 = t_grids
    t_split_stage1 = t_yiji_used
    t_split_stage2 = t_yiji_used + env.t_zhengliu

    idx_stage1 = _find_time_indices(t_history, t_split_stage1)
    idx_stage2 = _find_time_indices(t_history, t_split_stage2)

    if len(idx_stage1) < 2:
        raise ValueError("一级分离时刻只出现了一次，无法区分分离前后状态。")
    if len(idx_stage2) < 2:
        raise ValueError("整流罩分离时刻只出现了一次，无法区分分离前后状态。")

    i1_end = int(idx_stage1[0])
    i2_start = int(idx_stage1[1])
    i2_end = int(idx_stage2[0])
    i3_start = int(idx_stage2[1])

    x_hist = y_history.T
    u_hist = np.asarray(control_at(t_history), dtype=float)

    x1 = x_hist[:, : i1_end + 1]
    u1 = u_hist[:, : i1_end + 1]
    t1_full = t_history[: i1_end + 1]

    x3 = x_hist[:, i2_start : i2_end + 1]
    u3 = u_hist[:, i2_start : i2_end + 1]
    t3_full = t_history[i2_start : i2_end + 1]

    x4 = x_hist[:, i3_start :]
    u4 = u_hist[:, i3_start :]
    t4_full = t_history[i3_start :]

    np.savez(
        save_path,
        X1=x1,
        U1=u1,
        X3=x3,
        U3=u3,
        X4=x4,
        U4=u4,
        t1=t1_full,
        t3=t3_full,
        t4=t4_full,
    )


def run_replay(cfg: ReplayConfig):
    """读取初猜控制，按细步长回放动力学，并输出给 visual 使用。"""
    env = EarthEnv()

    input_path = _resolve_input_npz(cfg.input_npz)
    data = np.load(input_path)

    required = ["X1", "X3", "X4", "U1", "U3", "U4", "t1", "t3", "t4"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"初猜文件缺少字段: {missing}")

    p1_fault, mdot1_fault, t_yiji_used = _build_fault_stage1_params(env, cfg.fault_mode, cfg.te, cfg.kappa)

    stage1 = Rocket(thrust=env.P1, mdot=env.mdot1, name="Stage-1", Cd=env.Cd, S=env.S)
    stage1_fault = Rocket(thrust=p1_fault, mdot=mdot1_fault, name="Stage-1-Fault", Cd=env.Cd, S=env.S)
    stage2 = Rocket(thrust=env.P2, mdot=env.mdot2, name="Stage-2", Cd=env.Cd, S=env.S)

    control_at, t_grids, t_end = _build_control_lookup(data, cfg, t_yiji_used, env)
    t_history, y_history = _integrate_replay(
        env=env,
        cfg=cfg,
        stage1=stage1,
        stage1_fault=stage1_fault,
        stage2=stage2,
        t_yiji_used=t_yiji_used,
        t_end=t_end,
        t_grids=t_grids,
        control_at=control_at,
    )

    out_raw = Path(cfg.replay_result_npz)
    save_path = out_raw if out_raw.is_absolute() else RESULTS_DIR / out_raw.name
    _export_replay_npz(save_path, t_history, y_history, control_at, t_grids, t_yiji_used, env)

    print(f"回放完成，结果已保存: {save_path}")

    compare_path = None
    if cfg.compare_npz:
        compare_path = str(_resolve_input_npz(cfg.compare_npz))

    plot_from_npz(
        str(save_path),
        env=env,
        compare_npz=compare_path,
        label_current="Replay",
        label_compare="Compare",
        show=True,
    )


if __name__ == "__main__":
    config = ReplayConfig()
    run_replay(config)
