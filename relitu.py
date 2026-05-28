from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from core.env_models import EarthEnv
from core.rocket_stage import Rocket
from core.utils import interpolate_solution, rk4_step


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CHUCAI = "biaozhundandao.npz"

# 热力图定义为相对于标准工况 T4 的额外时长
T4_BASELINE = 240.3031

# 故障模式
# 1: 推力下降，一级工作时间不变（mdot 不变）
# 2: 推力和秒耗同比下降，一级工作时间延长
FAULT_MODE = 1

# 参考步长
dt = 1.0


def _resolve_guess_path(npz_name_or_path: str | Path) -> Path:
    raw = Path(npz_name_or_path)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates = [BASE_DIR / raw, RESULTS_DIR / raw.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"未找到初猜文件: {npz_name_or_path}\n已尝试:\n{checked}")


def _fault_stage1_dynamics_symbolic(t, x, u, env: EarthEnv, thrust_expr, mdot_expr):
    r = x[0:3]
    v = x[3:6]
    m = x[6]

    phi = u[0]
    psi = u[1]
    gamma = 0.0

    GB = ca.vertcat(
        ca.horzcat(
            ca.cos(phi) * ca.cos(psi),
            ca.cos(phi) * ca.sin(psi) * ca.sin(gamma) - ca.sin(phi) * ca.cos(gamma),
            ca.cos(phi) * ca.sin(psi) * ca.cos(gamma) + ca.sin(phi) * ca.sin(gamma),
        ),
        ca.horzcat(
            ca.sin(phi) * ca.cos(psi),
            ca.sin(phi) * ca.sin(psi) * ca.sin(gamma) + ca.cos(phi) * ca.cos(gamma),
            ca.sin(phi) * ca.sin(psi) * ca.cos(gamma) - ca.cos(phi) * ca.sin(gamma),
        ),
        ca.horzcat(
            -ca.sin(psi),
            ca.sin(phi) * ca.sin(psi) * ca.cos(gamma) - ca.cos(phi) * ca.sin(gamma),
            ca.cos(psi) * ca.cos(gamma),
        ),
    )

    g = env.gravity(r)
    _, _, h = env.llh(r, t)
    rho = env.atmosphere(h)

    omega_vec = ca.DM(env.omega_e_faguan)
    r_launch = ca.DM(env.R_fashe)
    v_rel = v - ca.cross(omega_vec, r + r_launch)
    vr = ca.sqrt(ca.dot(v_rel, v_rel) + 1e-6)

    q = 0.5 * rho * vr**2
    v_hat = v_rel / vr
    drag = -env.Cd * q * env.S * v_hat

    u_earth = GB @ ca.DM([1.0, 0.0, 0.0])
    a = (thrust_expr / m) * u_earth + g + drag / m
    return ca.vertcat(v, a, -mdot_expr)


def solve_t4(te_val: float, kappa_val: float, chucai_path: str | Path = CHUCAI):
    """返回 (T4_opt, Delta_T4)，若失败则返回 (np.nan, np.nan)。"""
    if not (0.0 <= kappa_val < 1.0):
        return np.nan, np.nan

    opti = ca.Opti()
    env = EarthEnv()

    try:
        data = np.load(_resolve_guess_path(chucai_path))
        X1_saved, X3_saved, X4_saved = data["X1"], data["X3"], data["X4"]
        U1_saved, U3_saved, U4_saved = data["U1"], data["U3"], data["U4"]
        t1_saved, t3_saved, t4_saved = data["t1"], data["t3"], data["t4"]
    except Exception as exc:
        print(f"  [ERROR] 初猜加载失败: {exc}")
        return np.nan, np.nan

    # 故障模式参数
    if FAULT_MODE == 1:
        p1_fault = env.P1 * (1.0 - kappa_val)
        mdot1_fault = env.mdot1
        t_yiji_used = env.t_yiji
    elif FAULT_MODE == 2:
        p1_fault = env.P1 * (1.0 - kappa_val)
        mdot1_fault = env.mdot1 * (1.0 - kappa_val)
        t_yiji_used = te_val + (env.t_yiji - te_val) / (1.0 - kappa_val)
    else:
        print(f"  [ERROR] 未知 FAULT_MODE={FAULT_MODE}")
        return np.nan, np.nan

    T1 = te_val
    T2 = t_yiji_used - te_val
    T3 = env.t_zhengliu
    T4_guess = 235.0

    N1 = int(T1 / dt)
    N2 = int(T2 / dt)
    N3 = int(T3 / dt)
    N4 = int(T4_guess / dt)

    if T2 <= 0.0 or N1 <= 0 or N2 <= 0 or N3 <= 0 or N4 <= 0:
        return np.nan, np.nan

    t1_duration = T1
    t2_duration = T2
    t3_duration = T3
    t4_guess_scalar = T4_guess

    dt2 = t2_duration / N2
    dt3 = t3_duration / N3

    # 为当前工况构造初猜网格
    t_1 = np.linspace(0.0, t1_duration, N1 + 1)
    t_2 = np.linspace(t1_duration, t1_duration + t2_duration, N2 + 1)
    t_3 = np.linspace(t1_duration + t2_duration, t1_duration + t2_duration + t3_duration, N3 + 1)
    t_4_guess = np.linspace(
        t1_duration + t2_duration + t3_duration,
        t1_duration + t2_duration + t3_duration + t4_guess_scalar,
        N4 + 1,
    )

    # 插值初猜
    X_cha_1 = interpolate_solution(t_1, t1_saved, X1_saved)
    X_cha_2 = interpolate_solution(t_2, t1_saved, X1_saved)
    X_cha_3 = interpolate_solution(t_3, t3_saved, X3_saved)
    X_cha_4 = interpolate_solution(t_4_guess, t4_saved, X4_saved)

    U_cha_1 = interpolate_solution(t_1, t1_saved, U1_saved)
    U_cha_2 = interpolate_solution(t_2, t1_saved, U1_saved)
    U_cha_3 = interpolate_solution(t_3, t3_saved, U3_saved)
    U_cha_4 = interpolate_solution(t_4_guess, t4_saved, U4_saved)

    # 优化变量
    T4 = opti.variable()
    opti.subject_to(opti.bounded(200.0, T4, 400.0))
    opti.set_initial(T4, t4_guess_scalar)

    dt4 = T4 / N4

    X1 = ca.DM(X_cha_1)
    U1 = ca.DM(U_cha_1)

    X2 = opti.variable(7, N2 + 1)
    U2 = opti.variable(2, N2 + 1)
    X3 = opti.variable(7, N3 + 1)
    U3 = opti.variable(2, N3 + 1)
    X4 = opti.variable(7, N4 + 1)
    U4 = opti.variable(2, N4 + 1)

    opti.set_initial(X2, X_cha_2)
    opti.set_initial(X3, X_cha_3)
    opti.set_initial(X4, X_cha_4)
    opti.set_initial(U2, U_cha_2)
    opti.set_initial(U3, U_cha_3)
    opti.set_initial(U4, U_cha_4)

    # 控制约束
    dphi_max = dt * 0.8 * np.pi / 180.0
    for k in range(N2):
        opti.subject_to(U2[0, k + 1] - U2[0, k] <= dphi_max)
        opti.subject_to(U2[0, k] - U2[0, k + 1] <= dphi_max)

    opti.subject_to(X4[6, :] >= env.m_gan)

    opti.subject_to(opti.bounded(-60 / 180.0 * np.pi, U2[0, :], 90 / 180.0 * np.pi))
    opti.subject_to(opti.bounded(-60 / 180.0 * np.pi, U3[0, :], 90 / 180.0 * np.pi))
    opti.subject_to(opti.bounded(-60 / 180.0 * np.pi, U4[0, :], 90 / 180.0 * np.pi))
    opti.subject_to(opti.bounded(-5 / 180.0 * np.pi, U2[1, :], 3 / 180.0 * np.pi))
    opti.subject_to(opti.bounded(-6 / 180.0 * np.pi, U3[1, :], 2 / 180.0 * np.pi))
    opti.subject_to(opti.bounded(-7 / 180.0 * np.pi, U4[1, :], 1 / 180.0 * np.pi))

    # 动力学约束
    ode2 = lambda t, x, u: _fault_stage1_dynamics_symbolic(t, x, u, env, p1_fault, mdot1_fault)
    stage2 = Rocket(thrust=env.P2, mdot=env.mdot2, name="Stage-2")
    ode_stage2 = lambda t, x, u: stage2.dynamics(t, x, u, env)

    for k in range(N2):
        opti.subject_to(
            X2[:, k + 1] == rk4_step(ode2, t_2[k], X2[:, k], U2[:, k], U2[:, k + 1], dt2)
        )

    for k in range(N3):
        opti.subject_to(
            X3[:, k + 1] == rk4_step(ode_stage2, t_3[k], X3[:, k], U3[:, k], U3[:, k + 1], dt3)
        )

    for k in range(N4):
        t_current = t1_duration + t2_duration + t3_duration + k * dt4
        opti.subject_to(
            X4[:, k + 1]
            == rk4_step(ode_stage2, t_current, X4[:, k], U4[:, k], U4[:, k + 1], dt4)
        )

    # 段间连接
    opti.subject_to(X2[:, 0] == X1[:, -1])
    opti.subject_to(U2[:, 0] == U1[:, -1])

    opti.subject_to(X3[0:6, 0] == X2[0:6, -1])
    opti.subject_to(X3[6, 0] == X2[6, -1] - env.m_pao)
    opti.subject_to(U3[:, 0] == U2[:, -1])

    opti.subject_to(X4[0:6, 0] == X3[0:6, -1])
    opti.subject_to(X4[6, 0] == X3[6, -1] - env.m_zhengliu)
    opti.subject_to(U4[:, 0] == U3[:, -1])

    # 终端轨道约束
    r_final = X4[0:3, -1]
    v_final = X4[3:6, -1]
    r_eci_final, v_eci_final = env.ECI(r_final, v_final)
    a_fin, e_fin, i_fin, Omega_fin, omega_fin, f_fin = env.orbit_six(r_eci_final, v_eci_final)

    a_t = env.target["a"]
    e_t = env.target["e"]
    i_t = env.target["i_deg"]
    Omega_t = env.target["Omega_deg"]
    omega_t = env.target["omega_deg"]
    f_t = env.target["f_deg"]

    tol_a = 1e-5
    tol_e = 1e-5
    tol_i = 1e-5
    tol_O = 1e-5
    tol_w = 1e-5
    tol_f = 1e-5

    opti.subject_to(opti.bounded(-tol_a / a_t, (a_fin - a_t) / a_t, tol_a / a_t))
    opti.subject_to(opti.bounded(-tol_e / e_t, (e_fin - e_t) / e_t, tol_e / e_t))
    opti.subject_to(opti.bounded(-tol_i / i_t, (i_fin - i_t) / i_t, tol_i / i_t))

    err_O = env.wrap_angle_deg(Omega_fin - Omega_t)
    err_w = env.wrap_angle_deg(omega_fin - omega_t)
    err_f = env.wrap_angle_deg(f_fin - f_t)

    opti.subject_to(opti.bounded(-tol_O, err_O, tol_O))
    opti.subject_to(opti.bounded(-tol_w, err_w, tol_w))
    opti.subject_to(opti.bounded(-tol_f, err_f, tol_f))

    # 目标函数
    w_ctrl = 1000.0
    smoothness = (
        ca.sumsqr(ca.diff(ca.diff(U2, 1, 1), 1, 1))
        + ca.sumsqr(ca.diff(ca.diff(U3, 1, 1), 1, 1))
        + ca.sumsqr(ca.diff(ca.diff(U4, 1, 1), 1, 1))
    )
    opti.minimize(T4 + w_ctrl * smoothness)

    opti.solver(
        "ipopt",
        {"expand": True},
        {
            "linear_solver": "mumps",
            "mumps_mem_percent": 4000,
            "mumps_pivtol": 1e-6,
            "mumps_pivtolmax": 1e-4,
            "tol": 1e-8,
            "acceptable_tol": 1e-6,
            "max_iter": 3000,
        },
    )

    try:
        sol = opti.solve()
        t4_opt = float(sol.value(T4))
        delta_t4 = t4_opt - T4_BASELINE
        return t4_opt, delta_t4
    except Exception:
        return np.nan, np.nan


def plot_heatmap(npz_path):
    """绘制热力图"""

    npz_file = Path(npz_path)
    if not npz_file.is_absolute():
        candidates = [
            npz_file,
            BASE_DIR / npz_file,
            RESULTS_DIR / npz_file,
            RESULTS_DIR / npz_file.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                npz_file = candidate
                break

    if not npz_file.exists():
        print(f"未找到 npz 文件: {npz_file}")
        return

    try:
        data = np.load(npz_file)
        te_list = data["te"]
        kappa_list = data["kappa"]
        delta_t4_matrix = data["Delta_T4"]
    except KeyError as exc:
        print(f"npz 文件缺少必要字段: {exc}")
        return
    except Exception as exc:
        print(f"读取 npz 失败: {exc}")
        return

    if te_list is None or kappa_list is None or delta_t4_matrix is None:
        print("绘图输入不完整：请提供(te_list, kappa_list, delta_t4_matrix)或npz_path。")
        return

    TE, KAPPA = np.meshgrid(te_list, kappa_list)

    finite_mask = np.isfinite(delta_t4_matrix)
    if not np.any(finite_mask):
        print("无可用数据，跳过绘图。")
        return

    plt.figure(figsize=(10, 7))

    vmin = np.nanmin(delta_t4_matrix)
    vmax = np.nanmax(delta_t4_matrix)
    levels = np.linspace(vmin, vmax, 50)

    contour = plt.contourf(TE, KAPPA, delta_t4_matrix, levels=levels, cmap="jet")
    cbar = plt.colorbar(contour)
    cbar.set_label(r"Additional $T_4$ Burn Time $\Delta T_4$ (s)", fontsize=12)

    contour_lines = plt.contour(
        TE,
        KAPPA,
        delta_t4_matrix,
        levels=15,
        colors="black",
        linewidths=0.5,
    )
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.1f")

    plt.gca().set_facecolor("lightgray")
    plt.xlabel('Failure Time $t_f$ (s)', fontsize=12)
    plt.ylabel(r'Thrust Loss Ratio $\kappa$', fontsize=12) # 对应你之前的定义
    plt.title(r'Required $T_4$ Compensation for Mission Recovery', fontsize=14)

    plt.xticks(np.arange(50, 191, 10))
    plt.yticks([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_png = RESULTS_DIR / "relitu.png"
    plt.savefig(output_png, dpi=300)
    print(f"热力图已保存: {output_png}")
    plt.show()

def main():
    te_list = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    kappa_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    t4_matrix = np.full((len(kappa_list), len(te_list)), np.nan)
    delta_t4_matrix = np.full((len(kappa_list), len(te_list)), np.nan)

    total_cases = len(te_list) * len(kappa_list)
    print(f"开始扫描，共 {total_cases} 个工况；FAULT_MODE={FAULT_MODE}")

    for i, kappa in enumerate(kappa_list):
        for j, te in enumerate(te_list):
            print(f"正在计算 te={te}s, kappa={kappa:.3f} ...", end=" ")
            t4_opt, delta_t4 = solve_t4(te, kappa)
            t4_matrix[i, j] = t4_opt
            delta_t4_matrix[i, j] = delta_t4

            if np.isnan(t4_opt):
                print("[失败/不可达]")
            else:
                print(f"[成功] T4={t4_opt:.3f}s, DeltaT4={delta_t4:.3f}s")

    output_npz = RESULTS_DIR / "relitu.npz"
    np.savez(
        output_npz,
        te=np.array(te_list),
        kappa=np.array(kappa_list),
        T4_opt=t4_matrix,
        Delta_T4=delta_t4_matrix,
        T4_baseline=T4_BASELINE,
        fault_mode=FAULT_MODE,
    )
    print(f"数据已保存: {output_npz}")

    plot_heatmap(output_npz)
    """ plot_heatmap("relitu.npz") """


if __name__ == "__main__":
    main()
