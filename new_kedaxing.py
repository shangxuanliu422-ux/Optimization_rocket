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
dt = 1.0

# 与原脚本一致的离散设置
N2 = 50
N3 = 7
N4 = 240

TE_LIST = [95, 105, 115, 125, 135, 145, 155, 165, 170, 175, 180, 185, 190, 193, 195, 196, 197, 198, 199, 200]

def _resolve_guess_path(npz_name_or_path: str) -> Path:
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
    """一级故障段动力学：推力和秒耗都带 kappa 的符号表达式。"""
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


# ================= 核心：边界求解封装 =================
def solve_boundary(te_val, chucai_te: str | Path, kappa_guess=0.1):
    """求解给定故障时间下的最大可容忍推力下降比例（一级可延时）。"""
    opti = ca.Opti()
    env = EarthEnv()

    stage2 = Rocket(thrust=env.P2, mdot=env.mdot2, name="Stage-2", Cd=env.Cd, S=env.S)

    try:
        solution_data = np.load(_resolve_guess_path(str(chucai_te)))
        X1_saved, X3_saved, X4_saved = solution_data["X1"], solution_data["X3"], solution_data["X4"]
        U1_saved, U3_saved, U4_saved = solution_data["U1"], solution_data["U3"], solution_data["U4"]
        t1_saved, t3_saved, t4_saved = solution_data["t1"], solution_data["t3"], solution_data["t4"]
    except Exception:
        print("未找到初猜文件或格式不对！请检查初猜 npz")
        return None

    # 1) kappa 作为决策变量
    kappa = opti.variable()
    opti.subject_to(kappa >= 0.0)
    opti.subject_to(kappa <= 1.0)

    # 2) 故障段参数表达式（推力和秒耗都下降）
    P1_fault = env.P1 * (1 - kappa)
    mdot1_fault = env.mdot1 * (1 - kappa)

    # 3) 动态时间定义（一级可延长）
    T2_expr = (env.t_yiji - te_val) / (1 - kappa)
    T3 = env.t_zhengliu
    T4 = float(t4_saved[-1] - env.t_yiji - env.t_zhengliu)

    dt2_expr = T2_expr / N2
    dt3 = T3 / N3
    dt4 = T4 / N4

    # 获取故障瞬间状态
    X_start_fault = interpolate_solution([te_val], t1_saved, X1_saved)[:, 0]
    U_start_fault = interpolate_solution([te_val], t1_saved, U1_saved)[:, 0]

    # 根据 guess 生成初猜网格
    T2_guess = (env.t_yiji - te_val) / (1 - kappa_guess)
    t_2_guess = np.linspace(te_val, te_val + T2_guess, N2 + 1)
    t_3 = np.linspace(te_val + T2_guess, te_val + T2_guess + T3, N3 + 1)
    t_4 = np.linspace(te_val + T2_guess + T3, te_val + T2_guess + T3 + T4, N4 + 1)

    X_cha_2 = interpolate_solution(t_2_guess, t1_saved, X1_saved)
    X_cha_3 = interpolate_solution(t_3, t3_saved, X3_saved)
    X_cha_4 = interpolate_solution(t_4, t4_saved, X4_saved)

    U_cha_2 = interpolate_solution(t_2_guess, t1_saved, U1_saved)
    U_cha_3 = interpolate_solution(t_3, t3_saved, U3_saved)
    U_cha_4 = interpolate_solution(t_4, t4_saved, U4_saved)

    # 优化变量
    X2 = opti.variable(7, N2 + 1)
    U2 = opti.variable(2, N2 + 1)
    X3 = opti.variable(7, N3 + 1)
    U3 = opti.variable(2, N3 + 1)
    X4 = opti.variable(7, N4 + 1)
    U4 = opti.variable(2, N4 + 1)

    # 初猜
    opti.set_initial(kappa, kappa_guess)
    opti.set_initial(X2, X_cha_2)
    opti.set_initial(X3, X_cha_3)
    opti.set_initial(X4, X_cha_4)
    opti.set_initial(U2, U_cha_2)
    opti.set_initial(U3, U_cha_3)
    opti.set_initial(U4, U_cha_4)

    # 5) 约束条件
    dphi_max = dt * 0.8 * np.pi / 180.0
    for k in range(N2):
        opti.subject_to(opti.bounded(-dphi_max, U2[0, k + 1] - U2[0, k], dphi_max))

    opti.subject_to(opti.bounded(-60 / 180 * np.pi, U2[0, :], 90 / 180 * np.pi))
    opti.subject_to(opti.bounded(-60 / 180 * np.pi, U3[0, :], 90 / 180 * np.pi))
    opti.subject_to(opti.bounded(-60 / 180 * np.pi, U4[0, :], 90 / 180 * np.pi))
    opti.subject_to(opti.bounded(-5 / 180 * np.pi, U2[1, :], 3 / 180 * np.pi))
    opti.subject_to(opti.bounded(-6 / 180 * np.pi, U3[1, :], 2 / 180 * np.pi))
    opti.subject_to(opti.bounded(-7 / 180 * np.pi, U4[1, :], 1 / 180 * np.pi))
    opti.subject_to(X4[6, :] >= 0.0)

    # 6) 动力学积分
    opti.subject_to(X2[:, 0] == X_start_fault)
    opti.subject_to(U2[:, 0] == U_start_fault)

    t_start2 = te_val
    ode2_fault = lambda t, x, u: _fault_stage1_dynamics_symbolic(t, x, u, env, P1_fault, mdot1_fault)
    for k in range(N2):
        t_k = t_start2 + k * dt2_expr
        opti.subject_to(
            X2[:, k + 1] == rk4_step(ode2_fault, t_k, X2[:, k], U2[:, k], U2[:, k + 1], dt2_expr)
        )

    t_start3 = t_start2 + T2_expr
    ode3 = lambda t, x, u: stage2.dynamics(t, x, u, env)
    for k in range(N3):
        t_k = t_start3 + k * dt3
        opti.subject_to(
            X3[:, k + 1] == rk4_step(ode3, t_k, X3[:, k], U3[:, k], U3[:, k + 1], dt3)
        )

    t_start4 = t_start3 + T3
    ode4 = lambda t, x, u: stage2.dynamics(t, x, u, env)
    for k in range(N4):
        t_k = t_start4 + k * dt4
        opti.subject_to(
            X4[:, k + 1] == rk4_step(ode4, t_k, X4[:, k], U4[:, k], U4[:, k + 1], dt4)
        )

    # 段间分离跳变
    opti.subject_to(X3[0:6, 0] == X2[0:6, -1])
    opti.subject_to(X3[6, 0] == X2[6, -1] - env.m_pao)
    opti.subject_to(U3[:, 0] == U2[:, -1])

    opti.subject_to(X4[0:6, 0] == X3[0:6, -1])
    opti.subject_to(X4[6, 0] == X3[6, -1] - env.m_zhengliu)
    opti.subject_to(U4[:, 0] == U3[:, -1])

    # 7) 终端轨道约束
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

    err_O = ca.fmod(Omega_fin - Omega_t + 180.0, 360.0) - 180.0
    err_w = ca.fmod(omega_fin - omega_t + 180.0, 360.0) - 180.0
    err_f = ca.fmod(f_fin - f_t + 180.0, 360.0) - 180.0
    opti.subject_to(opti.bounded(-tol_O, err_O, tol_O))
    opti.subject_to(opti.bounded(-tol_w, err_w, tol_w))
    opti.subject_to(opti.bounded(-tol_f, err_f, tol_f))

    # 8) 目标函数
    objective = -kappa
    opti.minimize(objective)

    # 9) 求解器
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

    # 10) 求解与异常处理
    try:
        sol = opti.solve()
        print(f"--- te={te_val}s 求解成功! 最大 kappa={sol.value(kappa):.3f} ---")
        print("h_final =", opti.debug.value(r_final))
        print("v_final =", opti.debug.value(v_final))
        return sol.value(kappa)
    except Exception:
        k_fail = opti.debug.value(kappa)
        print(f"--- te={te_val}s 求解失败，最后停留 kappa={k_fail:.3f} ---")
        return k_fail


if __name__ == "__main__":
    kappa_max_list = []

    print("开始进行可达性边界扫描计算...")
    for te in TE_LIST:
        guess = 0
        k_max = solve_boundary(te, CHUCAI, kappa_guess=guess)
        kappa_max_list.append(k_max)

    print(kappa_max_list)

    # ======== 保存数据为 npz 文件 ========
    output_npz = RESULTS_DIR / "kedaxing111.npz"
    output_png = RESULTS_DIR / "kedaxing111.png"

    np.savez(output_npz, 
             te_list=np.array(TE_LIST), 
             kappa_max_list=np.array(kappa_max_list))
    print(f"数据已保存至 {output_npz}")

    # ======== 绘图 ========
    npzfile = np.load(output_npz)
    te_list = npzfile['te_list']
    kappa_max_list = npzfile['kappa_max_list']
    plt.figure(figsize=(12, 7))
    plt.plot(te_list, kappa_max_list, 'bo-', linewidth=2.5, markersize=3)

    plt.xlabel("Failure Time $t_f$ (s)", fontsize=15)
    plt.ylabel(r"Max Thrust Loss Ratio $\kappa_{max}$", fontsize=15)
    plt.title("Reachable Domain (Propellant Depletion Cutoff Mode)", fontsize=17, pad=20)

    plt.grid(True, linestyle="--", alpha=0.7)

    plt.fill_between(TE_LIST, 0, kappa_max_list, alpha=0.2, color="green", label="Reconfigurable")
    plt.fill_between(TE_LIST, kappa_max_list, 1, alpha=0.2, color="red", label="Unrecoverable")

    plt.ylim(0, 1)
    plt.xlim(95, 200)
    plt.xticks(range(95, 201, 5))

    for te, kappa in zip(TE_LIST, kappa_max_list):
        plt.text(
            te,
            kappa + 0.04,
            f"{kappa:.3f}",
            fontsize=8,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
        )

    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"图片已保存至 {output_png}")
    plt.show()
