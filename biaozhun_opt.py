from dataclasses import dataclass, field
import os
from pathlib import Path
import numpy as np
import casadi as ca

# 包名.模块名
from core.env_models import EarthEnv # 火箭仿真环境参数，用到的llh、重力等函数
from core.rocket_stage import Rocket # 单级火箭动力学模型
from core.utils import interpolate_solution # 插值函数
from core.ocp_blocks import (
    apply_control_angle_bounds,
    apply_dphi_rate_limit,
    add_rk4_segment_constraints,
    apply_mass_lower_bound,
    configure_ipopt_solver,
)
from core.visual import plot_from_npz # 画图函数

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INIT_GUESS_FILE = str(RESULTS_DIR / "biaozhundandao.npz")
RESULT_FILE = str(RESULTS_DIR / "biaozhundandao_new.npz")
COMPARE_FILE = str(RESULTS_DIR / "biaozhundandao.npz")
# 是否启用一级程序角 phi 的变化率限制
ENABLE_DPHI_LIMIT = True
# 是否给控制量加平滑项，抑制姿态突变
ENABLE_SMOOTHNESS = True

@dataclass
class OptimizeConfig:
	# 时间离散步长，决定状态和控制的网格密度。
	dt: float = 1.0
	# 第四段时间的初始猜测值，优化会围绕这个值调整。
	T3_guess: float = 239.0

	# 约束
	# 第四段时间下界，避免优化出过短的不合理方案。
	T3_min: float = 200.0
	# 第四段时间上界，限制搜索范围。
	T3_max: float = 400.0
	# 一级程序角每一步允许变化的最大幅度，单位是“度/步”的参数形式。
	dphi_max_deg_per_step: float = 0.8

	# 目标与容差
	# 目标轨道六要素，和环境类中的 target 保持一致。
	target: dict = field(        # 每次实例化 独立 互不影响
		default_factory=lambda: {
			"a_km": 6778.0,
			"e": 0.01,
			"i_deg": 27.57,
			"Omega_deg": 335.0,
			"f_deg": 0.0,
			"omega_deg": 148.0,
		}
	)
	# 轨道根数的容差，越小代表末端轨道约束越严格。
	tolerances: dict = field(
		default_factory=lambda: {
			"a": 1e-5,
			"e": 1e-5,
			"i": 1e-5,
			"Omega": 1e-5,
			"omega": 1e-5,
			"f": 1e-5,
		}
	)

	# 权重与求解器
	# 控制平滑项权重，越大越偏向控制平顺。
	w_ctrl: float = 100000.0

	# IPOPT 求解器参数集中放这里，后续调参更方便。
	solver_opts: dict = field(
		default_factory=lambda: {
			"expand": True,
			"linear_solver": "mumps",
			"mumps_mem_percent": 4000,
			"mumps_pivtol": 1e-6,
			"mumps_pivtolmax": 1e-4,
			"tol": 1e-8,
			"acceptable_tol": 1e-6,
			"max_iter": 3000,
		}
	)

def build_and_solve(cfg: OptimizeConfig, make_plot=False):
    """构建并求解三段式入轨优化问题。

    流程是：先准备环境和动力学，再读初猜、建变量、加约束、设目标，
    最后用 IPOPT 求解并保存结果，必要时直接画图。
    """
    env = EarthEnv(target=cfg.target)
    # 一级和二级动力学对象，推力和质量流率都从 env 里统一取。
    stage1 = Rocket(thrust=env.P1, mdot=env.mdot1, name="Stage-1", Cd=env.Cd, S=env.S)
    stage2 = Rocket(thrust=env.P2, mdot=env.mdot2, name="Stage-2", Cd=env.Cd, S=env.S)

    dt = cfg.dt
    T1 = env.t_yiji
    T2 = env.t_zhengliu
    N1 = int(T1 / dt)
    N2 = int(T2 / dt)
    N3 = int(cfg.T3_guess / dt)

    # Opti 是 CasADi 提供的非线性规划容器。
    opti = ca.Opti()

    # 第四段持续时间作为优化变量，由求解器自动调整。
    T3 = opti.variable()
    opti.subject_to(T3 >= cfg.T3_min)
    opti.subject_to(T3 <= cfg.T3_max)
    opti.set_initial(T3, cfg.T3_guess)

    dt_4 = T3 / N3

    # 状态统一采用 [r(3), v(3), m]，控制统一采用 [phi, psi]。
    X2 = opti.variable(7, N1 + 1)
    U2 = opti.variable(2, N1 + 1)
    X3 = opti.variable(7, N2 + 1)
    U3 = opti.variable(2, N2 + 1)
    X4 = opti.variable(7, N3 + 1)
    U4 = opti.variable(2, N3 + 1)

    t_2 = np.linspace(0.0, T1, N1 + 1)
    t_3 = np.linspace(T1, T1 + T2, N2 + 1)
    t_4_guess = np.linspace(T1 + T2, T1 + T2 + cfg.T3_guess, N3 + 1)

    # 载入初猜（如果存在）
    # 好的初猜通常能明显提高求解速度和成功率。
    if os.path.exists(INIT_GUESS_FILE):
        print("chucai yes")
        data = np.load(INIT_GUESS_FILE)
        X_cha_2 = interpolate_solution(t_2, data["t1"], data["X1"])
        X_cha_3 = interpolate_solution(t_3, data["t3"], data["X3"])
        X_cha_4 = interpolate_solution(t_4_guess, data["t4"], data["X4"])

        U_cha_2 = interpolate_solution(t_2, data["t1"], data["U1"])
        U_cha_3 = interpolate_solution(t_3, data["t3"], data["U3"])
        U_cha_4 = interpolate_solution(t_4_guess, data["t4"], data["U4"])
    else:
        # 没有历史结果时，就退化成常值状态和零控制的粗糙初值。
        print("chucai no")
        X_cha_2 = np.tile(env.y0.reshape(-1, 1), (1, N1 + 1))
        X_cha_3 = np.tile(env.y0.reshape(-1, 1), (1, N2 + 1))
        X_cha_4 = np.tile(env.y0.reshape(-1, 1), (1, N3 + 1))
        U_cha_2 = np.zeros((2, N1 + 1))
        U_cha_3 = np.zeros((2, N2 + 1))
        U_cha_4 = np.zeros((2, N3 + 1))

    opti.set_initial(X2, X_cha_2)
    opti.set_initial(X3, X_cha_3)
    opti.set_initial(X4, X_cha_4)
    opti.set_initial(U2, U_cha_2)
    opti.set_initial(U3, U_cha_3)
    opti.set_initial(U4, U_cha_4)

    # 控制约束
    # 一级程序角变化率约束，抑制姿态突变。
    if ENABLE_DPHI_LIMIT:
        apply_dphi_rate_limit(opti, U2, dt, cfg.dphi_max_deg_per_step)

    apply_mass_lower_bound(opti, X4, env.m_gan)

    # 控制角上下界，代表火箭姿态的可实现范围。
    apply_control_angle_bounds(
        opti,
        [
            (U2, -60, 90, -5, 3),
            (U3, -60, 90, -6, 2),
            (U4, -60, 90, -7, 1),
        ],
    )

    # 动力学离散
    # 用 RK4 把连续动力学离散成每个网格点上的等式约束。
    ode1 = lambda t, x, u: stage1.dynamics(t, x, u, env)
    add_rk4_segment_constraints(opti, X2, U2, N1, ode1, dt, t_grid=t_2)

    ode2 = lambda t, x, u: stage2.dynamics(t, x, u, env)
    add_rk4_segment_constraints(opti, X3, U3, N2, ode2, dt, t_grid=t_3)

    ode3 = lambda t, x, u: stage2.dynamics(t, x, u, env)
    add_rk4_segment_constraints(opti, X4, U4, N3, ode3, dt_4, t0=T1 + T2)

    # 边界条件
    # 初始状态：位置原点、速度来自地球自转、质量为 m01。
    opti.subject_to(X2[0:3, 0] == ca.DM(env.r0))
    opti.subject_to(X2[3:6, 0] == ca.DM(env.v0))
    opti.subject_to(X2[6, 0] == env.m01)

    k_vert = int(16.0 / dt)
    # 前 16 秒强制垂直上升，对应发射初段的姿态保持。
    for k in range(k_vert + 1):
        opti.subject_to(U2[0, k] == np.pi / 2)

    # 分段拼接：位置速度连续，质量在分离时扣除相应质量。
    opti.subject_to(X3[0:6, 0] == X2[0:6, -1])
    opti.subject_to(X3[6, 0] == X2[6, -1] - env.m_pao)
    opti.subject_to(U3[:, 0] == U2[:, -1])

    opti.subject_to(X4[0:6, 0] == X3[0:6, -1])
    opti.subject_to(X4[6, 0] == X3[6, -1] - env.m_zhengliu)
    opti.subject_to(U4[:, 0] == U3[:, -1])

    # 末端轨道约束
    # 把末端状态转成 ECI，再计算轨道六根数与目标轨道对比。
    r_final = X4[0:3, -1]
    v_final = X4[3:6, -1]
    r_eci, v_eci = env.ECI(r_final, v_final)
    a_fin, e_fin, i_fin, O_fin, w_fin, f_fin = env.orbit_six(r_eci, v_eci)

    a_t = env.target["a"]
    e_t = env.target["e"]
    i_t = env.target["i_deg"]
    O_t = env.target["Omega_deg"]
    w_t = env.target["omega_deg"]
    f_t = env.target["f_deg"]

    tol = cfg.tolerances
    opti.subject_to(opti.bounded(-tol["a"] / a_t, (a_fin - a_t) / a_t, tol["a"] / a_t))
    opti.subject_to(opti.bounded(-tol["e"] / e_t, (e_fin - e_t) / e_t, tol["e"] / e_t))
    opti.subject_to(opti.bounded(-tol["i"] / i_t, (i_fin - i_t) / i_t, tol["i"] / i_t))

    err_O = env.wrap_angle_deg(O_fin - O_t)
    err_w = env.wrap_angle_deg(w_fin - w_t)
    err_f = env.wrap_angle_deg(f_fin - f_t)

    opti.subject_to(opti.bounded(-tol["Omega"], err_O, tol["Omega"]))
    opti.subject_to(opti.bounded(-tol["omega"], err_w, tol["omega"]))
    opti.subject_to(opti.bounded(-tol["f"], err_f, tol["f"]))

    # smoothness 越大，程序角越平滑，但求解也可能更困难。
    if ENABLE_SMOOTHNESS:
        smoothness = (ca.sumsqr(ca.diff(U2, 1, 1)) + ca.sumsqr(ca.diff(ca.diff(U2, 1, 1), 1, 1)) +
              ca.sumsqr(ca.diff(U3, 1, 1)) + ca.sumsqr(ca.diff(ca.diff(U3, 1, 1), 1, 1)) +
              ca.sumsqr(ca.diff(U4, 1, 1)) + ca.sumsqr(ca.diff(ca.diff(U4, 1, 1), 1, 1)))
        opti.minimize(T3 + cfg.w_ctrl * smoothness)
    else:
        opti.minimize(T3)

    # 求解器
    # IPOPT 适合这种稠密非线性约束问题，是 CasADi 常用搭配。
    configure_ipopt_solver(opti, cfg.solver_opts)

    print("开始求解 ...")
    # 如果这里失败，通常是初猜、约束或数值尺度需要调整。
    sol = opti.solve()

    T3_opt = float(sol.value(T3))
    t4_opt = np.linspace(T1 + T2, T1 + T2 + T3_opt, N3 + 1)

    X2_v = sol.value(X2)
    U2_v = sol.value(U2)
    X3_v = sol.value(X3)
    U3_v = sol.value(U3)
    X4_v = sol.value(X4)
    U4_v = sol.value(U4)

    np.savez(
        RESULT_FILE,
        X1=X2_v,
        U1=U2_v,
        X3=X3_v,
        U3=U3_v,
        X4=X4_v,
        U4=U4_v,
        t1=t_2,
        t3=t_3,
        t4=t4_opt,
    )

    print(f"求解完成，结果已保存: {os.path.abspath(RESULT_FILE)}")
    print(f"优化后 T4={T3_opt:.4f}s")

    if make_plot:
        # 画图函数会把刚保存的 npz 读取出来，生成轨迹和控制图。
        plot_from_npz(RESULT_FILE, env=env, compare_npz=COMPARE_FILE, label_current="Unlimited", label_compare="Limited", show=True)


if __name__ == "__main__":
    # 直接运行 main_opt.py 时，默认求解并自动画图。
    config = OptimizeConfig()
    build_and_solve(config, make_plot=True)
