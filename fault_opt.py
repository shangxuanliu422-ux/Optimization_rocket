from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np

from core.env_models import EarthEnv
from core.rocket_stage import Rocket
from core.utils import interpolate_solution
from core.ocp_blocks import (
	apply_control_angle_bounds,
	apply_dphi_rate_limit,
	add_rk4_segment_constraints,
	apply_mass_lower_bound,
	configure_ipopt_solver,
)
from core.visual import plot_from_npz


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INIT_GUESS_FILE = str(RESULTS_DIR / "biaozhundandao.npz")
RESULT_FILE = str(RESULTS_DIR / "fault_case.npz")
COMPARE_FILE = str(RESULTS_DIR / "biaozhundandao.npz")
# 是否启用一级程序角 phi 的变化率限制
ENABLE_DPHI_LIMIT = True
# 是否给控制量加平滑项，抑制姿态突变
ENABLE_SMOOTHNESS = True


@dataclass
class FaultOptimizeConfig:
	"""故障工况优化参数。"""

	dt: float = 1.0
	fault_mode: int = 2 # 0: 无故障, 1: 推力下降时长不变, 2: 推力+秒耗下降且一级延时
	te: float = 155.0
	kappa: float = 0.103

	dphi_max_deg_per_step: float = 0.8
	w_ctrl: float = 10.0

	target: dict = field(
		default_factory=lambda: {
			"a_km": 6778.0,
			"e": 0.01,
			"i_deg": 27.57,
			"Omega_deg": 335.0,
			"f_deg": 0.0,
			"omega_deg": 148.0,
		}
	)
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


def _fault_stage1_params(env: EarthEnv, fault_mode: int, te: float, kappa: float):
	"""根据故障模式给出一级故障后参数。"""
	if fault_mode == 0:
		return env.P1, env.mdot1, env.t_yiji

	if fault_mode == 1:
		return env.P1 * (1.0 - kappa), env.mdot1, env.t_yiji

	if fault_mode == 2:
		t_yiji_used = te + (env.t_yiji - te) / (1.0 - kappa)
		return env.P1 * (1.0 - kappa), env.mdot1 * (1.0 - kappa), t_yiji_used

	raise ValueError(f"不支持的 fault_mode: {fault_mode}")


def build_and_solve_fault(cfg: FaultOptimizeConfig, make_plot: bool = False):
	"""构建并求解故障工况下的入轨优化。"""
	env = EarthEnv(target=cfg.target)

	init_guess_path = Path(INIT_GUESS_FILE)
	if not init_guess_path.exists():
		raise FileNotFoundError(f"初猜文件不存在: {init_guess_path}")

	data = np.load(init_guess_path)
	required = ["X1", "U1", "X3", "U3", "X4", "U4", "t1", "t3", "t4"]
	missing = [k for k in required if k not in data]
	if missing:
		raise KeyError(f"初猜文件缺少字段: {missing}")

	P1_fault, mdot1_fault, t_yiji_used = _fault_stage1_params(
		env, cfg.fault_mode, cfg.te, cfg.kappa
	)
	stage1_fault = Rocket(thrust=P1_fault, mdot=mdot1_fault, name="Stage-1-Fault", Cd=env.Cd, S=env.S)
	stage2 = Rocket(thrust=env.P2, mdot=env.mdot2, name="Stage-2", Cd=env.Cd, S=env.S)

	dt = cfg.dt
	if not (0.0 < cfg.te < t_yiji_used):
		raise ValueError(f"故障时刻 te 必须在 (0, {t_yiji_used}) 内，当前 te={cfg.te}")

	T1 = float(cfg.te)
	T2 = float(t_yiji_used - cfg.te)
	T3 = float(env.t_zhengliu)
	T4_end_fixed = float(data["t4"][-1] + (t_yiji_used - env.t_yiji))
	T4_fixed = float(T4_end_fixed - (T1 + T2 + T3))

	N1 = max(1, int(np.round(T1 / dt)))
	N2 = max(1, int(np.round(T2 / dt)))
	N3 = max(1, int(np.round(T3 / dt)))
	N4 = max(1, int(np.round(T4_fixed / dt)))

	if min(N1, N2, N3, N4) < 1:
		raise ValueError("时间网格过粗或时长过短，导致某段网格点不足。")

	t_1 = np.linspace(0.0, T1, N1 + 1)
	t_2 = np.linspace(T1, T1 + T2, N2 + 1)
	t_3 = np.linspace(T1 + T2, T1 + T2 + T3, N3 + 1)
	t_4_fixed = np.linspace(T1 + T2 + T3, T1 + T2 + T3 + T4_fixed, N4 + 1)
	dt_2 = T2 / N2

	X1_ref = interpolate_solution(t_1, data["t1"], data["X1"])
	U1_ref = interpolate_solution(t_1, data["t1"], data["U1"])
	X2_guess = interpolate_solution(t_2, data["t1"], data["X1"])
	U2_guess = interpolate_solution(t_2, data["t1"], data["U1"])
	X3_guess = interpolate_solution(t_3, data["t3"], data["X3"])
	U3_guess = interpolate_solution(t_3, data["t3"], data["U3"])
	X4_guess = interpolate_solution(t_4_fixed, data["t4"], data["X4"])
	U4_guess = interpolate_solution(t_4_fixed, data["t4"], data["U4"])

	X1 = ca.DM(X1_ref)
	U1 = ca.DM(U1_ref)

	opti = ca.Opti()
	dt_4 = T4_fixed / N4

	X2 = opti.variable(7, N2 + 1)
	U2 = opti.variable(2, N2 + 1)
	X3 = opti.variable(7, N3 + 1)
	U3 = opti.variable(2, N3 + 1)
	X4 = opti.variable(7, N4 + 1)
	U4 = opti.variable(2, N4 + 1)

	opti.set_initial(X2, X2_guess)
	opti.set_initial(U2, U2_guess)
	opti.set_initial(X3, X3_guess)
	opti.set_initial(U3, U3_guess)
	opti.set_initial(X4, X4_guess)
	opti.set_initial(U4, U4_guess)

	apply_mass_lower_bound(opti, X4, env.m_gan)
	apply_control_angle_bounds(
		opti,
		[
			(U2, -60, 90, -5, 3),
			(U3, -60, 90, -6, 2),
			(U4, -60, 90, -7, 1),
		],
	)

	if ENABLE_DPHI_LIMIT:
		apply_dphi_rate_limit(opti, U2, dt_2, cfg.dphi_max_deg_per_step)

	ode2 = lambda t, x, u: stage1_fault.dynamics(t, x, u, env)
	add_rk4_segment_constraints(opti, X2, U2, N2, ode2, dt_2, t_grid=t_2)

	ode3 = lambda t, x, u: stage2.dynamics(t, x, u, env)
	add_rk4_segment_constraints(opti, X3, U3, N3, ode3, dt, t_grid=t_3)

	ode4 = lambda t, x, u: stage2.dynamics(t, x, u, env)
	add_rk4_segment_constraints(opti, X4, U4, N4, ode4, dt_4, t0=T1 + T2 + T3)

	opti.subject_to(X2[:, 0] == X1[:, -1])
	opti.subject_to(U2[:, 0] == U1[:, -1])

	opti.subject_to(X3[0:6, 0] == X2[0:6, -1])
	opti.subject_to(X3[6, 0] == X2[6, -1] - env.m_pao)
	opti.subject_to(U3[:, 0] == U2[:, -1])

	opti.subject_to(X4[0:6, 0] == X3[0:6, -1])
	opti.subject_to(X4[6, 0] == X3[6, -1] - env.m_zhengliu)
	opti.subject_to(U4[:, 0] == U3[:, -1])

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

	err_O = ca.fmod(O_fin - O_t + 180.0, 360.0) - 180.0
	err_w = ca.fmod(w_fin - w_t + 180.0, 360.0) - 180.0
	err_f = ca.fmod(f_fin - f_t + 180.0, 360.0) - 180.0
	opti.subject_to(opti.bounded(-tol["Omega"], err_O, tol["Omega"]))
	opti.subject_to(opti.bounded(-tol["omega"], err_w, tol["omega"]))
	opti.subject_to(opti.bounded(-tol["f"], err_f, tol["f"]))

	if ENABLE_SMOOTHNESS:
		smoothness = (
			ca.sumsqr(ca.diff(U2, 1, 1)) + ca.sumsqr(ca.diff(ca.diff(U2, 1, 1), 1, 1))
			+ ca.sumsqr(ca.diff(U3, 1, 1)) + ca.sumsqr(ca.diff(ca.diff(U3, 1, 1), 1, 1))
			+ ca.sumsqr(ca.diff(U4, 1, 1)) + ca.sumsqr(ca.diff(ca.diff(U4, 1, 1), 1, 1))
		)
		opti.minimize(cfg.w_ctrl * smoothness)
	else:
		opti.minimize(0)

	configure_ipopt_solver(opti, cfg.solver_opts)

	print("开始求解故障工况 ...")
	sol = opti.solve()

	t_4_opt = np.linspace(T1 + T2 + T3, T1 + T2 + T3 + T4_fixed, N4 + 1)

	X1_v = np.array(X1)
	U1_v = np.array(U1)
	X2_v = sol.value(X2)
	U2_v = sol.value(U2)
	X3_v = sol.value(X3)
	U3_v = sol.value(U3)
	X4_v = sol.value(X4)
	U4_v = sol.value(U4)

	X1_full = np.hstack([X1_v, X2_v[:, 1:]])
	U1_full = np.hstack([U1_v, U2_v[:, 1:]])
	t1_full = np.hstack([t_1, t_2[1:]])

	np.savez(
		RESULT_FILE,
		X1=X1_full,
		U1=U1_full,
		X3=X3_v,
		U3=U3_v,
		X4=X4_v,
		U4=U4_v,
		t1=t1_full,
		t3=t_3,
		t4=t_4_opt,
	)

	print(f"故障优化完成，结果已保存: {Path(RESULT_FILE).resolve()}")
	print(f"一级故障模式={cfg.fault_mode}, te={cfg.te:.2f}s, kappa={cfg.kappa:.4f}")
	print(f"固定第四段时长 T4={T4_fixed:.4f}s")

	if make_plot:
		plot_from_npz(
			RESULT_FILE,
			env=env,
			compare_npz=COMPARE_FILE,
			label_current="Fault",
			label_compare="Nominal",
			show=True,
		)


if __name__ == "__main__":
	config = FaultOptimizeConfig()
	build_and_solve_fault(config, make_plot=True)