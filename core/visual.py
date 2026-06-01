from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

try:
	from .env_models import EarthEnv
except ImportError:
	# 兼容直接运行 `python core/visual.py` 的情况。
	from env_models import EarthEnv


# ----------------------------------------------------------------------
# Matplotlib 全局样式
# ----------------------------------------------------------------------
# 这里统一设置字体和线宽，后面每张图只需要关心自己的坐标轴含义。
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["lines.linewidth"] = 3


def _resolve_npz_path(npz_path):
	"""尽量稳健地解析 npz 结果文件路径。

	绘图脚本可能从工程根目录运行，也可能从 core 目录或其他目录运行。
	这里按常见位置依次查找，避免相对路径变化导致找不到结果文件。
	"""
	raw = Path(npz_path)
	if raw.is_file():
		return raw

	cwd = Path.cwd()
	module_dir = Path(__file__).resolve().parent
	project_root = module_dir.parent

	candidates = [
		cwd / npz_path,
		module_dir / npz_path,
		project_root / npz_path,
		project_root / "results" / raw.name,
	]

	for p in candidates:
		if p.is_file():
			return p

	checked = "\n".join(str(p) for p in candidates)
	raise FileNotFoundError(
		f"找不到 npz 文件: {npz_path}\n已尝试路径:\n{checked}\n"
		f"建议把文件放在 {project_root / 'results'}，或传入绝对路径。"
	)


def _concat_solution(data):
	"""把分段优化结果拼接成一条连续轨迹。

	结果文件里按飞行阶段保存：
	- X1/U1/t1：一级主动段；
	- X3/U3/t3：级间或二级短时工作段；
	- X4/U4/t4：后续入轨段。

	画图时通常只关心完整时间历程，因此先在时间方向上拼接。
	"""
	X_opt = np.hstack([data["X1"], data["X3"], data["X4"]])
	U_opt = np.hstack([data["U1"], data["U3"], data["U4"]])
	t_opt = np.hstack([data["t1"], data["t3"], data["t4"]])
	return X_opt, U_opt, t_opt


def _compute_llh_history(r_sim, t_opt, env):
	"""逐点计算经度、纬度和高度。

	动力学状态使用发射坐标系下的位置，绘图时需要转换成地理量。
	高度 h 也会被气动后处理复用，用于查大气密度。
	"""
	lat = np.zeros_like(t_opt, dtype=float)
	lon = np.zeros_like(t_opt, dtype=float)
	h = np.zeros_like(t_opt, dtype=float)

	for k, tk in enumerate(t_opt):
		lat_k, lon_k, h_k = env.llh(r_sim[:, k], tk)
		lat[k] = float(lat_k)
		lon[k] = float(lon_k)
		h[k] = float(h_k)

	return lat, lon, h


def _compute_aero_loads(X_opt, U_opt, h, env):
	"""计算相对气流攻角和 q-alpha 载荷指标。

	这里的 q-alpha 不是完整弯矩，只是常用的气动载荷趋势指标：

	    q = 0.5 * rho * V_rel^2
	    q_alpha = q * alpha

	如果要得到真实弯矩，还需要继续乘法向力导数、参考面积、参考长度
	或压力中心到质心的力臂等气动参数。
	"""
	r_sim = X_opt[0:3, :]
	v_sim = X_opt[3:6, :]

	# 大气随地球自转。先算当前位置处大气的速度，再得到火箭相对气流速度。
	r_geocentric = r_sim + env.R_fashe.reshape(3, 1)
	v_atm = np.cross(env.omega_e_faguan.reshape(3, 1), r_geocentric, axis=0)
	v_rel = v_sim - v_atm
	v_rel_norm = np.linalg.norm(v_rel, axis=0)

	# theta_rel 是相对气流方向的俯仰角；alpha 是程序俯仰角与气流方向的差。
	# 起飞瞬间相对速度接近 0，方向角没有物理意义，因此用程序角兜底。
	theta_rel = np.arctan2(v_rel[1, :], np.sqrt(v_rel[0, :] ** 2 + v_rel[2, :] ** 2))
	theta_rel = np.where(v_rel_norm < 1.0, U_opt[0, :], theta_rel)
	alpha = U_opt[0, :] - theta_rel

	rho = env.atmosphere(h)
	q = 0.5 * rho * v_rel_norm**2
	q_alpha = q * alpha

	return {
		"v_rel": v_rel,
		"v_rel_norm": v_rel_norm,
		"theta_rel": theta_rel,
		"theta_deg": np.degrees(theta_rel),
		"alpha": alpha,
		"alpha_deg": np.degrees(alpha),
		"q": q,
		"q_alpha": q_alpha,
		"q_alpha_abs": np.abs(q_alpha),
	}


def _compute_terminal_orbit(r_end, v_end, env):
	"""计算末端轨道六根数，便于检查是否进入目标轨道附近。"""
	r_eci_end, v_eci_end = env.ECI(r_end, v_end)
	a, e, i_deg, Omega_deg, omega_deg, f_deg = env.orbit_six(r_eci_end, v_eci_end)

	return {
		"a": float(a),
		"e": float(e),
		"i_deg": float(i_deg),
		"Omega_deg": float(Omega_deg),
		"omega_deg": float(omega_deg),
		"f_deg": float(f_deg),
	}


def compute_derived_history(X_opt, U_opt, t_opt, env=None):
	"""从优化结果计算绘图和保存用的派生量。

	结果文件的核心信息仍然是 X/U/t。这里把攻角、速度倾角和 q-alpha
	集中算出来，既可在 npz 里保存，也可让旧 npz 在绘图时即时补算。
	"""
	env = env or EarthEnv()
	r_sim = X_opt[0:3, :]
	v_sim = X_opt[3:6, :]

	lat, lon, h = _compute_llh_history(r_sim, t_opt, env)
	aero = _compute_aero_loads(X_opt, U_opt, h, env)

	return {
		"lat": lat,
		"lon": lon,
		"h": h,
		"speed": np.linalg.norm(v_sim, axis=0),
		"v_rel_norm": aero["v_rel_norm"],
		"theta": aero["theta_rel"],
		"theta_deg": aero["theta_deg"],
		"alpha": aero["alpha"],
		"alpha_deg": aero["alpha_deg"],
		"q": aero["q"],
		"q_alpha": aero["q_alpha"],
		"q_alpha_abs": aero["q_alpha_abs"],
		"aero": aero,
	}


def _orbit_points_km(a_km, e, i_deg, Omega_deg, omega_deg, n=500):
	"""根据轨道六根数重建一圈理论椭圆轨道点，单位为 km。"""
	theta = np.linspace(0, 2 * np.pi, n)
	r = a_km * (1 - e**2) / (1 + e * np.cos(theta))
	orb_points = np.vstack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)])

	Omega_rad = np.deg2rad(Omega_deg)
	i_rad = np.deg2rad(i_deg)
	omega_rad = np.deg2rad(omega_deg)

	R3_Omega = np.array(
		[
			[np.cos(Omega_rad), -np.sin(Omega_rad), 0],
			[np.sin(Omega_rad), np.cos(Omega_rad), 0],
			[0, 0, 1],
		],
		dtype=float,
	)
	R1_i = np.array(
		[
			[1, 0, 0],
			[0, np.cos(i_rad), -np.sin(i_rad)],
			[0, np.sin(i_rad), np.cos(i_rad)],
		],
		dtype=float,
	)
	R3_omega = np.array(
		[
			[np.cos(omega_rad), -np.sin(omega_rad), 0],
			[np.sin(omega_rad), np.cos(omega_rad), 0],
			[0, 0, 1],
		],
		dtype=float,
	)

	return R3_Omega @ R1_i @ R3_omega @ orb_points


def _caption_axis(ax, caption, y=-0.28, fontsize=13):
	"""给子图下方添加论文式小标题，例如“(a) Altitude”。"""
	ax.text(
		0.5,
		y,
		caption,
		transform=ax.transAxes,
		ha="center",
		va="top",
		fontsize=fontsize,
	)


def _caption_figure(fig, caption, y=0.02, fontsize=14):
	"""给单图下方添加标题。"""
	fig.text(0.5, y, caption, ha="center", va="bottom", fontsize=fontsize)


def _style_xy_axis(ax, xlabel, ylabel, label_fontsize, tick_fontsize, grid_alpha=0.25):
	"""统一二维坐标轴样式，减少每张图里的重复代码。"""
	ax.set_xlabel(xlabel, fontsize=label_fontsize)
	ax.set_ylabel(ylabel, fontsize=label_fontsize)
	ax.tick_params(axis="both", labelsize=tick_fontsize)
	ax.grid(alpha=grid_alpha)


def _save_figure(fig, figures_dir, filename):
	"""统一保存设置，保证输出图片边界比较紧凑。"""
	fig.savefig(figures_dir / filename, bbox_inches="tight", pad_inches=0)


def _plot_state_panel(t_opt, h, v_sim, m_sim, aero, figures_dir):
	"""绘制高度、速度、质量和相对速度倾角。"""
	label_fontsize = 23
	tick_fontsize = 20
	caption_fontsize = 23

	fig, axs = plt.subplots(2, 2, figsize=(14, 13))
	axs = axs.ravel()
	state_plots = [
		(axs[0], t_opt, 0.001 * h, "Time (s)", "Altitude (km)", "(a) Altitude"),
		(axs[1], t_opt, np.linalg.norm(v_sim, axis=0), "Time (s)", "Velocity (m/s)", "(b) Velocity"),
		(axs[2], t_opt, 0.001 * m_sim, "Time (s)", "Mass (t)", "(c) Mass"),
		(
			axs[3],
			t_opt,
			np.degrees(aero["theta_rel"]),
			"Time (s)",
			r"$\theta$ (deg)",
			"(d) Velocity inclination angle",
		),
	]

	for ax, x_data, y_data, xlabel, ylabel, caption in state_plots:
		ax.plot(x_data, y_data)
		_style_xy_axis(ax, xlabel, ylabel, label_fontsize, tick_fontsize)
		_caption_axis(ax, caption, y=-0.18, fontsize=caption_fontsize)

	axs[3].yaxis.set_major_locator(MultipleLocator(30))
	fig.tight_layout(rect=[0, 0.05, 1, 0.95])
	_save_figure(fig, figures_dir, "flight_curve.pdf")


def _plot_program_angles(t_opt, U_opt):
	"""绘制程序俯仰角 phi 和偏航角 psi。"""
	label_fontsize = 28
	tick_fontsize = 25
	caption_fontsize = 28

	fig, axs = plt.subplots(1, 2, figsize=(16, 9))

	axs[0].plot(t_opt, np.degrees(U_opt[0, :]), linewidth=3)
	_style_xy_axis(axs[0], "t (s)", "Phi (deg)", label_fontsize, tick_fontsize, grid_alpha=1.0)
	_caption_axis(axs[0], "(a) Phi", y=-0.25, fontsize=caption_fontsize)

	axs[1].plot(t_opt, np.degrees(U_opt[1, :]), linewidth=3)
	_style_xy_axis(axs[1], "t (s)", "Psi (deg)", label_fontsize, tick_fontsize, grid_alpha=1.0)
	_caption_axis(axs[1], "(b) Psi", y=-0.25, fontsize=caption_fontsize)

	fig.tight_layout(rect=[0, 0.08, 1, 0.93])


def _plot_aero_loads(t_opt, aero, figures_dir):
	"""绘制攻角和有符号 q-alpha 载荷指标。

	这里保持 q_alpha 的正负号，不取绝对值。这样可以看出攻角换符号时
	载荷方向也发生了反向；如果只关心大小，再画 abs(q_alpha) 即可。
	"""
	label_fontsize = 28
	tick_fontsize = 25
	caption_fontsize = 28

	fig, axs = plt.subplots(1, 2, figsize=(16, 9))

	axs[0].plot(t_opt, aero["alpha_deg"], linewidth=3)
	_style_xy_axis(axs[0], "Time (s)", r"$\alpha$ (deg)", label_fontsize, tick_fontsize)
	_caption_axis(axs[0], r"(a) Angle of attack $\alpha$", y=-0.17, fontsize=caption_fontsize)

	axs[1].plot(t_opt, aero["q_alpha"], linewidth=3)
	_style_xy_axis(axs[1], "Time (s)", r"$q\alpha$ (Pa rad)", label_fontsize, tick_fontsize)
	_caption_axis(axs[1], r"(b) $q\alpha$ load index", y=-0.17, fontsize=caption_fontsize)

	fig.tight_layout(rect=[0, 0.08, 1, 0.93])
	_save_figure(fig, figures_dir, "aero_load_curves.pdf")


def _plot_component_curves(t_opt, X_opt):
	"""绘制速度三分量和位置三分量，主要用于调试轨迹形状。"""
	fig_vel, ax_vel = plt.subplots()
	for idx, label in zip([3, 4, 5], ["Vx", "Vy", "Vz"]):
		ax_vel.plot(t_opt, X_opt[idx, :], label=label)
	ax_vel.set_xlabel("Time (s)")
	ax_vel.set_ylabel("Velocity (m/s)")
	ax_vel.legend()
	ax_vel.grid()
	_caption_figure(fig_vel, "(a) Velocity Components")
	fig_vel.tight_layout(rect=[0, 0.08, 1, 1])

	fig_pos, ax_pos = plt.subplots()
	for idx, label in zip([0, 1, 2], ["X", "Y", "Z"]):
		ax_pos.plot(t_opt, X_opt[idx, :], label=label)
	ax_pos.set_xlabel("Time (s)")
	ax_pos.set_ylabel("Position (m)")
	ax_pos.legend()
	ax_pos.grid()
	_caption_figure(fig_pos, "(a) Position Components")
	fig_pos.tight_layout(rect=[0, 0.08, 1, 1])


def _plot_space_results(lon, lat, h, orbit, figures_dir):
	"""绘制经纬高轨迹和末端理论轨道。"""
	fontsize = 25
	tick_fontsize = 20
	caption_fontsize = 29

	pos_orb = _orbit_points_km(
		orbit["a"] / 1000.0,
		orbit["e"],
		orbit["i_deg"],
		orbit["Omega_deg"],
		orbit["omega_deg"],
	)

	fig = plt.figure(figsize=(22, 9))

	# 左图：经度-纬度-高度，展示地面覆盖方向和上升高度。
	ax_llh = fig.add_subplot(121, projection="3d")
	ax_llh.plot(lon, lat, 0.001 * h)
	ax_llh.set_xlabel("Longitude (deg)", fontsize=fontsize, labelpad=14)
	ax_llh.set_ylabel("Latitude (deg)", fontsize=fontsize, labelpad=14)
	ax_llh.set_zlabel("Altitude (km)", fontsize=fontsize, labelpad=14)
	ax_llh.tick_params(axis="both", labelsize=tick_fontsize)
	ax_llh.grid()

	# 右图：用末端轨道六根数重建一圈轨道，与地球位置做直观对照。
	ax_orb = fig.add_subplot(122, projection="3d")
	ax_orb.plot(pos_orb[0, :], pos_orb[1, :], pos_orb[2, :], "b", linewidth=3, label="Orbit")
	ax_orb.scatter(0, 0, 0, c="r", s=80, marker="o", label="Earth Center")

	r_earth = 6371.0
	u = np.linspace(0, 2 * np.pi, 20)
	v = np.linspace(0, np.pi, 20)
	x_e = r_earth * np.outer(np.cos(u), np.sin(v))
	y_e = r_earth * np.outer(np.sin(u), np.sin(v))
	z_e = r_earth * np.outer(np.ones_like(u), np.cos(v))
	ax_orb.plot_surface(x_e, y_e, z_e, color="cyan", alpha=0.3, edgecolor="none")

	ax_orb.set_xlabel("X (km)", fontsize=fontsize, labelpad=14)
	ax_orb.set_ylabel("Y (km)", fontsize=fontsize, labelpad=14)
	ax_orb.set_zlabel("Z (km)", fontsize=fontsize, labelpad=14)
	ax_orb.tick_params(axis="both", labelsize=tick_fontsize)
	ax_orb.grid(True)
	ax_orb.axis("equal")
	ax_orb.view_init(elev=30, azim=30)
	ax_orb.legend(fontsize=fontsize)

	fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.98, wspace=-0.08)
	fig.text(
		0.25,
		0.045,
		"(a) Longitude-Latitude-Altitude trajectory",
		ha="center",
		va="bottom",
		fontsize=caption_fontsize,
	)
	fig.text(0.75, 0.045, "(b) Terminal orbit", ha="center", va="bottom", fontsize=caption_fontsize)


def _plot_control_comparison(
	t_opt,
	U_opt,
	compare_npz,
	label_current,
	label_compare,
	figures_dir,
):
	"""如果给定对比结果文件，就叠加两组程序角曲线。"""
	cmp_resolved = _resolve_npz_path(compare_npz)
	cmp_data = np.load(cmp_resolved)
	_, U_cmp, t_cmp = _concat_solution(cmp_data)

	label_fontsize = 28
	tick_fontsize = 25
	caption_fontsize = 28

	fig, axs = plt.subplots(1, 2, figsize=(17, 9))

	axs[0].plot(t_opt, np.degrees(U_opt[0, :]), label=label_current, linewidth=3)
	axs[0].plot(t_cmp, np.degrees(U_cmp[0, :]), "--", label=label_compare, linewidth=3)
	_style_xy_axis(axs[0], "Times (s)", r"$\varphi$ (deg)", label_fontsize, tick_fontsize, grid_alpha=0.2)
	axs[0].yaxis.set_major_locator(MultipleLocator(30))
	axs[0].legend(fontsize=tick_fontsize)
	_caption_axis(axs[0], r"(a) Pitch program angle $\varphi$", y=-0.17, fontsize=caption_fontsize)

	axs[1].plot(t_opt, np.degrees(U_opt[1, :]), label=label_current, linewidth=3)
	axs[1].plot(t_cmp, np.degrees(U_cmp[1, :]), "--", label=label_compare, linewidth=3)
	_style_xy_axis(axs[1], "Times (s)", r"$\psi$ (deg)", label_fontsize, tick_fontsize, grid_alpha=0.2)
	axs[1].legend(fontsize=tick_fontsize)
	_caption_axis(axs[1], r"(b) Yaw program angle $\psi$", y=-0.17, fontsize=caption_fontsize)

	fig.tight_layout(rect=[0, 0.12, 1, 0.92])
	_save_figure(fig, figures_dir, "phi_psi_limited_and_unlimited.pdf")


def _plot_aero_theta_comparison(
	t_opt,
	aero,
	compare_npz,
	env,
	label_current,
	label_compare,
	figures_dir,
):
	"""叠加故障/标称的 alpha、theta 和 q-alpha 曲线。"""
	cmp_resolved = _resolve_npz_path(compare_npz)
	cmp_data = np.load(cmp_resolved)
	X_cmp, U_cmp, t_cmp = _concat_solution(cmp_data)
	derived_cmp = compute_derived_history(X_cmp, U_cmp, t_cmp, env)

	label_fontsize = 23
	tick_fontsize = 20
	caption_fontsize = 23

	fig, axs = plt.subplots(1, 3, figsize=(22, 7.5))
	plots = [
		(
			aero["alpha_deg"],
			derived_cmp["alpha_deg"],
			r"$\alpha$ (deg)",
			r"(a) Angle of attack $\alpha$",
		),
		(
			aero["theta_deg"],
			derived_cmp["theta_deg"],
			r"$\theta$ (deg)",
			r"(b) Velocity inclination angle $\theta$",
		),
		(
			aero["q_alpha"],
			derived_cmp["q_alpha"],
			r"$q\alpha$ (Pa rad)",
			r"(c) $q\alpha$ load index",
		),
	]

	for ax, (current_y, compare_y, ylabel, caption) in zip(axs, plots):
		ax.plot(t_opt, current_y, label=label_current, linewidth=3)
		ax.plot(t_cmp, compare_y, "--", label=label_compare, linewidth=3)
		_style_xy_axis(ax, "Time (s)", ylabel, label_fontsize, tick_fontsize, grid_alpha=0.25)
		_caption_axis(ax, caption, y=-0.23, fontsize=caption_fontsize)

	axs[0].legend(fontsize=tick_fontsize)
	axs[1].legend(fontsize=tick_fontsize)
	axs[2].legend(fontsize=tick_fontsize)
	fig.tight_layout(rect=[0, 0.13, 1, 0.95])
	_save_figure(fig, figures_dir, "alpha_theta_qalpha_comparison.pdf")


def plot_from_npz(
	npz_path,
	env=None,
	compare_npz=None,
	label_current="Constrained",
	label_compare="Unconstrained",
	show=True,
):
	"""读取优化结果并生成完整可视化图像。

	主要输出：
	- `flight_curve.pdf`：高度、速度、质量、相对速度倾角；
	- `aero_load_curves.pdf`：攻角和有符号 q-alpha 载荷指标；
	- `phi_psi_limited_and_unlimited.pdf`：可选的程序角对比图；
	- `alpha_theta_qalpha_comparison.pdf`：可选的攻角/速度倾角/q-alpha 对比图。

	函数也会返回处理后的数据字典，便于后续脚本继续分析。
	"""
	env = env or EarthEnv()

	# 1) 读取结果并完成基础后处理。
	npz_resolved = _resolve_npz_path(npz_path)
	data = np.load(npz_resolved)
	X_opt, U_opt, t_opt = _concat_solution(data)

	# 状态格式统一为 [r(3), v(3), m]。
	r_sim = X_opt[0:3, :]
	v_sim = X_opt[3:6, :]
	m_sim = X_opt[6, :]

	derived = compute_derived_history(X_opt, U_opt, t_opt, env)
	lat = derived["lat"]
	lon = derived["lon"]
	h = derived["h"]
	aero = derived["aero"]
	orbit = _compute_terminal_orbit(r_sim[:, -1], v_sim[:, -1], env)

	print(f"Final altitude: {h[-1]:.3f} m")
	print(f"Final speed: {np.linalg.norm(v_sim[:, -1]):.3f} m/s")
	print(f"a: {orbit['a'] / 1000:.3f} km, e: {orbit['e']:.3f}, i: {orbit['i_deg']:.3f} deg")
	print(
		f"Omega: {orbit['Omega_deg']:.3f} deg, "
		f"omega: {orbit['omega_deg']:.3f} deg, "
		f"f: {orbit['f_deg']:.3f} deg"
	)

	# 2) 准备图片保存目录。
	figures_dir = Path("figures")
	figures_dir.mkdir(exist_ok=True)

	# 3) 按主题生成各类图像。
	_plot_state_panel(t_opt, h, v_sim, m_sim, aero, figures_dir)
	_plot_program_angles(t_opt, U_opt)
	_plot_aero_loads(t_opt, aero, figures_dir)
	_plot_component_curves(t_opt, X_opt)
	_plot_space_results(lon, lat, h, orbit, figures_dir)

	if compare_npz is not None:
		_plot_control_comparison(t_opt, U_opt, compare_npz, label_current, label_compare, figures_dir)
		_plot_aero_theta_comparison(t_opt, aero, compare_npz, env, label_current, label_compare, figures_dir)

	if show:
		plt.show()

	return {
		"t": t_opt,
		"X": X_opt,
		"U": U_opt,
		"lat": lat,
		"lon": lon,
		"h": h,
		"derived": derived,
		"aero": aero,
		"orbit": orbit,
	}


if __name__ == "__main__":
	# 直接运行本文件时，默认读取标准弹道结果，并叠加一组对比结果。
	plot_from_npz("results/biaozhundandao.npz", compare_npz="results/biaozhundandao_unlimited.npz")
	# 如果不需要对比图，可以改成：
	# plot_from_npz("results/biaozhundandao.npz")
