import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
	from .env_models import EarthEnv # 用点.表示"就在我所在的这个包里"
except ImportError:
	# 兼容直接运行 `python core/visual.py` 的场景。
	from env_models import EarthEnv # 相对导入不能在直接运行的脚本中用


def _resolve_npz_path(npz_path):
	"""尽量稳健地解析 npz 文件路径。

	因为你可能在工程根目录运行，也可能在 Optimization_part 目录运行，
	所以这里会尝试多个候选位置，避免简单相对路径导致找不到文件。
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
		f"建议把文件放在 {project_root / 'results'} 或传绝对路径"
	)


def _concat_solution(data):
	"""把三段优化结果拼接成一条连续轨迹。

	优化结果是分段保存的，画图前需要先按时间顺序拼起来。
	"""
	X_opt = np.hstack([data["X1"], data["X3"], data["X4"]])
	U_opt = np.hstack([data["U1"], data["U3"], data["U4"]])
	t_opt = np.hstack([data["t1"], data["t3"], data["t4"]])
	return X_opt, U_opt, t_opt


def _orbit_points_km(a_km, e, i_deg, Omega_deg, omega_deg, n=500):
	"""根据轨道六根数重建一圈理想椭圆轨道点。"""
	theta = np.linspace(0, 2 * np.pi, n)
	r = a_km * (1 - e**2) / (1 + e * np.cos(theta))
	x_orb = r * np.cos(theta)
	y_orb = r * np.sin(theta)
	z_orb = np.zeros_like(theta)

	Omega_rad = np.deg2rad(Omega_deg)
	i_rad = np.deg2rad(i_deg)
	omega_rad = np.deg2rad(omega_deg)

	R3_W = np.array(
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
	R3_w = np.array(
		[
			[np.cos(omega_rad), -np.sin(omega_rad), 0],
			[np.sin(omega_rad), np.cos(omega_rad), 0],
			[0, 0, 1],
		],
		dtype=float,
	)

	Q_pX = R3_W @ R1_i @ R3_w
	orb_points = np.vstack([x_orb, y_orb, z_orb])
	return Q_pX @ orb_points


def plot_from_npz(
	npz_path,
	env=None,
	compare_npz=None,
	label_current="Current",
	label_compare="Compare",
	show=True,
):
	"""读取优化结果并生成完整可视化图像。

	包括：速度、质量、高度、速度倾角、控制角、速度/位置分量、经纬高、轨道图，
	以及可选的控制对比图。
	"""
	env = env or EarthEnv()

	npz_resolved = _resolve_npz_path(npz_path)
	data = np.load(npz_resolved)
	# 先把分段结果拼接成连续结果，再做后处理。
	X_opt, U_opt, t_opt = _concat_solution(data)

	# 状态格式统一为 [r(3), v(3), m]。
	r_sim = X_opt[0:3, :]
	v_sim = X_opt[3:6, :]
	m_sim = X_opt[6, :]

	# llh curve
	# 逐点计算经纬度和高度，方便展示火箭飞行轨迹在地球坐标中的位置。
	lat = np.zeros_like(t_opt, dtype=float)
	lon = np.zeros_like(t_opt, dtype=float)
	h = np.zeros_like(t_opt, dtype=float)
	for k, tk in enumerate(t_opt):
		lat_k, lon_k, h_k = env.llh(r_sim[:, k], tk)
		lat[k] = float(lat_k)
		lon[k] = float(lon_k)
		h[k] = float(h_k)

	# terminal orbit elements
	# 末端轨道六根数用于验证最终是否接近目标轨道。
	r_end = r_sim[:, -1]
	v_end = v_sim[:, -1]
	r_eci_end, v_eci_end = env.ECI(r_end, v_end)
	a, e, i_deg, Omega_deg, omega_deg, f_deg = env.orbit_six(r_eci_end, v_eci_end)

	a = float(a)
	e = float(e)
	i_deg = float(i_deg)
	Omega_deg = float(Omega_deg)
	omega_deg = float(omega_deg)
	f_deg = float(f_deg)

	print(f"Final altitude: {h[-1]:.3f} m")
	print(f"Final speed: {np.linalg.norm(v_end):.3f} m/s")
	print(f"a: {a / 1000:.3f} km, e: {e:.6f}, i: {i_deg:.3f} deg")
	print(f"Omega: {Omega_deg:.3f} deg, omega: {omega_deg:.3f} deg, f: {f_deg:.3f} deg")

	# 速度矢量在 x-y 平面的方向角，可作为一个直观的速度方向参考。
	theta_speed = np.arctan2(X_opt[4, :], X_opt[3, :])

	# Main panel
	plt.figure(figsize=(18, 10))

	plt.subplot(2, 3, 1)
	plt.plot(t_opt, np.linalg.norm(v_sim, axis=0))
	plt.xlabel("Time (s)")
	plt.ylabel("V (m/s)")
	plt.title("Velocity")
	plt.grid()

	plt.subplot(2, 3, 2)
	plt.plot(t_opt, m_sim)
	plt.xlabel("Time (s)")
	plt.ylabel("M (kg)")
	plt.title("Mass")
	plt.grid()

	plt.subplot(2, 3, 3)
	plt.plot(t_opt, 0.001 * h)
	plt.xlabel("Time (s)")
	plt.ylabel("Altitude (km)")
	plt.title("Altitude")
	plt.grid()

	plt.subplot(2, 3, 4)
	plt.plot(t_opt, np.rad2deg(theta_speed))
	plt.xlabel("Time (s)")
	plt.ylabel("Launch Speed Angle (deg)")
	plt.title("Launch Speed Angle")
	plt.grid()

	plt.subplot(2, 3, 5)
	plt.plot(t_opt, np.degrees(U_opt[0, :]), linewidth=2)
	plt.xlabel("t (s)")
	plt.ylabel("Phi (deg)")
	plt.title("Phi")
	plt.grid()

	plt.subplot(2, 3, 6)
	plt.plot(t_opt, np.degrees(U_opt[1, :]), linewidth=2)
	plt.xlabel("t (s)")
	plt.ylabel("Psi (deg)")
	plt.title("Psi")
	plt.grid()

	# Velocity components
	plt.figure()
	plt.plot(t_opt, X_opt[3, :], label="Vx")
	plt.plot(t_opt, X_opt[4, :], label="Vy")
	plt.plot(t_opt, X_opt[5, :], label="Vz")
	plt.xlabel("Time (s)")
	plt.ylabel("Velocity (m/s)")
	plt.title("Velocity Components")
	plt.legend()
	plt.grid()

	# Position components
	plt.figure()
	plt.plot(t_opt, X_opt[0, :], label="X")
	plt.plot(t_opt, X_opt[1, :], label="Y")
	plt.plot(t_opt, X_opt[2, :], label="Z")
	plt.xlabel("Time (s)")
	plt.ylabel("Position (m)")
	plt.title("Position Components")
	plt.legend()
	plt.grid()

	# LLH 3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.plot(lon, lat, 0.001 * h)
	ax.set_xlabel("Longitude (deg)")
	ax.set_ylabel("Latitude (deg)")
	ax.set_zlabel("Altitude (km)")
	ax.set_title("Latitude, Longitude, Altitude")
	ax.grid()

	# Compare controls if provided
	# 如果给了对比文件，就把 phi 和 psi 曲线叠加，方便看控制策略差异。
	if compare_npz is not None:
		cmp_resolved = _resolve_npz_path(compare_npz)
		cmp_data = np.load(cmp_resolved)
		_, U_cmp, t_cmp = _concat_solution(cmp_data)

		plt.figure(figsize=(16, 6))
		plt.subplot(1, 2, 1)
		plt.plot(t_opt, np.degrees(U_opt[0, :]), "--",label=label_current, linewidth=2)
		plt.plot(t_cmp, np.degrees(U_cmp[0, :]), label=label_compare, linewidth=2)
		plt.xlabel("t (s)")
		plt.ylabel("Phi (deg)")
		plt.title("Phi")
		plt.grid()
		plt.legend()

		plt.subplot(1, 2, 2)
		plt.plot(t_opt, np.degrees(U_opt[1, :]), "--",label=label_current, linewidth=2)
		plt.plot(t_cmp, np.degrees(U_cmp[1, :]), label=label_compare, linewidth=2)
		plt.xlabel("t (s)")
		plt.ylabel("Psi (deg)")
		plt.title("Psi")
		plt.grid()
		plt.legend()

	# Orbit 3D
	# 用最终轨道六根数重建理论轨道，与仿真末端轨道做对照。
	pos_orb = _orbit_points_km(a / 1000.0, e, i_deg, Omega_deg, omega_deg)

	fig_orb = plt.figure()
	ax_orb = fig_orb.add_subplot(111, projection="3d")
	ax_orb.plot(pos_orb[0, :], pos_orb[1, :], pos_orb[2, :], "b", linewidth=2, label="Orbit")
	ax_orb.scatter(0, 0, 0, c="r", s=80, marker="o", label="Earth Center")

	r_earth = 6371.0
	u = np.linspace(0, 2 * np.pi, 20)
	v = np.linspace(0, np.pi, 20)
	x_e = r_earth * np.outer(np.cos(u), np.sin(v))
	y_e = r_earth * np.outer(np.sin(u), np.sin(v))
	z_e = r_earth * np.outer(np.ones_like(u), np.cos(v))
	ax_orb.plot_surface(x_e, y_e, z_e, color="cyan", alpha=0.3, edgecolor="none")

	ax_orb.set_xlabel("X (km)")
	ax_orb.set_ylabel("Y (km)")
	ax_orb.set_zlabel("Z (km)")
	ax_orb.set_title("Orbit")
	ax_orb.grid(True)
	ax_orb.axis("equal")
	ax_orb.view_init(elev=30, azim=30)
	ax_orb.legend()

	plt.tight_layout()
	if show:
		plt.show()

	return {
		"t": t_opt,
		"X": X_opt,
		"U": U_opt,
		"lat": lat,
		"lon": lon,
		"h": h,
		"orbit": {
			"a": a,
			"e": e,
			"i_deg": i_deg,
			"Omega_deg": Omega_deg,
			"omega_deg": omega_deg,
			"f_deg": f_deg,
		},
	}


if __name__ == "__main__":
	# Example:
	# python visual.py
	plot_from_npz("results/biaozhundandao.npz",compare_npz = "results/biaozhundandao_unlimited.npz")
	""" plot_from_npz("results/biaozhundandao.npz") """