import numpy as np
import casadi as ca


def interpolate_solution(t_target, t_source, X_source):
	"""将历史解插值到当前时间网格。

	常用于把上一轮优化得到的状态/控制轨迹，映射到当前网格上作为初始猜测。
	"""
	t_target = np.asarray(t_target, dtype=float)
	t_source = np.asarray(t_source, dtype=float)
	X_source = np.asarray(X_source, dtype=float)

	n_states = X_source.shape[0]
	X_interp = np.zeros((n_states, len(t_target)), dtype=float)

	try:
		from scipy import interpolate

		for i in range(n_states):
			# 线性插值更稳，适合作为初猜，不容易引入额外振荡。
			f = interpolate.interp1d(
				t_source,
				X_source[i, :],
				kind="linear",
				bounds_error=False,
				fill_value="extrapolate",
			)
			X_interp[i, :] = f(t_target)
	except Exception:
		# scipy 不可用时，退化为 np.interp
		# 这样即便没有 scipy，脚本也能继续跑，只是插值能力简单一些。
		for i in range(n_states):
			X_interp[i, :] = np.interp(t_target, t_source, X_source[i, :])

	return X_interp


def rk4_step(ode_func, t, x, u_k, u_k1, dt):
	"""通用 RK4 四阶龙格库塔积分。

	ode_func 需要满足签名 ode_func(t, x, u)。
	这里把一个步长内的控制量用前后两个节点和中点近似，适合分段控制离散。
	"""
	u_mid = (u_k + u_k1) / 2
	k1 = ode_func(t, x, u_k)
	k2 = ode_func(t + dt / 2, x + dt / 2 * k1, u_mid)
	k3 = ode_func(t + dt / 2, x + dt / 2 * k2, u_mid)
	k4 = ode_func(t + dt, x + dt * k3, u_k1)
	return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def to_ca_vec(x):
	"""把普通数组转成 CasADi DM，方便塞进符号表达式。"""
	if isinstance(x, (ca.SX, ca.MX, ca.DM)):
		return x
	return ca.DM(x)
