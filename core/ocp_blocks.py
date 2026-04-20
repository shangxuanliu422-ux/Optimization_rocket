import numpy as np

from .utils import rk4_step


def apply_control_angle_bounds(opti, control_specs):
    """批量添加控制角上下界。

    control_specs 每项格式:
    (U, phi_min_deg, phi_max_deg, psi_min_deg, psi_max_deg)
    """
    deg2rad = np.pi / 180.0
    for U, phi_min_deg, phi_max_deg, psi_min_deg, psi_max_deg in control_specs:
        opti.subject_to(opti.bounded(phi_min_deg * deg2rad, U[0, :], phi_max_deg * deg2rad))
        opti.subject_to(opti.bounded(psi_min_deg * deg2rad, U[1, :], psi_max_deg * deg2rad))


def apply_dphi_rate_limit(opti, U, dt, dphi_max_deg_per_step):
    """给单段控制 U 施加 phi 变化率限制。"""
    dphi_max = dt * dphi_max_deg_per_step * np.pi / 180.0
    n_steps = U.shape[1] - 1
    for k in range(n_steps):
        opti.subject_to(U[0, k + 1] - U[0, k] <= dphi_max)
        opti.subject_to(U[0, k] - U[0, k + 1] <= dphi_max)


def add_rk4_segment_constraints(opti, X, U, n_steps, ode, dt, t_grid=None, t0=0.0):
    """给单段轨迹添加 RK4 离散动力学约束。"""
    for k in range(n_steps):
        t_k = t_grid[k] if t_grid is not None else t0 + k * dt
        opti.subject_to(X[:, k + 1] == rk4_step(ode, t_k, X[:, k], U[:, k], U[:, k + 1], dt))


def apply_mass_lower_bound(opti, X, m_min):
    """给状态段添加质量下界约束。"""
    opti.subject_to(X[6, :] >= m_min)


def configure_ipopt_solver(opti, solver_opts):
    """统一配置 IPOPT 求解器。"""
    p_opts = {"expand": solver_opts.get("expand", True)}
    s_opts = dict(solver_opts)
    s_opts.pop("expand", None)
    opti.solver("ipopt", p_opts, s_opts)
