import casadi as ca
import numpy as np


class Rocket:
    """单级火箭动力学模型。

    这个类只负责动力学，不负责优化器搭建。
    主流程只需要给它状态、控制和环境对象，就能拿到状态导数。
    """

    def __init__(self, thrust, mdot, name="Stage", Cd=0.4, S=11.341):
        # thrust: 推力；mdot: 质量流率；Cd/S: 阻力模型参数。
        self.P = float(thrust)
        self.mdot = float(mdot)
        self.name = name
        self.Cd = float(Cd)
        self.S = float(S)

    @staticmethod
    def _is_ca_value(x):
        """判断是否为 CasADi 类型。"""
        return isinstance(x, (ca.SX, ca.MX, ca.DM))

    def _is_ca_backend(self, *vals):
        """只要输入里有 CasADi 对象，就整体切换到 CasADi 分支。"""
        return any(self._is_ca_value(v) for v in vals)

    def _to_vec(self, arr, use_ca):
        """把 numpy 数组转换成当前后端可用的向量。"""
        return ca.DM(arr) if use_ca else np.asarray(arr, dtype=float)

    @staticmethod
    def _gb_matrix(phi, psi, gamma, use_ca):
        """箭体系到发惯系的姿态矩阵 GB。
        这里沿用原脚本的展开式，保证和旧结果一致。
        """
        if use_ca:
            return ca.vertcat(
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

        return np.array(
            [
                [
                    np.cos(phi) * np.cos(psi),
                    np.cos(phi) * np.sin(psi) * np.sin(gamma) - np.sin(phi) * np.cos(gamma),
                    np.cos(phi) * np.sin(psi) * np.cos(gamma) + np.sin(phi) * np.sin(gamma),
                ],
                [
                    np.sin(phi) * np.cos(psi),
                    np.sin(phi) * np.sin(psi) * np.sin(gamma) + np.cos(phi) * np.cos(gamma),
                    np.sin(phi) * np.sin(psi) * np.cos(gamma) - np.cos(phi) * np.sin(gamma),
                ],
                [
                    -np.sin(psi),
                    np.sin(phi) * np.sin(psi) * np.cos(gamma) - np.cos(phi) * np.sin(gamma),
                    np.cos(psi) * np.cos(gamma),
                ],
            ],
            dtype=float,
        )

    def dynamics(self, t, x, u, env):
        """火箭状态方程。

        状态 x = [r(3), v(3), m]。
        控制 u = [phi, psi]。
        返回值是 x_dot，适合直接用于 RK4 和最优控制离散约束。
        """
        r = x[0:3]
        v = x[3:6]
        m = x[6]

        use_ca = self._is_ca_backend(r, v, m, u, t)

        # 重力
        # 重力模型由 env 统一管理，动力学里直接调用即可。
        g = env.gravity(r)

        # 程序角
        # 程序角决定推力在箭体系中的指向。
        phi = u[0]
        psi = u[1]
        gamma = 0.0
        GB = self._gb_matrix(phi, psi, gamma, use_ca)

        # 气动力
        # 先算当前高度，再由高度得到大气密度。
        _, _, h = env.llh(r, t)
        rho = env.atmosphere(h)

        omega_vec = self._to_vec(env.omega_e_faguan, use_ca)
        R_launch = self._to_vec(env.R_fashe, use_ca)

        # 火箭相对大气的速度：火箭速度减去地球自转带来的大气速度。
        v_rel = v - (ca.cross(omega_vec, r + R_launch) if use_ca else np.cross(omega_vec, r + R_launch))
        # 加一个小量避免速度接近 0 时数值不稳定。
        vr = ca.sqrt(ca.dot(v_rel, v_rel) + 1e-6) if use_ca else np.sqrt(np.dot(v_rel, v_rel) + 1e-6)

        # q 是动压，drag 是阻力向量，方向与相对速度相反。
        q = 0.5 * rho * vr**2
        v_hat = v_rel / vr
        drag = -self.Cd * q * self.S * v_hat

        # 推力先在箭体系里沿 x 轴，再通过姿态矩阵转到发惯系。
        u_body = self._to_vec([1.0, 0.0, 0.0], use_ca)
        u_earth = GB @ u_body
        a = (self.P / m) * u_earth + g + drag / m

        # 标准状态导数：位置导数=速度，速度导数=加速度，质量导数=-mdot。
        if use_ca:
            return ca.vertcat(v, a, -self.mdot)
        return np.concatenate([v, a, np.array([-self.mdot], dtype=float)])