import casadi as ca
import numpy as np


class EarthEnv:
    """集中管理火箭仿真环境参数，并提供 numpy/casadi 双后端物理模型。"""
    # 这个类的作用是把“与物理环境相关的公共量”统一收口。
    # 这样动力学和优化主流程就不会到处重复写地球参数、坐标变换和轨道公式。

    def __init__(self, **overrides): # 接收任意数量的关键字参数
        # 目标轨道六要素（与 biaozhundandao.py 一致）
        self.target = {
            "a_km": 6778.0,
            "e": 0.01,
            "i_deg": 27.57,
            "Omega_deg": 335.0,
            "f_deg": 0.0,
            "omega_deg": 148.0,
        }

        # 发射场与地球参数
        self.E = 110.95
        self.B_0 = 19.61
        self.Am = 110.0
        self.GM = 3.986004418e14
        self.R_fashedian = 6375742.0
        self.omega_e = 7.292115e-05
        self.a0 = 6378137.0
        self.b0 = 6356752.3142
        self.J2 = 0.00108263
        self.ae = 6378137.0

        # 火箭与任务参数
        self.number1 = 12
        self.number2 = 2
        self.mdot1_per_engine = 500.0
        self.mdot2_per_engine = 405.0
        self.c_eff1 = 3800.0 # 一级发动机有效排气速度（m/s），与秒耗量一起决定推力
        self.c_eff2 = 3800.0
        self.t_yiji = 200.0
        self.t_zhengliu = 7.0
        self.m01 = 1560000.0
        self.m_pao = 30000.0
        self.m_zhengliu = 3200.0
        self.m_gan = 10000.0

        # 气动与大气参数
        self.Cd = 0.4
        self.S = 11.341
        self.rho0 = 1.225
        self.hs = 7200.0
        self.h_atm_cutoff = 120000.0

        # 允许外部覆盖基础参数，比如 EarthEnv(E=120.5, Cd=0.35)
        for key, value in overrides.items():
            if key == "target" and isinstance(value, dict):
                self.target.update(value)
            else:
                setattr(self, key, value)

        self._refresh_derived()

    def _refresh_derived(self): # 计算一些派生属性
        # 目标轨道半长轴（m）
        self.target["a"] = self.target["a_km"] * 1000.0

        # 一级/二级推力与秒耗量
        self.mdot1 = self.mdot1_per_engine * self.number1
        self.mdot2 = self.mdot2_per_engine * self.number2
        self.P1 = self.mdot1 * self.c_eff1
        self.P2 = self.mdot2 * self.c_eff2

        # 地球参数与发射点地心纬度
        self.e2 = (self.a0**2 - self.b0**2) / self.a0**2
        self.phi_0_rad = np.arctan((1 - self.e2) * np.tan(np.deg2rad(self.B_0)))
        self.phi_0 = np.rad2deg(self.phi_0_rad)

        # 发射点随地球自转线速度
        self.v_e = self.omega_e * self.R_fashedian * np.cos(np.deg2rad(self.B_0))

        # 初始状态
        self.r0 = np.array([0.0, 0.0, 0.0], dtype=float)
        self.v0 = np.array(
            [
                self.v_e * np.sin(np.deg2rad(self.Am)),
                0.0,
                self.v_e * np.cos(np.deg2rad(self.Am)),
            ],
            dtype=float,
        )
        self.y0 = np.concatenate([self.r0, self.v0, np.array([self.m01], dtype=float)])

        # 发射坐标系到地心坐标系矩阵（地心 = EG * 发射）
        alpha0 = np.deg2rad(self.Am)
        lambda0 = np.deg2rad(self.E)
        phi0 = np.deg2rad(self.B_0)
        self.EG = np.array(
            [
                [
                    -np.sin(alpha0) * np.sin(lambda0)
                    - np.cos(alpha0) * np.sin(phi0) * np.cos(lambda0),
                    np.cos(phi0) * np.cos(lambda0),
                    -np.cos(alpha0) * np.sin(lambda0)
                    + np.sin(alpha0) * np.sin(phi0) * np.cos(lambda0),
                ],
                [
                    np.sin(alpha0) * np.cos(lambda0)
                    - np.cos(alpha0) * np.sin(phi0) * np.sin(lambda0),
                    np.cos(phi0) * np.sin(lambda0),
                    np.cos(alpha0) * np.cos(lambda0)
                    + np.sin(alpha0) * np.sin(phi0) * np.sin(lambda0),
                ],
                [
                    np.cos(alpha0) * np.cos(phi0),
                    np.sin(phi0),
                    -np.sin(alpha0) * np.cos(phi0),
                ],
            ],
            dtype=float,
        )

        # 地球自转角速度在发惯系下表示
        self.omega_e_faguan = np.array(
            [
                self.omega_e * np.cos(np.deg2rad(self.B_0)) * np.cos(np.deg2rad(self.Am)),
                self.omega_e * np.sin(np.deg2rad(self.B_0)),
                self.omega_e * -np.cos(np.deg2rad(self.B_0)) * np.sin(np.deg2rad(self.Am)),
            ],
            dtype=float,
        )

        # 发惯系下，地心到发射点的矢量
        mu0 = np.deg2rad(self.B_0 - self.phi_0)
        self.R_fashe = np.array(
            [
                -self.R_fashedian * np.sin(mu0) * np.cos(np.deg2rad(self.Am)),
                self.R_fashedian * np.cos(mu0),
                self.R_fashedian * np.sin(mu0) * np.sin(np.deg2rad(self.Am)),
            ],
            dtype=float,
        )

    @staticmethod # 静态方法，不需要self里的值，只跟传入的参数有关
    def is_casadi_value(x):
        return isinstance(x, (ca.SX, ca.MX, ca.DM)) # x是np返回False，是CasADi返回True

    def get_lib(self, x):
        return ca if self.is_casadi_value(x) else np

    def _is_ca_backend(self, *values): # 接收任意数量的参数
        for v in values:                      # 遍历所有传入的参数
            if self.is_casadi_value(v):       # 如果有任何一个是 CasADi 类型
                return True                   # 立即返回 True
        return False                          # 全部都不是才返回 False

    def _to_vec(self, arr, use_ca): # 传入arr数组，根据use_ca决定返回CasADi DM还是NumPy array
        # use_ca 来自 _is_ca_backend的判断
        return ca.DM(arr) if use_ca else np.asarray(arr, dtype=float)

    @staticmethod
    def _ca_rot_z(angle):
        return ca.vertcat(
            ca.horzcat(ca.cos(angle), ca.sin(angle), 0),
            ca.horzcat(-ca.sin(angle), ca.cos(angle), 0),
            ca.horzcat(0, 0, 1),
        )

    @staticmethod
    def wrap_angle_deg(angle_deg):
        """Return the shortest signed angle difference in degrees."""
        angle_rad = angle_deg * np.pi / 180.0
        return ca.atan2(ca.sin(angle_rad), ca.cos(angle_rad)) * 180.0 / np.pi
    
    def atmosphere(self, h):
        use_ca = self._is_ca_backend(h)
        rho = self.rho0 * (ca.exp(-h / self.hs) if use_ca else np.exp(-h / self.hs))
        if use_ca:
            return ca.if_else(h < self.h_atm_cutoff, rho, 0.0)
        return np.where(np.asarray(h) < self.h_atm_cutoff, rho, 0.0)

    def gravity(self, r_faguan):
        """发惯系下的重力加速度。

        使用带 J2 修正的近似重力模型：
        - 中心引力给出主项
        - J2 项表示地球扁率带来的非球形扰动
        输入 r_faguan 是相对发射点的位移，内部会加上发射点的固定偏置。
        """
        use_ca = self._is_ca_backend(r_faguan)
        r = r_faguan
        R_fashe = self._to_vec(self.R_fashe, use_ca)
        omega_vec = self._to_vec(self.omega_e_faguan, use_ca)

        r1 = r + R_fashe
        rnorm = ca.sqrt(ca.dot(r1, r1)) if use_ca else np.linalg.norm(r1)

        phi_z = (
            ca.asin(ca.dot(r1, omega_vec) / (rnorm * self.omega_e))
            if use_ca
            else np.arcsin(np.dot(r1, omega_vec) / (rnorm * self.omega_e))
        )

        sin_phi = ca.sin(phi_z) if use_ca else np.sin(phi_z)
        g_r = self.GM / (rnorm**2) * (
            1
            + 1.5
            * self.J2
            * (self.ae**2)
            / (rnorm**2)
            * (1 - 5 * sin_phi**2)
        )
        g_omega = (
            2
            * self.GM
            / (rnorm**2)
            * 1.5
            * self.J2
            * (self.ae**2)
            / (rnorm**2)
            * sin_phi
        )

        return -g_r * (r1 / rnorm) - g_omega * omega_vec

    def llh(self, r_faguan, t):
        """把发惯系位置转换成地理纬度、经度和高度。

        这个函数常用于：
        - 画飞行轨迹
        - 计算空气密度
        - 看火箭在地球表面的覆盖区域
        """
        use_ca = self._is_ca_backend(r_faguan, t)
        R_fashe = self._to_vec(self.R_fashe, use_ca)
        EG = self._to_vec(self.EG, use_ca)

        r_sim1 = r_faguan + R_fashe
        r_sim2 = EG @ r_sim1

        omega_jiao = self.omega_e * t
        E_jiao = self._ca_rot_z(omega_jiao) if use_ca else np.array(
            [
                [np.cos(omega_jiao), np.sin(omega_jiao), 0.0],
                [-np.sin(omega_jiao), np.cos(omega_jiao), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        r_digu = E_jiao @ r_sim2
        Xc, Yc, Zc = r_digu[0], r_digu[1], r_digu[2]

        rnorm = ca.sqrt(Xc**2 + Yc**2 + Zc**2) if use_ca else np.sqrt(Xc**2 + Yc**2 + Zc**2)
        lat_rad = ca.asin(Zc / rnorm) if use_ca else np.arcsin(Zc / rnorm)
        lon_rad = ca.atan2(Yc, Xc) if use_ca else np.arctan2(Yc, Xc)

        lat = lat_rad * 180.0 / np.pi
        lon = lon_rad * 180.0 / np.pi

        sin_lat = ca.sin(lat_rad) if use_ca else np.sin(lat_rad)
        cos_lat = ca.cos(lat_rad) if use_ca else np.cos(lat_rad)
        R_surface = (self.a0 * self.b0) / np.sqrt(
            self.a0**2 * sin_lat**2 + self.b0**2 * cos_lat**2
        )
        h = rnorm - R_surface

        return lat, lon, h

    def ECI(self, r_faguan, v_faguan, Omega_G=0.0):
        """把发惯系状态转到地心惯性系(ECI)。"""
        use_ca = self._is_ca_backend(r_faguan, v_faguan)
        EG = self._to_vec(self.EG, use_ca)
        R_fashe = self._to_vec(self.R_fashe, use_ca)

        EI = self._ca_rot_z(Omega_G) if use_ca else np.array(
            [
                [np.cos(Omega_G), np.sin(Omega_G), 0.0],
                [-np.sin(Omega_G), np.cos(Omega_G), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        r_sim2 = EG @ (r_faguan + R_fashe)
        v_sim2 = EG @ v_faguan
        r_eci = EI @ r_sim2
        v_eci = EI @ v_sim2
        return r_eci, v_eci

    def orbit_six(self, r_vec, v_vec):
        """根据位置和速度计算轨道六根数。

        返回值顺序：
        a, e, i, Omega, omega, f
        分别对应半长轴、偏心率、倾角、升交点赤经、近地点幅角、真近点角。
        """
        use_ca = self._is_ca_backend(r_vec, v_vec)

        if use_ca:
            rnorm = ca.sqrt(ca.dot(r_vec, r_vec) + 1e-6)
            vnorm = ca.sqrt(ca.dot(v_vec, v_vec) + 1e-6)
            h_vec = ca.cross(r_vec, v_vec)
            hnorm = ca.sqrt(ca.dot(h_vec, h_vec) + 1e-6)
            e_vec = ca.cross(v_vec, h_vec) / self.GM - r_vec / rnorm
            e = ca.sqrt(ca.dot(e_vec, e_vec) + 1e-12)
            k = ca.DM([0.0, 0.0, 1.0])
            n_vec = ca.cross(k, h_vec)
            n = ca.sqrt(ca.dot(n_vec, n_vec) + 1e-12)

            i_rad = ca.acos(h_vec[2] / hnorm)
            Omega_rad = ca.if_else(n > 1e-8, ca.atan2(n_vec[1], n_vec[0]), 0.0)
            omega_rad = ca.if_else(
                ca.logic_and(n > 1e-8, e > 1e-12),
                ca.atan2(ca.dot(ca.cross(n_vec, e_vec), h_vec) / hnorm, ca.dot(n_vec, e_vec)),
                0.0,
            )
            f_rad = ca.if_else(
                e > 1e-12,
                ca.atan2(ca.dot(ca.cross(e_vec, r_vec), h_vec) / hnorm, ca.dot(e_vec, r_vec)),
                ca.atan2(ca.dot(ca.cross(n_vec, r_vec), h_vec) / hnorm, ca.dot(n_vec, r_vec)),
            )

            eps = vnorm**2 / 2 - self.GM / rnorm
            a = -self.GM / (2 * eps)
            i_deg = i_rad * 180.0 / np.pi
            Omega_deg = ca.fmod(Omega_rad * 180.0 / np.pi + 360.0, 360.0)
            omega_deg = ca.fmod(omega_rad * 180.0 / np.pi + 360.0, 360.0)
            f_deg = ca.fmod(f_rad * 180.0 / np.pi + 360.0, 360.0)
            return a, e, i_deg, Omega_deg, omega_deg, f_deg

        rnorm = np.linalg.norm(r_vec) + 1e-12
        vnorm = np.linalg.norm(v_vec) + 1e-12
        h_vec = np.cross(r_vec, v_vec)
        hnorm = np.linalg.norm(h_vec) + 1e-12
        e_vec = np.cross(v_vec, h_vec) / self.GM - r_vec / rnorm
        e = np.linalg.norm(e_vec)

        k = np.array([0.0, 0.0, 1.0])
        n_vec = np.cross(k, h_vec)
        n = np.linalg.norm(n_vec)

        i_rad = np.arccos(np.clip(h_vec[2] / hnorm, -1.0, 1.0))
        Omega_rad = np.arctan2(n_vec[1], n_vec[0]) if n > 1e-8 else 0.0

        if n > 1e-8 and e > 1e-12:
            omega_rad = np.arctan2(np.dot(np.cross(n_vec, e_vec), h_vec) / hnorm, np.dot(n_vec, e_vec))
        else:
            omega_rad = 0.0

        if e > 1e-12:
            f_rad = np.arctan2(np.dot(np.cross(e_vec, r_vec), h_vec) / hnorm, np.dot(e_vec, r_vec))
        else:
            f_rad = np.arctan2(np.dot(np.cross(n_vec, r_vec), h_vec) / hnorm, np.dot(n_vec, r_vec))

        eps = vnorm**2 / 2 - self.GM / rnorm
        a = -self.GM / (2 * eps)

        i_deg = np.degrees(i_rad)
        Omega_deg = np.mod(np.degrees(Omega_rad) + 360.0, 360.0)
        omega_deg = np.mod(np.degrees(omega_rad) + 360.0, 360.0)
        f_deg = np.mod(np.degrees(f_rad) + 360.0, 360.0)
        return a, e, i_deg, Omega_deg, omega_deg, f_deg

    def export_legacy_params(self):
        """导出与旧脚本同名的参数字典，便于 donglixue_fault 这类脚本平滑迁移。"""
        # 这相当于把类里的属性重新导成旧式全局变量风格，方便逐步重构旧代码。
        return {
            "target": dict(self.target),
            "number1": self.number1,
            "mdot1": self.mdot1,
            "c_eff1": self.c_eff1,
            "P1": self.P1,
            "E": self.E,
            "B_0": self.B_0,
            "Am": self.Am,
            "m01": self.m01,
            "GM": self.GM,
            "R_fashedian": self.R_fashedian,
            "omega_e": self.omega_e,
            "v_e": self.v_e,
            "t_yiji": self.t_yiji,
            "m_pao": self.m_pao,
            "m_gan": self.m_gan,
            "number2": self.number2,
            "mdot2": self.mdot2,
            "c_eff2": self.c_eff2,
            "P2": self.P2,
            "t_zhengliu": self.t_zhengliu,
            "m_zhengliu": self.m_zhengliu,
            "a0": self.a0,
            "b0": self.b0,
            "e2": self.e2,
            "phi_0_rad": self.phi_0_rad,
            "phi_0": self.phi_0,
            "EG": self.EG.copy(), # 防止外部修改影响内部状态
            "omega_e_faguan": self.omega_e_faguan.copy(),
            "R_fashe": self.R_fashe.copy(),
            "v0": self.v0.copy(),
            "R0": self.r0.copy(),
            "y0": self.y0.copy(),
            "Cd": self.Cd,
            "S": self.S,
        }
