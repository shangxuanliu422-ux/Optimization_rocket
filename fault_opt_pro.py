from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import casadi as ca
import numpy as np

from core.env_models import EarthEnv
from core.ocp_blocks import (
    add_rk4_segment_constraints,
    apply_control_angle_bounds,
    apply_dphi_rate_limit,
    apply_mass_lower_bound,
    configure_ipopt_solver,
)
from core.rocket_stage import Rocket
from core.utils import interpolate_solution
from core.visual import plot_from_npz


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INIT_GUESS_FILE = RESULTS_DIR / "biaozhundandao.npz"
COMPARE_FILE = RESULTS_DIR / "biaozhundandao.npz"

STAGE1_TIMED = "timed"
STAGE1_FREE = "free"
T4_FIXED = "fixed"
T4_FREE = "free"

ENABLE_DPHI_LIMIT = True
ENABLE_SMOOTHNESS = True
ENABLE_ALPHA_LIMIT = True
ALPHA_MIN_DEG = -60.0
ALPHA_MAX_DEG = 30.0
ENABLE_QALPHA_LIMIT = True
QALPHA_LIMIT = 5000.0  # Pa rad = N rad / m^2
DPHI_MAX_DEG_PER_STEP = 2

# =============================================================================
# Direct-run settings
# =============================================================================
# Strategy 1: stage-1 timed cutoff at 200 s, fixed T4.
# Strategy 2: stage-1 timed cutoff at 200 s, optimized T4.
# Strategy 3: stage-1 free cutoff before propellant depletion, fixed T4.
# Strategy 4: stage-1 free cutoff before propellant depletion, optimized T4.
STRATEGY_ID = 1

TE = 195
KAPPA = 0.0014
DT = 1.0
MAKE_PLOT = True
RUN_ALL_STRATEGIES = False

# T4 settings.
T4_GUESS = 239.0
T4_MIN = 180.0
T4_MAX = 340.0
T4_FIXED_DURATION = 239.0262

# Free stage-1 cutoff settings.
STAGE1_FREE_MIN_AFTER_FAULT = 1e-3
STAGE1_FREE_GUESS = None  # None means initialize near the nominal 200 s cutoff.

# Emergency cutoff settings. When stage-1 thrust is almost gone, minimizing T4
# alone can treat stage-1 free cutoff as a free coast-time variable. This penalty
# makes the optimizer prefer jettisoning stage 1 immediately after the fault.
AUTO_EMERGENCY_CUTOFF = False
EMERGENCY_KAPPA_THRESHOLD = 0.95
EMERGENCY_STAGE1_TIME_WEIGHT = 10.0
W_STAGE1_TIME = 0.0


@dataclass
class FaultProConfig:
    """Unified fault optimizer for four strategy combinations.

    The physical fault is always:
        P1_fault = P1 * (1 - kappa)
        mdot1_fault = mdot1 * (1 - kappa)

    stage1_cutoff_strategy:
        "timed": stage 1 shuts down at env.t_yiji, normally 200 s.
        "free": optimizer chooses the post-fault burn duration before propellant depletion.

    t4_strategy:
        "fixed": T4 keeps the nominal duration from the initial guess file.
        "free": optimizer chooses T4 inside [T4_min, T4_max].

    When stage 1 shuts down before the nominal propellant budget is depleted,
    the remaining stage-1 propellant is discarded at stage separation. Otherwise
    the second stage would unrealistically carry unused first-stage propellant.
    """

    dt: float = 1.0
    te: float = 50.0
    kappa: float = 0.2

    stage1_cutoff_strategy: str = STAGE1_TIMED
    t4_strategy: str = T4_FREE

    # Only used by free stage-1 cutoff. Keep tiny but positive to avoid zero-length RK4.
    stage1_free_min_after_fault: float = 1e-3
    stage1_free_guess: float | None = None

    # T4 settings. T4_guess is used for mesh sizing when T4 is free.
    T4_guess: float = 239.0
    T4_min: float = 180.0
    T4_max: float = 340.0
    T4_fixed: float | None = None

    dphi_max_deg_per_step: float = DPHI_MAX_DEG_PER_STEP
    w_ctrl: float = 1000.0
    w_stage1_time: float = 0.0
    enable_alpha_limit: bool = ENABLE_ALPHA_LIMIT
    alpha_min_deg: float = ALPHA_MIN_DEG
    alpha_max_deg: float = ALPHA_MAX_DEG
    enable_qalpha_limit: bool = ENABLE_QALPHA_LIMIT
    qalpha_limit: float = QALPHA_LIMIT

    init_guess_file: Path = INIT_GUESS_FILE
    compare_file: Path = COMPARE_FILE
    result_file: Path | None = None

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


def _resolve_input_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates = [
        raw,
        BASE_DIR / raw,
        RESULTS_DIR / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Initial guess file not found: {path}\nChecked:\n{checked}")


def _resolve_output_path(cfg: FaultProConfig) -> Path:
    if cfg.result_file is not None:
        raw = Path(cfg.result_file)
        return raw if raw.is_absolute() else RESULTS_DIR / raw.name

    name = f"fault_opt_pro_stage1-{cfg.stage1_cutoff_strategy}_t4-{cfg.t4_strategy}.npz"
    return RESULTS_DIR / name


def _strategy_to_modes(strategy_id: int) -> tuple[str, str]:
    strategy_map = {
        1: (STAGE1_TIMED, T4_FIXED),
        2: (STAGE1_TIMED, T4_FREE),
        3: (STAGE1_FREE, T4_FIXED),
        4: (STAGE1_FREE, T4_FREE),
    }
    try:
        return strategy_map[int(strategy_id)]
    except KeyError as exc:
        raise ValueError("STRATEGY_ID must be one of 1, 2, 3, 4") from exc


def _direct_config(strategy_id: int = STRATEGY_ID) -> FaultProConfig:
    stage1_cutoff, t4_strategy = _strategy_to_modes(strategy_id)
    return FaultProConfig(
        dt=DT,
        te=TE,
        kappa=KAPPA,
        stage1_cutoff_strategy=stage1_cutoff,
        t4_strategy=t4_strategy,
        stage1_free_min_after_fault=STAGE1_FREE_MIN_AFTER_FAULT,
        stage1_free_guess=STAGE1_FREE_GUESS,
        T4_guess=T4_GUESS,
        T4_min=T4_MIN,
        T4_max=T4_MAX,
        T4_fixed=T4_FIXED_DURATION,
        w_stage1_time=_stage1_time_weight_for_kappa(KAPPA),
        enable_alpha_limit=ENABLE_ALPHA_LIMIT,
        alpha_min_deg=ALPHA_MIN_DEG,
        alpha_max_deg=ALPHA_MAX_DEG,
        enable_qalpha_limit=ENABLE_QALPHA_LIMIT,
        qalpha_limit=QALPHA_LIMIT,
    )


def _validate_config(cfg: FaultProConfig, env: EarthEnv) -> None:
    if cfg.stage1_cutoff_strategy not in {STAGE1_TIMED, STAGE1_FREE}:
        raise ValueError(
            f"stage1_cutoff_strategy must be '{STAGE1_TIMED}' or '{STAGE1_FREE}', "
            f"got {cfg.stage1_cutoff_strategy!r}"
        )
    if cfg.t4_strategy not in {T4_FIXED, T4_FREE}:
        raise ValueError(f"t4_strategy must be '{T4_FIXED}' or '{T4_FREE}', got {cfg.t4_strategy!r}")
    if not (0.0 <= cfg.kappa < 1.0):
        raise ValueError(f"kappa must satisfy 0 <= kappa < 1, got {cfg.kappa}")
    if not (0.0 < cfg.te < env.t_yiji):
        raise ValueError(f"te must be in (0, {env.t_yiji}), got {cfg.te}")
    if cfg.dt <= 0.0:
        raise ValueError(f"dt must be positive, got {cfg.dt}")
    if cfg.alpha_min_deg >= cfg.alpha_max_deg:
        raise ValueError(
            f"alpha_min_deg must be smaller than alpha_max_deg, "
            f"got {cfg.alpha_min_deg} >= {cfg.alpha_max_deg}"
        )


def _nominal_t4_duration(data: np.lib.npyio.NpzFile, env: EarthEnv) -> float:
    t4 = np.asarray(data["t4"], dtype=float)
    if len(t4) >= 2:
        return float(t4[-1] - t4[0])
    return float(t4[-1] - env.t_yiji - env.t_zhengliu)


def _stage1_duration_plan(env: EarthEnv, cfg: FaultProConfig) -> tuple[float, float, float]:
    scheduled_after_fault = env.t_yiji - cfg.te

    if cfg.stage1_cutoff_strategy == STAGE1_TIMED:
        return scheduled_after_fault, scheduled_after_fault, scheduled_after_fault

    max_after_fault = scheduled_after_fault / (1.0 - cfg.kappa)
    min_after_fault = max(float(cfg.stage1_free_min_after_fault), 1e-9)
    if min_after_fault > max_after_fault:
        raise ValueError(
            "stage1_free_min_after_fault is larger than the propellant-depletion limit: "
            f"{min_after_fault} > {max_after_fault}"
        )

    guess = cfg.stage1_free_guess
    if guess is None:
        guess = min_after_fault if cfg.w_stage1_time > 0.0 else scheduled_after_fault
    guess = float(np.clip(guess, min_after_fault, max_after_fault))
    return min_after_fault, max_after_fault, guess


def _stage1_time_weight_for_kappa(kappa: float, base_weight: float = W_STAGE1_TIME) -> float:
    if AUTO_EMERGENCY_CUTOFF and kappa >= EMERGENCY_KAPPA_THRESHOLD:
        return max(float(base_weight), float(EMERGENCY_STAGE1_TIME_WEIGHT))
    return float(base_weight)


def _make_mesh_count(duration: float, dt: float) -> int:
    return max(1, int(np.round(float(duration) / float(dt))))


def _smoothness_cost(*controls) -> ca.MX:
    cost = 0
    for U in controls:
        d1 = ca.diff(U, 1, 1)
        cost += ca.sumsqr(d1)
        if U.shape[1] >= 3:
            cost += ca.sumsqr(ca.diff(d1, 1, 1))
    return cost


def _apply_alpha_limit(
    opti: ca.Opti,
    env: EarthEnv,
    segment_specs,
    alpha_min_deg: float,
    alpha_max_deg: float,
) -> None:
    omega_vec = ca.DM(env.omega_e_faguan)
    r_launch = ca.DM(env.R_fashe)
    alpha_min = np.deg2rad(float(alpha_min_deg))
    alpha_max = np.deg2rad(float(alpha_max_deg))
    eps = 1e-9

    for X, U, n_steps, dt, t0 in segment_specs:
        for k in range(n_steps + 1):
            t_k = t0 + k * dt
            r = X[0:3, k]
            v = X[3:6, k]
            v_rel = v - ca.cross(omega_vec, r + r_launch)
            horizontal_speed = ca.sqrt(v_rel[0] ** 2 + v_rel[2] ** 2 + eps)
            theta_rel = ca.atan2(v_rel[1], horizontal_speed)
            alpha = U[0, k] - theta_rel
            opti.subject_to(opti.bounded(alpha_min, alpha, alpha_max))


def _apply_qalpha_limit(opti: ca.Opti, env: EarthEnv, segment_specs, qalpha_limit: float) -> None:
    omega_vec = ca.DM(env.omega_e_faguan)
    r_launch = ca.DM(env.R_fashe)
    eps = 1e-9

    for X, U, n_steps, dt, t0 in segment_specs:
        for k in range(n_steps + 1):
            t_k = t0 + k * dt
            r = X[0:3, k]
            v = X[3:6, k]
            _, _, h = env.llh(r, t_k)
            rho = env.atmosphere(h)

            v_rel = v - ca.cross(omega_vec, r + r_launch)
            v_rel_sq = ca.dot(v_rel, v_rel)
            horizontal_speed = ca.sqrt(v_rel[0] ** 2 + v_rel[2] ** 2 + eps)
            theta_rel = ca.atan2(v_rel[1], horizontal_speed)
            alpha = U[0, k] - theta_rel
            q_alpha = 0.5 * rho * v_rel_sq * alpha
            opti.subject_to(opti.bounded(-qalpha_limit, q_alpha, qalpha_limit))


def build_and_solve_fault_pro(cfg: FaultProConfig, make_plot: bool = False) -> Path:
    env = EarthEnv(target=cfg.target)
    _validate_config(cfg, env)

    init_guess_path = _resolve_input_path(cfg.init_guess_file)
    data = np.load(init_guess_path)
    required = ["X1", "U1", "X3", "U3", "X4", "U4", "t1", "t3", "t4"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Initial guess file is missing required arrays: {missing}")

    P1_fault = env.P1 * (1.0 - cfg.kappa)
    mdot1_fault = env.mdot1 * (1.0 - cfg.kappa)
    stage1_fault = Rocket(thrust=P1_fault, mdot=mdot1_fault, name="Stage-1-Fault", Cd=env.Cd, S=env.S)
    stage2 = Rocket(thrust=env.P2, mdot=env.mdot2, name="Stage-2", Cd=env.Cd, S=env.S)

    T1 = float(cfg.te)
    T3 = float(env.t_zhengliu)
    T2_min, T2_max, T2_guess = _stage1_duration_plan(env, cfg)
    T2_mesh_ref = env.t_yiji - cfg.te

    T4_nominal = _nominal_t4_duration(data, env)
    if cfg.t4_strategy == T4_FIXED:
        T4_guess = float(T4_nominal if cfg.T4_fixed is None else cfg.T4_fixed)
        T4_fixed = T4_guess
    else:
        T4_guess = float(cfg.T4_guess)
        T4_fixed = None

    N1 = _make_mesh_count(T1, cfg.dt)
    N2 = _make_mesh_count(T2_mesh_ref, cfg.dt)
    N3 = _make_mesh_count(T3, cfg.dt)
    N4 = _make_mesh_count(T4_guess, cfg.dt)

    t_1 = np.linspace(0.0, T1, N1 + 1)
    t_2_guess = np.linspace(T1, T1 + T2_guess, N2 + 1)
    t_3_guess = np.linspace(T1 + T2_guess, T1 + T2_guess + T3, N3 + 1)
    t_4_guess = np.linspace(T1 + T2_guess + T3, T1 + T2_guess + T3 + T4_guess, N4 + 1)

    X1_ref = interpolate_solution(t_1, data["t1"], data["X1"])
    U1_ref = interpolate_solution(t_1, data["t1"], data["U1"])
    X2_guess = interpolate_solution(t_2_guess, data["t1"], data["X1"])
    U2_guess = interpolate_solution(t_2_guess, data["t1"], data["U1"])
    X3_guess = interpolate_solution(t_3_guess, data["t3"], data["X3"])
    U3_guess = interpolate_solution(t_3_guess, data["t3"], data["U3"])
    X4_guess = interpolate_solution(t_4_guess, data["t4"], data["X4"])
    U4_guess = interpolate_solution(t_4_guess, data["t4"], data["U4"])

    X1 = ca.DM(X1_ref)
    U1 = ca.DM(U1_ref)

    opti = ca.Opti()

    if cfg.stage1_cutoff_strategy == STAGE1_FREE:
        T2 = opti.variable()
        opti.subject_to(opti.bounded(T2_min, T2, T2_max))
        opti.set_initial(T2, T2_guess)
    else:
        T2 = T2_guess

    if cfg.t4_strategy == T4_FREE:
        T4 = opti.variable()
        opti.subject_to(opti.bounded(cfg.T4_min, T4, cfg.T4_max))
        opti.set_initial(T4, T4_guess)
    else:
        T4 = T4_fixed

    dt_2 = T2 / N2
    dt_3 = T3 / N3
    dt_4 = T4 / N4

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
        apply_dphi_rate_limit(opti, U3, dt_3, cfg.dphi_max_deg_per_step)
        apply_dphi_rate_limit(opti, U4, dt_4, cfg.dphi_max_deg_per_step)
    segment_specs = [
        (X2, U2, N2, dt_2, T1),
        (X3, U3, N3, dt_3, T1 + T2),
        (X4, U4, N4, dt_4, T1 + T2 + T3),
    ]
    if cfg.enable_alpha_limit:
        _apply_alpha_limit(opti, env, segment_specs, cfg.alpha_min_deg, cfg.alpha_max_deg)
    if cfg.enable_qalpha_limit:
        _apply_qalpha_limit(opti, env, segment_specs, cfg.qalpha_limit)

    ode_stage1_fault = lambda t, x, u: stage1_fault.dynamics(t, x, u, env)
    ode_stage2 = lambda t, x, u: stage2.dynamics(t, x, u, env)

    add_rk4_segment_constraints(opti, X2, U2, N2, ode_stage1_fault, dt_2, t0=T1)
    add_rk4_segment_constraints(opti, X3, U3, N3, ode_stage2, dt_3, t0=T1 + T2)
    add_rk4_segment_constraints(opti, X4, U4, N4, ode_stage2, dt_4, t0=T1 + T2 + T3)

    opti.subject_to(X2[:, 0] == X1[:, -1])
    opti.subject_to(U2[:, 0] == U1[:, -1])

    stage1_propellant_budget = env.mdot1 * env.t_yiji
    stage1_propellant_burned = env.mdot1 * T1 + mdot1_fault * T2
    leftover_stage1_propellant = stage1_propellant_budget - stage1_propellant_burned
    if cfg.stage1_cutoff_strategy == STAGE1_FREE:
        opti.subject_to(leftover_stage1_propellant >= -1e-6)
    elif leftover_stage1_propellant < -1e-6:
        raise ValueError(
            "Timed stage-1 cutoff burns more than the nominal stage-1 propellant budget. "
            f"leftover_stage1_propellant={leftover_stage1_propellant}"
        )

    opti.subject_to(X3[0:6, 0] == X2[0:6, -1])
    opti.subject_to(X3[6, 0] == X2[6, -1] - env.m_pao - leftover_stage1_propellant)
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

    err_O = env.wrap_angle_deg(O_fin - O_t)
    err_w = env.wrap_angle_deg(w_fin - w_t)
    err_f = env.wrap_angle_deg(f_fin - f_t)
    opti.subject_to(opti.bounded(-tol["Omega"], err_O, tol["Omega"]))
    opti.subject_to(opti.bounded(-tol["omega"], err_w, tol["omega"]))
    opti.subject_to(opti.bounded(-tol["f"], err_f, tol["f"]))

    objective = 0
    if cfg.t4_strategy == T4_FREE:
        objective += T4
    if ENABLE_SMOOTHNESS:
        objective += cfg.w_ctrl * _smoothness_cost(U2, U3, U4)
    if cfg.stage1_cutoff_strategy == STAGE1_FREE and cfg.w_stage1_time != 0.0:
        objective += cfg.w_stage1_time * T2
    opti.minimize(objective)

    configure_ipopt_solver(opti, cfg.solver_opts)

    print("Starting fault_opt_pro solve ...")
    print(
        "Strategy: "
        f"stage1_cutoff={cfg.stage1_cutoff_strategy}, "
        f"t4={cfg.t4_strategy}, te={cfg.te:.3f}, kappa={cfg.kappa:.4f}"
    )
    if cfg.enable_alpha_limit:
        print(f"Alpha limit: {cfg.alpha_min_deg:.3f} deg <= alpha <= {cfg.alpha_max_deg:.3f} deg")
    if cfg.enable_qalpha_limit:
        print(f"q-alpha limit: |q*alpha| <= {cfg.qalpha_limit:.3f} Pa rad")
    if cfg.stage1_cutoff_strategy == STAGE1_FREE:
        print(f"Stage-1 post-fault burn bounds: [{T2_min:.6f}, {T2_max:.6f}] s")
        print(f"Stage-1 post-fault burn initial guess = {T2_guess:.6f} s")
        print(f"Stage-1 post-fault mesh reference = {T2_mesh_ref:.6f} s, N2 = {N2}")
        print(f"Stage-1 time penalty weight = {cfg.w_stage1_time:.6g}")
    if cfg.enable_qalpha_limit:
        print(f"q-alpha path limit = {cfg.qalpha_limit:.6g} Pa rad")
    else:
        print("q-alpha path limit disabled")

    sol = opti.solve()

    T2_opt = float(sol.value(T2)) if cfg.stage1_cutoff_strategy == STAGE1_FREE else float(T2_guess)
    T4_opt = float(sol.value(T4)) if cfg.t4_strategy == T4_FREE else float(T4_fixed)
    leftover_stage1_propellant_opt = (
        float(sol.value(leftover_stage1_propellant))
        if cfg.stage1_cutoff_strategy == STAGE1_FREE
        else float(leftover_stage1_propellant)
    )

    t_2_opt = np.linspace(T1, T1 + T2_opt, N2 + 1)
    t_3_opt = np.linspace(T1 + T2_opt, T1 + T2_opt + T3, N3 + 1)
    t_4_opt = np.linspace(T1 + T2_opt + T3, T1 + T2_opt + T3 + T4_opt, N4 + 1)

    X1_v = np.asarray(X1)
    U1_v = np.asarray(U1)
    X2_v = sol.value(X2)
    U2_v = sol.value(U2)
    X3_v = sol.value(X3)
    U3_v = sol.value(U3)
    X4_v = sol.value(X4)
    U4_v = sol.value(U4)

    X1_full = np.hstack([X1_v, X2_v[:, 1:]])
    U1_full = np.hstack([U1_v, U2_v[:, 1:]])
    t1_full = np.hstack([t_1, t_2_opt[1:]])

    result_path = _resolve_output_path(cfg)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        result_path,
        X1=X1_full,
        U1=U1_full,
        X3=X3_v,
        U3=U3_v,
        X4=X4_v,
        U4=U4_v,
        t1=t1_full,
        t3=t_3_opt,
        t4=t_4_opt,
        te=np.array(cfg.te),
        kappa=np.array(cfg.kappa),
        P1_fault=np.array(P1_fault),
        mdot1_fault=np.array(mdot1_fault),
        leftover_stage1_propellant=np.array(leftover_stage1_propellant_opt),
        stage1_cutoff_strategy=np.array(cfg.stage1_cutoff_strategy),
        t4_strategy=np.array(cfg.t4_strategy),
        stage1_cutoff_time=np.array(T1 + T2_opt),
        stage1_after_fault_duration=np.array(T2_opt),
        T4_duration=np.array(T4_opt),
    )

    print(f"fault_opt_pro done: {result_path.resolve()}")
    print(f"Stage-1 cutoff time = {T1 + T2_opt:.6f} s")
    print(f"Discarded leftover stage-1 propellant = {leftover_stage1_propellant_opt:.6f} kg")
    print(f"T4 duration = {T4_opt:.6f} s")

    if make_plot:
        plot_from_npz(
            str(result_path),
            env=env,
            compare_npz=str(_resolve_input_path(cfg.compare_file)),
            label_current="Fault",
            label_compare="Nominal",
            show=True,
        )

    return result_path


def _build_config_from_args() -> tuple[FaultProConfig, bool, bool]:
    parser = argparse.ArgumentParser(description="Unified fault optimizer with four strategy combinations.")
    parser.add_argument("--strategy", type=int, choices=[1, 2, 3, 4], default=STRATEGY_ID)
    parser.add_argument("--stage1-cutoff", choices=[STAGE1_TIMED, STAGE1_FREE], default=None)
    parser.add_argument("--t4", choices=[T4_FIXED, T4_FREE], default=None)
    parser.add_argument("--te", type=float, default=TE)
    parser.add_argument("--kappa", type=float, default=KAPPA)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--t4-guess", type=float, default=T4_GUESS)
    parser.add_argument("--t4-min", type=float, default=T4_MIN)
    parser.add_argument("--t4-max", type=float, default=T4_MAX)
    parser.add_argument("--t4-fixed", type=float, default=T4_FIXED_DURATION)
    parser.add_argument("--stage1-free-min-after-fault", type=float, default=STAGE1_FREE_MIN_AFTER_FAULT)
    parser.add_argument("--stage1-free-guess", type=float, default=STAGE1_FREE_GUESS)
    parser.add_argument("--w-stage1-time", type=float, default=None)
    parser.add_argument("--alpha-min-deg", type=float, default=ALPHA_MIN_DEG)
    parser.add_argument("--alpha-max-deg", type=float, default=ALPHA_MAX_DEG)
    parser.add_argument("--no-alpha-limit", action="store_true")
    parser.add_argument("--qalpha-limit", type=float, default=QALPHA_LIMIT)
    parser.add_argument("--no-qalpha-limit", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--init-guess", type=Path, default=INIT_GUESS_FILE)
    parser.add_argument("--compare", type=Path, default=COMPARE_FILE)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()

    stage1_cutoff, t4_strategy = _strategy_to_modes(args.strategy)
    if args.stage1_cutoff is not None:
        stage1_cutoff = args.stage1_cutoff
    if args.t4 is not None:
        t4_strategy = args.t4
    w_stage1_time = (
        _stage1_time_weight_for_kappa(args.kappa)
        if args.w_stage1_time is None
        else float(args.w_stage1_time)
    )

    cfg = FaultProConfig(
        dt=args.dt,
        te=args.te,
        kappa=args.kappa,
        stage1_cutoff_strategy=stage1_cutoff,
        t4_strategy=t4_strategy,
        stage1_free_min_after_fault=args.stage1_free_min_after_fault,
        stage1_free_guess=args.stage1_free_guess,
        T4_guess=args.t4_guess,
        T4_min=args.t4_min,
        T4_max=args.t4_max,
        T4_fixed=args.t4_fixed,
        w_stage1_time=w_stage1_time,
        enable_alpha_limit=not args.no_alpha_limit,
        alpha_min_deg=args.alpha_min_deg,
        alpha_max_deg=args.alpha_max_deg,
        enable_qalpha_limit=not args.no_qalpha_limit,
        qalpha_limit=args.qalpha_limit,
        init_guess_file=args.init_guess,
        compare_file=args.compare,
        result_file=args.output,
    )
    return cfg, args.plot, args.run_all


def main() -> None:
    if len(sys.argv) == 1:
        cfg = _direct_config()
        make_plot = MAKE_PLOT
        run_all = RUN_ALL_STRATEGIES
    else:
        cfg, make_plot, run_all = _build_config_from_args()

    if not run_all:
        build_and_solve_fault_pro(cfg, make_plot=make_plot)
        return

    for stage1_cutoff in [STAGE1_TIMED, STAGE1_FREE]:
        for t4_strategy in [T4_FIXED, T4_FREE]:
            cfg_case = replace(
                cfg,
                stage1_cutoff_strategy=stage1_cutoff,
                t4_strategy=t4_strategy,
                result_file=None,
            )
            build_and_solve_fault_pro(cfg_case, make_plot=False)


if __name__ == "__main__":
    main()
