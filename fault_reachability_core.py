from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np

from core.env_models import EarthEnv
from core.ocp_blocks import (
    add_rk4_segment_constraints,
    apply_control_angle_bounds,
    apply_dphi_rate_limit,
    apply_mass_lower_bound,
)
from core.rocket_stage import Rocket
from core.utils import interpolate_solution


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

INIT_GUESS_FILE = RESULTS_DIR / "biaozhundandao.npz"

S0 = "S0"
S1 = "S1"
S2 = "S2"
S3 = "S3"

TERMINAL_L0 = "L0"
TERMINAL_L1 = "L1"
TERMINAL_L2 = "L2"
TERMINAL_L3 = "L3"


@dataclass(frozen=True)
class StrategySpec:
    name: str
    optimize_stage1_time: bool
    optimize_t4: bool
    description: str


STRATEGIES = {
    S0: StrategySpec(S0, False, False, "fixed stage-1 timing, fixed T4"),
    S1: StrategySpec(S1, True, False, "stage-1 timing compensation, fixed T4"),
    S2: StrategySpec(S2, False, True, "fixed stage-1 timing, T4 compensation"),
    S3: StrategySpec(S3, True, True, "joint stage-1 and T4 compensation"),
}

TERMINAL_LEVELS = {
    TERMINAL_L0: ("a", "e", "i", "Omega", "omega", "f"),
    TERMINAL_L1: ("a", "e", "i", "Omega", "omega"),
    TERMINAL_L2: ("a", "Omega", "omega"),
    TERMINAL_L3: ("Omega", "omega"),
}


@dataclass
class ReachabilityConfig:
    dt: float = 1.0
    init_guess_file: Path = INIT_GUESS_FILE
    result_file: Path = RESULTS_DIR / "fault_strategy_reachability.npz"

    T4_guess: float = 239.0
    T4_min: float = 180.0
    T4_max: float = 450.0
    T4_fixed: float | None = None
    final_mass_min: float | None = None
    stage1_free_min_after_fault: float = 1e-3

    terminal_active: tuple[str, ...] = TERMINAL_LEVELS[TERMINAL_L0]

    dphi_max_deg_per_step: float = 0.8
    enable_dphi_limit: bool = True

    enable_smoothness: bool = True
    w_smoothness: float = 0.0
    w_stage1_time: float = 0.0

    enable_alpha_limit: bool = True
    alpha_limit_deg: float = 60.0
    enable_qalpha_limit: bool = True
    qalpha_limit: float = 5000.0

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
            "max_iter": 500,
            "print_level": 0,
            "sb": "yes",
        }
    )


@dataclass
class CaseResult:
    reachable: bool
    te: float
    kappa: float
    strategy: str
    T1_sep: float = np.nan
    T1_dep: float = np.nan
    T4_duration: float = np.nan
    delta_T1: float = np.nan
    eta1: float = np.nan
    delta_T4: float = np.nan
    leftover_stage1_propellant: float = np.nan
    final_mass: float = np.nan
    message: str = ""


def resolve_input_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates = [raw, BASE_DIR / raw, RESULTS_DIR / raw.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Initial guess file not found: {path}\nChecked:\n{checked}")


def nominal_t4_duration(data: np.lib.npyio.NpzFile, env: EarthEnv) -> float:
    t4 = np.asarray(data["t4"], dtype=float)
    if len(t4) >= 2:
        return float(t4[-1] - t4[0])
    return float(t4[-1] - env.t_yiji - env.t_zhengliu)


def make_mesh_count(duration: float, dt: float) -> int:
    return max(1, int(np.round(float(duration) / float(dt))))


def smoothness_cost(*controls) -> ca.MX:
    cost = 0
    for U in controls:
        d1 = ca.diff(U, 1, 1)
        cost += ca.sumsqr(d1)
        if U.shape[1] >= 3:
            cost += ca.sumsqr(ca.diff(d1, 1, 1))
    return cost


def apply_alpha_limit(opti: ca.Opti, env: EarthEnv, segment_specs, alpha_limit_deg: float) -> None:
    omega_vec = ca.DM(env.omega_e_faguan)
    r_launch = ca.DM(env.R_fashe)
    alpha_limit = np.deg2rad(float(alpha_limit_deg))
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
            opti.subject_to(opti.bounded(-alpha_limit, alpha, alpha_limit))


def apply_qalpha_limit(opti: ca.Opti, env: EarthEnv, segment_specs, qalpha_limit: float) -> None:
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


def configure_reachability_solver(opti: ca.Opti, solver_opts: dict) -> None:
    p_opts = {
        "expand": solver_opts.get("expand", True),
        "print_time": False,
    }
    s_opts = dict(solver_opts)
    s_opts.pop("expand", None)
    s_opts.pop("print_time", None)
    opti.solver("ipopt", p_opts, s_opts)


def _terminal_orbit_constraints(
    opti: ca.Opti,
    env: EarthEnv,
    r_final,
    v_final,
    tolerances: dict,
    active: tuple[str, ...],
) -> None:
    r_eci, v_eci = env.ECI(r_final, v_final)
    a_fin, e_fin, i_fin, O_fin, w_fin, f_fin = env.orbit_six(r_eci, v_eci)

    a_t = env.target["a"]
    e_t = env.target["e"]
    i_t = env.target["i_deg"]
    O_t = env.target["Omega_deg"]
    w_t = env.target["omega_deg"]
    f_t = env.target["f_deg"]

    active_set = set(active)
    tol = tolerances

    if "a" in active_set:
        opti.subject_to(opti.bounded(-tol["a"] / a_t, (a_fin - a_t) / a_t, tol["a"] / a_t))
    if "e" in active_set:
        opti.subject_to(opti.bounded(-tol["e"] / e_t, (e_fin - e_t) / e_t, tol["e"] / e_t))
    if "i" in active_set:
        opti.subject_to(opti.bounded(-tol["i"] / i_t, (i_fin - i_t) / i_t, tol["i"] / i_t))
    if "Omega" in active_set:
        opti.subject_to(opti.bounded(-tol["Omega"], env.wrap_angle_deg(O_fin - O_t), tol["Omega"]))
    if "omega" in active_set:
        opti.subject_to(opti.bounded(-tol["omega"], env.wrap_angle_deg(w_fin - w_t), tol["omega"]))
    if "f" in active_set:
        opti.subject_to(opti.bounded(-tol["f"], env.wrap_angle_deg(f_fin - f_t), tol["f"]))


def _case_failure_metadata(env: EarthEnv, te: float, kappa: float) -> tuple[float, float, float]:
    scheduled_after_fault = env.t_yiji - te
    depletion_after_fault = scheduled_after_fault / (1.0 - kappa)
    depletion_time = te + depletion_after_fault
    return scheduled_after_fault, depletion_after_fault, depletion_time


def solve_reachability_case(
    te: float,
    kappa: float,
    strategy_name: str,
    cfg: ReachabilityConfig,
    data: np.lib.npyio.NpzFile | None = None,
    env: EarthEnv | None = None,
) -> CaseResult:
    env = env or EarthEnv(target=cfg.target)
    if data is None:
        data = np.load(resolve_input_path(cfg.init_guess_file))

    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy_name!r}; choose from {list(STRATEGIES)}")
    strategy = STRATEGIES[strategy_name]

    if not (0.0 <= kappa < 1.0):
        return CaseResult(False, te, kappa, strategy_name, message="kappa outside [0, 1)")
    if not (0.0 <= te < env.t_yiji):
        return CaseResult(False, te, kappa, strategy_name, message=f"te outside [0, {env.t_yiji})")

    try:
        return _solve_reachability_case_impl(te, kappa, strategy, cfg, data, env)
    except Exception as exc:
        return CaseResult(False, te, kappa, strategy_name, message=str(exc))


def _solve_reachability_case_impl(
    te: float,
    kappa: float,
    strategy: StrategySpec,
    cfg: ReachabilityConfig,
    data: np.lib.npyio.NpzFile,
    env: EarthEnv,
) -> CaseResult:
    required = ["X1", "U1", "X3", "U3", "X4", "U4", "t1", "t3", "t4"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Initial guess file is missing arrays: {missing}")

    T1 = float(te)
    T3 = float(env.t_zhengliu)
    T2_nominal, T2_depletion, T1_dep = _case_failure_metadata(env, T1, kappa)
    T2_free_min = max(float(cfg.stage1_free_min_after_fault), 1e-9)
    T2_guess = T2_nominal
    T2_mesh_ref = T2_nominal

    T4_nominal = nominal_t4_duration(data, env)
    T4_fixed = T4_nominal if cfg.T4_fixed is None else float(cfg.T4_fixed)
    T4_guess = float(cfg.T4_guess if strategy.optimize_t4 else T4_fixed)

    if T2_guess <= 0.0:
        raise ValueError("post-fault stage-1 duration is not positive")

    N2 = make_mesh_count(T2_mesh_ref, cfg.dt)
    N3 = make_mesh_count(T3, cfg.dt)
    N4 = make_mesh_count(T4_guess, cfg.dt)

    t_2_guess = np.linspace(T1, T1 + T2_guess, N2 + 1)
    t_3_guess = np.linspace(T1 + T2_guess, T1 + T2_guess + T3, N3 + 1)
    t_4_guess = np.linspace(T1 + T2_guess + T3, T1 + T2_guess + T3 + T4_guess, N4 + 1)

    x_start_fault = interpolate_solution([T1], data["t1"], data["X1"])[:, 0]
    u_start_fault = interpolate_solution([T1], data["t1"], data["U1"])[:, 0]

    X2_guess = interpolate_solution(t_2_guess, data["t1"], data["X1"])
    U2_guess = interpolate_solution(t_2_guess, data["t1"], data["U1"])
    X3_guess = interpolate_solution(t_3_guess, data["t3"], data["X3"])
    U3_guess = interpolate_solution(t_3_guess, data["t3"], data["U3"])
    X4_guess = interpolate_solution(t_4_guess, data["t4"], data["X4"])
    U4_guess = interpolate_solution(t_4_guess, data["t4"], data["U4"])

    P1_fault = env.P1 * (1.0 - kappa)
    mdot1_fault = env.mdot1 * (1.0 - kappa)

    stage1_fault = Rocket(thrust=P1_fault, mdot=mdot1_fault, name="Stage-1-Fault", Cd=env.Cd, S=env.S)
    stage2 = Rocket(thrust=env.P2, mdot=env.mdot2, name="Stage-2", Cd=env.Cd, S=env.S)

    opti = ca.Opti()

    if strategy.optimize_stage1_time:
        T2 = opti.variable()
        opti.subject_to(opti.bounded(T2_free_min, T2, T2_depletion))
        opti.set_initial(T2, T2_guess)
    else:
        T2 = T2_nominal

    if strategy.optimize_t4:
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

    m_min = env.m_gan if cfg.final_mass_min is None else float(cfg.final_mass_min)
    apply_mass_lower_bound(opti, X4, m_min)
    apply_control_angle_bounds(
        opti,
        [
            (U2, -60, 90, -5, 3),
            (U3, -60, 90, -6, 2),
            (U4, -60, 90, -7, 1),
        ],
    )
    if cfg.enable_dphi_limit:
        apply_dphi_rate_limit(opti, U2, dt_2, cfg.dphi_max_deg_per_step)
    segment_specs = [
        (X2, U2, N2, dt_2, T1),
        (X3, U3, N3, dt_3, T1 + T2),
        (X4, U4, N4, dt_4, T1 + T2 + T3),
    ]
    if cfg.enable_alpha_limit:
        apply_alpha_limit(opti, env, segment_specs, cfg.alpha_limit_deg)
    if cfg.enable_qalpha_limit:
        apply_qalpha_limit(opti, env, segment_specs, cfg.qalpha_limit)

    ode_stage1_fault = lambda t, x, u: stage1_fault.dynamics(t, x, u, env)
    ode_stage2 = lambda t, x, u: stage2.dynamics(t, x, u, env)

    add_rk4_segment_constraints(opti, X2, U2, N2, ode_stage1_fault, dt_2, t0=T1)
    add_rk4_segment_constraints(opti, X3, U3, N3, ode_stage2, dt_3, t0=T1 + T2)
    add_rk4_segment_constraints(opti, X4, U4, N4, ode_stage2, dt_4, t0=T1 + T2 + T3)

    opti.subject_to(X2[:, 0] == ca.DM(x_start_fault))
    opti.subject_to(U2[:, 0] == ca.DM(u_start_fault))

    stage1_propellant_budget = env.mdot1 * env.t_yiji
    stage1_propellant_burned = env.mdot1 * T1 + mdot1_fault * T2
    leftover_stage1_propellant = stage1_propellant_budget - stage1_propellant_burned
    if strategy.optimize_stage1_time:
        opti.subject_to(leftover_stage1_propellant >= -1e-6)
    elif leftover_stage1_propellant < -1e-6:
        raise ValueError("fixed stage-1 timing exceeds propellant budget")

    opti.subject_to(X3[0:6, 0] == X2[0:6, -1])
    opti.subject_to(X3[6, 0] == X2[6, -1] - env.m_pao - leftover_stage1_propellant)
    opti.subject_to(U3[:, 0] == U2[:, -1])

    opti.subject_to(X4[0:6, 0] == X3[0:6, -1])
    opti.subject_to(X4[6, 0] == X3[6, -1] - env.m_zhengliu)
    opti.subject_to(U4[:, 0] == U3[:, -1])

    _terminal_orbit_constraints(
        opti,
        env,
        X4[0:3, -1],
        X4[3:6, -1],
        cfg.tolerances,
        cfg.terminal_active,
    )

    objective = 0
    if strategy.optimize_t4:
        objective += T4
    if strategy.optimize_stage1_time:
        if cfg.w_stage1_time != 0.0:
            objective += cfg.w_stage1_time * T2
    if cfg.enable_smoothness and cfg.w_smoothness > 0.0:
        objective += cfg.w_smoothness * smoothness_cost(U2, U3, U4)
    opti.minimize(objective)

    configure_reachability_solver(opti, cfg.solver_opts)
    sol = opti.solve()

    T2_opt = float(sol.value(T2)) if strategy.optimize_stage1_time else float(T2_nominal)
    T4_opt = float(sol.value(T4)) if strategy.optimize_t4 else float(T4_fixed)
    leftover_opt = (
        float(sol.value(leftover_stage1_propellant))
        if strategy.optimize_stage1_time
        else float(leftover_stage1_propellant)
    )
    final_mass = float(sol.value(X4[6, -1]))

    delta_T1 = T1 + T2_opt - env.t_yiji
    delta_T4 = T4_opt - T4_nominal
    eta_den = T1_dep - T1
    eta1 = T2_opt / eta_den if eta_den > 1e-9 else 0.0

    return CaseResult(
        reachable=True,
        te=float(te),
        kappa=float(kappa),
        strategy=strategy.name,
        T1_sep=float(T1 + T2_opt),
        T1_dep=float(T1_dep),
        T4_duration=float(T4_opt),
        delta_T1=float(delta_T1),
        eta1=float(eta1),
        delta_T4=float(delta_T4),
        leftover_stage1_propellant=float(leftover_opt),
        final_mass=float(final_mass),
        message="ok",
    )


def scan_strategy_grid(
    te_values,
    kappa_values,
    strategy_names,
    cfg: ReachabilityConfig,
) -> dict:
    env = EarthEnv(target=cfg.target)
    data = np.load(resolve_input_path(cfg.init_guess_file))
    te_values = np.asarray(te_values, dtype=float)
    kappa_values = np.asarray(kappa_values, dtype=float)
    strategy_names = list(strategy_names)

    shape = (len(strategy_names), len(kappa_values), len(te_values))
    reachable = np.zeros(shape, dtype=bool)
    delta_T1 = np.full(shape, np.nan)
    eta1 = np.full(shape, np.nan)
    delta_T4 = np.full(shape, np.nan)
    T1_sep = np.full(shape, np.nan)
    T1_dep = np.full(shape, np.nan)
    T4_duration = np.full(shape, np.nan)
    leftover = np.full(shape, np.nan)
    final_mass = np.full(shape, np.nan)
    messages = np.empty(shape, dtype=object)

    total = len(strategy_names) * len(kappa_values) * len(te_values)
    done = 0
    for s_idx, strategy_name in enumerate(strategy_names):
        for k_idx, kappa in enumerate(kappa_values):
            for t_idx, te in enumerate(te_values):
                done += 1
                print(f"[{done}/{total}] {strategy_name}: te={te:.3f}, kappa={kappa:.4f}")
                res = solve_reachability_case(te, kappa, strategy_name, cfg, data=data, env=env)
                reachable[s_idx, k_idx, t_idx] = res.reachable
                delta_T1[s_idx, k_idx, t_idx] = res.delta_T1
                eta1[s_idx, k_idx, t_idx] = res.eta1
                delta_T4[s_idx, k_idx, t_idx] = res.delta_T4
                T1_sep[s_idx, k_idx, t_idx] = res.T1_sep
                T1_dep[s_idx, k_idx, t_idx] = res.T1_dep
                T4_duration[s_idx, k_idx, t_idx] = res.T4_duration
                leftover[s_idx, k_idx, t_idx] = res.leftover_stage1_propellant
                final_mass[s_idx, k_idx, t_idx] = res.final_mass
                messages[s_idx, k_idx, t_idx] = res.message
                print("  reachable" if res.reachable else f"  failed: {res.message[:120]}")

    return {
        "te": te_values,
        "kappa": kappa_values,
        "strategies": np.asarray(strategy_names),
        "reachable": reachable,
        "delta_T1": delta_T1,
        "eta1": eta1,
        "delta_T4": delta_T4,
        "T1_sep": T1_sep,
        "T1_dep": T1_dep,
        "T4_duration": T4_duration,
        "leftover_stage1_propellant": leftover,
        "final_mass": final_mass,
        "messages": messages,
        "terminal_active": np.asarray(cfg.terminal_active),
        "T4_nominal": np.array(nominal_t4_duration(data, env)),
        "final_mass_min": np.array(env.m_gan if cfg.final_mass_min is None else cfg.final_mass_min),
    }


def save_scan_npz(path: str | Path, scan: dict) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = RESULTS_DIR / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **scan)
    print(f"Saved scan: {path}")
    return path


def load_scan_npz(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        candidates = [path, BASE_DIR / path, RESULTS_DIR / path.name]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def scan_terminal_relaxation_grid(
    te_values,
    kappa_values,
    level_names,
    cfg: ReachabilityConfig,
    strategy_name: str = S0,
) -> dict:
    level_names = list(level_names)
    all_reachable = []
    all_messages = []

    for level_name in level_names:
        if level_name not in TERMINAL_LEVELS:
            raise ValueError(f"Unknown terminal level {level_name!r}")
        cfg_level = ReachabilityConfig(
            dt=cfg.dt,
            init_guess_file=cfg.init_guess_file,
            result_file=cfg.result_file,
            T4_guess=cfg.T4_guess,
            T4_min=cfg.T4_min,
            T4_max=cfg.T4_max,
            T4_fixed=cfg.T4_fixed,
            final_mass_min=cfg.final_mass_min,
            stage1_free_min_after_fault=cfg.stage1_free_min_after_fault,
            terminal_active=TERMINAL_LEVELS[level_name],
            dphi_max_deg_per_step=cfg.dphi_max_deg_per_step,
            enable_dphi_limit=cfg.enable_dphi_limit,
            enable_smoothness=cfg.enable_smoothness,
            w_smoothness=cfg.w_smoothness,
            w_stage1_time=cfg.w_stage1_time,
            enable_alpha_limit=cfg.enable_alpha_limit,
            alpha_limit_deg=cfg.alpha_limit_deg,
            enable_qalpha_limit=cfg.enable_qalpha_limit,
            qalpha_limit=cfg.qalpha_limit,
            target=cfg.target,
            tolerances=cfg.tolerances,
            solver_opts=cfg.solver_opts,
        )
        scan = scan_strategy_grid(te_values, kappa_values, [strategy_name], cfg_level)
        all_reachable.append(scan["reachable"][0])
        all_messages.append(scan["messages"][0])

    return {
        "te": np.asarray(te_values, dtype=float),
        "kappa": np.asarray(kappa_values, dtype=float),
        "levels": np.asarray(level_names),
        "strategy": np.asarray(strategy_name),
        "reachable": np.asarray(all_reachable, dtype=bool),
        "messages": np.asarray(all_messages, dtype=object),
    }
