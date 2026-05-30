"""servo_tf.py — Discrete servo plant Gp(s) and closed-loop simulation.

Transfer function (continuous):
    Gp(s) = (0.174s + 0.3744) / (s² + 0.785s + 0.3744)

Properties:
    DC gain = 0.3744 / 0.3744 = 1.0
    ωn = sqrt(0.3744) ≈ 0.612 rad/s
    ζ = 0.785 / (2·0.612) ≈ 0.641  (well-damped underdamped)
    Zero at s = -0.3744/0.174 ≈ -2.15

Disturbance model: additive output disturbance d(k) applied to the measured
output, i.e., y_eff(k) = y_plant(k) + d(k). This models a load perturbation
seen by the controller as an apparent change in plant output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.signal import cont2discrete


# ── Discrete plant ─────────────────────────────────────────────────────────────

def build_discrete_plant(num: list[float], den: list[float], Ts: float):
    """Discretize Gp(s) using ZOH and return (Ad, Bd, Cd, Dd, nx)."""
    sys_d = cont2discrete((num, den), Ts, method="zoh")
    num_d, den_d, _ = sys_d
    # Convert TF to state-space via scipy
    from scipy.signal import tf2ss
    A, B, C, D = tf2ss(num_d.squeeze(), den_d.squeeze())
    return A, B, C, D


# ── Controller interface ───────────────────────────────────────────────────────

class ServoController(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: dict) -> float: ...


# ── Simulation ─────────────────────────────────────────────────────────────────

@dataclass
class ServoSimResult:
    t: np.ndarray
    r: np.ndarray
    y: np.ndarray
    y_plant: np.ndarray       # plant output before disturbance
    e: np.ndarray
    u: np.ndarray
    disturbance: np.ndarray
    controller_name: str


def simulate_servo(
    controller: ServoController,
    controller_name: str,
    *,
    num: list[float],
    den: list[float],
    Ts: float,
    T_final: float,
    ref_step_time: float,
    ref_value: float,
    dist_start: float,
    dist_end: float,
    dist_mag: float,
    u_min: float = 0.0,
    u_max: float = 1.0,
    du_max: float = 0.08,
    seed: int = 8,
) -> ServoSimResult:
    """Run closed-loop servo simulation with the given controller.

    The plant state is integrated forward using discrete-time ZOH matrices.
    The disturbance is additive on the measured output (load disturbance).
    """
    N = round(T_final / Ts)
    t = np.arange(N) * Ts

    # Reference: step at ref_step_time
    r = np.where(t >= ref_step_time, ref_value, 0.0)

    # Disturbance signal (additive output disturbance)
    d = np.zeros(N)
    k_start = round(dist_start / Ts)
    k_end = round(dist_end / Ts)
    d[k_start:k_end] = dist_mag

    # Discretize plant
    Ad, Bd, Cd, Dd = build_discrete_plant(num, den, Ts)
    nx = Ad.shape[0]
    x = np.zeros(nx)

    y_plant = np.zeros(N)
    y_eff = np.zeros(N)     # observed output (with disturbance)
    u_arr = np.zeros(N)
    e_arr = np.zeros(N)

    controller.reset()
    prev_u = 0.0
    integral_e = 0.0
    prev_e = 0.0

    for k in range(N):
        # Measured output
        y_now = float(Cd @ x + Dd * (prev_u if k > 0 else 0.0))
        y_plant[k] = y_now
        y_eff[k] = y_now + d[k]

        e = r[k] - y_eff[k]
        de = (e - prev_e) / Ts if k > 0 else 0.0
        integral_e = np.clip(integral_e + e * Ts, -10.0, 10.0)

        obs = {
            "t": t[k], "k": k,
            "r": r[k], "y": y_eff[k],
            "e": e, "de": de,
            "integral_e": integral_e,
            "disturbance_flag": float(d[k] != 0),
            "previous_u": prev_u,
        }
        u_raw = controller.act(obs)

        # Rate limiter
        u_clipped = np.clip(u_raw, u_min, u_max)
        delta = u_clipped - prev_u
        delta = np.clip(delta, -du_max, du_max)
        u = np.clip(prev_u + delta, u_min, u_max)

        u_arr[k] = u
        e_arr[k] = e

        # Advance plant state (using u applied at this step)
        x = Ad @ x + Bd.flatten() * u
        prev_u = u
        prev_e = e

    return ServoSimResult(
        t=t, r=r, y=y_eff, y_plant=y_plant,
        e=e_arr, u=u_arr, disturbance=d,
        controller_name=controller_name,
    )
