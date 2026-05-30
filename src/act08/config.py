"""config.py — Load and expose r2et_config.yaml as a namespace."""
from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_FILE = _ROOT / "configs" / "r2et_config.yaml"


def _load() -> dict[str, Any]:
    with open(_CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


_raw: dict[str, Any] = _load()

SEED: int = _raw["seed"]

# ── Servo ──────────────────────────────────────────────────────────────────────
class _Servo:
    Ts: float = _raw["servo"]["Ts"]
    T_final: float = _raw["servo"]["T_final"]
    N: int = round(_raw["servo"]["T_final"] / _raw["servo"]["Ts"])
    ref_step_time: float = _raw["servo"]["reference_step_time"]
    ref_value: float = _raw["servo"]["reference_value"]
    dist_start: float = _raw["servo"]["disturbance_start"]
    dist_end: float = _raw["servo"]["disturbance_end"]
    dist_mag: float = _raw["servo"]["disturbance_magnitude"]
    u_min: float = _raw["servo"]["u_min"]
    u_max: float = _raw["servo"]["u_max"]
    du_max: float = _raw["servo"]["du_max"]
    plant_num: list[float] = _raw["servo"]["plant"]["numerator"]
    plant_den: list[float] = _raw["servo"]["plant"]["denominator"]


Servo = _Servo()

# ── Coordination (R2ET) ────────────────────────────────────────────────────────
class _Coord:
    Ts: float = _raw["coordination"]["Ts"]
    horizon: int = _raw["coordination"]["horizon"]
    N_max: float = _raw["coordination"]["N_max"]
    n_target: float = _raw["coordination"]["n_target"]
    risk_threshold: float = _raw["coordination"]["risk_threshold"]
    arrival_nominal: float = _raw["coordination"]["arrival_nominal"]
    arrival_pulse: float = _raw["coordination"]["arrival_pulse"]
    pulse_start: int = _raw["coordination"]["pulse_start"]
    pulse_end: int = _raw["coordination"]["pulse_end"]
    jam_start: int = _raw["coordination"]["jam_start"]
    jam_end: int = _raw["coordination"]["jam_end"]
    jam_loss: float = _raw["coordination"]["jam_loss"]
    u_min: float = _raw["coordination"]["u_min"]
    u_max: float = _raw["coordination"]["u_max"]
    admission_min: float = _raw["coordination"]["admission_min"]
    admission_max: float = _raw["coordination"]["admission_max"]
    split_min: float = _raw["coordination"]["split_min"]
    split_max: float = _raw["coordination"]["split_max"]
    k_discharge: float = _raw["coordination"]["k_discharge"]


Coord = _Coord()

# ── Training ───────────────────────────────────────────────────────────────────
class _Training:
    deep_servo_samples: int = _raw["training"]["deep_servo_samples"]
    deep_coord_samples: int = _raw["training"]["deep_coord_samples"]
    train_split: float = _raw["training"]["train_split"]
    val_split: float = _raw["training"]["val_split"]
    test_split: float = _raw["training"]["test_split"]
    deep_epochs: int = _raw["training"]["deep_epochs"]
    batch_size: int = _raw["training"]["batch_size"]
    lr: float = _raw["training"]["learning_rate"]
    wd: float = _raw["training"]["weight_decay"]
    q_servo_episodes: int = _raw["training"]["q_servo_episodes"]
    q_coord_episodes: int = _raw["training"]["q_coord_episodes"]
    q_gamma: float = _raw["training"]["q_gamma"]
    q_alpha: float = _raw["training"]["q_alpha"]
    q_eps_start: float = _raw["training"]["q_epsilon_start"]
    q_eps_end: float = _raw["training"]["q_epsilon_end"]
    q_servo_horizon: int = _raw["training"]["q_servo_horizon"]
    q_coord_horizon: int = _raw["training"]["q_coord_horizon"]


Training = _Training()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = _ROOT
ARTIFACTS = _ROOT / "artifacts"
FIGURES = _ROOT / "figures"
REPORT = _ROOT / "report"

for _d in [
    ARTIFACTS / "data", ARTIFACTS / "models", ARTIFACTS / "metrics",
    ARTIFACTS / "tables", ARTIFACTS / "training_logs", ARTIFACTS / "final",
    FIGURES / "servo", FIGURES / "coordination", FIGURES / "training",
    FIGURES / "diagrams", FIGURES / "final",
]:
    _d.mkdir(parents=True, exist_ok=True)
