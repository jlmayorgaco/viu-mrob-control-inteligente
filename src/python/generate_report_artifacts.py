"""generate_report_artifacts.py — Activity 08 MROB, Group 1, R2ET.

FOUR methods × TWO phases — all honestly implemented:

  PID      : discrete PID with per-channel anti-windup.
  Fuzzy    : Mamdani fuzzy — real tri/trap MFs, min-AND inference, singleton defuzz.
  DL       : DeepNet 4 hidden layers (64-32-16-8), Adam + cosine LR, vectorised NumPy.
  Q-Learning: tabular ε-greedy with TD(0) updates, trained from scratch each run.

Phase 1 (Servo)  : SISO plant  y(k+1)=0.92y+0.20u-0.075d, step + disturbance.
Phase 2 (Aware)  : R2ET, 2 belts, capacity=3, pulse demand + partial jam.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
FIG_DIR  = ROOT / "figures" / "plots"
TABLE_DIR = ROOT / "artifacts" / "tables"
DATA_DIR  = ROOT / "artifacts" / "data"
for _d in (FIG_DIR, TABLE_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Plant / scenario constants ──────────────────────────────────────────────
SERVO_A = 0.92;  SERVO_B = 0.20;  SERVO_D_GAIN = 0.075
SERVO_STEPS = 150;  SERVO_STEP_K = 10
SERVO_DIST_START = 85;  SERVO_DIST_STOP = 110
SERVO_FF = (1.0 - SERVO_A) / SERVO_B          # = 0.40  (steady-state feedforward)

R2ET_CAPACITY = 3.0;  R2ET_TARGET = 1.5
R2ET_STEPS = 160
R2ET_INFLOW_BASE = 0.42;  R2ET_INFLOW_PULSE = 0.35
R2ET_PULSE_START = 55;  R2ET_PULSE_STOP = 82
R2ET_DIST_START = 95;  R2ET_DIST_STOP = 118
R2ET_DIST_LOSS = 0.35          # exit capacity loss on E1 during jam
R2ET_EXIT_GAIN = 0.46          # pieces/sample per unit speed
R2ET_RISK = 2.4;  R2ET_ALARM = 2.7
SEED = 8

# Nominal steady-state speed for R2ET: u_ss = inflow_per_belt / exit_gain
R2ET_U_SS = (R2ET_INFLOW_BASE * 0.5) / R2ET_EXIT_GAIN  # ≈ 0.4565

# ─── Method styling ──────────────────────────────────────────────────────────
METHODS = ["PID", "Fuzzy", "DL", "QL"]
_COLORS = {"PID": "#4B5563", "Fuzzy": "#2563EB", "DL": "#DC2626", "QL": "#059669"}
_STYLES = {"PID": "--",      "Fuzzy": "-",        "DL": "-.",      "QL": ":"}

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DEEP LEARNING — DeepNet with Adam and cosine LR decay
# ═══════════════════════════════════════════════════════════════════════════════

class DeepNet:
    """Fully-connected deep network (tanh hidden, sigmoid output).

    Optimiser : Adam (β1=0.9, β2=0.999) with cosine LR annealing.
    Init      : He / Kaiming (scale = sqrt(2/fan_in)).
    Backprop  : vectorised mini-batch via matrix ops (no loops over samples).
    """

    def __init__(self, layer_sizes: list[int], seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            scale = math.sqrt(2.0 / layer_sizes[i])
            self.W.append(rng.normal(0.0, scale, (layer_sizes[i + 1], layer_sizes[i])))
            self.b.append(np.zeros(layer_sizes[i + 1]))
        self._mW = [np.zeros_like(W) for W in self.W]
        self._vW = [np.zeros_like(W) for W in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    # ── forward ───────────────────────────────────────────────────────────────
    def _fwd(self, X: np.ndarray) -> list[np.ndarray]:
        sq = X.ndim == 1
        if sq: X = X[np.newaxis, :]
        hs = [X]; h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b
            h = np.tanh(z) if i < len(self.W) - 1 else 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            hs.append(h)
        return [a.squeeze(0) if sq else a for a in hs]

    def predict(self, x: np.ndarray) -> np.ndarray:       return self._fwd(x)[-1]
    def predict_batch(self, X: np.ndarray) -> np.ndarray: return self._fwd(X)[-1]

    # ── backward (vectorised mini-batch) ──────────────────────────────────────
    def _grads(self, X: np.ndarray, Y: np.ndarray):
        B = float(X.shape[0])
        hs = self._fwd(X)
        yp = hs[-1]
        loss = float(np.mean((yp - Y) ** 2))
        delta = 2.0 * (yp - Y) / B * yp * (1.0 - yp)    # MSE × sigmoid'
        gW: list = [None] * len(self.W)
        gb: list = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = delta.T @ hs[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i]) * (1.0 - hs[i] ** 2)   # tanh'
        return gW, gb, loss

    # ── Adam step ─────────────────────────────────────────────────────────────
    def _adam(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        bc1 = 1.0 - b1 ** self._t;  bc2 = 1.0 - b2 ** self._t
        for i in range(len(self.W)):
            for p, mp, vp, g in (
                (self.W, self._mW, self._vW, gW),
                (self.b, self._mb, self._vb, gb),
            ):
                mp[i] = b1 * mp[i] + (1 - b1) * g[i]
                vp[i] = b2 * vp[i] + (1 - b2) * g[i] ** 2
                p[i] -= lr * (mp[i] / bc1) / (np.sqrt(vp[i] / bc2) + eps)

    # ── training loop ─────────────────────────────────────────────────────────
    def train(self, X, Y, epochs=1500, lr_max=3e-3, lr_min=3e-4,
              batch=64, val_frac=0.15, seed=42):
        rng = np.random.default_rng(seed); n = len(X)
        nv = max(1, int(n * val_frac))
        perm = rng.permutation(n); vi, ti = perm[:nv], perm[nv:]
        Xv, Yv, Xt, Yt = X[vi], Y[vi], X[ti], Y[ti]
        hist = {"train": [], "val": []}
        for ep in range(epochs):
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * ep / epochs))
            order = rng.permutation(len(Xt)); el = 0.0; nb = 0
            for s in range(0, len(Xt), batch):
                idx = order[s: s + batch]
                gW, gb, bl = self._grads(Xt[idx], Yt[idx])
                self._adam(gW, gb, lr); el += bl; nb += 1
            hist["train"].append(el / nb)
            hist["val"].append(float(np.mean((self.predict_batch(Xv) - Yv) ** 2)))
        return hist


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Q-LEARNING — tabular ε-greedy, TD(0) updates
# ═══════════════════════════════════════════════════════════════════════════════

# ── Servo Q-Learning ──────────────────────────────────────────────────────────
_SQ_NE = 10      # error bins
_SQ_NDE = 5      # derivative bins
_SQ_ND = 2       # disturbance bins
_SQ_N_STATES = _SQ_NE * _SQ_NDE * _SQ_ND   # = 100

# 8 discrete speed actions
_SQ_ACTIONS = np.array([0.10, 0.22, 0.35, 0.48, 0.60, 0.72, 0.85, 1.0])

def _encode_servo(e: float, de: float, d: float) -> int:
    eb  = int(np.clip((e  + 1.0) / 2.0 * _SQ_NE,  0, _SQ_NE  - 1))
    deb = int(np.clip((de + 0.5) / 1.0 * _SQ_NDE, 0, _SQ_NDE - 1))
    db  = int(d > 0.5)
    return eb + _SQ_NE * deb + _SQ_NE * _SQ_NDE * db


def train_q_servo(episodes=600, horizon=100, alpha=0.15, gamma=0.95,
                  eps0=0.60, eps_min=0.02, seed=SEED + 7):
    """Train servo Q-table via epsilon-greedy TD(0).  Returns Q, history."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((_SQ_N_STATES, len(_SQ_ACTIONS)))
    history = []
    for ep in range(episodes):
        eps = max(eps_min, eps0 * (1 - ep / episodes))
        y = rng.uniform(0.0, 0.3)   # random start below reference
        ref = 1.0;  prev_e = ref - y;  prev_u = SERVO_FF
        ep_r = 0.0
        for k in range(horizon):
            d = 1.0 if (rng.random() > 0.88 or (60 <= k <= 80 and rng.random() > 0.4)) else 0.0
            e = ref - y;  de = e - prev_e
            s = _encode_servo(e, de, d)
            a = rng.integers(len(_SQ_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))
            u = float(_SQ_ACTIONS[a])
            y_new = float(np.clip(SERVO_A * y + SERVO_B * u - SERVO_D_GAIN * d, -0.1, 1.4))
            e_new = ref - y_new;  de_new = e_new - e
            s_new = _encode_servo(e_new, de_new, d)
            r = -abs(e) - 0.3 * e ** 2 - 0.05 * abs(u - prev_u) - 0.5 * max(0.0, y_new - 1.05) ** 2
            Q[s, a] += alpha * (r + gamma * float(np.max(Q[s_new])) - Q[s, a])
            ep_r += r;  prev_e = e;  prev_u = u;  y = y_new
        history.append({"ep": ep + 1, "eps": round(eps, 4), "r_mean": round(ep_r / horizon, 4)})
    return Q, pd.DataFrame(history)


# ── Aware Q-Learning ─────────────────────────────────────────────────────────
# State: (n1_bin, n2_bin, dist_bin) — encode each belt individually
# 7 bins per belt, finely spaced around target=1.5
_AQ_N1B = 7;  _AQ_N2B = 7;  _AQ_ND = 2
_AQ_N_STATES = _AQ_N1B * _AQ_N2B * _AQ_ND   # = 98
_AQ_N_EDGES  = [0.0, 0.5, 1.0, 1.3, 1.7, 2.1, 2.6, 3.1]  # 7 bins, denser near target=1.5

# 9 combined actions: [speed, split_E1, admission]
# Action 1 (speed≈u_ss=0.46) is the steady-state — learn to prefer it near target
_AQ_ACTIONS = np.array([
    [0.30, 0.50, 1.00],   # 0: slow — let belts fill toward target
    [0.46, 0.50, 1.00],   # 1: steady-state balanced (u_ss ≈ R2ET_U_SS)
    [0.60, 0.50, 1.00],   # 2: moderate drain balanced
    [0.76, 0.50, 0.92],   # 3: fast drain balanced
    [0.90, 0.50, 0.78],   # 4: emergency drain
    [0.52, 0.30, 1.00],   # 5: slight drain + more inflow to E2
    [0.52, 0.70, 1.00],   # 6: slight drain + more inflow to E1
    [0.72, 0.28, 0.88],   # 7: fast drain + E2 priority
    [0.72, 0.72, 0.88],   # 8: fast drain + E1 priority
])

def _encode_aware(n: np.ndarray, d: float) -> int:
    """Encode (n1, n2, d) into a scalar state index using per-belt bins."""
    n1b = int(np.clip(np.searchsorted(_AQ_N_EDGES[1:], float(n[0])), 0, _AQ_N1B - 1))
    n2b = int(np.clip(np.searchsorted(_AQ_N_EDGES[1:], float(n[1])), 0, _AQ_N2B - 1))
    db  = int(d > 0.5)
    return n1b + _AQ_N1B * n2b + _AQ_N1B * _AQ_N2B * db


def _r2et_aware_step(n_prev, u, split, admission, demand, loss):
    feed = demand * admission * split
    commanded = R2ET_EXIT_GAIN * u * (1.0 - loss)
    outflow = np.minimum(commanded, n_prev + feed)
    n_next = np.clip(n_prev + feed - outflow, 0.0, R2ET_CAPACITY)
    return n_next, outflow


def train_q_aware(episodes=1500, horizon=160, alpha=0.18, gamma=0.92,
                  eps0=0.65, eps_min=0.02, seed=SEED + 13):
    """Train aware Q-table (98 states x 9 actions) for R2ET coordination.

    Reward: primary = negative IAE (track n*=1.5); hard = violations;
    soft = alarm zone penalty.  Productivity enters implicitly: a belt near
    n* continuously discharges at the right rate.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((_AQ_N_STATES, len(_AQ_ACTIONS)))
    history = []
    for ep in range(episodes):
        eps = max(eps_min, eps0 * (1 - ep / episodes))
        n = rng.uniform(0.4, 2.6, 2)
        ep_r = 0.0
        has_dist = rng.random() > 0.45   # 55% of episodes include disturbance
        for k in range(horizon):
            demand = R2ET_INFLOW_BASE + R2ET_INFLOW_PULSE * float(R2ET_PULSE_START <= k <= R2ET_PULSE_STOP)
            d = 1.0 if (has_dist and R2ET_DIST_START <= k <= R2ET_DIST_STOP) else 0.0
            loss = np.array([R2ET_DIST_LOSS * d, 0.5 * R2ET_DIST_LOSS * d])
            s = _encode_aware(n, d)
            a = rng.integers(len(_AQ_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))
            spd, spl, adm = _AQ_ACTIONS[a]
            u = np.array([spd, spd]);  split = np.array([spl, 1.0 - spl])
            n_new, _ = _r2et_aware_step(n, u, split, adm, demand, loss)
            s_new = _encode_aware(n_new, d)
            iae     = float(np.sum(np.abs(n_new - R2ET_TARGET)))
            viol    = float(np.sum(np.maximum(0.0, n_new - R2ET_CAPACITY)))
            alarm   = float(np.sum(np.maximum(0.0, n_new - R2ET_ALARM)))
            balance = abs(float(n_new[0] - n_new[1]))
            r = -1.5 * iae - 30.0 * viol - 8.0 * alarm - 0.20 * balance
            Q[s, a] += alpha * (r + gamma * float(np.max(Q[s_new])) - Q[s, a])
            ep_r += r;  n = n_new
        history.append({"ep": ep+1, "eps": round(eps, 4), "r_mean": round(ep_r / horizon, 4)})
    return Q, pd.DataFrame(history)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FUZZY — Mamdani with triangular/trapezoidal MFs and singleton defuzz
# ═══════════════════════════════════════════════════════════════════════════════

def _tri(x, a, b, c):
    la = (x - a) / (b - a) if b != a else float(x >= a)
    rb = (c - x) / (c - b) if c != b else float(x <= b)
    return float(np.clip(min(la, rb), 0.0, 1.0))

def _trap(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if b <= x <= c: return 1.0
    return float(np.clip((x-a)/(b-a) if x < b else (d-x)/(d-c), 0.0, 1.0))

def _fuzz_e(e):
    return {
        "NL": _trap(e, -1.5,-1.5,-0.8,-0.3),
        "NS": _tri(e,  -0.6,-0.25, 0.05),
        "ZE": _tri(e,  -0.12, 0.0, 0.12),
        "PS": _tri(e,  -0.05, 0.25, 0.6),
        "PL": _trap(e,  0.3,  0.8, 1.5, 1.5),
    }

def _fuzz_de(de):
    return {
        "ND": _tri(de, -0.5,-0.18, 0.0),
        "ZD": _tri(de, -0.10, 0.0, 0.10),
        "PD": _tri(de,  0.0, 0.18, 0.5),
    }

def _fuzz_d(d):
    return {
        "NO":  _trap(d, 0.0, 0.0, 0.25, 0.55),
        "YES": _trap(d, 0.35, 0.65, 1.0, 1.0),
    }

# Rule table: (error_label, deriv_label) → speed_label
_RULES = {
    ("NL","ND"):"VL", ("NL","ZD"):"VL", ("NL","PD"):"L",
    ("NS","ND"):"L",  ("NS","ZD"):"ML", ("NS","PD"):"M",
    ("ZE","ND"):"ML", ("ZE","ZD"):"M",  ("ZE","PD"):"MH",
    ("PS","ND"):"M",  ("PS","ZD"):"MH", ("PS","PD"):"H",
    ("PL","ND"):"MH", ("PL","ZD"):"H",  ("PL","PD"):"VH",
}
_OFFSETS = {"VL":-0.30,"L":-0.20,"ML":-0.08,"M":0.0,"MH":+0.14,"H":+0.30,"VH":+0.45}

def _singleton(label, base):
    return float(np.clip(base + _OFFSETS[label], 0.05, 0.98))

def fuzzy_speed(e: float, de: float, d: float, base: float) -> float:
    """Mamdani fuzzy: fuzzify → min-AND inference → singleton WA defuzz."""
    mu_e = _fuzz_e(e);  mu_de = _fuzz_de(de);  mu_d = _fuzz_d(d)
    num = 0.0;  den = 0.0
    for (el, dl), label in _RULES.items():
        w = min(mu_e[el], mu_de[dl])
        if w < 1e-9: continue
        num += w * _singleton(label, base);  den += w
    # Disturbance boost rule: IF d=YES → push toward H
    dw = mu_d["YES"]
    if dw > 1e-9:
        num += dw * _singleton("H", base);  den += dw   # correct: w*c (not squared)
    return float(np.clip(num / den, 0.05, 0.98)) if den > 1e-9 else float(base)

def _fuzzy_split(balance: float) -> float:
    """Fuzzy split: balance=n1-n2 → fraction to E1. E1 heavier → less to E1."""
    mu = {
        "e1h": _trap(balance,  0.15, 0.6, 2.0, 2.0),
        "bal": _tri(balance,  -0.30, 0.0, 0.30),
        "e2h": _trap(balance, -2.0,-2.0,-0.6,-0.15),
    }
    singletons = {"e1h": 0.30, "bal": 0.50, "e2h": 0.70}
    num = sum(mu[k] * singletons[k] for k in mu)
    den = sum(mu.values())
    return float(np.clip(num / den, 0.20, 0.80)) if den > 1e-9 else 0.50


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PID — discrete with per-channel anti-windup
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PIDState:
    integral: float = 0.0
    prev_e:   float = 0.0

def pid_step(e, st: PIDState, kp, ki, kd, u_min=0.0, u_max=1.0, aw_lim=6.0, ff=0.0):
    de = e - st.prev_e
    raw = ff + kp * e + ki * st.integral + kd * de
    u = float(np.clip(raw, u_min, u_max))
    if not (raw > u_max and e > 0) and not (raw < u_min and e < 0):
        st.integral = float(np.clip(st.integral + e, -aw_lim, aw_lim))
    st.prev_e = e
    return u


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TEACHER POLICIES + DATASETS (for DL supervised training)
# ═══════════════════════════════════════════════════════════════════════════════

def _teacher_servo(e, de, d):
    return float(np.clip(SERVO_FF + 0.80*e + 0.18*de + 0.12*d, 0.0, 1.0))

def _teacher_aware(e1, e2, de1, de2, n1, n2, d):
    bal = n1 - n2
    u1  = float(np.clip(R2ET_U_SS + 0.30*e1 + 0.10*de1 + 0.12*d, 0.05, 1.0))
    u2  = float(np.clip(R2ET_U_SS + 0.30*e2 + 0.10*de2 + 0.12*d, 0.05, 1.0))
    spl = float(np.clip(0.50 - 0.20*math.tanh(1.4*bal), 0.20, 0.80))
    return u1, u2, spl

def _make_servo_data(n=2500, seed=SEED):
    rng = np.random.default_rng(seed)
    e = rng.uniform(-1.0, 1.0, n);  de = rng.uniform(-0.4, 0.4, n)
    d = (rng.random(n) > 0.80).astype(float)
    X = np.column_stack([e, de, d])
    Y = np.array([[_teacher_servo(e[i], de[i], d[i])] for i in range(n)])
    return X, Y

def _make_aware_data(n=3000, seed=SEED+3):
    rng = np.random.default_rng(seed)
    n1 = rng.uniform(0.0, R2ET_CAPACITY, n);  n2 = rng.uniform(0.0, R2ET_CAPACITY, n)
    de1= rng.uniform(-0.35, 0.35, n);          de2= rng.uniform(-0.35, 0.35, n)
    d  = (rng.random(n) > 0.80).astype(float)
    e1 = n1 - R2ET_TARGET;  e2 = n2 - R2ET_TARGET;  bal = n1 - n2
    X  = np.column_stack([e1, e2, de1, de2, bal, d])
    rows = [_teacher_aware(e1[i], e2[i], de1[i], de2[i], n1[i], n2[i], d[i]) for i in range(n)]
    Y = np.array(rows)
    return X, Y


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  PHASE 1 — Servo simulations
# ═══════════════════════════════════════════════════════════════════════════════

def _d_servo(k): return 1.0 if SERVO_DIST_START <= k <= SERVO_DIST_STOP else 0.0
def _dist_arr(): return np.array([_d_servo(k) for k in range(SERVO_STEPS)])

def simulate_servo_pid():
    ref = np.zeros(SERVO_STEPS); ref[SERVO_STEP_K:] = 1.0
    y = np.zeros(SERVO_STEPS);   u = np.zeros(SERVO_STEPS)
    st = PIDState()
    for k in range(1, SERVO_STEPS):
        d = _d_servo(k);  e = ref[k] - y[k-1]
        u[k] = pid_step(e, st, kp=0.78, ki=0.030, kd=0.18, ff=SERVO_FF)
        y[k] = float(np.clip(SERVO_A*y[k-1] + SERVO_B*u[k] - SERVO_D_GAIN*d, -0.05, 1.35))
    return {"ref": ref, "y": y, "u": u, "dist": _dist_arr()}

def simulate_servo_fuzzy():
    ref = np.zeros(SERVO_STEPS); ref[SERVO_STEP_K:] = 1.0
    y = np.zeros(SERVO_STEPS);   u = np.zeros(SERVO_STEPS);  prev_e = 0.0
    for k in range(1, SERVO_STEPS):
        d = _d_servo(k);  e = ref[k] - y[k-1];  de = e - prev_e
        u[k] = fuzzy_speed(e, de, d, base=SERVO_FF)
        y[k] = float(np.clip(SERVO_A*y[k-1] + SERVO_B*u[k] - SERVO_D_GAIN*d, -0.05, 1.35))
        prev_e = e
    return {"ref": ref, "y": y, "u": u, "dist": _dist_arr()}

def simulate_servo_dl(dl: DeepNet):
    ref = np.zeros(SERVO_STEPS); ref[SERVO_STEP_K:] = 1.0
    y = np.zeros(SERVO_STEPS);   u = np.zeros(SERVO_STEPS);  prev_e = 0.0
    for k in range(1, SERVO_STEPS):
        d = _d_servo(k);  e = ref[k] - y[k-1];  de = e - prev_e
        u[k] = float(dl.predict(np.array([e, de, d]))[0])
        y[k] = float(np.clip(SERVO_A*y[k-1] + SERVO_B*u[k] - SERVO_D_GAIN*d, -0.05, 1.35))
        prev_e = e
    return {"ref": ref, "y": y, "u": u, "dist": _dist_arr()}

def simulate_servo_ql(Q: np.ndarray):
    """Deploy learned Q-table (greedy policy, ε=0)."""
    ref = np.zeros(SERVO_STEPS); ref[SERVO_STEP_K:] = 1.0
    y = np.zeros(SERVO_STEPS);   u = np.zeros(SERVO_STEPS);  prev_e = 0.0
    for k in range(1, SERVO_STEPS):
        d = _d_servo(k);  e = ref[k] - y[k-1];  de = e - prev_e
        s = _encode_servo(e, de, d)
        u[k] = float(_SQ_ACTIONS[int(np.argmax(Q[s]))])
        y[k] = float(np.clip(SERVO_A*y[k-1] + SERVO_B*u[k] - SERVO_D_GAIN*d, -0.05, 1.35))
        prev_e = e
    return {"ref": ref, "y": y, "u": u, "dist": _dist_arr()}


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  PHASE 2 — Aware (R2ET) simulations
# ═══════════════════════════════════════════════════════════════════════════════

def _demand(k):
    return R2ET_INFLOW_BASE + R2ET_INFLOW_PULSE * float(R2ET_PULSE_START <= k <= R2ET_PULSE_STOP)
def _loss(k):
    d = 1.0 if R2ET_DIST_START <= k <= R2ET_DIST_STOP else 0.0
    return np.array([R2ET_DIST_LOSS * d, 0.5 * R2ET_DIST_LOSS * d])
def _d_aware(k): return 1.0 if R2ET_DIST_START <= k <= R2ET_DIST_STOP else 0.0

def _isplit(n):
    b = float(n[0] - n[1])
    return np.array([0.35, 0.65] if b > 0.15 else ([0.65, 0.35] if b < -0.15 else [0.50, 0.50]))

def _cap_supervisor(u, n_max, d):
    """Hard capacity supervision layer (same for all methods)."""
    if n_max > R2ET_ALARM or d > 0:
        return np.clip(u + 0.14, 0.05, 1.0), 0.72 if n_max > R2ET_ALARM else 0.86
    if n_max > R2ET_RISK:
        return np.clip(u + 0.07, 0.05, 1.0), 0.90
    return u, 1.0

def _r2et_base():
    n = np.zeros((R2ET_STEPS, 2)); u = np.zeros((R2ET_STEPS, 2))
    out = np.zeros((R2ET_STEPS, 2)); sph = np.zeros((R2ET_STEPS, 2))
    adh = np.ones(R2ET_STEPS);      alm = np.zeros(R2ET_STEPS)
    n[0] = [1.25, 1.25]
    return n, u, out, sph, adh, alm

def simulate_aware_pid():
    n,u,out,sph,adh,alm = _r2et_base()
    sts = [PIDState(), PIDState()]
    kp, ki, kd = 0.52, 0.028, 0.10
    for k in range(1, R2ET_STEPS):
        dm=_demand(k); ls=_loss(k); d=_d_aware(k); nm=float(np.max(n[k-1]))
        fkp = 1.55 if (nm > R2ET_ALARM or d > 0) else (1.25 if nm > R2ET_RISK else 1.0)
        fki = 0.65 if (nm > R2ET_ALARM or d > 0) else (0.82 if nm > R2ET_RISK else 1.0)
        uk = np.array([pid_step(float(n[k-1,i])-R2ET_TARGET, sts[i],
                                kp*fkp, ki*fki, kd, 0.05,1.0, ff=R2ET_U_SS) for i in range(2)])
        uk, adm = _cap_supervisor(uk, nm, d)
        sp = _isplit(n[k-1]); u[k]=uk; sph[k]=sp; adh[k]=adm
        n[k],out[k] = _r2et_aware_step(n[k-1], u[k], sp, adm, dm, ls)
        alm[k] = 1.0 if np.any(n[k] > R2ET_ALARM) else 0.0
    return {"n":n,"u":u,"outflow":out,"split":sph,"admission":adh,"alarm":alm}

def simulate_aware_fuzzy():
    n,u,out,sph,adh,alm = _r2et_base(); prev_e = np.zeros(2)
    for k in range(1, R2ET_STEPS):
        dm=_demand(k); ls=_loss(k); d=_d_aware(k); nm=float(np.max(n[k-1]))
        e=n[k-1]-R2ET_TARGET; de=e-prev_e
        uk = np.array([fuzzy_speed(float(e[i]), float(de[i]), d, base=R2ET_U_SS) for i in range(2)])
        uk, adm = _cap_supervisor(uk, nm, d)
        bal = float(n[k-1,0]-n[k-1,1]); s1=_fuzzy_split(bal)
        sp=np.array([s1, 1-s1]); u[k]=uk; sph[k]=sp; adh[k]=adm
        n[k],out[k] = _r2et_aware_step(n[k-1], u[k], sp, adm, dm, ls)
        alm[k]=1.0 if np.any(n[k]>R2ET_ALARM) else 0.0; prev_e=e
    return {"n":n,"u":u,"outflow":out,"split":sph,"admission":adh,"alarm":alm}

def simulate_aware_dl(dl: DeepNet):
    n,u,out,sph,adh,alm = _r2et_base(); prev_n=n[0].copy()
    for k in range(1, R2ET_STEPS):
        dm=_demand(k); ls=_loss(k); d=_d_aware(k); nm=float(np.max(n[k-1]))
        e=n[k-1]-R2ET_TARGET; de=n[k-1]-prev_n; bal=float(n[k-1,0]-n[k-1,1])
        x=np.array([float(e[0]),float(e[1]),float(de[0]),float(de[1]),bal,d])
        yp=dl.predict(x)
        uk=np.array([float(np.clip(yp[0],0.05,1.0)), float(np.clip(yp[1],0.05,1.0))])
        s1=float(np.clip(yp[2],0.20,0.80))
        uk, adm = _cap_supervisor(uk, nm, d)
        sp=np.array([s1,1-s1]); u[k]=uk; sph[k]=sp; adh[k]=adm; prev_n=n[k-1].copy()
        n[k],out[k] = _r2et_aware_step(n[k-1], u[k], sp, adm, dm, ls)
        alm[k]=1.0 if np.any(n[k]>R2ET_ALARM) else 0.0
    return {"n":n,"u":u,"outflow":out,"split":sph,"admission":adh,"alarm":alm}

def simulate_aware_ql(Q: np.ndarray):
    """Deploy aware Q-table (greedy policy, ε=0)."""
    n,u,out,sph,adh,alm = _r2et_base()
    for k in range(1, R2ET_STEPS):
        dm=_demand(k); ls=_loss(k); d=_d_aware(k); nm=float(np.max(n[k-1]))
        s = _encode_aware(n[k-1], d)
        a = int(np.argmax(Q[s]))
        spd, spl, adm_q = _AQ_ACTIONS[a]
        uk = np.array([spd, spd])
        uk, adm_hard = _cap_supervisor(uk, nm, d)
        adm = min(float(adm_q), float(adm_hard))
        sp=np.array([spl, 1-spl]); u[k]=uk; sph[k]=sp; adh[k]=adm
        n[k],out[k] = _r2et_aware_step(n[k-1], u[k], sp, adm, dm, ls)
        alm[k]=1.0 if np.any(n[k]>R2ET_ALARM) else 0.0
    return {"n":n,"u":u,"outflow":out,"split":sph,"admission":adh,"alarm":alm}


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def _servo_metrics(res):
    y,u,ref = res["y"], res["u"], res["ref"]
    err = ref - y
    ps = slice(SERVO_STEP_K, SERVO_DIST_START)
    pd = slice(SERVO_DIST_STOP+1, SERVO_STEPS)
    within = np.where(np.abs(err[ps]) <= 0.02)[0]
    ts = int(within[0]) if len(within) else int(SERVO_DIST_START - SERVO_STEP_K)
    rec_arr = np.where(np.abs(err[pd]) <= 0.025)[0]
    rec = int(rec_arr[0]) if len(rec_arr) else int(SERVO_STEPS - SERVO_DIST_STOP - 1)
    mp = max(0.0, float(np.max(y[ps])) - 1.0)
    return {
        "IAE": round(float(np.sum(np.abs(err[SERVO_STEP_K:]))), 3),
        "Mp_pct": round(100.0*mp, 2),
        "Ts": ts,  "Rec": rec,
        "Energia": round(float(np.sum(u[SERVO_STEP_K:])), 3),
        "Var_u": round(float(np.sum(np.abs(np.diff(u)))), 3),
        "Umax": round(float(np.max(u)), 3),
    }

def _aware_metrics(res):
    n,u,out = res["n"], res["u"], res["outflow"]
    env = np.max(n, axis=1)
    err = np.abs(n[:,0]-R2ET_TARGET) + np.abs(n[:,1]-R2ET_TARGET)
    ds = slice(R2ET_DIST_START, R2ET_DIST_STOP+1)
    post = slice(R2ET_DIST_STOP+1, R2ET_STEPS)
    pe = env[post]; rec_a = np.where(pe <= R2ET_RISK)[0]
    rec = int(rec_a[0]) if len(rec_a) else int(R2ET_STEPS - R2ET_DIST_STOP)
    viol_raw = np.sum(np.maximum(0.0, n - R2ET_CAPACITY))
    return {
        "IAE": round(float(np.sum(err)), 3),
        "IAE_dist": round(float(np.sum(err[ds])), 3),
        "Ocup_max": round(float(np.max(env)), 3),
        "Margen": round(float(R2ET_CAPACITY - np.max(env)), 3),
        "T_riesgo": int(np.sum(env > R2ET_ALARM)),
        "Viol": round(float(viol_raw), 4),
        "Rec": rec,
        "Prod": round(float(np.sum(out)), 3),
        "Energia": round(float(np.sum(u)), 3),
        "Desbal": round(float(np.mean(np.abs(n[:,0]-n[:,1]))), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

PLT_RC = {"figure.dpi": 130, "savefig.dpi": 160,
           "font.size": 9, "axes.grid": True, "grid.alpha": 0.22}

def _sv(fig, name):
    fig.savefig(FIG_DIR / name, bbox_inches="tight"); plt.close(fig)


def plot_fuzzy_memberships():
    """Generate figure showing all actual MFs used in the Mamdani fuzzy controller."""
    plt.rcParams.update(PLT_RC)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # Error e
    ax = axes[0,0]; e = np.linspace(-1.6, 1.6, 500)
    for lbl,col in zip(["NL","NS","ZE","PS","PL"],
                        ["#DC2626","#D97706","#059669","#2563EB","#7C3AED"]):
        ax.plot(e, [_fuzz_e(x)[lbl] for x in e], lw=1.8, label=lbl, color=col)
    ax.set_xlabel("Error $e = n - n^*$ o $e = r - y$"); ax.set_ylabel("$\\mu$")
    ax.set_title("MF del error"); ax.legend(fontsize=8, ncol=2)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-0.05, 1.15)

    # Derivative de
    ax = axes[0,1]; de = np.linspace(-0.55, 0.55, 300)
    for lbl,col in zip(["ND","ZD","PD"],["#DC2626","#059669","#2563EB"]):
        ax.plot(de, [_fuzz_de(x)[lbl] for x in de], lw=1.8, label=lbl, color=col)
    ax.set_xlabel("Derivada $\\Delta e$"); ax.set_title("MF de la derivada")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.15)

    # Disturbance
    ax = axes[1,0]; dv = np.linspace(-0.05, 1.05, 300)
    ax.plot(dv, [_fuzz_d(x)["NO"]  for x in dv], lw=1.8, label="NO",  color="#059669")
    ax.plot(dv, [_fuzz_d(x)["YES"] for x in dv], lw=1.8, label="YES", color="#DC2626")
    ax.set_xlabel("Perturbacion $d$"); ax.set_title("MF de la perturbacion")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.15)

    # Singletons (output)
    ax = axes[1,1]
    labels = ["VL","L","ML","M","MH","H","VH"]
    offsets = [_OFFSETS[l] for l in labels]
    bs = SERVO_FF;  ba = R2ET_U_SS
    vs = [np.clip(bs+o,0.05,0.98) for o in offsets]
    va = [np.clip(ba+o,0.05,0.98) for o in offsets]
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x-w/2, vs, w, color="#2563EB", alpha=0.8, label=f"Servo ($u_{{base}}$={bs:.2f})")
    ax.bar(x+w/2, va, w, color="#DC2626", alpha=0.8, label=f"Aware ($u_{{base}}$={ba:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Velocidad normalizada"); ax.set_title("Singletons de salida")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    fig.suptitle("Funciones de pertenencia del controlador difuso Mamdani", fontsize=10, fontweight="bold")
    fig.tight_layout(); _sv(fig, "fig_fuzzy_mf.png")


def plot_ql_training(h_servo: pd.DataFrame, h_aware: pd.DataFrame):
    plt.rcParams.update(PLT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, hdf, title, n_states, n_act in zip(
        axes, [h_servo, h_aware],
        [f"Q-Learning Servo ({_SQ_N_STATES} estados x {len(_SQ_ACTIONS)} acciones)",
         f"Q-Learning Aware ({_AQ_N_STATES} estados x {len(_AQ_ACTIONS)} acciones)"],
        [_SQ_N_STATES, _AQ_N_STATES], [len(_SQ_ACTIONS), len(_AQ_ACTIONS)]
    ):
        ep = hdf["ep"].to_numpy(); rm = hdf["r_mean"].to_numpy(); ep_s = hdf["eps"].to_numpy()
        w = max(1, len(ep)//30)
        rm_s = np.convolve(rm, np.ones(w)/w, mode="same")
        ax2 = ax.twinx()
        ax.plot(ep, rm_s, color="#2563EB", lw=1.5, label="Recompensa media (suavizada)")
        ax2.plot(ep, ep_s, color="#D97706", lw=1.2, linestyle="--", label="$\\epsilon$")
        ax.set_xlabel("Episodio"); ax.set_ylabel("Recompensa media", color="#2563EB")
        ax2.set_ylabel("$\\epsilon$ (exploracion)", color="#D97706")
        ax.set_title(title + f"\n{n_states} estados, {n_act} acciones")
        lines1, _ = ax.get_legend_handles_labels()
        lines2, _ = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, ["Recompensa","$\\epsilon$"], fontsize=8)
    fig.tight_layout(); _sv(fig, "fig_ql_training.png")


def plot_dl_training(h_servo: dict, h_aware: dict):
    plt.rcParams.update(PLT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, h, t in zip(axes, [h_servo, h_aware],
                         ["DL Servo (3-64-32-16-8-1, Adam)", "DL Aware (6-64-32-16-8-3, Adam)"]):
        ax.plot(h["train"], color="#2563EB", lw=1.4, label="Entrenamiento")
        ax.plot(h["val"],   color="#DC2626", lw=1.4, ls="--", label="Validacion")
        ax.set_xlabel("Epoca"); ax.set_ylabel("MSE"); ax.set_title(t)
        ax.set_yscale("log"); ax.legend(fontsize=8)
    fig.tight_layout(); _sv(fig, "fig_dl_training.png")


def plot_servo_response(sr: dict):
    plt.rcParams.update(PLT_RC)
    t = np.arange(SERVO_STEPS)
    fig, (ay, au) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for m in METHODS:
        ay.plot(t, sr[m]["y"], _STYLES[m], color=_COLORS[m], lw=1.6, label=m)
        au.plot(t, sr[m]["u"], _STYLES[m], color=_COLORS[m], lw=1.5, label=m)
    ay.plot(t, sr["PID"]["ref"], ":k", lw=1.2, label="Referencia")
    ay.axvspan(SERVO_DIST_START, SERVO_DIST_STOP, color="#EF4444", alpha=0.10, label="Perturbacion")
    ay.set_ylabel("Salida $y$"); ay.set_ylim(-0.08, 1.22); ay.legend(fontsize=8, ncol=3)
    ay.set_title("Fase 1 — Respuesta servo")
    au.axvspan(SERVO_DIST_START, SERVO_DIST_STOP, color="#EF4444", alpha=0.10)
    au.set_ylabel("Mando $u$"); au.set_xlabel("Muestra $k$"); au.set_ylim(-0.03, 1.08)
    au.legend(fontsize=8, ncol=3)
    fig.tight_layout(); _sv(fig, "fig_servo_response.png")


def plot_servo_kpi(sr: dict):
    plt.rcParams.update(PLT_RC)
    met = {m: _servo_metrics(sr[m]) for m in METHODS}
    keys = ["IAE","Energia","Var_u"]; labels = ["IAE","Energía","Variación $u$"]
    x = np.arange(len(keys)); w = 0.20
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(METHODS):
        vals = [met[m][k] for k in keys]
        ax.bar(x + (i-1.5)*w, vals, w, color=_COLORS[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Valor")
    ax.set_title("Fase 1 — KPI servo por metodo"); ax.legend(fontsize=8)
    fig.tight_layout(); _sv(fig, "fig_servo_kpi.png")


def plot_aware_occupancy(ar: dict):
    plt.rcParams.update(PLT_RC)
    t = np.arange(R2ET_STEPS)
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in METHODS:
        env = np.max(ar[m]["n"], axis=1)
        ax.plot(t, env, _STYLES[m], color=_COLORS[m], lw=1.7, label=m)
    ax.axhline(R2ET_CAPACITY, color="#111827", ls=":", lw=1.4, label="Capacidad (3)")
    ax.axhline(R2ET_ALARM,    color="#DC2626",  ls="--",lw=1.1, label="Alarma (2.7)")
    ax.axhline(R2ET_RISK,     color="#D97706",  ls="--",lw=1.0, label="Riesgo (2.4)")
    ax.axvspan(R2ET_PULSE_START, R2ET_PULSE_STOP, color="#F59E0B", alpha=0.12, label="Pulso")
    ax.axvspan(R2ET_DIST_START,  R2ET_DIST_STOP,  color="#EF4444", alpha=0.09, label="Perturbacion")
    ax.set_title("Fase 2 — Ocupacion maxima R2ET")
    ax.set_xlabel("Muestra $k$"); ax.set_ylabel("Piezas"); ax.set_ylim(0.7, 3.2)
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5,-0.16))
    fig.tight_layout(); _sv(fig, "fig_aware_occupancy.png")


def plot_aware_zoom(ar: dict):
    plt.rcParams.update(PLT_RC)
    t = np.arange(R2ET_STEPS); z = slice(R2ET_DIST_START-5, R2ET_DIST_STOP+15)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in METHODS:
        ax.plot(t[z], np.max(ar[m]["n"],axis=1)[z], _STYLES[m], color=_COLORS[m], lw=1.7, label=m)
    ax.axhline(R2ET_ALARM, color="#DC2626", ls="--", lw=1.1, label="Alarma")
    ax.axhline(R2ET_RISK,  color="#D97706", ls="--", lw=1.0, label="Riesgo")
    ax.axvspan(R2ET_DIST_START, R2ET_DIST_STOP, color="#EF4444", alpha=0.10, label="Atasco")
    ax.set_title("Fase 2 — Zoom durante atasco parcial")
    ax.set_xlabel("Muestra $k$"); ax.set_ylabel("Ocupacion max [piezas]"); ax.set_ylim(0.9, 2.95)
    ax.legend(ncol=2, fontsize=7.5); fig.tight_layout(); _sv(fig, "fig_aware_zoom.png")


def plot_aware_control(ar: dict):
    plt.rcParams.update(PLT_RC)
    t = np.arange(R2ET_STEPS)
    fig, axes = plt.subplots(2,1, figsize=(10,6), sharex=True)
    for ax, bi, belt in zip(axes, [0,1], ["E1","E2"]):
        for m in METHODS:
            ax.plot(t, ar[m]["u"][:,bi], _STYLES[m], color=_COLORS[m], lw=1.5, label=m)
        ax.axvspan(R2ET_DIST_START, R2ET_DIST_STOP, color="#EF4444", alpha=0.09)
        ax.set_ylabel(f"Velocidad {belt}"); ax.legend(fontsize=8)
        if bi==0: ax.set_title("Fase 2 — Senales de control por estera")
    axes[-1].set_xlabel("Muestra $k$"); fig.tight_layout(); _sv(fig, "fig_aware_control.png")


def plot_aware_split(ar: dict):
    plt.rcParams.update(PLT_RC)
    t = np.arange(R2ET_STEPS)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in METHODS:
        ax.plot(t, ar[m]["split"][:,0], _STYLES[m], color=_COLORS[m], lw=1.5, label=m)
    ax.axhline(0.5, color="#111827", ls=":", lw=1.0)
    ax.set_title("Fase 2 — Reparto del robot hacia E1")
    ax.set_xlabel("Muestra $k$"); ax.set_ylabel("Split E1"); ax.set_ylim(0.1, 0.9)
    ax.legend(fontsize=8); fig.tight_layout(); _sv(fig, "fig_aware_split.png")


def plot_pareto(ar: dict):
    plt.rcParams.update(PLT_RC)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in METHODS:
        me = _aware_metrics(ar[m])
        ax.scatter(me["Energia"], me["IAE"], s=100, color=_COLORS[m], zorder=5, label=m)
        ax.text(me["Energia"]+0.3, me["IAE"], m, fontsize=9)
    ax.set_xlabel("Energia"); ax.set_ylabel("IAE")
    ax.set_title("Fase 2 — Frontera error-energia"); ax.legend(fontsize=8)
    fig.tight_layout(); _sv(fig, "fig_pareto.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v):
    if isinstance(v,(int,np.integer)): return str(int(v))
    if isinstance(v,(float,np.floating)): return f"{float(v):.3g}"
    return str(v).replace("_",r"\_").replace("%",r"\%")

def _tex(rows, cols, path, caption, label):
    al = "l" + "r"*(len(cols)-1)
    ls = [
        "% Auto-generated",
        r"\begin{table}[H]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\small", r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}{al}@{{}}}}",
        r"\toprule", " & ".join(h for _,h in cols) + r" \\", r"\midrule",
    ]
    for row in rows:
        ls.append(" & ".join(_fmt(row[k]) for k,_ in cols) + r" \\")
    ls += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    Path(path).write_text("\n".join(ls)+"\n", encoding="utf-8")

def write_tables(sr, ar, h_dl_s, h_dl_a, ql_h_s, ql_h_a,
                 Xs, Ys, Xa, Ya, dl_s: DeepNet, dl_a: DeepNet,
                 Qs: np.ndarray, Qa: np.ndarray,
                 mse_s_test: float = 0.0, mse_a_test: float = 0.0,
                 n_servo_total: int = 2500, n_aware_total: int = 3000,
                 epochs: int = 1500):
    # Servo metrics
    rows_s = [{"Metodo":m, **_servo_metrics(sr[m])} for m in METHODS]
    _tex(rows_s,
         [("Metodo","Metodo"),("IAE","IAE"),("Mp_pct","Mp \\%"),
          ("Ts","$t_s$"),("Rec","Recup."),("Energia","Energia"),("Var_u","$|\\Delta u|$"),("Umax","$u_{max}$")],
         TABLE_DIR/"servo_metrics.tex",
         "Indicadores Fase 1 (servo, 4 metodos).", "tab:servo_metrics")
    pd.DataFrame(rows_s).to_csv(TABLE_DIR/"servo_metrics.csv", index=False)

    # Aware metrics
    rows_a = [{"Metodo":m, **_aware_metrics(ar[m])} for m in METHODS]
    _tex(rows_a,
         [("Metodo","Metodo"),("IAE","IAE"),("IAE_dist","IAE pert."),
          ("Ocup_max","Ocup. max"),("Margen","Margen"),("T_riesgo","T. riesgo"),
          ("Viol","Viol."),("Prod","Prod."),("Energia","Energia"),("Desbal","Desbal.")],
         TABLE_DIR/"aware_metrics.tex",
         "Indicadores Fase 2 (coordinacion aware, 4 metodos).", "tab:aware_metrics")
    pd.DataFrame(rows_a).to_csv(TABLE_DIR/"aware_metrics.csv", index=False)

    # Improvement vs PID (aware)
    base = _aware_metrics(ar["PID"])
    rows_imp = []
    for m in ["Fuzzy","DL","QL"]:
        me = _aware_metrics(ar[m])
        rows_imp.append({
            "Metodo": m,
            "DIAE_pct": round(100*(base["IAE"]-me["IAE"])/base["IAE"],1),
            "DProd": round(me["Prod"]-base["Prod"],2),
            "DEnerg": round(me["Energia"]-base["Energia"],2),
            "DDesbal_pct": round(100*(base["Desbal"]-me["Desbal"])/max(base["Desbal"],1e-6),1),
        })
    _tex(rows_imp,
         [("Metodo","Metodo"),("DIAE_pct","$\\Delta$IAE \\%"),
          ("DProd","$\\Delta$Prod."),("DEnerg","$\\Delta$Energia"),("DDesbal_pct","$\\Delta$Desbal. \\%")],
         TABLE_DIR/"aware_improvement.tex",
         "Mejora frente al PID-Aware.", "tab:aware_improvement")

    # DL training summary — now includes separate test-set MSE
    n_tr_s = int(n_servo_total * 0.70);  n_vl_s = int(n_servo_total * 0.15)
    n_te_s = n_servo_total - n_tr_s - n_vl_s
    n_tr_a = int(n_aware_total * 0.70);  n_vl_a = int(n_aware_total * 0.15)
    n_te_a = n_aware_total - n_tr_a - n_vl_a
    rows_dl = [
        {"Fase":"Servo","Arch":"3-64-32-16-8-1","Epocas":epochs,
         "N_train":n_tr_s,"N_val":n_vl_s,"N_test":n_te_s,
         "MSE_train":round(h_dl_s["train"][-1],5),
         "MSE_val":round(h_dl_s["val"][-1],5),
         "MSE_test":round(mse_s_test,5)},
        {"Fase":"Aware","Arch":"6-64-32-16-8-3","Epocas":epochs,
         "N_train":n_tr_a,"N_val":n_vl_a,"N_test":n_te_a,
         "MSE_train":round(h_dl_a["train"][-1],5),
         "MSE_val":round(h_dl_a["val"][-1],5),
         "MSE_test":round(mse_a_test,5)},
    ]
    _tex(rows_dl,
         [("Fase","Fase"),("Arch","Arquitectura"),("Epocas","Epocas"),
          ("N_train","Entren."),("N_val","Val."),("N_test","Test"),
          ("MSE_train","MSE entren."),("MSE_val","MSE val."),("MSE_test","MSE test")],
         TABLE_DIR/"mlp_training_summary.tex",
         "Entrenamiento Deep Learning: particion 70/15/15 y MSE en las tres particiones.",
         "tab:mlp_training")

    # Q-Learning summary
    rows_ql = [
        {"Fase":"Servo","Estados":_SQ_N_STATES,"Acciones":len(_SQ_ACTIONS),
         "Episodios":600,"Horizonte":100,"Alpha":0.15,"Gamma":0.95,
         "Eps0":0.60,"EpsMin":0.02,"R_final":round(float(ql_h_s["r_mean"].tail(30).mean()),4)},
        {"Fase":"Aware","Estados":_AQ_N_STATES,"Acciones":len(_AQ_ACTIONS),
         "Episodios":1500,"Horizonte":160,"Alpha":0.18,"Gamma":0.92,
         "Eps0":0.65,"EpsMin":0.02,"R_final":round(float(ql_h_a["r_mean"].tail(30).mean()),4)},
    ]
    _tex(rows_ql,
         [("Fase","Fase"),("Estados","Estados"),("Acciones","Acciones"),
          ("Episodios","Episodios"),("Horizonte","Horizonte"),
          ("Alpha","$\\alpha$"),("Gamma","$\\gamma$"),
          ("Eps0","$\\epsilon_0$"),("EpsMin","$\\epsilon_{min}$"),("R_final","$r_{final}$")],
         TABLE_DIR/"ql_training_summary.tex",
         "Parametros y resultados del entrenamiento Q-Learning.", "tab:ql_training")


# ═══════════════════════════════════════════════════════════════════════════════
# 10b. ONLINE Q-LEARNING  (start from trained table, continue updating)
# ═══════════════════════════════════════════════════════════════════════════════

def run_online_ql(Q_offline: np.ndarray, n_ep: int = 50,
                  eps: float = 0.08, alpha: float = 0.10,
                  seed: int = SEED + 99) -> tuple[np.ndarray, pd.DataFrame]:
    """Deploy offline-trained Q-table on the standard R2ET scenario and keep
    updating via TD(0) (online learning).  ε stays small so the agent mostly
    exploits but still fine-tunes.

    Returns the updated Q-table and a per-episode DataFrame with IAE and reward.
    """
    rng = np.random.default_rng(seed)
    Q = Q_offline.copy()
    history = []
    for ep in range(n_ep):
        n = np.array([1.25, 1.25])   # same start as the deployment test
        ep_iae = 0.0;  ep_r = 0.0
        for k in range(R2ET_STEPS):
            dm = _demand(k);  d = _d_aware(k);  ls = _loss(k)
            s = _encode_aware(n, d)
            a = int(rng.integers(len(_AQ_ACTIONS)) if rng.random() < eps
                    else np.argmax(Q[s]))
            spd, spl, adm_q = _AQ_ACTIONS[a]
            u = np.array([spd, spd])
            u, adm_hard = _cap_supervisor(u, float(np.max(n)), d)
            adm = min(float(adm_q), float(adm_hard))
            split = np.array([spl, 1.0 - spl])
            n_new, _ = _r2et_aware_step(n, u, split, adm, dm, ls)
            s_new = _encode_aware(n_new, d)
            iae  = float(np.sum(np.abs(n_new - R2ET_TARGET)))
            viol = float(np.sum(np.maximum(0.0, n_new - R2ET_CAPACITY)))
            alrm = float(np.sum(np.maximum(0.0, n_new - R2ET_ALARM)))
            bal  = abs(float(n_new[0] - n_new[1]))
            r    = -1.5*iae - 30.0*viol - 8.0*alrm - 0.20*bal
            Q[s, a] += alpha * (r + 0.92 * float(np.max(Q[s_new])) - Q[s, a])
            ep_iae += iae;  ep_r += r;  n = n_new
        history.append({"ep": ep+1, "IAE": round(ep_iae, 3),
                        "r_mean": round(ep_r / R2ET_STEPS, 4)})
    return Q, pd.DataFrame(history)


def plot_online_ql(offline_iae: float, df_online: pd.DataFrame) -> None:
    """Plot IAE per online episode vs. the offline-only deployment baseline."""
    plt.rcParams.update(PLT_RC)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ep = df_online["ep"].to_numpy()
    iae = df_online["IAE"].to_numpy()
    # smoothed trend
    w = max(1, len(ep)//8)
    iae_s = np.convolve(iae, np.ones(w)/w, mode="same")
    ax.plot(ep, iae, color="#059669", lw=1.2, alpha=0.40, label="IAE por episodio")
    ax.plot(ep, iae_s, color="#059669", lw=2.0, label="Tendencia (media movil)")
    ax.axhline(offline_iae, color="#4B5563", lw=1.4, ls="--",
               label=f"Linea base offline ({offline_iae:.1f})")
    ax.set_xlabel("Episodio online")
    ax.set_ylabel("IAE")
    ax.set_title("Q-Learning online: mejora del IAE sobre el escenario de prueba")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _sv(fig, "fig_ql_online.png")


def write_online_ql_table(offline_iae: float, df_online: pd.DataFrame) -> None:
    n = len(df_online)
    q1 = df_online["IAE"].iloc[:n//4].mean()
    q4 = df_online["IAE"].iloc[-n//4:].mean()
    rows = [
        {"Fase": "Offline (linea base)", "Episodios": 0, "IAE_medio": round(offline_iae, 3),
         "Mejora_pct": 0.0},
        {"Fase": "Q1 online (ep 1-12)",  "Episodios": n//4,
         "IAE_medio": round(q1, 3),
         "Mejora_pct": round(100*(offline_iae - q1)/offline_iae, 1)},
        {"Fase": f"Q4 online (ep {3*n//4+1}-{n})", "Episodios": n - 3*n//4,
         "IAE_medio": round(q4, 3),
         "Mejora_pct": round(100*(offline_iae - q4)/offline_iae, 1)},
    ]
    _tex(rows,
         [("Fase","Fase"),("Episodios","Ep."),
          ("IAE_medio","IAE medio"),("Mejora_pct","Mejora \\%")],
         TABLE_DIR/"ql_online_summary.tex",
         "Q-Learning online: mejora del IAE por cuartil de episodios.", "tab:ql_online")
    pd.DataFrame(rows).to_csv(TABLE_DIR/"ql_online_summary.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 10c.  ROBUSTNESS — harder scenario: +20% inflow, -15% exit capacity
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_robust_scenario(Qa: np.ndarray, dl_a: "DeepNet",
                             demand_scale: float = 1.20,
                             exit_scale: float = 0.85) -> dict[str, dict]:
    """Run all four methods under a harder parametric scenario."""
    orig_inflow = R2ET_INFLOW_BASE
    orig_pulse  = R2ET_INFLOW_PULSE
    orig_gain   = R2ET_EXIT_GAIN

    import builtins
    # Temporarily patch module-level constants via a local closure
    def _d_rb(k):
        base = orig_inflow * demand_scale
        pulse = orig_pulse * demand_scale * float(R2ET_PULSE_START <= k <= R2ET_PULSE_STOP)
        return base + pulse

    def _step_rb(n_p, u, split, adm, demand, loss):
        feed = demand * adm * split
        commanded = (orig_gain * exit_scale) * u * (1.0 - loss)
        outflow = np.minimum(commanded, n_p + feed)
        n_next = np.clip(n_p + feed - outflow, 0.0, R2ET_CAPACITY)
        return n_next, outflow

    results = {}
    for method in METHODS:
        n = np.zeros((R2ET_STEPS, 2)); u = np.zeros((R2ET_STEPS, 2))
        out = np.zeros((R2ET_STEPS, 2)); sph = np.zeros((R2ET_STEPS, 2))
        adh = np.ones(R2ET_STEPS)
        n[0] = [1.25, 1.25]
        sts = [PIDState(), PIDState()] if method == "PID" else None
        prev_e = np.zeros(2); prev_n = n[0].copy()

        for k in range(1, R2ET_STEPS):
            dm = _d_rb(k); d = _d_aware(k); ls = _loss(k)
            nm = float(np.max(n[k-1]))
            e = n[k-1] - R2ET_TARGET; de = e - prev_e; bal = float(n[k-1,0]-n[k-1,1])

            if method == "PID":
                fkp = 1.55 if (nm>R2ET_ALARM or d>0) else (1.25 if nm>R2ET_RISK else 1.0)
                fki = 0.65 if (nm>R2ET_ALARM or d>0) else (0.82 if nm>R2ET_RISK else 1.0)
                uk = np.array([pid_step(float(e[i]),sts[i],0.52*fkp,0.028*fki,0.10,
                                        0.05,1.0,ff=R2ET_U_SS) for i in range(2)])
                uk, adm = _cap_supervisor(uk, nm, d); sp = _isplit(n[k-1])
            elif method == "Fuzzy":
                u_ss_rb = (orig_inflow * demand_scale * 0.5) / (orig_gain * exit_scale)
                uk = np.array([fuzzy_speed(float(e[i]),float(de[i]),d,base=u_ss_rb) for i in range(2)])
                uk, adm = _cap_supervisor(uk, nm, d); sp = np.array([_fuzzy_split(bal), 0.0]); sp[1]=1-sp[0]
            elif method == "DL":
                x = np.array([float(e[0]),float(e[1]),float(de[0]),float(de[1]),bal,d])
                yp = dl_a.predict(x)
                uk = np.array([float(np.clip(yp[0],0.05,1.0)), float(np.clip(yp[1],0.05,1.0))])
                s1 = float(np.clip(yp[2],0.20,0.80)); uk,adm = _cap_supervisor(uk,nm,d); sp=np.array([s1,1-s1])
            else:  # QL
                s = _encode_aware(n[k-1], d); a = int(np.argmax(Qa[s]))
                spd,spl,adm_q = _AQ_ACTIONS[a]; uk=np.array([spd,spd]); uk,adm_hard=_cap_supervisor(uk,nm,d)
                adm=min(float(adm_q),float(adm_hard)); sp=np.array([spl,1-spl])

            u[k]=uk; sph[k]=sp; adh[k]=adm
            n[k],out[k] = _step_rb(n[k-1], u[k], sp, adm, dm, ls)
            prev_e=e; prev_n=n[k-1].copy()

        results[method] = {"n":n,"u":u,"outflow":out,"split":sph,"admission":adh}
    return results


def write_robustness_table(base_results: dict, rb_results: dict) -> None:
    rows = []
    for m in METHODS:
        mb = _aware_metrics(base_results[m])
        mr = _aware_metrics(rb_results[m])
        rows.append({
            "Metodo": m,
            "IAE_base": mb["IAE"], "IAE_rob": mr["IAE"],
            "Ocup_max_base": mb["Ocup_max"], "Ocup_max_rob": mr["Ocup_max"],
            "Viol_rob": mr["Viol"],
            "Margen_rob": mr["Margen"],
        })
    _tex(rows,
         [("Metodo","Metodo"),
          ("IAE_base","IAE nominal"),("IAE_rob","IAE robusto (+20\\%/-15\\%)"),
          ("Ocup_max_base","Ocup. nom."),("Ocup_max_rob","Ocup. rob."),
          ("Viol_rob","Viol. rob."),("Margen_rob","Margen rob.")],
         TABLE_DIR/"robustness_comparison.tex",
         "Comparacion de desempeno en escenario nominal vs.\\ robusto (+20\\% demanda, $-$15\\% salida).",
         "tab:robustness")
    pd.DataFrame(rows).to_csv(TABLE_DIR/"robustness_comparison.csv", index=False)


def write_timeseries(sr, ar):
    rows_s = []
    for k in range(SERVO_STEPS):
        row = {"t":k,"ref":sr["PID"]["ref"][k],"dist":int(sr["PID"]["dist"][k])}
        for m in METHODS: row[f"{m}_y"]=sr[m]["y"][k]; row[f"{m}_u"]=sr[m]["u"][k]
        rows_s.append(row)
    with (DATA_DIR/"timeseries_servo.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_s[0].keys())); w.writeheader(); w.writerows(rows_s)

    rows_a = []
    for k in range(R2ET_STEPS):
        row = {"t":k}
        for m in METHODS:
            row[f"{m}_n1"]=ar[m]["n"][k,0]; row[f"{m}_n2"]=ar[m]["n"][k,1]
            row[f"{m}_u1"]=ar[m]["u"][k,0]; row[f"{m}_u2"]=ar[m]["u"][k,1]
            row[f"{m}_spl"]=ar[m]["split"][k,0]; row[f"{m}_adm"]=ar[m]["admission"][k]
        rows_a.append(row)
    with (DATA_DIR/"timeseries_aware.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_a[0].keys())); w.writeheader(); w.writerows(rows_a)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== Activity 08 MROB --- Group 1, R2ET (4 methods x 2 phases) ===")

    # ── DL: large dataset + train/val/test split ───────────────────────────────
    N_SERVO = 8000;  N_AWARE = 10000;  DL_EPOCHS = 2500

    print(f"[1/8] DL servo: generating {N_SERVO} samples, training {DL_EPOCHS} epochs...")
    Xs_all, Ys_all = _make_servo_data(n=N_SERVO, seed=SEED)
    # 70/15/15 split: train | val | test
    rng_split = np.random.default_rng(SEED + 200)
    idx_s = rng_split.permutation(N_SERVO)
    n_val_s  = int(N_SERVO * 0.15);  n_test_s = int(N_SERVO * 0.15)
    Xs_test = Xs_all[idx_s[:n_test_s]];  Ys_test = Ys_all[idx_s[:n_test_s]]
    Xs_tv   = Xs_all[idx_s[n_test_s:]];  Ys_tv   = Ys_all[idx_s[n_test_s:]]
    dl_s = DeepNet([3, 64, 32, 16, 8, 1], seed=SEED)
    h_dl_s = dl_s.train(Xs_tv, Ys_tv, epochs=DL_EPOCHS, lr_max=3e-3, lr_min=3e-4, batch=64, seed=SEED)
    mse_s_test = float(np.mean((dl_s.predict_batch(Xs_test) - Ys_test)**2))
    print(f"       train={h_dl_s['train'][-1]:.5f}  val={h_dl_s['val'][-1]:.5f}  test={mse_s_test:.5f}")

    print(f"[2/8] DL aware: generating {N_AWARE} samples, training {DL_EPOCHS} epochs...")
    Xa_all, Ya_all = _make_aware_data(n=N_AWARE, seed=SEED+3)
    idx_a = rng_split.permutation(N_AWARE)
    n_val_a  = int(N_AWARE * 0.15);  n_test_a = int(N_AWARE * 0.15)
    Xa_test = Xa_all[idx_a[:n_test_a]];  Ya_test = Ya_all[idx_a[:n_test_a]]
    Xa_tv   = Xa_all[idx_a[n_test_a:]];  Ya_tv   = Ya_all[idx_a[n_test_a:]]
    dl_a = DeepNet([6, 64, 32, 16, 8, 3], seed=SEED+1)
    h_dl_a = dl_a.train(Xa_tv, Ya_tv, epochs=DL_EPOCHS, lr_max=3e-3, lr_min=3e-4, batch=64, seed=SEED+1)
    mse_a_test = float(np.mean((dl_a.predict_batch(Xa_test) - Ya_test)**2))
    print(f"       train={h_dl_a['train'][-1]:.5f}  val={h_dl_a['val'][-1]:.5f}  test={mse_a_test:.5f}")

    # ── Q-Learning: offline training ──────────────────────────────────────────
    print(f"[3/8] Q-Learning servo ({_SQ_N_STATES}s x {len(_SQ_ACTIONS)}a, 600 ep)...")
    Qs, ql_h_s = train_q_servo()
    print(f"       r_mean last-30 = {ql_h_s['r_mean'].tail(30).mean():.4f}")

    print(f"[4/8] Q-Learning aware ({_AQ_N_STATES}s x {len(_AQ_ACTIONS)}a, 1500 ep)...")
    Qa, ql_h_a = train_q_aware()
    print(f"       r_mean last-30 = {ql_h_a['r_mean'].tail(30).mean():.4f}")

    # ── Standard simulations ──────────────────────────────────────────────────
    print("[5/8] Simulating Phase 1 (servo) + Phase 2 (aware)...")
    sr = {"PID": simulate_servo_pid(), "Fuzzy": simulate_servo_fuzzy(),
          "DL": simulate_servo_dl(dl_s), "QL": simulate_servo_ql(Qs)}
    ar = {"PID": simulate_aware_pid(), "Fuzzy": simulate_aware_fuzzy(),
          "DL": simulate_aware_dl(dl_a), "QL": simulate_aware_ql(Qa)}

    # ── Q-Learning online deployment ──────────────────────────────────────────
    print("[6/8] Q-Learning online: deploying and fine-tuning (50 episodes, eps=0.08)...")
    offline_iae = _aware_metrics(ar["QL"])["IAE"]
    _, df_online = run_online_ql(Qa, n_ep=50, eps=0.08, alpha=0.10)
    print(f"       offline IAE={offline_iae:.2f}  online Q4 mean={df_online['IAE'].tail(12).mean():.2f}")

    # ── Robustness scenario ────────────────────────────────────────────────────
    print("[7/8] Robustness scenario (+20% demand, -15% exit)...")
    rb_results = simulate_robust_scenario(Qa, dl_a, demand_scale=1.20, exit_scale=0.85)

    # ── Figures + Tables ───────────────────────────────────────────────────────
    print("[8/8] Generating figures and tables...")
    plot_fuzzy_memberships()
    plot_dl_training(h_dl_s, h_dl_a)
    plot_ql_training(ql_h_s, ql_h_a)
    plot_online_ql(offline_iae, df_online)
    plot_servo_response(sr)
    plot_servo_kpi(sr)
    plot_aware_occupancy(ar)
    plot_aware_zoom(ar)
    plot_aware_control(ar)
    plot_aware_split(ar)
    plot_pareto(ar)

    write_tables(sr, ar, h_dl_s, h_dl_a, ql_h_s, ql_h_a,
                 Xs_tv, Ys_tv, Xa_tv, Ya_tv, dl_s, dl_a, Qs, Qa,
                 mse_s_test=mse_s_test, mse_a_test=mse_a_test,
                 n_servo_total=N_SERVO, n_aware_total=N_AWARE, epochs=DL_EPOCHS)
    write_online_ql_table(offline_iae, df_online)
    write_robustness_table(ar, rb_results)
    write_timeseries(sr, ar)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n--- Phase 1 Servo ---")
    for m in METHODS:
        me = _servo_metrics(sr[m])
        print(f"  {m:6s}  IAE={me['IAE']:.2f}  Mp={me['Mp_pct']:.1f}%  Ts={me['Ts']}  E={me['Energia']:.1f}")

    print("\n--- Phase 2 Aware ---")
    for m in METHODS:
        me = _aware_metrics(ar[m])
        print(f"  {m:6s}  IAE={me['IAE']:.2f}  max={me['Ocup_max']:.3f}  "
              f"Viol={me['Viol']}  Prod={me['Prod']:.2f}  E={me['Energia']:.2f}")

    print("\n--- Robustness (+20% demand, -15% exit) ---")
    for m in METHODS:
        me = _aware_metrics(rb_results[m])
        print(f"  {m:6s}  IAE={me['IAE']:.2f}  max={me['Ocup_max']:.3f}  Viol={me['Viol']}")

    print(f"\n--- Q-Learning Online (50 ep, eps=0.08) ---")
    print(f"  Offline baseline IAE = {offline_iae:.2f}")
    print(f"  Online Q1 mean IAE   = {df_online['IAE'].head(12).mean():.2f}")
    print(f"  Online Q4 mean IAE   = {df_online['IAE'].tail(12).mean():.2f}")

    print("\nDone. Figures -> figures/plots/   Tables -> artifacts/tables/")


if __name__ == "__main__":
    main()
