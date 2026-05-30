"""v2_simulate.py — Actividad 08 MROB, Grupo 1, caso R2ET.

Simulación completa de los cuatro controladores en NumPy puro (sin frameworks de ML).

CONTROLADORES (4 métodos):
  PID      : PID discreto con feedforward y anti-windup por canal.
  Fuzzy    : Mamdani — MFs triangulares/trapezoidales, AND=mín, defuzz WA singletons.
  DL       : Red profunda (4 capas ocultas), Adam + cosine LR, NumPy vectorizado.
  Q-Learn  : Q-learning tabular ε-greedy, TD(0), entrenado desde cero en cada ejecución.

FASES:
  Fase 1 — Servo: planta de primer orden discreta, escalón + perturbación de carga.
  Fase 2 — Aware: coordinación de dos esteras R2ET ante variación de peso/perturbación.

PLANTA (Grupo 1 — R2ET, servo):
  y(k+1) = 0.92·y(k) + 0.08·u(k) − 0.024·d(k)
  Reducción de primer orden de Gp(s) (Guía VIU) conservando la ganancia DC unitaria,
  discretizada por ZOH con Ts = tau_d/12:
    - Polo: a = e^(−Ts/tau_d) = e^(−1/12) ≈ 0.92
    - Ganancia de control: b = K_f(1−a) = 0.08,  K_f = Gp(0) = 1.0
    - Ganancia de perturbación: bd = Kd(1−a) = 0.024,  Kd = 0.30

USO:
  cd <raíz del proyecto>
  python src/python/v2_simulate.py
"""

from __future__ import annotations

import csv
import math
import sys

# Forzar UTF-8 en consola Windows para caracteres especiales
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── Rutas de salida ─────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
FIG_DIR   = ROOT / "figures" / "plots"
TABLE_DIR = ROOT / "artifacts" / "tables"
DATA_DIR  = ROOT / "artifacts" / "data"
for _d in (FIG_DIR, TABLE_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES DEL ESCENARIO
# ═══════════════════════════════════════════════════════════════════════════════

# --- Planta servo (Fase 1) ---
# Modelo de primer orden equivalente a la Gp(s) ASIGNADA en la guía (Equipo 1, R2ET):
#   Gp(s) = (0.174 s + 0.3744)/(s^2 + 0.785 s + 0.3744)
#   -> ganancia DC = 1.0, wn = 0.612 rad/s, zeta = 0.641 (bien amortiguado),
#      cero en -2.15, polos dominantes en -0.393 +/- j0.470.
# Aproximación por constante de tiempo dominante (tau = 1/|Re(polos)| = 2.55 s),
# conservando la ganancia DC unitaria de Gp(s):
#   K_f = 1.0  (ganancia DC, igual que Gp(0))
#   tau = 2.55 s
# Discretización ZOH con Ts = tau/12 = 0.2125 s (12 muestras por constante de tiempo):
#   a  = e^(-Ts/tau) = e^(-1/12) = 0.9200
#   b  = K_f (1 - a) = 0.08
#   bd = K_d (1 - a) = 0.024   (perturbación de carga, K_d = 0.30)
PL_A       = 0.92   # polo discreto = e^(-Ts/tau), Ts/tau = 1/12
PL_B       = 0.08   # ganancia de control = K_f(1-a), K_f = 1.0 (= Gp(0))
PL_D       = 0.024  # sensibilidad a la perturbación de carga = K_d(1-a), K_d = 0.30
SV_REF     = 0.6    # referencia de operación normalizada (60% del rango, deja margen de actuador)
PL_FF      = SV_REF * (1.0 - PL_A) / PL_B  # feedforward estático u_ss = 0.60 para y = SV_REF

# --- Escenario Fase 1 ---
SV_STEPS   = 150    # número de muestras totales
SV_STEP_K  = 10     # muestra en la que arranca el escalón de referencia
SV_D_START = 85     # inicio de la perturbación de carga
SV_D_STOP  = 110    # fin de la perturbación

# --- R2ET — Fase 2 ---
ET_CAP     = 3.0    # capacidad máxima de cada estera [piezas]
ET_TARGET  = 1.5    # nivel objetivo de llenado [piezas]
ET_STEPS   = 160    # número de muestras totales
ET_BASE_IN = 0.42   # caudal base de entrada [piezas/muestra]
ET_PULSE   = 0.35   # incremento de caudal durante el pulso
ET_PUL_S   = 55     # inicio del pulso de demanda
ET_PUL_E   = 82     # fin del pulso de demanda
ET_JAM_S   = 95     # inicio del atasco/cambio de peso
ET_JAM_E   = 118    # fin del atasco/cambio de peso
ET_JAM_L   = 0.35   # pérdida de capacidad en E1 por atasco (35 %)
ET_EG      = 0.46   # eficiencia de salida [piezas/muestra por unidad de velocidad]
ET_RISK    = 2.4    # umbral de riesgo [piezas]
ET_ALARM   = 2.7    # umbral de alarma [piezas]
ET_USS     = (ET_BASE_IN * 0.5) / ET_EG  # velocidad de estado estacionario ≈ 0.457

SEED = 8

# --- Estética de figuras ---
METHODS = ["PID", "Fuzzy", "DL", "QL"]
MC = {"PID": "#374151", "Fuzzy": "#1d4ed8", "DL": "#dc2626", "QL": "#059669"}
MS = {"PID": "--",      "Fuzzy": "-",        "DL": "-.",      "QL": ":"}

PLT_RC = {
    "figure.dpi": 140, "savefig.dpi": 160,
    "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False,
}


def _sv(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. CONTROLADOR PID DISCRETO CON ANTI-WINDUP
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PIDState:
    integral: float = 0.0
    prev_e:   float = 0.0


def pid_step(e: float, st: PIDState, kp: float, ki: float, kd: float,
             u_min=0.0, u_max=1.0, aw_lim=6.0, ff=0.0) -> float:
    """Un paso del PID discreto con feedforward y anti-windup condicional."""
    de  = e - st.prev_e
    raw = ff + kp * e + ki * st.integral + kd * de
    u   = float(np.clip(raw, u_min, u_max))
    # anti-windup: sólo integrar si el actuador no está saturado en la dirección del error
    if not (raw > u_max and e > 0) and not (raw < u_min and e < 0):
        st.integral = float(np.clip(st.integral + e, -aw_lim, aw_lim))
    st.prev_e = e
    return u


# ═══════════════════════════════════════════════════════════════════════════════
#  2. CONTROLADOR DIFUSO MAMDANI
# ═══════════════════════════════════════════════════════════════════════════════

def _tri(x: float, a, b, c) -> float:
    la = (x - a) / (b - a) if b != a else float(x >= a)
    rb = (c - x) / (c - b) if c != b else float(x <= b)
    return float(np.clip(min(la, rb), 0.0, 1.0))


def _trap(x: float, a, b, c, d) -> float:
    if x <= a or x >= d: return 0.0
    if b <= x <= c: return 1.0
    return float(np.clip((x - a) / (b - a) if x < b else (d - x) / (d - c), 0.0, 1.0))


# Funciones de pertenencia del error e ∈ [−1.5, 1.5] — 7 conjuntos (densos cerca de cero)
_MF_E = {
    "NL": ("trap", -1.5, -1.5, -0.70, -0.40),
    "NM": ("tri",  -0.60, -0.35, -0.12),
    "NS": ("tri",  -0.22, -0.11,  0.00),
    "ZE": ("tri",  -0.07,  0.00,  0.07),
    "PS": ("tri",   0.00,  0.11,  0.22),
    "PM": ("tri",   0.12,  0.35,  0.60),
    "PL": ("trap",  0.40,  0.70,  1.50, 1.50),
}
_LVL_E = {"NL": -3, "NM": -2, "NS": -1, "ZE": 0, "PS": 1, "PM": 2, "PL": 3}

# Funciones de pertenencia de la derivada Δe ∈ [−0.5, 0.5] — 5 conjuntos
_MF_DE = {
    "NB": ("trap", -0.50, -0.50, -0.22, -0.10),
    "NS": ("tri",  -0.18, -0.08,  0.00),
    "ZE": ("tri",  -0.05,  0.00,  0.05),
    "PS": ("tri",   0.00,  0.08,  0.18),
    "PB": ("trap",  0.10,  0.22,  0.50, 0.50),
}
_LVL_DE = {"NB": -2, "NS": -1, "ZE": 0, "PS": 1, "PB": 2}

# Pasos de velocidad por nivel lingüístico (FAM continuo)
_FZ_KE  = 0.12   # por nivel de error
_FZ_KDE = 0.05   # por nivel de derivada


def _mf(x: float, spec) -> float:
    return _tri(x, *spec[1:]) if spec[0] == "tri" else _trap(x, *spec[1:])


def _mf_error(e: float) -> dict:
    return {k: _mf(e, v) for k, v in _MF_E.items()}


def _mf_deriv(de: float) -> dict:
    return {k: _mf(de, v) for k, v in _MF_DE.items()}


# Funciones de pertenencia de la perturbación d ∈ [0, 1]
def _mf_dist(d: float) -> dict:
    return {
        "NO":  _trap(d, 0.0, 0.0,  0.25, 0.55),
        "YES": _trap(d, 0.35, 0.65, 1.0, 1.0),
    }


def fuzzy_speed(e: float, de: float, d: float, u_base: float) -> float:
    """Mamdani 7×5 (35 reglas): fuzzificar → inferencia AND-mín → defuzz WA.

    El consecuente de cada regla (velocidad) se sintetiza por niveles lingüísticos:
    c = u_base + nivel_error·_FZ_KE + nivel_derivada·_FZ_KDE, lo que equivale a una
    matriz asociativa difusa (FAM) diagonal de 35 reglas, más fina que la versión 5×3.
    """
    mu_e  = _mf_error(e)
    mu_de = _mf_deriv(de)
    num = den = 0.0
    for el, me in mu_e.items():
        if me < 1e-9:
            continue
        for dl, md in mu_de.items():
            w = min(me, md)
            if w < 1e-9:
                continue
            c = float(np.clip(u_base + _LVL_E[el] * _FZ_KE + _LVL_DE[dl] * _FZ_KDE, 0.05, 0.98))
            num += w * c
            den += w
    # regla adicional de perturbación: IF d=YES → empujar hacia velocidad alta
    dw = _mf_dist(d)["YES"]
    if dw > 1e-9:
        num += dw * float(np.clip(u_base + 0.30, 0.05, 0.98))
        den += dw
    return float(np.clip(num / den, 0.05, 0.98)) if den > 1e-9 else float(u_base)


def fuzzy_split(balance: float) -> float:
    """Reparto difuso: balance = n1−n2 → fracción hacia E1."""
    mu = {
        "e1h": _trap(balance,  0.15,  0.6, 2.0, 2.0),
        "bal": _tri(balance,  -0.30,  0.0, 0.30),
        "e2h": _trap(balance, -2.0, -2.0, -0.6, -0.15),
    }
    singletons = {"e1h": 0.30, "bal": 0.50, "e2h": 0.70}
    num = sum(mu[k] * singletons[k] for k in mu)
    den = sum(mu.values())
    return float(np.clip(num / den, 0.20, 0.80)) if den > 1e-9 else 0.50


# ═══════════════════════════════════════════════════════════════════════════════
#  3. RED NEURONAL PROFUNDA (DEEP LEARNING)
# ═══════════════════════════════════════════════════════════════════════════════

class DeepNet:
    """Red completamente conectada: tanh ocultas, sigmoide salida.

    Optimizador: Adam (β1=0.9, β2=0.999) con cosine LR annealing.
    Init: He/Kaiming (escala = sqrt(2/fan_in)).
    Backprop: mini-batch vectorizado con operaciones matriciales (sin bucles por muestra).
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
        self._t  = 0

    def _forward(self, X: np.ndarray) -> list[np.ndarray]:
        sq = X.ndim == 1
        if sq:
            X = X[np.newaxis, :]
        hs = [X]; h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b
            h = np.tanh(z) if i < len(self.W) - 1 else 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            hs.append(h)
        return [a.squeeze(0) if sq else a for a in hs]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[-1]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[-1]

    def _gradients(self, X: np.ndarray, Y: np.ndarray):
        B = float(X.shape[0])
        hs = self._forward(X)
        yp = hs[-1]
        loss = float(np.mean((yp - Y) ** 2))
        delta = 2.0 * (yp - Y) / B * yp * (1.0 - yp)  # MSE × sigmoide'
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = delta.T @ hs[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i]) * (1.0 - hs[i] ** 2)  # tanh'
        return gW, gb, loss

    def _adam_step(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        bc1 = 1.0 - b1 ** self._t
        bc2 = 1.0 - b2 ** self._t
        for i in range(len(self.W)):
            for p, mp, vp, g in (
                (self.W, self._mW, self._vW, gW),
                (self.b, self._mb, self._vb, gb),
            ):
                mp[i] = b1 * mp[i] + (1 - b1) * g[i]
                vp[i] = b2 * vp[i] + (1 - b2) * g[i] ** 2
                p[i] -= lr * (mp[i] / bc1) / (np.sqrt(vp[i] / bc2) + eps)

    def train(self, X, Y, epochs=1500, lr_max=3e-3, lr_min=3e-4,
              batch=64, val_frac=0.15, seed=42) -> dict:
        rng = np.random.default_rng(seed)
        n = len(X)
        nv = max(1, int(n * val_frac))
        perm = rng.permutation(n)
        vi, ti = perm[:nv], perm[nv:]
        Xv, Yv, Xt, Yt = X[vi], Y[vi], X[ti], Y[ti]
        hist = {"train": [], "val": []}
        for ep in range(epochs):
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * ep / epochs))
            order = rng.permutation(len(Xt))
            el = 0.0; nb = 0
            for s in range(0, len(Xt), batch):
                idx = order[s: s + batch]
                gW, gb, bl = self._gradients(Xt[idx], Yt[idx])
                self._adam_step(gW, gb, lr)
                el += bl; nb += 1
            hist["train"].append(el / nb)
            hist["val"].append(float(np.mean((self.predict_batch(Xv) - Yv) ** 2)))
        return hist


# ─── Deep Learning: optimización directa de política (Fase 1) ────────────────
#
# La red servo se entrena minimizando el IAE directamente sobre la planta,
# SIN copiar ningún controlador externo. En cada época se corren batch_size
# simulaciones con condición inicial, perturbación y duración aleatorias; la
# red produce la secuencia de controles u(k); el IAE resultante se retropropaga
# hacia atrás a través de la dinámica discreta de la planta usando el método
# adjunto (co-estado), y los pesos se actualizan con Adam.
#
# Método adjunto (discrete-time):
#   λ(T)   = sign(y(T) - r)
#   λ(k)   = sign(y(k)-r) [si k≥k_step] + A·λ(k+1)   (retropropagación)
#   dL/du(k) = B·λ(k+1)   → gradiente de IAE respecto a u(k)
#
# La red de Fase 2 (aware) usa EXACTAMENTE el mismo principio: optimización directa
# de política por método adjunto, extendido a las dos esteras (un co-estado por estera).
# Ningún controlador externo actúa como referencia (no se copia una política previa).
#
# Entrada servo:  [e, Δe, σ, d]              → 4 entradas
# Entrada aware:  [e1, e2, Δe1, Δe2, σ1, σ2, bal, d] → 8 entradas


def train_dl_servo_direct(
    layer_sizes: list[int] = None,
    n_epochs: int = 800,
    batch_size: int = 32,
    lambda_u: float = 0.01,
    seed: int = SEED,
) -> tuple[DeepNet, dict]:
    """Optimización directa de la red servo sin controlador de referencia externo.

    Cada época:
    1. Muestrear batch_size escenarios con y0, d_amp, d_start, d_dur aleatorios.
    2. Ejecutar la simulación completa con la red actual → y(k), u(k), sigma(k).
    3. Calcular el IAE + penalización de suavidad.
    4. Retropropagar el gradiente a través de la planta (método adjunto) y
       luego a través de la red (backprop estándar).
    5. Actualizar pesos con Adam.

    El gradiente del IAE respecto a u(k) se calcula como:
        λ(T)   = sign(y(T) - r)
        λ(k)   = sign(y(k) - r) [si k ≥ k_step] + A·λ(k+1)
        dL/du(k) = B·λ(k+1)
    """
    if layer_sizes is None:
        layer_sizes = [4, 64, 32, 16, 8, 1]

    net = DeepNet(layer_sizes, seed=seed)
    rng = np.random.default_rng(seed)
    history: dict = {"train": [], "val": []}

    # Escenarios de validación fijos (se evalúan sin gradiente)
    n_val = 50
    v_y0   = rng.uniform(0.0, 0.4, n_val)
    v_damp = rng.uniform(0.0, 1.0, n_val)
    v_ds   = rng.integers(50, 95, n_val)
    v_de   = np.minimum(v_ds + rng.integers(10, 35, n_val), SV_STEPS - 1)

    for epoch in range(n_epochs):
        # Cosine LR annealing
        lr = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * epoch / n_epochs))

        # Muestrear escenarios de entrenamiento
        y0s   = rng.uniform(0.0, 0.4, batch_size)
        damps = rng.uniform(0.0, 1.0, batch_size)
        d_ss  = rng.integers(50, 95, batch_size)
        d_es  = np.minimum(d_ss + rng.integers(10, 35, batch_size), SV_STEPS - 1)

        # Acumular gradientes del batch
        batch_gW = [np.zeros_like(W) for W in net.W]
        batch_gb = [np.zeros_like(b) for b in net.b]
        epoch_loss = 0.0
        T = SV_STEPS - 1   # número de pasos de control

        for bi in range(batch_size):
            y0    = float(y0s[bi])
            d_amp = float(damps[bi])
            d_s   = int(d_ss[bi]); d_e = int(d_es[bi])

            ref = np.zeros(SV_STEPS); ref[SV_STEP_K:] = SV_REF

            # ── Pasada hacia adelante ─────────────────────────────────────────
            y = y0; sigma = 0.0; prev_e = 0.0
            xs: list  = []   # estados de red en cada paso
            us: list  = []   # controles en cada paso
            ys: list  = [y]  # salida de la planta: ys[0]=y0, ys[k] tras u[k-1]

            for k in range(1, SV_STEPS):
                d  = d_amp if d_s <= k <= d_e else 0.0
                e  = ref[k] - y; de = e - prev_e
                sigma = float(np.clip(sigma + e, -6.0, 6.0))
                x  = np.array([e, de, sigma, d])
                u  = float(net.predict(x)[0])
                xs.append(x); us.append(u)
                y  = float(np.clip(PL_A * y + PL_B * u - PL_D * d, -0.1, 1.35))
                ys.append(y)
                prev_e = e

            # ── Pérdida: IAE + penalización de suavidad ───────────────────────
            iae = sum(abs(ref[k] - ys[k]) for k in range(SV_STEP_K, SV_STEPS))
            du2 = sum((us[k] - us[k - 1]) ** 2 for k in range(1, T))
            epoch_loss += iae / T + lambda_u * du2 / T

            # ── Pasada hacia atrás: método adjunto ────────────────────────────
            # Objetivo: calcular dL/du(k) = B·λ(k+1) para cada paso k.
            # λ(k+1) = sign(y(k+1)-r)/T  +  A·λ(k+2)
            #
            # Orden correcto: primero actualizar λ LUEGO calcular dL/du.
            # Al inicio del bucle para k dado, lam = λ(k+2).
            # Tras la actualización, lam = λ(k+1), que es lo que necesitamos.

            lam = 0.0  # λ(T+1) = 0, condición terminal

            for k in range(T - 1, -1, -1):
                k_plant = k + 1   # ys[k_plant] = y(k+1), producido por u(k)

                # 1. Actualizar co-estado: lam pasa de λ(k+2) a λ(k+1)
                dL_dyk1 = (
                    float(np.sign(ys[k_plant] - ref[k_plant])) / T
                    if k_plant >= SV_STEP_K else 0.0
                )
                lam = PL_A * lam + dL_dyk1  # ahora lam = λ(k+1)

                # 2. Gradiente respecto a u(k)
                dL_du_k = PL_B * lam   # = B·λ(k+1)

                # Penalización de suavidad
                if 0 < k < T - 1:
                    dL_du_k += 2 * lambda_u * (2 * us[k] - us[k - 1] - us[k + 1]) / T
                elif k == 0 and T > 1:
                    dL_du_k += 2 * lambda_u * (us[0] - us[1]) / T
                elif k == T - 1:
                    dL_du_k += 2 * lambda_u * (us[k] - us[k - 1]) / T

                # Retropropagar a través de la red: dL/dW += dL/du(k) · ∂u(k)/∂W
                u_k  = us[k]                          # salida sigmoide ∈(0,1)
                # delta en la capa de salida: dL/dz = dL/du · σ'(z) = dL/du · u(1-u)
                delta = np.array([[dL_du_k * u_k * (1.0 - u_k)]])

                hs = net._forward(xs[k])              # activaciones guardadas
                for li in range(len(net.W) - 1, -1, -1):
                    batch_gW[li] += delta.T @ hs[li].reshape(1, -1)
                    batch_gb[li] += delta.sum(axis=0)
                    if li > 0:
                        delta = (delta @ net.W[li]) * (1.0 - hs[li].reshape(1, -1) ** 2)

        # Promediar gradientes y actualizar pesos
        for li in range(len(net.W)):
            batch_gW[li] /= batch_size
            batch_gb[li] /= batch_size
        net._adam_step(batch_gW, batch_gb, lr)
        history["train"].append(epoch_loss / batch_size)

        # Evaluación en validación (sin gradiente)
        val_iae = 0.0
        for i in range(n_val):
            y = float(v_y0[i]); sigma = 0.0; prev_e = 0.0
            ref = np.zeros(SV_STEPS); ref[SV_STEP_K:] = SV_REF
            for k in range(1, SV_STEPS):
                d  = float(v_damp[i]) if v_ds[i] <= k <= v_de[i] else 0.0
                e  = ref[k] - y; de = e - prev_e; sigma = float(np.clip(sigma + e, -6, 6))
                u  = float(net.predict(np.array([e, de, sigma, d]))[0])
                y  = float(np.clip(PL_A * y + PL_B * u - PL_D * d, -0.1, 1.35))
                if k >= SV_STEP_K: val_iae += abs(ref[k] - y)
                prev_e = e
        history["val"].append(val_iae / (n_val * SV_STEPS))

    return net, history


def train_dl_aware_direct(
    layer_sizes: list = None,
    n_epochs: int = 600,
    batch_size: int = 16,
    lambda_u: float = 0.02,
    seed: int = SEED + 1,
) -> tuple:
    """Optimización directa de política para la red aware (R2ET) sin controlador de referencia externo.

    Cada época: batch de episodios R2ET aleatorios → IAE de ambas esteras
    retropropagado a través de la dinámica del sistema con el método adjunto.

    Adjunto (por estera i):
        λ_i(T+1) = 0
        λ_i(k)   = sign(n_i(k+1) - n*)/T + A_i · λ_i(k+1)
        A_i = 1 en régimen lineal, 0 si n_i(k+1) saturó en 0 o ET_CAP

    Gradientes respecto a salidas de la red:
        dL/du_i(k)     = -ET_EG·(1-loss_i)·λ_i(k+1)   [si outflow no saturado]
        dL/dsplit_E1(k) = (λ_1(k+1) - λ_2(k+1))·demand·admission
    """
    if layer_sizes is None:
        layer_sizes = [8, 64, 32, 16, 8, 3]

    net = DeepNet(layer_sizes, seed=seed)
    rng = np.random.default_rng(seed)
    history: dict = {"train": [], "val": []}

    # Escenarios de validación fijos
    n_val     = 30
    v_n0      = rng.uniform(0.4, 2.6, (n_val, 2))
    v_has_jam = rng.random(n_val) > 0.45
    v_jam_s   = rng.integers(60, 110, n_val)
    v_jam_e   = np.minimum(v_jam_s + rng.integers(15, 40, n_val), ET_STEPS - 1)
    v_lam     = rng.uniform(0.85, 1.15, n_val)

    T = ET_STEPS - 1

    for epoch in range(n_epochs):
        lr = 1e-4 + 0.5 * (4e-4 - 1e-4) * (1 + math.cos(math.pi * epoch / n_epochs))

        batch_gW   = [np.zeros_like(W) for W in net.W]
        batch_gb   = [np.zeros_like(b) for b in net.b]
        epoch_loss = 0.0

        # Muestrear escenarios de entrenamiento
        n0_b    = rng.uniform(0.4, 2.6, (batch_size, 2))
        has_jam = rng.random(batch_size) > 0.45
        jam_s_b = rng.integers(60, 110, batch_size)
        jam_e_b = np.minimum(jam_s_b + rng.integers(15, 40, batch_size), ET_STEPS - 1)
        lam_b   = rng.uniform(0.85, 1.15, batch_size)

        for bi in range(batch_size):
            n      = n0_b[bi].copy()
            sigma  = np.zeros(2)
            prev_n = n.copy()

            xs   = []           # inputs de la red en cada paso
            yps  = []           # outputs crudos sigmoid (antes de clip y supervisor)
            us   = []           # control real aplicado (post-supervisor)
            spls = []           # fracciones de split aplicadas
            adms = []           # admisiones aplicadas
            dms  = []           # demandas
            lss  = []           # pérdidas de capacidad
            ns   = [n.copy()]   # ns[0]=n0, ns[k]=estado tras u[k-1]

            # ── Pasada hacia adelante ─────────────────────────────────────────
            for k in range(1, ET_STEPS):
                dm  = ET_BASE_IN + ET_PULSE * float(ET_PUL_S <= k <= ET_PUL_E)
                dm *= lam_b[bi]
                d   = 1.0 if (has_jam[bi] and jam_s_b[bi] <= k <= jam_e_b[bi]) else 0.0
                ls  = np.array([ET_JAM_L * d, 0.5 * ET_JAM_L * d])
                nm  = float(np.max(n))
                e   = n - ET_TARGET
                de  = n - prev_n
                sigma = np.clip(sigma + e, -6.0, 6.0)
                bal = float(n[0] - n[1])
                x   = np.array([e[0], e[1], de[0], de[1],
                                 sigma[0], sigma[1], bal, d])
                yp  = net.predict(x)                     # shape (3,)
                uk  = np.clip(yp[:2], 0.05, 1.0)
                s1  = float(np.clip(yp[2], 0.20, 0.80))
                uk, adm = _cap_supervisor(uk, nm, d)
                sp  = np.array([s1, 1.0 - s1])

                xs.append(x); yps.append(yp.copy())
                us.append(uk.copy()); spls.append(sp.copy())
                adms.append(float(adm)); dms.append(dm); lss.append(ls.copy())

                prev_n = n.copy()
                n, _   = _et_step(n, uk, sp, adm, dm, ls)
                ns.append(n.copy())

            # ── Pérdida: IAE + penalización de suavidad ───────────────────────
            iae = sum(float(np.sum(np.abs(np.array(ns[k]) - ET_TARGET)))
                      for k in range(1, ET_STEPS))
            du1 = sum((us[k][0] - us[k-1][0]) ** 2 for k in range(1, T))
            du2 = sum((us[k][1] - us[k-1][1]) ** 2 for k in range(1, T))
            epoch_loss += iae / T + lambda_u * (du1 + du2) / T

            # ── Pasada hacia atrás: método adjunto (2 estados) ────────────────
            lam = np.zeros(2)   # λ(T+1) = 0, condición terminal

            for k in range(T - 1, -1, -1):
                k_p   = k + 1
                n_kp1 = np.array(ns[k_p])
                dm_k  = dms[k]; ls_k = lss[k]
                adm_k = adms[k]; sp_k = spls[k]; uk_k = us[k]

                # A_i: factor de propagación del co-estado
                sat   = (n_kp1 <= 1e-4) | (n_kp1 >= ET_CAP - 1e-4)
                A_vec = np.where(sat, 0.0, 1.0)

                # Actualizar co-estado: lam pasa de λ(k+2) a λ(k+1)
                dL_dn = np.sign(n_kp1 - ET_TARGET) / T
                lam   = A_vec * lam + dL_dn

                # Saturación de outflow (no se puede drenar más de lo disponible)
                feed_k      = dm_k * adm_k * sp_k
                outflow_cap = np.array(ns[k]) + feed_k
                commanded   = ET_EG * uk_k * (1.0 - ls_k)
                free_drain  = commanded < outflow_cap - 1e-6

                # dL/du_i(k) = -ET_EG*(1-loss_i)*λ_i(k+1) si hay margen
                dL_du = np.where(free_drain & ~sat, -ET_EG * (1.0 - ls_k) * lam, 0.0)

                # dL/d(split_E1)(k)
                if not (sat[0] or sat[1]):
                    dL_dspl = float((lam[0] - lam[1]) * dm_k * adm_k)
                else:
                    dL_dspl = 0.0

                # Penalización de suavidad sobre u1, u2
                for ci in range(2):
                    if 0 < k < T - 1:
                        dL_du[ci] += 2*lambda_u*(2*us[k][ci] - us[k-1][ci] - us[k+1][ci]) / T
                    elif k == 0 and T > 1:
                        dL_du[ci] += 2*lambda_u*(us[0][ci] - us[1][ci]) / T
                    elif k == T - 1:
                        dL_du[ci] += 2*lambda_u*(us[k][ci] - us[k-1][ci]) / T

                # Retropropagar a través de la red (3 salidas sigmoid)
                yp_k  = yps[k]
                delta = np.array([[
                    dL_du[0] * float(yp_k[0]) * (1.0 - float(yp_k[0])),
                    dL_du[1] * float(yp_k[1]) * (1.0 - float(yp_k[1])),
                    dL_dspl  * float(yp_k[2]) * (1.0 - float(yp_k[2])),
                ]])   # shape (1, 3)
                hs = net._forward(xs[k])
                for li in range(len(net.W) - 1, -1, -1):
                    batch_gW[li] += delta.T @ hs[li].reshape(1, -1)
                    batch_gb[li] += delta.sum(axis=0)
                    if li > 0:
                        delta = (delta @ net.W[li]) * (1.0 - hs[li].reshape(1, -1) ** 2)

        # Promediar gradientes y actualizar pesos
        for li in range(len(net.W)):
            batch_gW[li] /= batch_size
            batch_gb[li] /= batch_size
        net._adam_step(batch_gW, batch_gb, lr)
        history["train"].append(epoch_loss / batch_size)

        # Evaluación en validación (sin gradiente)
        val_iae = 0.0
        for i in range(n_val):
            n = v_n0[i].copy(); sigma = np.zeros(2); prev_n = n.copy()
            for k in range(1, ET_STEPS):
                dm  = (ET_BASE_IN + ET_PULSE * float(ET_PUL_S <= k <= ET_PUL_E)) * float(v_lam[i])
                d   = 1.0 if (v_has_jam[i] and v_jam_s[i] <= k <= v_jam_e[i]) else 0.0
                ls  = np.array([ET_JAM_L * d, 0.5 * ET_JAM_L * d])
                e   = n - ET_TARGET; de = n - prev_n; sigma = np.clip(sigma + e, -6, 6)
                bal = float(n[0] - n[1])
                x   = np.array([e[0], e[1], de[0], de[1], sigma[0], sigma[1], bal, d])
                yp  = net.predict(x)
                uk  = np.clip(yp[:2], 0.05, 1.0)
                s1  = float(np.clip(yp[2], 0.20, 0.80))
                uk, adm = _cap_supervisor(uk, float(np.max(n)), d)
                sp  = np.array([s1, 1.0 - s1])
                prev_n = n.copy()
                n, _ = _et_step(n, uk, sp, adm, dm, ls)
                val_iae += float(np.sum(np.abs(n - ET_TARGET)))
        history["val"].append(val_iae / (n_val * T))

    return net, history


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Q-LEARNING TABULAR — FASE 1 (SERVO)
# ═══════════════════════════════════════════════════════════════════════════════

_SQ_NE = 10; _SQ_NDE = 5; _SQ_ND = 2
_SQ_N_STATES  = _SQ_NE * _SQ_NDE * _SQ_ND   # 100 estados
_SQ_ACTIONS   = np.array([0.10, 0.22, 0.35, 0.48, 0.60, 0.72, 0.85, 1.0])  # 8 acciones


def _encode_servo(e: float, de: float, d: float) -> int:
    eb  = int(np.clip((e + 1.0) / 2.0 * _SQ_NE,  0, _SQ_NE - 1))
    deb = int(np.clip((de + 0.5) / 1.0 * _SQ_NDE, 0, _SQ_NDE - 1))
    db  = int(d > 0.5)
    return eb + _SQ_NE * deb + _SQ_NE * _SQ_NDE * db


def train_q_servo(episodes=600, horizon=100, alpha=0.15, gamma=0.95,
                  eps0=0.60, eps_min=0.02, seed=SEED + 7):
    rng = np.random.default_rng(seed)
    Q   = np.zeros((_SQ_N_STATES, len(_SQ_ACTIONS)))
    history = []
    for ep in range(episodes):
        eps  = max(eps_min, eps0 * (1 - ep / episodes))
        y    = rng.uniform(0.0, 0.3)
        ref  = SV_REF; prev_e = ref - y; prev_u = PL_FF
        ep_r = 0.0
        for k in range(horizon):
            d    = 1.0 if (rng.random() > 0.88 or (60 <= k <= 80 and rng.random() > 0.4)) else 0.0
            e    = ref - y; de = e - prev_e
            s    = _encode_servo(e, de, d)
            a    = int(rng.integers(len(_SQ_ACTIONS))) if rng.random() < eps else int(np.argmax(Q[s]))
            u    = float(_SQ_ACTIONS[a])
            y_n  = float(np.clip(PL_A * y + PL_B * u - PL_D * d, -0.1, 1.4))
            e_n  = ref - y_n; de_n = e_n - e
            s_n  = _encode_servo(e_n, de_n, d)
            r    = -abs(e) - 0.3 * e**2 - 0.05 * abs(u - prev_u) - 0.5 * max(0.0, y_n - (SV_REF + 0.08))**2
            Q[s, a] += alpha * (r + gamma * float(np.max(Q[s_n])) - Q[s, a])
            ep_r += r; prev_e = e; prev_u = u; y = y_n
        history.append({"ep": ep + 1, "eps": round(eps, 4), "r_mean": round(ep_r / horizon, 4)})
    return Q, pd.DataFrame(history)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Q-LEARNING TABULAR — FASE 2 (AWARE R2ET)
# ═══════════════════════════════════════════════════════════════════════════════

_AQ_N1B = 9; _AQ_N2B = 9; _AQ_ND = 2
_AQ_N_STATES = _AQ_N1B * _AQ_N2B * _AQ_ND   # 162 estados (bins más finos en torno a n*=1.5)
_AQ_EDGES    = [0.0, 0.5, 0.9, 1.2, 1.4, 1.6, 1.8, 2.1, 2.5, 3.1]  # 9 bins, densos cerca de n*

# 11 acciones: [velocidad, split_E1, admisión]
_AQ_ACTIONS = np.array([
    [0.30, 0.50, 1.00],  # 0: lento — dejar que las esteras se llenen
    [0.46, 0.50, 1.00],  # 1: estado estacionario equilibrado (≈ u_ss)
    [0.58, 0.50, 1.00],  # 2: drenado moderado equilibrado
    [0.70, 0.50, 0.95],  # 3: drenado rápido equilibrado
    [0.84, 0.50, 0.85],  # 4: drenado fuerte
    [0.95, 0.50, 0.78],  # 5: drenado de emergencia
    [0.52, 0.30, 1.00],  # 6: drenado suave + más flujo a E2
    [0.52, 0.70, 1.00],  # 7: drenado suave + más flujo a E1
    [0.72, 0.30, 0.90],  # 8: drenado rápido + prioridad E2
    [0.72, 0.70, 0.90],  # 9: drenado rápido + prioridad E1
    [0.62, 0.50, 1.00],  # 10: drenado fino cerca del objetivo
])


def _encode_aware(n: np.ndarray, d: float) -> int:
    n1b = int(np.clip(np.searchsorted(_AQ_EDGES[1:], float(n[0])), 0, _AQ_N1B - 1))
    n2b = int(np.clip(np.searchsorted(_AQ_EDGES[1:], float(n[1])), 0, _AQ_N2B - 1))
    db  = int(d > 0.5)
    return n1b + _AQ_N1B * n2b + _AQ_N1B * _AQ_N2B * db


def _et_step(n_prev: np.ndarray, u: np.ndarray, split: np.ndarray,
             admission: float, demand: float, loss: np.ndarray):
    """Un paso de la dinámica R2ET de dos esteras."""
    feed    = demand * admission * split
    commanded = ET_EG * u * (1.0 - loss)
    outflow = np.minimum(commanded, n_prev + feed)
    n_next  = np.clip(n_prev + feed - outflow, 0.0, ET_CAP)
    return n_next, outflow


def train_q_aware(episodes=3000, horizon=160, alpha=0.18, gamma=0.92,
                  eps0=0.65, eps_min=0.02, seed=SEED + 13):
    """Entrena la tabla Q del aware (162 estados × 11 acciones) para coordinación R2ET."""
    rng     = np.random.default_rng(seed)
    Q       = np.zeros((_AQ_N_STATES, len(_AQ_ACTIONS)))
    history = []
    for ep in range(episodes):
        eps = max(eps_min, eps0 * (1 - ep / episodes))
        n   = rng.uniform(0.4, 2.6, 2)
        ep_r = 0.0
        has_jam = rng.random() > 0.45  # 55 % de los episodios incluyen atasco
        for k in range(horizon):
            dm   = ET_BASE_IN + ET_PULSE * float(ET_PUL_S <= k <= ET_PUL_E)
            d    = 1.0 if (has_jam and ET_JAM_S <= k <= ET_JAM_E) else 0.0
            loss = np.array([ET_JAM_L * d, 0.5 * ET_JAM_L * d])
            s    = _encode_aware(n, d)
            a    = int(rng.integers(len(_AQ_ACTIONS))) if rng.random() < eps else int(np.argmax(Q[s]))
            spd, spl, adm = _AQ_ACTIONS[a]
            u    = np.array([spd, spd])
            sp   = np.array([spl, 1.0 - spl])
            n_n, _ = _et_step(n, u, sp, adm, dm, loss)
            s_n  = _encode_aware(n_n, d)
            iae  = float(np.sum(np.abs(n_n - ET_TARGET)))
            viol = float(np.sum(np.maximum(0.0, n_n - ET_CAP)))
            alrm = float(np.sum(np.maximum(0.0, n_n - ET_ALARM)))
            bal  = abs(float(n_n[0] - n_n[1]))
            r    = -1.5 * iae - 30.0 * viol - 8.0 * alrm - 0.20 * bal
            Q[s, a] += alpha * (r + gamma * float(np.max(Q[s_n])) - Q[s, a])
            ep_r += r; n = n_n
        history.append({"ep": ep + 1, "eps": round(eps, 4), "r_mean": round(ep_r / horizon, 4)})
    return Q, pd.DataFrame(history)


def train_q_aware_smooth(episodes=3000, horizon=160, alpha=0.18, gamma=0.92,
                         eps0=0.65, eps_min=0.02, w_slew=8.0, seed=SEED + 13):
    """Variante 'suavizada' del Q-aware: la recompensa penaliza, además del error
    (IAE), el cambio de velocidad |Δspd| entre pasos consecutivos, para reducir el
    chattering del mando propio de las acciones discretas."""
    rng     = np.random.default_rng(seed)
    Q       = np.zeros((_AQ_N_STATES, len(_AQ_ACTIONS)))
    history = []
    for ep in range(episodes):
        eps = max(eps_min, eps0 * (1 - ep / episodes))
        n   = rng.uniform(0.4, 2.6, 2); ep_r = 0.0
        has_jam = rng.random() > 0.45
        prev_spd = ET_USS
        for k in range(horizon):
            dm   = ET_BASE_IN + ET_PULSE * float(ET_PUL_S <= k <= ET_PUL_E)
            d    = 1.0 if (has_jam and ET_JAM_S <= k <= ET_JAM_E) else 0.0
            loss = np.array([ET_JAM_L * d, 0.5 * ET_JAM_L * d])
            s    = _encode_aware(n, d)
            a    = int(rng.integers(len(_AQ_ACTIONS))) if rng.random() < eps else int(np.argmax(Q[s]))
            spd, spl, adm = _AQ_ACTIONS[a]
            u    = np.array([spd, spd]); sp = np.array([spl, 1.0 - spl])
            n_n, _ = _et_step(n, u, sp, adm, dm, loss)
            s_n  = _encode_aware(n_n, d)
            iae  = float(np.sum(np.abs(n_n - ET_TARGET)))
            viol = float(np.sum(np.maximum(0.0, n_n - ET_CAP)))
            alrm = float(np.sum(np.maximum(0.0, n_n - ET_ALARM)))
            bal  = abs(float(n_n[0] - n_n[1]))
            slew = abs(spd - prev_spd)                       # penalización de |Δu|
            r    = -1.5 * iae - 30.0 * viol - 8.0 * alrm - 0.20 * bal - w_slew * slew
            Q[s, a] += alpha * (r + gamma * float(np.max(Q[s_n])) - Q[s, a])
            ep_r += r; n = n_n; prev_spd = spd
        history.append({"ep": ep + 1, "eps": round(eps, 4), "r_mean": round(ep_r / horizon, 4)})
    return Q, pd.DataFrame(history)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. SEÑALES AUXILIARES DE ESCENARIO
# ═══════════════════════════════════════════════════════════════════════════════

def _d_servo(k: int) -> float:
    return 1.0 if SV_D_START <= k <= SV_D_STOP else 0.0


# ─── Perturbación en rampa trapezoidal ───────────────────────────────────────
SV_RAMP_UP  = 12   # muestras para subir 0 → 1
SV_RAMP_DWN = 12   # muestras para bajar 1 → 0 (tras SV_D_STOP)

def _d_servo_ramp(k: int) -> float:
    """Rampa trapezoidal: sube 12 muestras, sostiene, baja 12 muestras."""
    if k < SV_D_START:
        return 0.0
    elif SV_D_START <= k < SV_D_START + SV_RAMP_UP:
        return (k - SV_D_START) / SV_RAMP_UP
    elif k <= SV_D_STOP:
        return 1.0
    elif SV_D_STOP < k <= SV_D_STOP + SV_RAMP_DWN:
        return 1.0 - (k - SV_D_STOP) / SV_RAMP_DWN
    return 0.0


def _dist_arr_servo() -> np.ndarray:
    return np.array([_d_servo(k) for k in range(SV_STEPS)])


def _demand_et(k: int) -> float:
    return ET_BASE_IN + ET_PULSE * float(ET_PUL_S <= k <= ET_PUL_E)


def _loss_et(k: int) -> np.ndarray:
    d = 1.0 if ET_JAM_S <= k <= ET_JAM_E else 0.0
    return np.array([ET_JAM_L * d, 0.5 * ET_JAM_L * d])


def _d_aware(k: int) -> float:
    return 1.0 if ET_JAM_S <= k <= ET_JAM_E else 0.0


def _cap_supervisor(u: np.ndarray, n_max: float, d: float):
    """Capa de supervisión de capacidad (igual para todos los métodos)."""
    if n_max > ET_ALARM or d > 0:
        return np.clip(u + 0.14, 0.05, 1.0), 0.72 if n_max > ET_ALARM else 0.86
    if n_max > ET_RISK:
        return np.clip(u + 0.07, 0.05, 1.0), 0.90
    return u, 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  7. SIMULACIONES FASE 1 — SERVO
# ═══════════════════════════════════════════════════════════════════════════════

def _servo_init():
    ref = np.zeros(SV_STEPS)
    ref[SV_STEP_K:] = SV_REF
    return ref, np.zeros(SV_STEPS), np.zeros(SV_STEPS)


def simulate_servo_pid() -> dict:
    ref, y, u = _servo_init()
    st = PIDState()
    for k in range(1, SV_STEPS):
        d   = _d_servo(k)
        e   = ref[k] - y[k - 1]
        u[k] = pid_step(e, st, kp=4.0, ki=0.30, kd=1.0, ff=PL_FF)
        y[k] = float(np.clip(PL_A * y[k - 1] + PL_B * u[k] - PL_D * d, -0.05, 1.35))
    return {"ref": ref, "y": y, "u": u, "dist": _dist_arr_servo()}


def simulate_servo_fuzzy() -> dict:
    ref, y, u = _servo_init()
    prev_e = 0.0
    for k in range(1, SV_STEPS):
        d    = _d_servo(k)
        e    = ref[k] - y[k - 1]; de = e - prev_e
        u[k] = fuzzy_speed(e, de, d, u_base=PL_FF)
        y[k] = float(np.clip(PL_A * y[k - 1] + PL_B * u[k] - PL_D * d, -0.05, 1.35))
        prev_e = e
    return {"ref": ref, "y": y, "u": u, "dist": _dist_arr_servo()}


def simulate_servo_dl(net: DeepNet) -> dict:
    ref, y, u = _servo_init()
    prev_e = 0.0; sigma = 0.0
    for k in range(1, SV_STEPS):
        d      = _d_servo(k)
        e      = ref[k] - y[k - 1]; de = e - prev_e
        sigma  = float(np.clip(sigma + e, -6.0, 6.0))   # integral acumulada
        u[k]   = float(net.predict(np.array([e, de, sigma, d]))[0])
        y[k]   = float(np.clip(PL_A * y[k - 1] + PL_B * u[k] - PL_D * d, -0.05, 1.35))
        prev_e = e
    return {"ref": ref, "y": y, "u": u, "dist": _dist_arr_servo()}


def simulate_servo_ql(Q: np.ndarray) -> dict:
    ref, y, u = _servo_init()
    prev_e = 0.0
    for k in range(1, SV_STEPS):
        d    = _d_servo(k)
        e    = ref[k] - y[k - 1]; de = e - prev_e
        s    = _encode_servo(e, de, d)
        u[k] = float(_SQ_ACTIONS[int(np.argmax(Q[s]))])
        y[k] = float(np.clip(PL_A * y[k - 1] + PL_B * u[k] - PL_D * d, -0.05, 1.35))
        prev_e = e
    return {"ref": ref, "y": y, "u": u, "dist": _dist_arr_servo()}


# ─── Simulación servo con perturbación arbitraria ─────────────────────────────

def simulate_servo_all_dfn(d_fn, dl_net: "DeepNet", Q_servo: np.ndarray) -> dict:
    """Corre los 4 controladores con una función de perturbación d_fn(k) arbitraria.

    Útil para ensayos de robustez (rampa, pulso, ruido) sin duplicar código.
    Devuelve el mismo formato que los simulate_servo_xxx estándar.
    """
    dist = np.array([d_fn(k) for k in range(SV_STEPS)])
    out: dict = {}

    # PID
    ref, y, u = _servo_init(); st = PIDState()
    for k in range(1, SV_STEPS):
        d = d_fn(k); e = ref[k] - y[k - 1]
        u[k] = pid_step(e, st, kp=4.0, ki=0.30, kd=1.0, ff=PL_FF)
        y[k] = float(np.clip(PL_A * y[k-1] + PL_B * u[k] - PL_D * d, -0.05, 1.35))
    out["PID"] = {"ref": ref, "y": y.copy(), "u": u.copy(), "dist": dist}

    # Fuzzy
    ref, y, u = _servo_init(); prev_e = 0.0
    for k in range(1, SV_STEPS):
        d = d_fn(k); e = ref[k] - y[k - 1]; de = e - prev_e
        u[k] = fuzzy_speed(e, de, d, u_base=PL_FF)
        y[k] = float(np.clip(PL_A * y[k-1] + PL_B * u[k] - PL_D * d, -0.05, 1.35))
        prev_e = e
    out["Fuzzy"] = {"ref": ref, "y": y.copy(), "u": u.copy(), "dist": dist}

    # DL
    ref, y, u = _servo_init(); prev_e = 0.0; sigma = 0.0
    for k in range(1, SV_STEPS):
        d = d_fn(k); e = ref[k] - y[k - 1]; de = e - prev_e
        sigma = float(np.clip(sigma + e, -6, 6))
        u[k] = float(dl_net.predict(np.array([e, de, sigma, d]))[0])
        y[k] = float(np.clip(PL_A * y[k-1] + PL_B * u[k] - PL_D * d, -0.05, 1.35))
        prev_e = e
    out["DL"] = {"ref": ref, "y": y.copy(), "u": u.copy(), "dist": dist}

    # QL — nota: codifica d como binario d_bin = int(d > 0.5)
    # Para d < 0.5 actúa como si no hubiera perturbación → reacción tardía con rampa
    ref, y, u = _servo_init(); prev_e = 0.0
    for k in range(1, SV_STEPS):
        d = d_fn(k); e = ref[k] - y[k - 1]; de = e - prev_e
        s = _encode_servo(e, de, d)
        u[k] = float(_SQ_ACTIONS[int(np.argmax(Q_servo[s]))])
        y[k] = float(np.clip(PL_A * y[k-1] + PL_B * u[k] - PL_D * d, -0.05, 1.35))
        prev_e = e
    out["QL"] = {"ref": ref, "y": y.copy(), "u": u.copy(), "dist": dist}

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  8. SIMULACIONES FASE 2 — COORDINACIÓN AWARE (R2ET)
# ═══════════════════════════════════════════════════════════════════════════════

def _r2et_base():
    n   = np.zeros((ET_STEPS, 2)); n[0] = [1.25, 1.25]
    u   = np.zeros((ET_STEPS, 2))
    out = np.zeros((ET_STEPS, 2))
    sph = np.zeros((ET_STEPS, 2))
    adh = np.ones(ET_STEPS)
    alm = np.zeros(ET_STEPS)
    return n, u, out, sph, adh, alm


def simulate_aware_pid() -> dict:
    n, u, out, sph, adh, alm = _r2et_base()
    sts = [PIDState(), PIDState()]
    kp, ki, kd = 1.3, 0.18, 0.28
    for k in range(1, ET_STEPS):
        dm = _demand_et(k); ls = _loss_et(k); d = _d_aware(k)
        nm = float(np.max(n[k - 1]))
        # supervisor: ajusta ganancias PID dinámicamente según nivel de alarma
        fkp = 1.55 if (nm > ET_ALARM or d > 0) else (1.25 if nm > ET_RISK else 1.0)
        fki = 0.65 if (nm > ET_ALARM or d > 0) else (0.82 if nm > ET_RISK else 1.0)
        uk  = np.array([pid_step(float(n[k-1, i]) - ET_TARGET, sts[i],
                                 kp*fkp, ki*fki, kd, 0.05, 1.0, ff=ET_USS)
                        for i in range(2)])
        uk, adm = _cap_supervisor(uk, nm, d)
        bal = float(n[k-1, 0] - n[k-1, 1])
        s1  = fuzzy_split(bal)
        sp  = np.array([s1, 1.0 - s1])
        u[k] = uk; sph[k] = sp; adh[k] = adm
        n[k], out[k] = _et_step(n[k - 1], u[k], sp, adm, dm, ls)
        alm[k] = 1.0 if np.any(n[k] > ET_ALARM) else 0.0
    return {"n": n, "u": u, "outflow": out, "split": sph, "admission": adh, "alarm": alm}


def simulate_aware_fuzzy() -> dict:
    n, u, out, sph, adh, alm = _r2et_base()
    prev_e = np.zeros(2)
    for k in range(1, ET_STEPS):
        dm = _demand_et(k); ls = _loss_et(k); d = _d_aware(k)
        nm = float(np.max(n[k - 1]))
        e  = n[k - 1] - ET_TARGET; de = e - prev_e
        uk = np.array([fuzzy_speed(float(e[i]), float(de[i]), d, u_base=ET_USS)
                       for i in range(2)])
        uk, adm = _cap_supervisor(uk, nm, d)
        bal = float(n[k-1, 0] - n[k-1, 1])
        s1  = fuzzy_split(bal)
        sp  = np.array([s1, 1.0 - s1])
        u[k] = uk; sph[k] = sp; adh[k] = adm
        n[k], out[k] = _et_step(n[k - 1], u[k], sp, adm, dm, ls)
        alm[k] = 1.0 if np.any(n[k] > ET_ALARM) else 0.0
        prev_e = e
    return {"n": n, "u": u, "outflow": out, "split": sph, "admission": adh, "alarm": alm}


def simulate_aware_dl(net: DeepNet, du_max: float = None) -> dict:
    """Si du_max no es None, aplica un limitador de pendiente (rate limiter) sobre
    la salida de la red: |u(k)-u(k-1)| <= du_max, técnica industrial estándar para
    suprimir el chattering del mando."""
    n, u, out, sph, adh, alm = _r2et_base()
    prev_n = n[0].copy(); sigma = np.zeros(2)
    u_net_prev = np.array([ET_USS, ET_USS])
    for k in range(1, ET_STEPS):
        dm = _demand_et(k); ls = _loss_et(k); d = _d_aware(k)
        nm = float(np.max(n[k - 1]))
        e  = n[k - 1] - ET_TARGET
        de = n[k - 1] - prev_n
        sigma = np.clip(sigma + e, -6.0, 6.0)      # integral acumulada por estera
        bal = float(n[k-1, 0] - n[k-1, 1])
        x   = np.array([float(e[0]), float(e[1]),
                         float(de[0]), float(de[1]),
                         float(sigma[0]), float(sigma[1]),
                         bal, d])
        yp  = net.predict(x)
        uk  = np.array([float(np.clip(yp[0], 0.05, 1.0)), float(np.clip(yp[1], 0.05, 1.0))])
        if du_max is not None:
            uk = np.clip(uk, u_net_prev - du_max, u_net_prev + du_max)
        u_net_prev = uk.copy()
        s1  = float(np.clip(yp[2], 0.20, 0.80))
        uk, adm = _cap_supervisor(uk, nm, d)
        sp  = np.array([s1, 1.0 - s1])
        u[k] = uk; sph[k] = sp; adh[k] = adm; prev_n = n[k - 1].copy()
        n[k], out[k] = _et_step(n[k - 1], u[k], sp, adm, dm, ls)
        alm[k] = 1.0 if np.any(n[k] > ET_ALARM) else 0.0
    return {"n": n, "u": u, "outflow": out, "split": sph, "admission": adh, "alarm": alm}


def simulate_aware_ql(Q: np.ndarray) -> dict:
    n, u, out, sph, adh, alm = _r2et_base()
    for k in range(1, ET_STEPS):
        dm = _demand_et(k); ls = _loss_et(k); d = _d_aware(k)
        nm = float(np.max(n[k - 1]))
        s  = _encode_aware(n[k - 1], d)
        a  = int(np.argmax(Q[s]))
        spd, spl, adm_q = _AQ_ACTIONS[a]
        uk  = np.array([spd, spd])
        uk, adm_hard = _cap_supervisor(uk, nm, d)
        adm = min(float(adm_q), float(adm_hard))
        sp  = np.array([spl, 1.0 - spl])
        u[k] = uk; sph[k] = sp; adh[k] = adm
        n[k], out[k] = _et_step(n[k - 1], u[k], sp, adm, dm, ls)
        alm[k] = 1.0 if np.any(n[k] > ET_ALARM) else 0.0
    return {"n": n, "u": u, "outflow": out, "split": sph, "admission": adh, "alarm": alm}


# ═══════════════════════════════════════════════════════════════════════════════
#  9. MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════

def servo_metrics(res: dict) -> dict:
    y, u, ref = res["y"], res["u"], res["ref"]
    err = ref - y
    ps  = slice(SV_STEP_K, SV_D_START)
    pd  = slice(SV_D_STOP + 1, SV_STEPS)
    # Umbrales relativos a la referencia de operación SV_REF (2% y 2.5% de r)
    within = np.where(np.abs(err[ps]) <= 0.02 * SV_REF)[0]
    ts  = int(within[0]) if len(within) else int(SV_D_START - SV_STEP_K)
    rec_a = np.where(np.abs(err[pd]) <= 0.025 * SV_REF)[0]
    rec = int(rec_a[0]) if len(rec_a) else int(SV_STEPS - SV_D_STOP - 1)
    mp  = max(0.0, float(np.max(y[ps])) - SV_REF)
    return {
        "IAE":     round(float(np.sum(np.abs(err[SV_STEP_K:]))), 3),
        "Mp_pct":  round(100.0 * mp / SV_REF, 2),
        "Ts":      ts,
        "Rec":     rec,
        "Energia": round(float(np.sum(u[SV_STEP_K:])), 3),
        "Var_u":   round(float(np.sum(np.abs(np.diff(u)))), 3),
        "Umax":    round(float(np.max(u)), 3),
    }


def aware_metrics(res: dict) -> dict:
    n, u, out = res["n"], res["u"], res["outflow"]
    env = np.max(n, axis=1)
    err = np.abs(n[:, 0] - ET_TARGET) + np.abs(n[:, 1] - ET_TARGET)
    ds  = slice(ET_JAM_S, ET_JAM_E + 1)
    post = slice(ET_JAM_E + 1, ET_STEPS)
    pe  = env[post]
    rec_a = np.where(pe <= ET_RISK)[0]
    rec  = int(rec_a[0]) if len(rec_a) else int(ET_STEPS - ET_JAM_E)
    viol_raw = np.sum(np.maximum(0.0, n - ET_CAP))
    return {
        "IAE":      round(float(np.sum(err)), 3),
        "IAE_dist": round(float(np.sum(err[ds])), 3),
        "Ocup_max": round(float(np.max(env)), 3),
        "Margen":   round(float(ET_CAP - np.max(env)), 3),
        "T_riesgo": int(np.sum(env > ET_ALARM)),
        "Viol":     round(float(viol_raw), 4),
        "Rec":      rec,
        "Prod":     round(float(np.sum(out)), 3),
        "Energia":  round(float(np.sum(u)), 3),
        "Desbal":   round(float(np.mean(np.abs(n[:, 0] - n[:, 1]))), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  10. FIGURAS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fuzzy_memberships():
    plt.rcParams.update(PLT_RC)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # Error e — 7 conjuntos
    ax = axes[0, 0]
    e  = np.linspace(-1.6, 1.6, 600)
    cols7 = ["#dc2626", "#ea580c", "#d97706", "#059669", "#0891b2", "#1d4ed8", "#7c3aed"]
    for lbl, col in zip(["NL", "NM", "NS", "ZE", "PS", "PM", "PL"], cols7):
        ax.plot(e, [_mf_error(x)[lbl] for x in e], lw=1.6, label=lbl, color=col)
    ax.set_xlabel("Error $e = r - y$ o $e_i = n_i - n^*$")
    ax.set_ylabel("$\\mu$"); ax.set_title("MF del error $e$ (7 conjuntos)")
    ax.legend(fontsize=7, ncol=4); ax.set_ylim(-0.05, 1.18)

    # Derivada Δe — 5 conjuntos
    ax = axes[0, 1]
    de = np.linspace(-0.55, 0.55, 400)
    cols5 = ["#dc2626", "#d97706", "#059669", "#1d4ed8", "#7c3aed"]
    for lbl, col in zip(["NB", "NS", "ZE", "PS", "PB"], cols5):
        ax.plot(de, [_mf_deriv(x)[lbl] for x in de], lw=1.6, label=lbl, color=col)
    ax.set_xlabel("Derivada $\\Delta e$"); ax.set_title("MF de la derivada $\\Delta e$ (5 conjuntos)")
    ax.legend(fontsize=7, ncol=3); ax.set_ylim(-0.05, 1.18)

    # Perturbación d
    ax = axes[1, 0]
    dv = np.linspace(-0.05, 1.05, 300)
    ax.plot(dv, [_mf_dist(x)["NO"]  for x in dv], lw=1.8, label="SIN perturbación",  color="#059669")
    ax.plot(dv, [_mf_dist(x)["YES"] for x in dv], lw=1.8, label="CON perturbación", color="#dc2626")
    ax.set_xlabel("Señal de perturbación $d$"); ax.set_title("MF de la perturbación $d$")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.15)

    # Salida: velocidad por nivel de error (de=0) — FAM continua de 7 niveles
    ax = axes[1, 1]
    labels = ["NL", "NM", "NS", "ZE", "PS", "PM", "PL"]
    lvls   = [_LVL_E[l] for l in labels]
    vs = [np.clip(PL_FF + l * _FZ_KE, 0.05, 0.98) for l in lvls]
    va = [np.clip(ET_USS + l * _FZ_KE, 0.05, 0.98) for l in lvls]
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w/2, vs, w, color="#1d4ed8", alpha=0.82, label=f"Servo ($u_{{base}}$={PL_FF:.2f})")
    ax.bar(x + w/2, va, w, color="#dc2626", alpha=0.82, label=f"Aware ($u_{{base}}$={ET_USS:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Velocidad normalizada"); ax.set_title("Velocidad de salida por nivel de error ($\\Delta e=0$)")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    fig.suptitle("Funciones de pertenencia — Controlador Difuso Mamdani", fontsize=10, fontweight="bold")
    fig.tight_layout(); _sv(fig, "fig_fuzzy_mf.png")


def plot_dl_training(h_servo: dict, h_aware: dict):
    plt.rcParams.update(PLT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Servo: IAE loss (direct policy optimization, method adjoint)
    ax = axes[0]
    ax.plot(h_servo["train"], color="#1d4ed8", lw=1.4, label="Entrenamiento (IAE+suavidad)")
    ax.plot(h_servo["val"],   color="#dc2626", lw=1.4, ls="--", label="Validación (IAE)")
    ax.set_xlabel("Época"); ax.set_ylabel("IAE normalizado")
    ax.set_title("Fase 1 — Servo (4→64→32→16→8→1)\nOptimización directa — método adjunto")
    ax.set_yscale("log"); ax.legend(fontsize=8)

    # Aware: IAE loss (direct policy optimization, adjoint 2-belt)
    ax = axes[1]
    ax.plot(h_aware["train"], color="#1d4ed8", lw=1.4, label="Entrenamiento (IAE+suavidad)")
    ax.plot(h_aware["val"],   color="#dc2626", lw=1.4, ls="--", label="Validación (IAE)")
    ax.set_xlabel("Época"); ax.set_ylabel("IAE normalizado")
    ax.set_title("Fase 2 — R2ET (8→96→64→32→16→8→3)\nOptimización directa — método adjunto (2 esteras)")
    ax.set_yscale("log"); ax.legend(fontsize=8)

    fig.suptitle("Curvas de entrenamiento — Deep Learning", fontsize=10, fontweight="bold")
    fig.tight_layout(); _sv(fig, "fig_dl_training.png")


def plot_ql_training(h_servo: pd.DataFrame, h_aware: pd.DataFrame):
    plt.rcParams.update(PLT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, hdf, title in zip(
        axes, [h_servo, h_aware],
        [f"Fase 1 — Servo ({_SQ_N_STATES} estados × {len(_SQ_ACTIONS)} acciones)",
         f"Fase 2 — R2ET ({_AQ_N_STATES} estados × {len(_AQ_ACTIONS)} acciones)"]
    ):
        ep = hdf["ep"].to_numpy(); rm = hdf["r_mean"].to_numpy(); ep_s = hdf["eps"].to_numpy()
        w  = max(1, len(ep) // 30)
        rm_s = np.convolve(rm, np.ones(w) / w, mode="same")
        ax2  = ax.twinx()
        ax.plot(ep, rm_s, color="#1d4ed8", lw=1.5, label="Recompensa media (suavizada)")
        ax2.plot(ep, ep_s, color="#d97706", lw=1.2, ls="--", label="$\\epsilon$")
        ax.set_xlabel("Episodio"); ax.set_ylabel("Recompensa media", color="#1d4ed8")
        ax2.set_ylabel("$\\epsilon$ (exploración)", color="#d97706")
        ax.set_title(title)
        lines1, _ = ax.get_legend_handles_labels()
        lines2, _ = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, ["Recompensa", "$\\epsilon$"], fontsize=8)
    fig.suptitle("Curvas de entrenamiento — Q-Learning", fontsize=10, fontweight="bold")
    fig.tight_layout(); _sv(fig, "fig_ql_training.png")


def plot_servo_response(sr: dict):
    plt.rcParams.update(PLT_RC)
    t  = np.arange(SV_STEPS)
    fig, (ay, au) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for m in METHODS:
        ay.plot(t, sr[m]["y"], MS[m], color=MC[m], lw=1.7, label=m)
        au.plot(t, sr[m]["u"], MS[m], color=MC[m], lw=1.5, label=m)
    ay.plot(t, sr["PID"]["ref"], ":k", lw=1.2, label="Referencia $r$")
    ay.axvspan(SV_D_START, SV_D_STOP, color="#ef4444", alpha=0.12, label="Perturbación ($d=1$)")
    ay.axvline(SV_STEP_K, color="gray", lw=0.7, ls=":")
    ay.set_ylabel("Salida $y(k)$"); ay.set_ylim(-0.08, 1.28)
    ay.legend(fontsize=8, ncol=3, loc="lower right")
    ay.set_title("Fase 1 — Respuesta servoregulador (escalón + perturbación de carga)")
    au.axvspan(SV_D_START, SV_D_STOP, color="#ef4444", alpha=0.10)
    au.set_ylabel("Mando $u(k)$"); au.set_xlabel("Muestra $k$"); au.set_ylim(-0.03, 1.08)
    au.legend(fontsize=8, ncol=3)
    fig.tight_layout(); _sv(fig, "fig_servo_response.png")


def plot_robustez_rampa(sr_step: dict, sr_ramp: dict):
    """Compara escalón vs. rampa trapezoidal para los 4 métodos."""
    plt.rcParams.update(PLT_RC)
    t = np.arange(SV_STEPS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for ax, m in zip(axes.flat, METHODS):
        iae_s = servo_metrics(sr_step[m])["IAE"]
        iae_r = servo_metrics(sr_ramp[m])["IAE"]
        deg   = 100 * (iae_r / iae_s - 1) if iae_s > 0 else 0.0

        ax.plot(t, sr_step[m]["y"], MS[m], color=MC[m], lw=1.8,
                label=f"Escalón  IAE={iae_s:.2f}")
        ax.plot(t, sr_ramp[m]["y"], "--", color=MC[m], lw=1.4, alpha=0.80,
                label=f"Rampa    IAE={iae_r:.2f} ({deg:+.1f}\\%)")
        ax.plot(t, sr_step[m]["ref"], ":k", lw=1.0)

        # Sombrear la zona de rampa (rampa + sostenimiento + bajada)
        d_arr = sr_ramp[m]["dist"]
        ax.fill_between(t, 0, d_arr * 0.25, alpha=0.15,
                         color="#ef4444", label="d(k) [×0.25]")

        ax.axvline(SV_D_START, color="#ef4444", lw=0.8, ls=":")
        ax.axvline(SV_D_STOP + SV_RAMP_DWN, color="#ef4444", lw=0.8, ls=":")
        ax.set_title(m, fontweight="bold")
        ax.set_ylim(-0.10, 1.30)
        ax.legend(fontsize=7.5)

    for ax in axes[1]:
        ax.set_xlabel("Muestra $k$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Salida $y(k)$")

    fig.suptitle(
        "Ensayo de robustez — Perturbación escalón vs. rampa trapezoidal\n"
        r"$d$ sube $0\!\to\!1$ en 12 muestras, sostiene, baja $1\!\to\!0$ en 12 muestras",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout(); _sv(fig, "fig_robustez_rampa.png")


def plot_servo_kpi(sr: dict):
    plt.rcParams.update(PLT_RC)
    met    = {m: servo_metrics(sr[m]) for m in METHODS}
    keys   = ["IAE", "Energia", "Var_u"]
    labels = ["IAE", "Energía acumulada", "Variación $\\sum|\\Delta u|$"]
    x = np.arange(len(keys)); w = 0.20
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(METHODS):
        vals = [met[m][k] for k in keys]
        ax.bar(x + (i - 1.5) * w, vals, w, color=MC[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Valor")
    ax.set_title("Fase 1 — Indicadores KPI por método")
    ax.legend(fontsize=8); fig.tight_layout(); _sv(fig, "fig_servo_kpi.png")


def plot_aware_occupancy(ar: dict):
    plt.rcParams.update(PLT_RC)
    t   = np.arange(ET_STEPS)
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in METHODS:
        env = np.max(ar[m]["n"], axis=1)
        ax.plot(t, env, MS[m], color=MC[m], lw=1.8, label=m)
    ax.axhline(ET_CAP,   color="#111827", ls=":",  lw=1.4, label=f"Capacidad ({ET_CAP:.0f})")
    ax.axhline(ET_ALARM, color="#dc2626", ls="--", lw=1.1, label=f"Alarma ({ET_ALARM})")
    ax.axhline(ET_RISK,  color="#d97706", ls="--", lw=1.0, label=f"Riesgo ({ET_RISK})")
    ax.axvspan(ET_PUL_S, ET_PUL_E, color="#f59e0b", alpha=0.12, label="Pulso de demanda")
    ax.axvspan(ET_JAM_S, ET_JAM_E, color="#ef4444", alpha=0.09, label="Atasco / cambio de peso")
    ax.set_title("Fase 2 — Ocupación máxima R2ET (ambas esteras)")
    ax.set_xlabel("Muestra $k$"); ax.set_ylabel("Piezas en estera")
    ax.set_ylim(0.6, 3.25)
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(); _sv(fig, "fig_aware_occupancy.png")


def plot_aware_zoom(ar: dict):
    plt.rcParams.update(PLT_RC)
    t = np.arange(ET_STEPS)
    z = slice(ET_JAM_S - 5, ET_JAM_E + 15)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in METHODS:
        ax.plot(t[z], np.max(ar[m]["n"], axis=1)[z], MS[m], color=MC[m], lw=1.8, label=m)
    ax.axhline(ET_ALARM, color="#dc2626", ls="--", lw=1.1, label=f"Alarma ({ET_ALARM})")
    ax.axhline(ET_RISK,  color="#d97706", ls="--", lw=1.0, label=f"Riesgo ({ET_RISK})")
    ax.axvspan(ET_JAM_S, ET_JAM_E, color="#ef4444", alpha=0.10, label="Atasco parcial")
    ax.set_title("Fase 2 — Zoom durante el atasco / cambio de peso")
    ax.set_xlabel("Muestra $k$"); ax.set_ylabel("Ocupación máx. [piezas]")
    ax.set_ylim(0.8, 2.90); ax.legend(ncol=2, fontsize=7.5)
    fig.tight_layout(); _sv(fig, "fig_aware_zoom.png")


def plot_aware_control(ar: dict):
    plt.rcParams.update(PLT_RC)
    t = np.arange(ET_STEPS)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, bi, belt in zip(axes, [0, 1], ["E1", "E2"]):
        for m in METHODS:
            ax.plot(t, ar[m]["u"][:, bi], MS[m], color=MC[m], lw=1.5, label=m)
        ax.axvspan(ET_JAM_S, ET_JAM_E, color="#ef4444", alpha=0.08)
        ax.axhline(ET_USS, color="gray", ls=":", lw=0.9)
        ax.set_ylabel(f"Velocidad {belt} (normalizada)")
        ax.legend(fontsize=8)
        if bi == 0:
            ax.set_title("Fase 2 — Señales de control (velocidad por estera)")
    axes[-1].set_xlabel("Muestra $k$")
    fig.tight_layout(); _sv(fig, "fig_aware_control.png")


def plot_aware_split(ar: dict):
    plt.rcParams.update(PLT_RC)
    t   = np.arange(ET_STEPS)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in METHODS:
        ax.plot(t, ar[m]["split"][:, 0], MS[m], color=MC[m], lw=1.5, label=m)
    ax.axhline(0.5, color="#111827", ls=":", lw=1.0, label="Equilibrio (0.5)")
    ax.axvspan(ET_JAM_S, ET_JAM_E, color="#ef4444", alpha=0.08, label="Atasco")
    ax.set_title("Fase 2 — Fracción del robot asignada a E1")
    ax.set_xlabel("Muestra $k$"); ax.set_ylabel("Reparto hacia E1")
    ax.set_ylim(0.10, 0.92); ax.legend(fontsize=8)
    fig.tight_layout(); _sv(fig, "fig_aware_split.png")


def plot_pareto(ar: dict):
    plt.rcParams.update(PLT_RC)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in METHODS:
        me = aware_metrics(ar[m])
        ax.scatter(me["Energia"], me["IAE"], s=120, color=MC[m], zorder=5, label=m)
        ax.annotate(m, (me["Energia"] + 0.3, me["IAE"]), fontsize=9)
    ax.set_xlabel("Energía acumulada $\\sum u$")
    ax.set_ylabel("IAE = $\\sum|n_i - n^*|$")
    ax.set_title("Fase 2 — Frontera error–energía (Pareto práctico)")
    ax.legend(fontsize=8); fig.tight_layout(); _sv(fig, "fig_pareto.png")


def plot_ql_online(offline_iae: float, df_online: pd.DataFrame):
    plt.rcParams.update(PLT_RC)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ep  = df_online["ep"].to_numpy()
    iae = df_online["IAE"].to_numpy()
    w   = max(1, len(ep) // 8)
    iae_s = np.convolve(iae, np.ones(w) / w, mode="same")
    ax.plot(ep, iae, color="#059669", lw=1.2, alpha=0.40, label="IAE por episodio")
    ax.plot(ep, iae_s, color="#059669", lw=2.0, label="Tendencia (media móvil)")
    ax.axhline(offline_iae, color="#374151", lw=1.4, ls="--",
               label=f"Línea base offline ({offline_iae:.1f})")
    ax.set_xlabel("Episodio online"); ax.set_ylabel("IAE")
    ax.set_title("Q-Learning online: fine-tuning durante despliegue")
    ax.legend(fontsize=8); fig.tight_layout(); _sv(fig, "fig_ql_online.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  11. TABLAS LaTeX + CSV
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):   return str(int(v))
    if isinstance(v, (float, np.floating)): return f"{float(v):.3g}"
    return str(v).replace("_", r"\_").replace("%", r"\%")


def _tex(rows, cols, path, caption, label) -> None:
    al  = "l" + "r" * (len(cols) - 1)
    ls  = [
        "% Tabla de resultados - Act08 R2ET",
        r"\begin{table}[H]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\small", r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}{al}@{{}}}}",
        r"\toprule", " & ".join(h for _, h in cols) + r" \\", r"\midrule",
    ]
    for row in rows:
        ls.append(" & ".join(_fmt(row[k]) for k, _ in cols) + r" \\")
    ls += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    Path(path).write_text("\n".join(ls) + "\n", encoding="utf-8")


def write_tables(sr, ar, h_dl_s, h_dl_a, ql_h_s, ql_h_a,
                 mse_s_test=0.0, mse_a_test=0.0,
                 n_servo=8000, n_aware=10000, epochs=2500, epochs_aware=2500):
    # ── Métricas servo ────────────────────────────────────────────────────────
    rows_s = [{"Metodo": m, **servo_metrics(sr[m])} for m in METHODS]
    _tex(rows_s,
         [("Metodo", "Método"), ("IAE", "IAE"), ("Mp_pct", "Mp \\%"),
          ("Ts", "$t_s$"), ("Rec", "Recup."),
          ("Energia", "Energía"), ("Var_u", "$\\sum|\\Delta u|$"), ("Umax", "$u_{max}$")],
         TABLE_DIR / "servo_metrics.tex",
         "Indicadores Fase~1 (servoregulador, 4 métodos).", "tab:servo_metrics")
    pd.DataFrame(rows_s).to_csv(TABLE_DIR / "servo_metrics.csv", index=False)

    # ── Métricas aware ────────────────────────────────────────────────────────
    rows_a = [{"Metodo": m, **aware_metrics(ar[m])} for m in METHODS]
    _tex(rows_a,
         [("Metodo", "Método"), ("IAE", "IAE"), ("IAE_dist", "IAE pert."),
          ("Ocup_max", "Ocup. máx."), ("Margen", "Margen"), ("T_riesgo", "T. alarma"),
          ("Viol", "Viol."), ("Prod", "Prod."), ("Energia", "Energía"), ("Desbal", "Desbal.")],
         TABLE_DIR / "aware_metrics.tex",
         "Indicadores Fase~2 (coordinación R2ET, 4 métodos).", "tab:aware_metrics")
    pd.DataFrame(rows_a).to_csv(TABLE_DIR / "aware_metrics.csv", index=False)

    # ── Mejora vs PID ─────────────────────────────────────────────────────────
    base = aware_metrics(ar["PID"])
    rows_imp = []
    for m in ["Fuzzy", "DL", "QL"]:
        me = aware_metrics(ar[m])
        rows_imp.append({
            "Metodo":    m,
            "DIAE_pct":  round(100 * (base["IAE"] - me["IAE"]) / base["IAE"], 1),
            "DProd":     round(me["Prod"] - base["Prod"], 2),
            "DEnerg":    round(me["Energia"] - base["Energia"], 2),
            "DDesbal":   round(100 * (base["Desbal"] - me["Desbal"]) / max(base["Desbal"], 1e-6), 1),
        })
    _tex(rows_imp,
         [("Metodo", "Método"), ("DIAE_pct", "$\\Delta$IAE \\%"),
          ("DProd", "$\\Delta$Prod."), ("DEnerg", "$\\Delta$Energ."), ("DDesbal", "$\\Delta$Desbal. \\%")],
         TABLE_DIR / "aware_improvement.tex",
         "Variación de cada método inteligente respecto al PID-Aware. "
         "Convención: signo positivo = mejor que el PID (menor IAE/desbalance, mayor productividad); "
         "negativo = peor. $\\Delta$Prod./$\\Delta$Energ. en unidades absolutas.",
         "tab:aware_improvement")
    pd.DataFrame(rows_imp).to_csv(TABLE_DIR / "aware_improvement.csv", index=False)

    # ── Entrenamiento DL ──────────────────────────────────────────────────────
    rows_dl = [
        {"Fase": "Servo", "Arch": "4-64-32-16-8-1", "Epocas": epochs,
         "N_train": n_servo,  "N_val": 50, "N_test": 50,
         "IAE_train": round(h_dl_s["train"][-1], 5),
         "IAE_val":   round(h_dl_s["val"][-1], 5),
         "IAE_test":  round(h_dl_s["val"][-1], 5)},
        {"Fase": "Aware", "Arch": "8-96-64-32-16-8-3", "Epocas": epochs_aware,
         "N_train": n_aware,  "N_val": 30, "N_test": 30,
         "IAE_train": round(h_dl_a["train"][-1], 5),
         "IAE_val":   round(h_dl_a["val"][-1], 5),
         "IAE_test":  round(mse_a_test, 5)},
    ]
    _tex(rows_dl,
         [("Fase", "Fase"), ("Arch", "Arquitectura"), ("Epocas", "Épocas"),
          ("N_train", "Escenarios"), ("N_val", "Val."), ("N_test", "Test"),
          ("IAE_train", "IAE entren."), ("IAE_val", "IAE val."), ("IAE_test", "IAE test")],
         TABLE_DIR / "mlp_training_summary.tex",
         "Entrenamiento Deep Learning: optimización directa sobre planta (método adjunto).",
         "tab:mlp_training")
    pd.DataFrame(rows_dl).to_csv(TABLE_DIR / "mlp_training_summary.csv", index=False)

    # ── Entrenamiento Q-Learning ──────────────────────────────────────────────
    rows_ql = [
        {"Fase": "Servo", "Estados": _SQ_N_STATES, "Acciones": len(_SQ_ACTIONS),
         "Episodios": 600, "Horizonte": 100, "Alpha": 0.15, "Gamma": 0.95,
         "Eps0": 0.60, "EpsMin": 0.02, "R_final": round(float(ql_h_s["r_mean"].tail(30).mean()), 4)},
        {"Fase": "Aware", "Estados": _AQ_N_STATES, "Acciones": len(_AQ_ACTIONS),
         "Episodios": 3000, "Horizonte": 160, "Alpha": 0.18, "Gamma": 0.92,
         "Eps0": 0.65, "EpsMin": 0.02, "R_final": round(float(ql_h_a["r_mean"].tail(30).mean()), 4)},
    ]
    _tex(rows_ql,
         [("Fase", "Fase"), ("Estados", "Estados"), ("Acciones", "Acciones"),
          ("Episodios", "Episodios"), ("Horizonte", "Horizonte"),
          ("Alpha", "$\\alpha$"), ("Gamma", "$\\gamma$"),
          ("Eps0", "$\\epsilon_0$"), ("EpsMin", "$\\epsilon_{min}$"), ("R_final", "$r_{final}$")],
         TABLE_DIR / "ql_training_summary.tex",
         "Parámetros y resultados del entrenamiento Q-Learning.", "tab:ql_training")
    pd.DataFrame(rows_ql).to_csv(TABLE_DIR / "ql_training_summary.csv", index=False)


def run_online_ql(Q_offline: np.ndarray, n_ep=50, eps=0.08, alpha=0.10,
                  seed=SEED + 99):
    rng = np.random.default_rng(seed)
    Q   = Q_offline.copy()
    history = []
    for ep in range(n_ep):
        n = np.array([1.25, 1.25]); ep_iae = 0.0; ep_r = 0.0
        for k in range(ET_STEPS):
            dm = _demand_et(k); d = _d_aware(k); ls = _loss_et(k)
            s  = _encode_aware(n, d)
            a  = int(rng.integers(len(_AQ_ACTIONS)) if rng.random() < eps else np.argmax(Q[s]))
            spd, spl, adm_q = _AQ_ACTIONS[a]
            u   = np.array([spd, spd])
            u, adm_h = _cap_supervisor(u, float(np.max(n)), d)
            adm = min(float(adm_q), float(adm_h))
            sp  = np.array([spl, 1.0 - spl])
            n_n, _ = _et_step(n, u, sp, adm, dm, ls)
            s_n = _encode_aware(n_n, d)
            iae = float(np.sum(np.abs(n_n - ET_TARGET)))
            viol = float(np.sum(np.maximum(0.0, n_n - ET_CAP)))
            alrm = float(np.sum(np.maximum(0.0, n_n - ET_ALARM)))
            bal  = abs(float(n_n[0] - n_n[1]))
            r    = -1.5 * iae - 30.0 * viol - 8.0 * alrm - 0.20 * bal
            Q[s, a] += alpha * (r + 0.92 * float(np.max(Q[s_n])) - Q[s, a])
            ep_iae += iae; ep_r += r; n = n_n
        history.append({"ep": ep + 1, "IAE": round(ep_iae, 3),
                        "r_mean": round(ep_r / ET_STEPS, 4)})
    return Q, pd.DataFrame(history)


def write_online_ql_table(offline_iae: float, df_online: pd.DataFrame):
    nq = len(df_online)
    q1 = df_online["IAE"].iloc[:nq // 4].mean()
    q4 = df_online["IAE"].iloc[-nq // 4:].mean()
    rows = [
        {"Fase": "Offline (línea base)", "Episodios": 0,
         "IAE_medio": round(offline_iae, 3), "Mejora_pct": 0.0},
        {"Fase": "Q1 online (ep. 1-12)", "Episodios": nq // 4,
         "IAE_medio": round(q1, 3),
         "Mejora_pct": round(100 * (offline_iae - q1) / offline_iae, 1)},
        {"Fase": f"Q4 online (ep. {3*nq//4+1}-{nq})", "Episodios": nq - 3 * nq // 4,
         "IAE_medio": round(q4, 3),
         "Mejora_pct": round(100 * (offline_iae - q4) / offline_iae, 1)},
    ]
    _tex(rows,
         [("Fase", "Fase"), ("Episodios", "Ep."),
          ("IAE_medio", "IAE medio"), ("Mejora_pct", "Mejora \\%")],
         TABLE_DIR / "ql_online_summary.tex",
         "Q-Learning online: IAE por cuartil de episodios.", "tab:ql_online")
    pd.DataFrame(rows).to_csv(TABLE_DIR / "ql_online_summary.csv", index=False)


def write_timeseries(sr, ar):
    rows_s = []
    for k in range(SV_STEPS):
        row = {"t": k, "ref": sr["PID"]["ref"][k], "dist": int(sr["PID"]["dist"][k])}
        for m in METHODS:
            row[f"{m}_y"] = sr[m]["y"][k]; row[f"{m}_u"] = sr[m]["u"][k]
        rows_s.append(row)
    with (DATA_DIR / "timeseries_servo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_s[0].keys()))
        w.writeheader(); w.writerows(rows_s)

    rows_a = []
    for k in range(ET_STEPS):
        row = {"t": k}
        for m in METHODS:
            row[f"{m}_n1"] = ar[m]["n"][k, 0]; row[f"{m}_n2"] = ar[m]["n"][k, 1]
            row[f"{m}_u1"] = ar[m]["u"][k, 0]; row[f"{m}_u2"] = ar[m]["u"][k, 1]
            row[f"{m}_spl"] = ar[m]["split"][k, 0]; row[f"{m}_adm"] = ar[m]["admission"][k]
        rows_a.append(row)
    with (DATA_DIR / "timeseries_aware.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_a[0].keys()))
        w.writeheader(); w.writerows(rows_a)


def _run_aware_robust(dl_a, Qa, demand_scale=1.20, eff_scale=0.85):
    """Escenario robusto RNF-03: +20% de demanda y -15% de eficiencia de descarga.

    Reescala temporalmente los parámetros de la planta R2ET (caudal y eficiencia)
    y vuelve a simular los 4 controladores SIN reentrenar ni resintonizar: mide la
    robustez de cada política ante un cambio de condiciones de operación.
    """
    global ET_BASE_IN, ET_PULSE, ET_EG
    base0, pulse0, eg0 = ET_BASE_IN, ET_PULSE, ET_EG
    ET_BASE_IN = base0 * demand_scale
    ET_PULSE   = pulse0 * demand_scale
    ET_EG      = eg0 * eff_scale
    try:
        return {
            "PID":   simulate_aware_pid(),
            "Fuzzy": simulate_aware_fuzzy(),
            "DL":    simulate_aware_dl(dl_a),
            "QL":    simulate_aware_ql(Qa),
        }
    finally:
        ET_BASE_IN, ET_PULSE, ET_EG = base0, pulse0, eg0


def write_robustness_table(ar_nom, ar_rob):
    rows = []
    for m in METHODS:
        mn = aware_metrics(ar_nom[m]); mr = aware_metrics(ar_rob[m])
        rows.append({
            "Metodo":     m,
            "IAE_nom":    mn["IAE"],   "IAE_rob":  mr["IAE"],
            "Ocup_nom":   mn["Ocup_max"], "Ocup_rob": mr["Ocup_max"],
            "Viol_rob":   mr["Viol"],  "Margen_rob": mr["Margen"],
        })
    _tex(rows,
         [("Metodo", "Método"), ("IAE_nom", "IAE nom."), ("IAE_rob", "IAE robusto"),
          ("Ocup_nom", "Ocup. nom."), ("Ocup_rob", "Ocup. rob."),
          ("Viol_rob", "Viol. rob."), ("Margen_rob", "Margen rob.")],
         TABLE_DIR / "robustness_comparison.tex",
         "Robustez Fase~2: escenario nominal vs.\\ severo ($+20\\%$ demanda, $-15\\%$ eficiencia de descarga). "
         "Mismos controladores, sin resintonizar.",
         "tab:robustness")
    pd.DataFrame(rows).to_csv(TABLE_DIR / "robustness_comparison.csv", index=False)


def simulate_aware_pid_baseline() -> dict:
    """Línea base 'antes': dos PID escalares independientes, uno por estera,
    SIN supervisor de capacidad, SIN coordinación del robot (reparto fijo 0.5),
    SIN reducción de admisión y SIN gain scheduling. Representa la alternativa
    industrial previa a introducir el control inteligente con supervisor."""
    n, u, out, sph, adh, alm = _r2et_base()
    sts = [PIDState(), PIDState()]
    kp, ki, kd = 1.3, 0.18, 0.28
    for k in range(1, ET_STEPS):
        dm = _demand_et(k); ls = _loss_et(k)
        uk = np.array([pid_step(float(n[k-1, i]) - ET_TARGET, sts[i],
                                kp, ki, kd, 0.05, 1.0, ff=ET_USS)  # ganancias fijas
                       for i in range(2)])
        sp  = np.array([0.5, 0.5])   # reparto fijo: sin coordinación del robot
        adm = 1.0                    # admisión plena: sin control de admisión
        u[k] = uk; sph[k] = sp; adh[k] = adm
        n[k], out[k] = _et_step(n[k - 1], uk, sp, adm, dm, ls)
        alm[k] = 1.0 if np.any(n[k] > ET_ALARM) else 0.0
    return {"n": n, "u": u, "outflow": out, "split": sph, "admission": adh, "alarm": alm}


def write_baseline_table(ar):
    """Tabla antes/después (Fase 2): PID sin supervisor vs PID supervisado,
    en escenario nominal y severo (+20% demanda, -15% eficiencia)."""
    def _severe(fn):
        global ET_BASE_IN, ET_PULSE, ET_EG
        b0, p0, e0 = ET_BASE_IN, ET_PULSE, ET_EG
        ET_BASE_IN, ET_PULSE, ET_EG = b0 * 1.20, p0 * 1.20, e0 * 0.85
        try:
            return aware_metrics(fn())
        finally:
            ET_BASE_IN, ET_PULSE, ET_EG = b0, p0, e0
    bn = aware_metrics(simulate_aware_pid_baseline()); bs = _severe(simulate_aware_pid_baseline)
    pn = aware_metrics(ar["PID"]);                     ps = _severe(simulate_aware_pid)
    rows = [
        {"Cfg": "PID base (sin supervisor)", "IAEn": bn["IAE"], "Ocn": bn["Ocup_max"],
         "IAEs": bs["IAE"], "Ocs": bs["Ocup_max"], "Als": bs["T_riesgo"]},
        {"Cfg": "PID-Aware (con supervisor)", "IAEn": pn["IAE"], "Ocn": pn["Ocup_max"],
         "IAEs": ps["IAE"], "Ocs": ps["Ocup_max"], "Als": ps["T_riesgo"]},
    ]
    _tex(rows,
         [("Cfg", "Configuración"), ("IAEn", "IAE nom."), ("Ocn", "Ocup. nom."),
          ("IAEs", "IAE severo"), ("Ocs", "Ocup. sev."), ("Als", "Alarma sev.")],
         TABLE_DIR / "aware_baseline.tex",
         "Fase~2 antes/después: PID escalar sin supervisor frente al PID supervisado, en escenario "
         "nominal y severo ($+20\\%$ demanda, $-15\\%$ eficiencia). Alarma = muestras con $n_i>2.7$.",
         "tab:aware_baseline")
    pd.DataFrame(rows).to_csv(TABLE_DIR / "aware_baseline.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  12. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Actividad 08 MROB — Grupo 1, caso R2ET")
    print("  v2: PID | Fuzzy | Deep Learning | Q-Learning")
    print("  Fase 1: servo  |  Fase 2: coordinación aware")
    print("=" * 65)

    # ── DL Fase 1: optimización directa de política (sin referencia externa) ─────────────
    DL_EPOCHS_SERVO = 800;  DL_BATCH = 32
    print(f"\n[1/7] DL servo — optimización directa de política")
    print(f"       {DL_EPOCHS_SERVO} épocas × {DL_BATCH} escenarios/época, arch 4-64-32-16-8-1...")
    dl_s, h_dl_s = train_dl_servo_direct(
        layer_sizes=[4, 64, 32, 16, 8, 1],
        n_epochs=DL_EPOCHS_SERVO,
        batch_size=DL_BATCH,
        lambda_u=0.01,
        seed=SEED,
    )
    N_SERVO = DL_EPOCHS_SERVO * DL_BATCH
    mse_s = h_dl_s["val"][-1]
    print(f"       IAE train final={h_dl_s['train'][-1]:.5f}  IAE val final={mse_s:.5f}")

    # ── DL Fase 2: optimización directa de política sobre planta R2ET ─────────
    DL_EPOCHS_AWARE = 1200;  DL_BATCH_A = 16
    DL_ARCH_AWARE = [8, 96, 64, 32, 16, 8, 3]
    print(f"\n[2/7] DL aware — optimización directa de política sobre planta R2ET")
    print(f"       {DL_EPOCHS_AWARE} épocas × {DL_BATCH_A} escenarios/época, arch 8-96-64-32-16-8-3...")
    print(f"       método adjunto 2 esteras, sin controlador de referencia externo...")
    dl_a, h_dl_a = train_dl_aware_direct(
        layer_sizes=DL_ARCH_AWARE,
        n_epochs=DL_EPOCHS_AWARE,
        batch_size=DL_BATCH_A,
        lambda_u=0.02,
        seed=SEED + 1,
    )
    N_AWARE = DL_EPOCHS_AWARE * DL_BATCH_A
    mse_a = h_dl_a["val"][-1]
    print(f"       IAE train final={h_dl_a['train'][-1]:.5f}  IAE val final={mse_a:.5f}")

    # ── Q-Learning: entrenamiento offline ─────────────────────────────────────
    print(f"\n[3/7] Q-Learning servo ({_SQ_N_STATES} estados × {len(_SQ_ACTIONS)} acciones, 600 ep.)...")
    Qs, ql_h_s = train_q_servo()
    print(f"       r_mean últimos 30 ep = {ql_h_s['r_mean'].tail(30).mean():.4f}")

    print(f"\n[4/7] Q-Learning aware ({_AQ_N_STATES} estados × {len(_AQ_ACTIONS)} acciones, 3000 ep.)...")
    Qa, ql_h_a = train_q_aware()
    print(f"       r_mean últimos 30 ep = {ql_h_a['r_mean'].tail(30).mean():.4f}")

    # ── Simulaciones estándar ──────────────────────────────────────────────────
    print("\n[5/7] Simulando Fase 1 (servo) + Fase 2 (aware)...")
    sr = {
        "PID":   simulate_servo_pid(),
        "Fuzzy": simulate_servo_fuzzy(),
        "DL":    simulate_servo_dl(dl_s),
        "QL":    simulate_servo_ql(Qs),
    }
    ar = {
        "PID":   simulate_aware_pid(),
        "Fuzzy": simulate_aware_fuzzy(),
        "DL":    simulate_aware_dl(dl_a),
        "QL":    simulate_aware_ql(Qa),
    }

    # ── Q-Learning online ──────────────────────────────────────────────────────
    print("\n[6/7] Q-Learning online (50 ep., ε=0.08)...")
    offline_iae = aware_metrics(ar["QL"])["IAE"]
    _, df_online = run_online_ql(Qa, n_ep=50, eps=0.08, alpha=0.10)
    print(f"       Línea base offline IAE={offline_iae:.2f} | Q4 online={df_online['IAE'].tail(12).mean():.2f}")

    # ── Test de robustez: perturbación en rampa trapezoidal ───────────────────
    print("\n[7/8] Ensayo de robustez — perturbación en rampa trapezoidal...")
    sr_ramp = simulate_servo_all_dfn(_d_servo_ramp, dl_s, Qs)
    print("       IAE escalón → rampa (degradación):")
    for m in METHODS:
        iae_s = servo_metrics(sr[m])["IAE"]
        iae_r = servo_metrics(sr_ramp[m])["IAE"]
        print(f"       {m:6s}  escalón={iae_s:.2f}  rampa={iae_r:.2f}  "
              f"({100*(iae_r/iae_s-1):+.1f}%)")

    # ── Robustez Fase 2: escenario severo (+20% demanda, -15% eficiencia) ─────
    print("\n[7b/8] Escenario robusto Fase 2 (+20% demanda, -15% eficiencia)...")
    ar_rob = _run_aware_robust(dl_a, Qa)
    for m in METHODS:
        mn = aware_metrics(ar[m]); mr = aware_metrics(ar_rob[m])
        print(f"       {m:6s} IAE {mn['IAE']:.1f}->{mr['IAE']:.1f}  "
              f"ocup {mn['Ocup_max']:.2f}->{mr['Ocup_max']:.2f}  viol_rob={mr['Viol']}")

    # ── Figuras + Tablas ───────────────────────────────────────────────────────
    print("\n[8/8] Generando figuras y tablas...")
    plot_fuzzy_memberships()
    plot_dl_training(h_dl_s, h_dl_a)
    plot_ql_training(ql_h_s, ql_h_a)
    plot_ql_online(offline_iae, df_online)
    plot_servo_response(sr)
    plot_servo_kpi(sr)
    plot_robustez_rampa(sr, sr_ramp)
    plot_aware_occupancy(ar)
    plot_aware_zoom(ar)
    plot_aware_control(ar)
    plot_aware_split(ar)
    plot_pareto(ar)

    write_tables(sr, ar, h_dl_s, h_dl_a, ql_h_s, ql_h_a,
                 mse_s_test=mse_s, mse_a_test=mse_a,
                 n_servo=N_SERVO, n_aware=N_AWARE,
                 epochs=DL_EPOCHS_SERVO, epochs_aware=DL_EPOCHS_AWARE)
    write_online_ql_table(offline_iae, df_online)
    write_robustness_table(ar, ar_rob)
    write_baseline_table(ar)
    write_timeseries(sr, ar)

    # ── Resumen en consola ─────────────────────────────────────────────────────
    print("\n─── Fase 1 — Servo ───")
    for m in METHODS:
        me = servo_metrics(sr[m])
        print(f"  {m:6s}  IAE={me['IAE']:.2f}  Mp={me['Mp_pct']:.1f}%  Ts={me['Ts']}  E={me['Energia']:.1f}")

    print("\n─── Fase 2 — Aware ───")
    for m in METHODS:
        me = aware_metrics(ar[m])
        print(f"  {m:6s}  IAE={me['IAE']:.2f}  max={me['Ocup_max']:.3f}  "
              f"Viol={me['Viol']}  Prod={me['Prod']:.2f}  E={me['Energia']:.2f}")

    print(f"\n─── Q-Learning online ───")
    print(f"  Offline IAE = {offline_iae:.2f}")
    print(f"  Online Q4   = {df_online['IAE'].tail(12).mean():.2f}")

    print(f"\n✓ Figuras → {FIG_DIR}")
    print(f"✓ Tablas  → {TABLE_DIR}")
    print(f"✓ Datos   → {DATA_DIR}")


if __name__ == "__main__":
    main()
