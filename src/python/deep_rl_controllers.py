"""Optional PyTorch module for Activity 08 controllers.

This file is intentionally optional. The MATLAB simulation can run without it.
Use it when the report needs a heavier Deep Learning / Q-learning / RL
experiment with external libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - only used interactively.
    raise SystemExit(
        "Install optional dependencies with: pip install -r src/python/requirements-deep.txt"
    ) from exc


@dataclass
class R2ETConfig:
    capacity: float = 3.0
    target: float = 1.5
    inflow_base: float = 0.42
    inflow_pulse: float = 0.35
    exit_gain: float = 0.46
    horizon: int = 160


class DeepController(nn.Module):
    """Deep MLP controller with a large servo-oriented hidden stack."""

    def __init__(self, state_dim: int = 6, hidden: int = 256, depth: int = 16):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 3), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns [u1, u2, split_e1] in normalized units."""
        return self.net(state)


class DQNPolicy(nn.Module):
    """DQN-style policy for discrete speed/split/admission actions."""

    def __init__(self, state_dim: int = 6, hidden: int = 128, actions: int = 15):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q(state)


class TabularQLearning:
    """Small tabular Q-learning baseline for discrete R2ET states."""

    def __init__(self, states: int = 40, actions: int = 6, alpha: float = 0.18, gamma: float = 0.92):
        self.q = np.zeros((states, actions), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        target = reward + self.gamma * float(np.max(self.q[next_state]))
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def act(self, state: int, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return int(np.random.randint(self.q.shape[1]))
        return int(np.argmax(self.q[state]))


def r2et_step(
    n: np.ndarray,
    control: np.ndarray,
    demand: float,
    disturbance: float,
    cfg: R2ETConfig,
) -> np.ndarray:
    """One-step R2ET dynamics used for synthetic training data."""
    u1, u2, split_e1 = control
    split = np.array([split_e1, 1.0 - split_e1])
    loss = np.array([0.35 * disturbance, 0.175 * disturbance])
    outflow = cfg.exit_gain * np.array([u1, u2]) * (1.0 - loss)
    n_next = n + demand * split - outflow
    return np.clip(n_next, 0.0, cfg.capacity + 0.4)


def make_state(n: np.ndarray, prev_n: np.ndarray, disturbance: float, cfg: R2ETConfig) -> np.ndarray:
    e = n - cfg.target
    de = n - prev_n
    return np.array([e[0], e[1], de[0], de[1], n[0] - n[1], disturbance], dtype=np.float32)


def heuristic_teacher(state: np.ndarray) -> np.ndarray:
    """Teacher policy for supervised warm start of the deep controller."""
    e1, e2, _de1, _de2, balance, disturbance = state
    u1 = np.clip(0.45 + 0.70 * e1 + 0.18 * disturbance, 0.05, 0.92)
    u2 = np.clip(0.45 + 0.70 * e2 + 0.18 * disturbance, 0.05, 0.92)
    split_e1 = np.clip(0.50 - 0.20 * np.tanh(1.4 * balance), 0.20, 0.80)
    return np.array([u1, u2, split_e1], dtype=np.float32)


def generate_teacher_dataset(samples: int = 4000, seed: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    states = []
    targets = []
    for _ in range(samples):
        n = rng.uniform(0.0, 3.2, size=2)
        prev_n = n + rng.normal(0.0, 0.15, size=2)
        disturbance = float(rng.random() > 0.78)
        state = make_state(n, prev_n, disturbance, R2ETConfig())
        states.append(state)
        targets.append(heuristic_teacher(state))
    return torch.tensor(np.array(states)), torch.tensor(np.array(targets))


def split_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed + 101)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    train_end = int(0.70 * len(idx))
    val_end = train_end + int(0.15 * len(idx))
    train_idx = torch.tensor(idx[:train_end], dtype=torch.long)
    val_idx = torch.tensor(idx[train_end:val_end], dtype=torch.long)
    test_idx = torch.tensor(idx[val_end:], dtype=torch.long)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[test_idx], y[test_idx]


def train_deep_controller(epochs: int = 500) -> DeepController:
    model = DeepController()
    x, y = generate_teacher_dataset()
    x_train, y_train, x_val, y_val, _x_test, _y_test = split_dataset(x, y)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    for epoch in range(epochs):
        pred = model(x_train)
        command_loss = mse(pred, y_train)
        saturation_loss = torch.mean(torch.relu(pred[:, :2] - 0.92) ** 2)
        energy_loss = 0.015 * torch.mean(pred[:, 0] + pred[:, 1])
        smooth_loss = 0.04 * torch.mean(torch.abs(pred[1:, :2] - pred[:-1, :2]))
        loss = command_loss + saturation_loss + energy_loss + smooth_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = mse(val_pred, y_val)
            print(f"epoch={epoch:03d} train_loss={loss.item():.5f} val_loss={val_loss.item():.5f}")
    return model


def main() -> None:
    model = train_deep_controller()
    torch.save(model.state_dict(), "artifacts/data/deep_controller_r2et.pt")
    print("Saved artifacts/data/deep_controller_r2et.pt")


if __name__ == "__main__":
    main()
