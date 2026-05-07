#!/usr/bin/env python3
"""
Fixture generator for ferrotorch-optim conformance suite — C6.1 SGD-family.

Reference: torch == 2.11.0
Pin:       All computations replicate the PyTorch algorithm exactly.
Output:    ferrotorch-optim/tests/conformance/fixtures.json

Usage:
    python3 scripts/regenerate_optim_fixtures.py

The script is intentionally runnable with OR without PyTorch installed.
  - With torch: validates against live torch output (comparison mode).
  - Without torch: runs the reference algorithm implemented here in pure
    Python (generation mode) and writes fixtures.json.

Sub-phases C6.2-C6.4 will EXTEND this script by appending new fixture
sections to the JSON output. Do NOT replace this file in those sub-phases;
add new fixture-generation functions and extend the `all_fixtures` dict.

Seeded initial params: torch.manual_seed(42), torch.randn(10), dtype=float32.
Gradient sequence:     torch.manual_seed(137), 5 × torch.randn(10) each step.
N steps:               5
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Seeded data (pre-generated from torch.manual_seed(42) / seed(137))
# These are the CANONICAL values for the entire C6.x series.
# ---------------------------------------------------------------------------

# torch.manual_seed(42); torch.randn(10).tolist()
INIT_PARAMS: list[float] = [
     0.3367,  0.1288,  0.2345,  0.2303, -1.1229,
    -0.1863,  2.2082, -0.6380,  0.4617,  0.2674,
]

# torch.manual_seed(137); [torch.randn(10).tolist() for _ in range(5)]
GRAD_STEPS: list[list[float]] = [
    [-0.4917,  0.3557, -0.3465,  0.6218,  1.5722,
      1.1409, -0.0960, -0.7693,  0.9757, -0.0737],
    [ 0.2162, -0.3476, -0.4131, -0.2786,  0.4397,
     -0.5476,  1.3618,  0.3791,  0.0870, -1.1205],
    [-0.1613,  1.0161, -0.7040, -0.2243, -0.0559,
      0.3891, -0.5073,  0.4729,  0.1440, -0.3407],
    [ 0.0516,  0.1965,  0.6048,  0.7213, -0.5534,
      0.3248, -0.1098, -1.2456,  0.4319,  0.1012],
    [-0.8547, -0.2256,  0.2893,  0.2490, -0.4065,
     -0.0572, -0.9133,  0.4128, -0.2097,  0.5714],
]


def _apply_sgd_step(
    params: list[float],
    grad: list[float],
    buf: list[float] | None,
    lr: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    nesterov: bool,
    step_idx: int,
) -> tuple[list[float], list[float] | None]:
    """Single SGD step (CPU, legacy path). Returns (new_params, new_buf)."""
    n = len(params)
    # Weight decay
    g = [grad[i] + weight_decay * params[i] for i in range(n)]

    if momentum > 0.0:
        if step_idx == 0 or buf is None:
            new_buf = g[:]
        else:
            new_buf = [momentum * buf[i] + (1.0 - dampening) * g[i] for i in range(n)]
        if nesterov:
            eff_g = [g[i] + momentum * new_buf[i] for i in range(n)]
        else:
            eff_g = new_buf[:]
    else:
        new_buf = None
        eff_g = g

    new_params = [params[i] - lr * eff_g[i] for i in range(n)]
    return new_params, new_buf


def gen_sgd_plain() -> dict[str, Any]:
    """SGD, lr=0.1, no momentum, no weight_decay — plain vanilla."""
    lr = 0.1
    params = INIT_PARAMS[:]
    steps: list[dict] = []
    for i, grad in enumerate(GRAD_STEPS):
        params, _ = _apply_sgd_step(params, grad, None, lr, 0.0, 0.0, 0.0, False, i)
        steps.append({"step": i + 1, "params": params[:]})
    return {"variant": "plain", "lr": lr, "momentum": 0.0, "weight_decay": 0.0,
            "nesterov": False, "dampening": 0.0, "steps": steps}


def gen_sgd_momentum() -> dict[str, Any]:
    """SGD with momentum=0.9."""
    lr, momentum = 0.1, 0.9
    params = INIT_PARAMS[:]
    buf = None
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, buf = _apply_sgd_step(params, grad, buf, lr, momentum, 0.0, 0.0, False, i)
        steps.append({"step": i + 1, "params": params[:], "momentum_buf": buf[:]})
    return {"variant": "momentum", "lr": lr, "momentum": momentum, "weight_decay": 0.0,
            "nesterov": False, "dampening": 0.0, "steps": steps}


def gen_sgd_nesterov() -> dict[str, Any]:
    """SGD with momentum=0.9, nesterov=True."""
    lr, momentum = 0.1, 0.9
    params = INIT_PARAMS[:]
    buf = None
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, buf = _apply_sgd_step(params, grad, buf, lr, momentum, 0.0, 0.0, True, i)
        steps.append({"step": i + 1, "params": params[:], "momentum_buf": buf[:]})
    return {"variant": "nesterov", "lr": lr, "momentum": momentum, "weight_decay": 0.0,
            "nesterov": True, "dampening": 0.0, "steps": steps}


def gen_sgd_weight_decay() -> dict[str, Any]:
    """SGD with weight_decay=0.01."""
    lr, wd = 0.1, 0.01
    params = INIT_PARAMS[:]
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, _ = _apply_sgd_step(params, grad, None, lr, 0.0, 0.0, wd, False, i)
        steps.append({"step": i + 1, "params": params[:]})
    return {"variant": "weight_decay", "lr": lr, "momentum": 0.0, "weight_decay": wd,
            "nesterov": False, "dampening": 0.0, "steps": steps}


def gen_sgd_dampening() -> dict[str, Any]:
    """SGD with momentum=0.9, dampening=0.5."""
    lr, momentum, dampening = 0.1, 0.9, 0.5
    params = INIT_PARAMS[:]
    buf = None
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, buf = _apply_sgd_step(params, grad, buf, lr, momentum, dampening, 0.0, False, i)
        steps.append({"step": i + 1, "params": params[:], "momentum_buf": buf[:]})
    return {"variant": "dampening", "lr": lr, "momentum": momentum, "weight_decay": 0.0,
            "nesterov": False, "dampening": dampening, "steps": steps}


# ---------------------------------------------------------------------------
# Adagrad
# ---------------------------------------------------------------------------

def _apply_adagrad_step(
    params: list[float],
    grad: list[float],
    sum_acc: list[float],
    lr: float,
    lr_decay: float,
    weight_decay: float,
    eps: float,
    step_idx: int,  # 0-based step index
) -> tuple[list[float], list[float]]:
    """Single Adagrad step. Returns (new_params, new_sum)."""
    n = len(params)
    step_count = step_idx + 1
    # Effective lr with decay
    clr = lr / (1.0 + (step_count - 1) * lr_decay)

    g = [grad[i] + weight_decay * params[i] for i in range(n)]
    new_sum = [sum_acc[i] + g[i] ** 2 for i in range(n)]
    new_params = [params[i] - clr * g[i] / (math.sqrt(new_sum[i]) + eps) for i in range(n)]
    return new_params, new_sum


def gen_adagrad_plain() -> dict[str, Any]:
    lr, eps = 0.1, 1e-10
    params = INIT_PARAMS[:]
    acc = [0.0] * 10
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, acc = _apply_adagrad_step(params, grad, acc, lr, 0.0, 0.0, eps, i)
        steps.append({"step": i + 1, "params": params[:], "sum_acc": acc[:]})
    return {"variant": "plain", "lr": lr, "lr_decay": 0.0, "weight_decay": 0.0,
            "eps": eps, "steps": steps}


def gen_adagrad_lr_decay() -> dict[str, Any]:
    lr, lr_decay, eps = 0.1, 0.01, 1e-10
    params = INIT_PARAMS[:]
    acc = [0.0] * 10
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, acc = _apply_adagrad_step(params, grad, acc, lr, lr_decay, 0.0, eps, i)
        steps.append({"step": i + 1, "params": params[:], "sum_acc": acc[:]})
    return {"variant": "lr_decay", "lr": lr, "lr_decay": lr_decay, "weight_decay": 0.0,
            "eps": eps, "steps": steps}


def gen_adagrad_weight_decay() -> dict[str, Any]:
    lr, wd, eps = 0.1, 0.01, 1e-10
    params = INIT_PARAMS[:]
    acc = [0.0] * 10
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, acc = _apply_adagrad_step(params, grad, acc, lr, 0.0, wd, eps, i)
        steps.append({"step": i + 1, "params": params[:], "sum_acc": acc[:]})
    return {"variant": "weight_decay", "lr": lr, "lr_decay": 0.0, "weight_decay": wd,
            "eps": eps, "steps": steps}


# ---------------------------------------------------------------------------
# ASGD
# ---------------------------------------------------------------------------

def _apply_asgd_step(
    params: list[float],
    grad: list[float],
    ax: list[float],
    step_count: int,  # 1-based after increment
    eta: float,
    mu: float,
    lr: float,
    lambd: float,
    alpha: float,
    t0: float,
    weight_decay: float,
) -> tuple[list[float], list[float], float, float]:
    """Single ASGD step (before state update). Returns (new_params, new_ax, new_eta, new_mu)."""
    n = len(params)
    g = [grad[i] + weight_decay * params[i] for i in range(n)]

    # p = p * (1 - lambd * eta) - eta * g
    new_params = [params[i] * (1.0 - lambd * eta) - eta * g[i] for i in range(n)]

    # ax update
    if mu != 1.0:
        new_ax = [ax[i] + mu * (new_params[i] - ax[i]) for i in range(n)]
    else:
        new_ax = new_params[:]

    step = float(step_count)
    new_eta = lr / (1.0 + lambd * lr * step) ** alpha
    new_mu = 1.0 / max(1.0, step - t0)
    return new_params, new_ax, new_eta, new_mu


def gen_asgd_plain() -> dict[str, Any]:
    """ASGD with small t0=0 so averaging kicks in from step 2."""
    lr, lambd, alpha, t0 = 0.1, 1e-4, 0.75, 0.0
    params = INIT_PARAMS[:]
    ax = params[:]
    eta = lr
    mu = 1.0
    step_count = 0
    steps = []
    for grad in GRAD_STEPS:
        step_count += 1
        params, ax, eta, mu = _apply_asgd_step(
            params, grad, ax, step_count, eta, mu, lr, lambd, alpha, t0, 0.0
        )
        steps.append({
            "step": step_count, "params": params[:], "ax": ax[:],
            "eta": eta, "mu": mu,
        })
    return {"variant": "plain", "lr": lr, "lambd": lambd, "alpha": alpha,
            "t0": t0, "weight_decay": 0.0, "steps": steps}


def gen_asgd_weight_decay() -> dict[str, Any]:
    lr, lambd, alpha, t0, wd = 0.1, 1e-4, 0.75, 0.0, 0.01
    params = INIT_PARAMS[:]
    ax = params[:]
    eta = lr
    mu = 1.0
    step_count = 0
    steps = []
    for grad in GRAD_STEPS:
        step_count += 1
        params, ax, eta, mu = _apply_asgd_step(
            params, grad, ax, step_count, eta, mu, lr, lambd, alpha, t0, wd
        )
        steps.append({
            "step": step_count, "params": params[:], "ax": ax[:],
            "eta": eta, "mu": mu,
        })
    return {"variant": "weight_decay", "lr": lr, "lambd": lambd, "alpha": alpha,
            "t0": t0, "weight_decay": wd, "steps": steps}


# ---------------------------------------------------------------------------
# Rprop
# ---------------------------------------------------------------------------

def _rprop_sign(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def _apply_rprop_step(
    params: list[float],
    grad: list[float],
    prev_grad: list[float],
    step_sizes: list[float],
    eta_minus: float,
    eta_plus: float,
    step_min: float,
    step_max: float,
    is_first: bool,
) -> tuple[list[float], list[float], list[float]]:
    """Single Rprop step. Returns (new_params, new_prev_grad, new_step_sizes)."""
    n = len(params)
    new_params = []
    new_prev_grad = []
    new_step_sizes = step_sizes[:]

    for i in range(n):
        g = grad[i]
        prev = prev_grad[i] if not is_first else 0.0
        product = g * prev

        if product > 0.0:
            new_step_sizes[i] = min(step_sizes[i] * eta_plus, step_max)
        elif product < 0.0:
            new_step_sizes[i] = max(step_sizes[i] * eta_minus, step_min)
        # == 0: unchanged

        eff_g = 0.0 if (product < 0.0) else g
        new_prev_grad.append(eff_g)
        new_params.append(params[i] - _rprop_sign(eff_g) * new_step_sizes[i])

    return new_params, new_prev_grad, new_step_sizes


def gen_rprop_plain() -> dict[str, Any]:
    lr = 0.01
    eta_minus, eta_plus = 0.5, 1.2
    step_min, step_max = 1e-6, 50.0
    params = INIT_PARAMS[:]
    prev_grad = [0.0] * 10
    step_sizes = [lr] * 10
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, prev_grad, step_sizes = _apply_rprop_step(
            params, grad, prev_grad, step_sizes, eta_minus, eta_plus, step_min, step_max, i == 0
        )
        steps.append({
            "step": i + 1, "params": params[:],
            "step_sizes": step_sizes[:], "prev_grad": prev_grad[:],
        })
    return {"variant": "plain", "lr": lr, "etas": [eta_minus, eta_plus],
            "step_sizes_range": [step_min, step_max], "steps": steps}


# ---------------------------------------------------------------------------
# RMSprop
# ---------------------------------------------------------------------------

def _apply_rmsprop_step(
    params: list[float],
    grad: list[float],
    square_avg: list[float],
    grad_avg: list[float] | None,
    mom_buf: list[float] | None,
    lr: float,
    alpha: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    centered: bool,
) -> tuple[list[float], list[float], list[float] | None, list[float] | None]:
    n = len(params)
    g = [grad[i] + weight_decay * params[i] for i in range(n)]

    new_sq = [alpha * square_avg[i] + (1.0 - alpha) * g[i] ** 2 for i in range(n)]

    if centered:
        new_ga = [alpha * grad_avg[i] + (1.0 - alpha) * g[i] for i in range(n)]  # type: ignore[index]
        avg = [math.sqrt(new_sq[i] - new_ga[i] ** 2 + eps) for i in range(n)]
    else:
        new_ga = None
        avg = [math.sqrt(new_sq[i] + eps) for i in range(n)]

    if momentum > 0.0:
        new_buf = [momentum * mom_buf[i] + g[i] / avg[i] for i in range(n)]  # type: ignore[index]
        new_params = [params[i] - lr * new_buf[i] for i in range(n)]
    else:
        new_buf = None
        new_params = [params[i] - lr * g[i] / avg[i] for i in range(n)]

    return new_params, new_sq, new_ga, new_buf


def gen_rmsprop_plain() -> dict[str, Any]:
    lr, alpha, eps = 0.01, 0.99, 1e-8
    params = INIT_PARAMS[:]
    sq = [0.0] * 10
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, sq, _, _ = _apply_rmsprop_step(
            params, grad, sq, None, None, lr, alpha, eps, 0.0, 0.0, False
        )
        steps.append({"step": i + 1, "params": params[:], "square_avg": sq[:]})
    return {"variant": "plain", "lr": lr, "alpha": alpha, "eps": eps,
            "weight_decay": 0.0, "momentum": 0.0, "centered": False, "steps": steps}


def gen_rmsprop_centered() -> dict[str, Any]:
    lr, alpha, eps = 0.01, 0.99, 1e-8
    params = INIT_PARAMS[:]
    sq = [0.0] * 10
    ga = [0.0] * 10
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, sq, ga, _ = _apply_rmsprop_step(
            params, grad, sq, ga, None, lr, alpha, eps, 0.0, 0.0, True
        )
        steps.append({"step": i + 1, "params": params[:],
                       "square_avg": sq[:], "grad_avg": ga[:]})
    return {"variant": "centered", "lr": lr, "alpha": alpha, "eps": eps,
            "weight_decay": 0.0, "momentum": 0.0, "centered": True, "steps": steps}


def gen_rmsprop_momentum() -> dict[str, Any]:
    lr, alpha, eps, momentum = 0.01, 0.99, 1e-8, 0.9
    params = INIT_PARAMS[:]
    sq = [0.0] * 10
    buf = [0.0] * 10
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, sq, _, buf = _apply_rmsprop_step(
            params, grad, sq, None, buf, lr, alpha, eps, 0.0, momentum, False
        )
        steps.append({"step": i + 1, "params": params[:],
                       "square_avg": sq[:], "momentum_buf": buf[:]})
    return {"variant": "momentum", "lr": lr, "alpha": alpha, "eps": eps,
            "weight_decay": 0.0, "momentum": momentum, "centered": False, "steps": steps}


def gen_rmsprop_weight_decay() -> dict[str, Any]:
    lr, alpha, eps, wd = 0.01, 0.99, 1e-8, 0.01
    params = INIT_PARAMS[:]
    sq = [0.0] * 10
    steps = []
    for i, grad in enumerate(GRAD_STEPS):
        params, sq, _, _ = _apply_rmsprop_step(
            params, grad, sq, None, None, lr, alpha, eps, wd, 0.0, False
        )
        steps.append({"step": i + 1, "params": params[:], "square_avg": sq[:]})
    return {"variant": "weight_decay", "lr": lr, "alpha": alpha, "eps": eps,
            "weight_decay": wd, "momentum": 0.0, "centered": False, "steps": steps}


# ---------------------------------------------------------------------------
# Optional: validate against live torch if available
# ---------------------------------------------------------------------------

def _torch_validate() -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        print("[regenerate_optim_fixtures] torch not installed — skipping live validation.")
        return

    print(f"[regenerate_optim_fixtures] torch {torch.__version__} found — running live validation.")

    init = torch.tensor(INIT_PARAMS, dtype=torch.float32)
    grads_t = [torch.tensor(g, dtype=torch.float32) for g in GRAD_STEPS]

    # SGD plain
    p = torch.nn.Parameter(init.clone())
    opt = torch.optim.SGD([p], lr=0.1)
    for g in grads_t:
        opt.zero_grad()
        p.grad = g.clone()
        opt.step()
    expected = p.detach().tolist()
    computed = gen_sgd_plain()["steps"][-1]["params"]
    max_err = max(abs(e - c) for e, c in zip(expected, computed))
    assert max_err < 1e-5, f"SGD plain mismatch: max_err={max_err}"
    print(f"  SGD plain OK (max_err={max_err:.2e})")

    # SGD momentum
    p = torch.nn.Parameter(init.clone())
    opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)
    for g in grads_t:
        opt.zero_grad()
        p.grad = g.clone()
        opt.step()
    expected = p.detach().tolist()
    computed = gen_sgd_momentum()["steps"][-1]["params"]
    max_err = max(abs(e - c) for e, c in zip(expected, computed))
    assert max_err < 1e-5, f"SGD momentum mismatch: max_err={max_err}"
    print(f"  SGD momentum OK (max_err={max_err:.2e})")

    # Adagrad plain
    p = torch.nn.Parameter(init.clone())
    opt = torch.optim.Adagrad([p], lr=0.1, eps=1e-10)
    for g in grads_t:
        opt.zero_grad()
        p.grad = g.clone()
        opt.step()
    expected = p.detach().tolist()
    computed = gen_adagrad_plain()["steps"][-1]["params"]
    max_err = max(abs(e - c) for e, c in zip(expected, computed))
    assert max_err < 1e-5, f"Adagrad plain mismatch: max_err={max_err}"
    print(f"  Adagrad plain OK (max_err={max_err:.2e})")

    # RMSprop plain
    p = torch.nn.Parameter(init.clone())
    opt = torch.optim.RMSprop([p], lr=0.01, alpha=0.99, eps=1e-8)
    for g in grads_t:
        opt.zero_grad()
        p.grad = g.clone()
        opt.step()
    expected = p.detach().tolist()
    computed = gen_rmsprop_plain()["steps"][-1]["params"]
    max_err = max(abs(e - c) for e, c in zip(expected, computed))
    assert max_err < 1e-5, f"RMSprop plain mismatch: max_err={max_err}"
    print(f"  RMSprop plain OK (max_err={max_err:.2e})")

    print("[regenerate_optim_fixtures] All live validations passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fixtures: dict[str, Any] = {
        "metadata": {
            "torch_version": "2.11.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rng_seed_params": 42,
            "rng_seed_grads": 137,
            "n_params": 10,
            "n_steps": 5,
            "note": (
                "C6.1: SGD-family fixtures. "
                "C6.2 will add Adam-family. "
                "C6.3 will add misc/utilities. "
                "C6.4 will add schedulers."
            ),
        },
        "sgd": [
            gen_sgd_plain(),
            gen_sgd_momentum(),
            gen_sgd_nesterov(),
            gen_sgd_weight_decay(),
            gen_sgd_dampening(),
        ],
        "adagrad": [
            gen_adagrad_plain(),
            gen_adagrad_lr_decay(),
            gen_adagrad_weight_decay(),
        ],
        "asgd": [
            gen_asgd_plain(),
            gen_asgd_weight_decay(),
        ],
        "rprop": [
            gen_rprop_plain(),
        ],
        "rmsprop": [
            gen_rmsprop_plain(),
            gen_rmsprop_centered(),
            gen_rmsprop_momentum(),
            gen_rmsprop_weight_decay(),
        ],
    }

    out_path = (
        Path(__file__).parent.parent
        / "ferrotorch-optim"
        / "tests"
        / "conformance"
        / "fixtures.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2)
    print(f"[regenerate_optim_fixtures] Wrote {out_path}")

    # Optional live validation (no-op if torch is absent).
    _torch_validate()

    print("[regenerate_optim_fixtures] Done. Exit 0.")


if __name__ == "__main__":
    main()
