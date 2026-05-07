#!/usr/bin/env python3
"""Regenerate Adam-family optimizer conformance fixtures.

Reference: torch == 2.11.0
Trajectories: 10-element parameter vector, 5-step update sequence, fixed seeds.

Each optimizer is implemented analytically here, matching the documented
algorithm from torch.optim.  This avoids a live torch dependency in CI.

Usage::

    python scripts/regenerate_optim_adam_fixtures.py \
        --output ferrotorch-optim/tests/conformance/fixtures_adam_family.json

The output file is committed to the repo.  CI never re-runs this script;
the JSON files are the source of truth.

C6.2 — Adam-family conformance (ferrotorch-optim).
Coordinate with C6.1 (SGD-family): uses separate files, no collision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


TORCH_VERSION = "2.11.0"
RNG_SEED = 42
N_PARAMS = 10
N_STEPS = 5


# ---------------------------------------------------------------------------
# Deterministic pseudo-random input data
# ---------------------------------------------------------------------------


def _make_params(n: int = N_PARAMS, seed: int = RNG_SEED) -> list[float]:
    """10-element param vector, each value in [-1, 1]."""
    vals = []
    for i in range(n):
        h = hashlib.md5(f"{seed}_{i}".encode()).hexdigest()
        v = (int(h[:8], 16) / 2**32) * 2 - 1
        vals.append(v)
    return vals


def _make_grads(step: int, n: int = N_PARAMS, seed: int = RNG_SEED) -> list[float]:
    """Gradient vector for a given step, each value in [-0.2, 0.2]."""
    vals = []
    for i in range(n):
        h = hashlib.md5(f"{seed}_g_{step}_{i}".encode()).hexdigest()
        v = (int(h[:8], 16) / 2**32) * 0.4 - 0.2
        vals.append(v)
    return vals


INIT_PARAMS: list[float] = _make_params()
GRADS: list[list[float]] = [_make_grads(s) for s in range(N_STEPS)]


# ---------------------------------------------------------------------------
# Adam
# Reference: Kingma & Ba (ICLR 2015); torch.optim.Adam
# ---------------------------------------------------------------------------


def run_adam(
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
) -> list[list[float]]:
    beta1, beta2 = betas
    params = list(INIT_PARAMS)
    m = [0.0] * N_PARAMS
    v = [0.0] * N_PARAMS
    v_max = [0.0] * N_PARAMS if amsgrad else None
    trajectory = [list(params)]
    for step in range(1, N_STEPS + 1):
        g = list(GRADS[step - 1])
        if weight_decay > 0:
            g = [g[i] + weight_decay * params[i] for i in range(N_PARAMS)]
        m = [beta1 * m[i] + (1 - beta1) * g[i] for i in range(N_PARAMS)]
        v = [beta2 * v[i] + (1 - beta2) * g[i] ** 2 for i in range(N_PARAMS)]
        bc1 = 1 - beta1**step
        bc2 = 1 - beta2**step
        new_params = []
        for i in range(N_PARAMS):
            m_hat = m[i] / bc1
            if amsgrad:
                assert v_max is not None
                # Rust stores raw (not bias-corrected) v in v_max, then
                # applies bias correction inside the denominator.
                v_max[i] = max(v_max[i], v[i])
                denom = math.sqrt(v_max[i] / bc2) + eps
            else:
                v_hat = v[i] / bc2
                denom = math.sqrt(v_hat) + eps
            new_params.append(params[i] - lr * m_hat / denom)
        params = new_params
        trajectory.append(list(params))
    return trajectory


# ---------------------------------------------------------------------------
# AdamW
# Reference: Loshchilov & Hutter (ICLR 2019); torch.optim.AdamW
# ---------------------------------------------------------------------------


def run_adamw(
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> list[list[float]]:
    beta1, beta2 = betas
    params = list(INIT_PARAMS)
    m = [0.0] * N_PARAMS
    v = [0.0] * N_PARAMS
    trajectory = [list(params)]
    for step in range(1, N_STEPS + 1):
        g = list(GRADS[step - 1])  # No L2 in gradient — decoupled decay
        m = [beta1 * m[i] + (1 - beta1) * g[i] for i in range(N_PARAMS)]
        v = [beta2 * v[i] + (1 - beta2) * g[i] ** 2 for i in range(N_PARAMS)]
        bc1 = 1 - beta1**step
        bc2 = 1 - beta2**step
        new_params = []
        for i in range(N_PARAMS):
            m_hat = m[i] / bc1
            v_hat = v[i] / bc2
            decayed = params[i] * (1 - lr * weight_decay)
            updated = decayed - lr * m_hat / (math.sqrt(v_hat) + eps)
            new_params.append(updated)
        params = new_params
        trajectory.append(list(params))
    return trajectory


# ---------------------------------------------------------------------------
# Adamax
# Reference: Kingma & Ba (ICLR 2015) Section 7; torch.optim.Adamax
# ---------------------------------------------------------------------------


def run_adamax(
    lr: float = 2e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> list[list[float]]:
    beta1, beta2 = betas
    params = list(INIT_PARAMS)
    m = [0.0] * N_PARAMS
    u = [0.0] * N_PARAMS  # exp infinity norm
    trajectory = [list(params)]
    for step in range(1, N_STEPS + 1):
        g = list(GRADS[step - 1])
        if weight_decay > 0:
            g = [g[i] + weight_decay * params[i] for i in range(N_PARAMS)]
        m = [beta1 * m[i] + (1 - beta1) * g[i] for i in range(N_PARAMS)]
        u = [max(beta2 * u[i], abs(g[i]) + eps) for i in range(N_PARAMS)]
        bc1 = 1 - beta1**step
        new_params = [params[i] - (lr / bc1) * m[i] / u[i] for i in range(N_PARAMS)]
        params = new_params
        trajectory.append(list(params))
    return trajectory


# ---------------------------------------------------------------------------
# NAdam
# Reference: Dozat (ICLR 2016 Workshop); torch.optim.NAdam
# ---------------------------------------------------------------------------


def run_nadam(
    lr: float = 2e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    momentum_decay: float = 4e-3,
) -> list[list[float]]:
    beta1, beta2 = betas
    params = list(INIT_PARAMS)
    m = [0.0] * N_PARAMS
    v = [0.0] * N_PARAMS
    mu_product = 1.0
    trajectory = [list(params)]
    for step in range(1, N_STEPS + 1):
        g = list(GRADS[step - 1])
        if weight_decay > 0:
            g = [g[i] + weight_decay * params[i] for i in range(N_PARAMS)]
        mu_t = beta1 * (1 - 0.5 * 0.96 ** (step * momentum_decay))
        mu_t1 = beta1 * (1 - 0.5 * 0.96 ** ((step + 1) * momentum_decay))
        mu_product *= mu_t
        mu_product_next = mu_product * mu_t1
        bc2 = 1 - beta2**step
        m = [beta1 * m[i] + (1 - beta1) * g[i] for i in range(N_PARAMS)]
        v = [beta2 * v[i] + (1 - beta2) * g[i] ** 2 for i in range(N_PARAMS)]
        new_params = []
        for i in range(N_PARAMS):
            # Rust: grad_component denominator is (1 - mu_product) after update,
            # momentum_component denominator is (1 - mu_product_next).
            m_hat = (
                mu_t1 * m[i] / (1 - mu_product_next)
                + (1 - mu_t) * g[i] / (1 - mu_product)
            )
            v_hat = v[i] / bc2
            new_params.append(params[i] - lr * m_hat / (math.sqrt(v_hat) + eps))
        params = new_params
        trajectory.append(list(params))
    return trajectory


# ---------------------------------------------------------------------------
# RAdam
# Reference: Liu et al. (ICLR 2020); torch.optim.RAdam
# ---------------------------------------------------------------------------


def run_radam(
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> list[list[float]]:
    beta1, beta2 = betas
    params = list(INIT_PARAMS)
    m = [0.0] * N_PARAMS
    v = [0.0] * N_PARAMS
    rho_inf = 2.0 / (1 - beta2) - 1
    trajectory = [list(params)]
    for step in range(1, N_STEPS + 1):
        g = list(GRADS[step - 1])
        if weight_decay > 0:
            g = [g[i] + weight_decay * params[i] for i in range(N_PARAMS)]
        m = [beta1 * m[i] + (1 - beta1) * g[i] for i in range(N_PARAMS)]
        v = [beta2 * v[i] + (1 - beta2) * g[i] ** 2 for i in range(N_PARAMS)]
        bc1 = 1 - beta1**step
        bc2 = 1 - beta2**step
        rho_t = rho_inf - 2 * step * beta2**step / bc2
        new_params = []
        for i in range(N_PARAMS):
            m_hat = m[i] / bc1
            if rho_t > 5:
                rect = math.sqrt(
                    (rho_t - 4)
                    * (rho_t - 2)
                    * rho_inf
                    / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                )
                v_hat = math.sqrt(v[i] / bc2) + eps
                new_params.append(params[i] - lr * rect * m_hat / v_hat)
            else:
                new_params.append(params[i] - lr * m_hat)
        params = new_params
        trajectory.append(list(params))
    return trajectory


# ---------------------------------------------------------------------------
# SparseAdam
# Reference: torch.optim.SparseAdam
# Dense inputs: equivalent to Adam without weight_decay/amsgrad.
# ---------------------------------------------------------------------------


def run_sparse_adam(
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> list[list[float]]:
    beta1, beta2 = betas
    params = list(INIT_PARAMS)
    m = [0.0] * N_PARAMS
    v = [0.0] * N_PARAMS
    trajectory = [list(params)]
    for step in range(1, N_STEPS + 1):
        g = list(GRADS[step - 1])
        m = [beta1 * m[i] + (1 - beta1) * g[i] for i in range(N_PARAMS)]
        v = [beta2 * v[i] + (1 - beta2) * g[i] ** 2 for i in range(N_PARAMS)]
        bc1 = 1 - beta1**step
        bc2 = 1 - beta2**step
        new_params = []
        for i in range(N_PARAMS):
            m_hat = m[i] / bc1
            v_hat = v[i] / bc2
            new_params.append(params[i] - lr * m_hat / (math.sqrt(v_hat) + eps))
        params = new_params
        trajectory.append(list(params))
    return trajectory


# ---------------------------------------------------------------------------
# Adafactor  (explicit lr, no relative step, no beta1)
# Reference: Shazeer & Stern (ICML 2018); torch.optim.Adafactor
# ---------------------------------------------------------------------------


def run_adafactor(
    lr: float = 1e-3,
    decay_rate: float = -0.8,
    eps_sq: float = 1e-30,
    eps_rms: float = 1e-3,
    weight_decay: float = 0.0,
) -> list[list[float]]:
    params = list(INIT_PARAMS)
    # 1-D params: no factoring, treat second moment as a full vector
    v = [0.0] * N_PARAMS
    trajectory = [list(params)]
    for step in range(1, N_STEPS + 1):
        g = list(GRADS[step - 1])
        # Rust: rho = min(1 - step^decay_rate, 1 - 1e-8)
        rho_t = min(1 - step**decay_rate, 1 - 1e-8)
        # Rust adds eps_sq to the gradient square inside the moving average:
        # full_sq[i] = rho * full_sq[i-1] + (1-rho) * (g^2 + eps_sq)
        v = [rho_t * v[i] + (1 - rho_t) * (g[i] ** 2 + eps_sq) for i in range(N_PARAMS)]
        rms_param = math.sqrt(sum(p**2 for p in params) / N_PARAMS)
        d = max(eps_rms, rms_param)
        new_params = []
        for i in range(N_PARAMS):
            # Rust non-factored path: u = g / (sqrt(full_sq) + 1e-30)
            # No RMS clipping is applied in this path.
            update = g[i] / (math.sqrt(v[i]) + 1e-30)
            updated = params[i] - lr * update
            if weight_decay > 0:
                updated -= lr * weight_decay * params[i]
            new_params.append(updated)
        params = new_params
        trajectory.append(list(params))
    return trajectory


# ---------------------------------------------------------------------------
# Adadelta
# Reference: Zeiler (2012); torch.optim.Adadelta
# ---------------------------------------------------------------------------


def run_adadelta(
    lr: float = 1.0,
    rho: float = 0.9,
    eps: float = 1e-6,
    weight_decay: float = 0.0,
) -> list[list[float]]:
    params = list(INIT_PARAMS)
    square_avg = [0.0] * N_PARAMS
    acc_delta = [0.0] * N_PARAMS
    trajectory = [list(params)]
    for step in range(1, N_STEPS + 1):
        g = list(GRADS[step - 1])
        if weight_decay > 0:
            g = [g[i] + weight_decay * params[i] for i in range(N_PARAMS)]
        square_avg = [
            rho * square_avg[i] + (1 - rho) * g[i] ** 2 for i in range(N_PARAMS)
        ]
        new_params = []
        new_acc_delta = []
        for i in range(N_PARAMS):
            std = math.sqrt(square_avg[i] + eps)
            delta = math.sqrt(acc_delta[i] + eps) / std * g[i]
            new_params.append(params[i] - lr * delta)
            new_acc_delta.append(rho * acc_delta[i] + (1 - rho) * delta**2)
        acc_delta = new_acc_delta
        params = new_params
        trajectory.append(list(params))
    return trajectory


# ---------------------------------------------------------------------------
# Assemble and serialise
# ---------------------------------------------------------------------------


def build_fixtures() -> dict:
    return {
        "metadata": {
            "torch_version": TORCH_VERSION,
            "generated_at": "2026-05-07T00:00:00+00:00",
            "rng_seed": RNG_SEED,
            "note": (
                "Trajectories computed analytically from documented algorithm, "
                "matching torch.optim " + TORCH_VERSION
            ),
        },
        "adam": [
            {
                "label": "adam_default",
                "lr": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adam(),
            },
            {
                "label": "adam_amsgrad",
                "lr": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": True,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adam(amsgrad=True),
            },
            {
                "label": "adam_weight_decay",
                "lr": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adam(weight_decay=0.01),
            },
            {
                "label": "adam_high_lr",
                "lr": 1e-2,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adam(lr=1e-2),
            },
        ],
        "adamw": [
            {
                "label": "adamw_default",
                "lr": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.01,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adamw(),
            },
            {
                "label": "adamw_no_decay",
                "lr": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adamw(weight_decay=0.0),
            },
            {
                "label": "adamw_high_decay",
                "lr": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.1,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adamw(weight_decay=0.1),
            },
        ],
        "adamax": [
            {
                "label": "adamax_default",
                "lr": 2e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adamax(),
            },
            {
                "label": "adamax_weight_decay",
                "lr": 2e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.01,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adamax(weight_decay=0.01),
            },
        ],
        "nadam": [
            {
                "label": "nadam_default",
                "lr": 2e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "momentum_decay": 4e-3,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_nadam(),
            },
        ],
        "radam": [
            {
                "label": "radam_default",
                "lr": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_radam(),
            },
        ],
        "sparse_adam": [
            {
                "label": "sparse_adam_default",
                "lr": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_sparse_adam(),
            },
        ],
        "adafactor": [
            {
                "label": "adafactor_explicit_lr",
                "lr": 1e-3,
                "beta1": None,
                "decay_rate": -0.8,
                "eps_sq": 1e-30,
                "eps_rms": 1e-3,
                "weight_decay": 0.0,
                "relative_step": False,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adafactor(),
            },
        ],
        "adadelta": [
            {
                "label": "adadelta_default",
                "lr": 1.0,
                "rho": 0.9,
                "eps": 1e-6,
                "weight_decay": 0.0,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adadelta(),
            },
            {
                "label": "adadelta_weight_decay",
                "lr": 1.0,
                "rho": 0.9,
                "eps": 1e-6,
                "weight_decay": 0.01,
                "init_params": INIT_PARAMS,
                "grads": GRADS,
                "trajectory": run_adadelta(weight_decay=0.01),
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = (
        Path(__file__).parent.parent
        / "ferrotorch-optim"
        / "tests"
        / "conformance"
        / "fixtures_adam_family.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help=f"Output path (default: {default_out})",
    )
    args = parser.parse_args()

    fixtures = build_fixtures()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2)
        f.write("\n")

    n_total = sum(
        len(v) for k, v in fixtures.items() if k != "metadata" and isinstance(v, list)
    )
    counts = ", ".join(
        f"{k}:{len(v)}"
        for k, v in fixtures.items()
        if k != "metadata" and isinstance(v, list)
    )
    print(
        f"Wrote {n_total} fixture groups ({counts}) to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
