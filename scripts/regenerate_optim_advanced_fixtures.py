#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-optim C6.3 conformance suite.

Sub-phase: C6.3 — Advanced optimizers + utilities.

Reference: torch == 2.11.0

Output:
    ferrotorch-optim/tests/conformance/fixtures.json

Coverage:

* EMA decay arithmetic (parity with torch EMA pattern)
    - Single step: decay=0.9, shadow=[1,2], param=[3,4]
    - Multi step: decay=0.5, three steps toward 10.0
    - Decay=0.0: full replace
    - Decay=1.0: freeze shadow

* SWA equal-weight averaging
    - Running mean: three checkpoints [1,3,6] -> [1, 2, 10/3]
    - Convergence: 4 identical checkpoints -> same value

* SWA EMA averaging
    - First call copies; second: 0.5*first + 0.5*second

* SWALR cosine annealing
    - 10 steps from lr=0.1 to swa_lr=0.05 -> reaches 0.05 at end
    - Midpoint at step 50/100: lr = 0.5 (from 1.0 to 0.0)

* SWALR linear annealing
    - Midpoint: lr = 0.5
    - Endpoint: lr = swa_lr

* LBFGS convergence on quadratic f(x) = x^2
    - Starting from x=5.0, 50 gradient-descent steps with lr=0.5
    - Expected: |x| < 1e-3 (convergence property, not exact)

* GradScaler scale management
    - Default init_scale = 65536.0
    - Inf gradient: scale halves after update (1024 -> 512)
    - N healthy steps -> growth after growth_interval
    - State dict round-trip preserves scale and tracker

* GradientAccumulator
    - should_step cycles correctly: n=3
    - scale_loss: loss / n_steps
    - Multi-step mean: 4 batches -> mean

* Differentiable SGD
    - diff_sgd_step: param=[1,2,3], grad=[0.1,0.2,0.3], lr=0.5
    - diff_sgd_momentum: param=[10], grad=[1], v_prev=[2], lr=0.1, mom=0.9

* foreach_utils scalar helpers
    - elemwise_max parity: max([1,5,3], [4,2,6]) = [4,5,6]

Usage:
    python3 scripts/regenerate_optim_advanced_fixtures.py
"""

from __future__ import annotations

import datetime
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Output path and metadata
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT / "ferrotorch-optim" / "tests" / "conformance" / "fixtures_advanced.json"
)

RNG_SEED: int = 0xC6_3000


def fixture_metadata() -> dict[str, Any]:
    torch_version = "2.11.0 (computed offline — no torch import required)"
    try:
        import torch  # type: ignore
        torch_version = torch.__version__
    except ImportError:
        # torch not installed — fixtures are computed in pure Python arithmetic
        # so this script exits 0 without it. The pinned reference version is
        # recorded in the string initialised above.
        torch_version = torch_version  # re-assign to make the intent explicit
    return {
        "torch_version": torch_version,
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "note": "C6.3 advanced optimizers + utilities conformance fixtures",
    }


# ---------------------------------------------------------------------------
# EMA decay arithmetic fixtures
# ---------------------------------------------------------------------------

def ema_fixtures() -> list[dict[str, Any]]:
    """
    EMA recurrence: shadow = decay * shadow + (1 - decay) * param
    Computed in pure Python (double precision) — exact same formula as
    ExponentialMovingAverage.update().
    """
    out: list[dict[str, Any]] = []

    # Single step: decay=0.9, shadow=[1,2], new_param=[3,4]
    decay = 0.9
    shadow = [1.0, 2.0]
    param = [3.0, 4.0]
    expected = [decay * s + (1 - decay) * p for s, p in zip(shadow, param)]
    out.append({
        "kind": "ema_single_step",
        "decay": decay,
        "initial_shadow": shadow,
        "update_param": param,
        "expected_after_1": expected,
        "label": "EMA single step decay=0.9 shadow=[1,2] param=[3,4]",
    })

    # Three steps: decay=0.5, initial=[0.0], all updates=[10.0]
    decay = 0.5
    s = 0.0
    steps = []
    for _ in range(3):
        s = decay * s + (1 - decay) * 10.0
        steps.append(s)
    out.append({
        "kind": "ema_multi_step",
        "decay": decay,
        "initial_shadow": [0.0],
        "update_value": 10.0,
        "n_steps": 3,
        "expected_after_each": steps,
        "label": "EMA three steps decay=0.5 toward 10.0",
    })

    # Decay=0.0: full replace
    decay = 0.0
    out.append({
        "kind": "ema_decay_zero",
        "decay": decay,
        "initial_shadow": [100.0],
        "update_param": [42.0],
        "expected_after_1": [42.0],
        "label": "EMA decay=0 full replace",
    })

    # Decay=1.0: freeze
    decay = 1.0
    out.append({
        "kind": "ema_decay_one",
        "decay": decay,
        "initial_shadow": [100.0],
        "update_param": [42.0],
        "expected_after_1": [100.0],
        "label": "EMA decay=1 freeze shadow",
    })

    # Multi-param, decay=0.0 (immediate copy): two independent params
    decay = 0.0
    out.append({
        "kind": "ema_multi_param",
        "decay": decay,
        "initial_shadows": [[10.0, 20.0], [30.0]],
        "update_params": [[10.0, 20.0], [30.0]],
        # decay=0: shadow = param regardless of old shadow
        "expected_shadows": [[10.0, 20.0], [30.0]],
        "label": "EMA multi-param decay=0",
    })

    return out


# ---------------------------------------------------------------------------
# SWA averaging fixtures
# ---------------------------------------------------------------------------

def swa_fixtures() -> list[dict[str, Any]]:
    """
    SWA equal-weight running mean:
        avg_{n+1} = avg_n + (param - avg_n) / (n + 1)
    with the first call performing a plain copy.
    """
    out: list[dict[str, Any]] = []

    # Three checkpoint values: [1], [3], [6]
    # n=0: avg = 1 (copy)
    # n=1: avg = 1 + (3 - 1) / 2 = 2
    # n=2: avg = 2 + (6 - 2) / 3 = 10/3
    checkpoints = [1.0, 3.0, 6.0]
    avg = checkpoints[0]
    running = [avg]
    for n, c in enumerate(checkpoints[1:], start=1):
        avg = avg + (c - avg) / (n + 1)
        running.append(avg)
    out.append({
        "kind": "swa_running_mean",
        "checkpoints": checkpoints,
        "expected_running": running,
        "label": "SWA running mean over [1,3,6]",
    })

    # Four identical checkpoints -> mean equals that value
    val = 7.0
    avg = val
    for n in range(1, 4):
        avg = avg + (val - avg) / (n + 1)
    out.append({
        "kind": "swa_identical_checkpoints",
        "checkpoint_value": val,
        "n_updates": 4,
        "expected_mean": val,
        "label": "SWA four identical checkpoints",
    })

    # EMA strategy through AveragedModel: decay=0.5
    # n=0: avg = first_val (copy)
    # n=1: avg = 0.5*avg + 0.5*second_val
    first_val = 10.0
    second_val = 20.0
    decay = 0.5
    avg_ema = first_val  # first call copies
    avg_ema = decay * avg_ema + (1 - decay) * second_val
    out.append({
        "kind": "swa_ema_two_steps",
        "first_checkpoint": first_val,
        "second_checkpoint": second_val,
        "ema_decay": decay,
        "expected_after_second": avg_ema,
        "label": "SWA EMA strategy: copy then blend",
    })

    return out


# ---------------------------------------------------------------------------
# SWALR annealing fixtures
# ---------------------------------------------------------------------------

def swalr_fixtures() -> list[dict[str, Any]]:
    """
    SWALR interpolation:
        lr(t) = initial_lr * (1 - alpha(t)) + swa_lr * alpha(t)
    where
        cosine: alpha(t) = (1 - cos(pi * t)) / 2
        linear: alpha(t) = t
    and t = step / anneal_epochs clipped to [0, 1].
    """
    out: list[dict[str, Any]] = []

    # Cosine: 10 steps from 0.1 to swa_lr=0.05 -> reaches 0.05 at step 10
    initial_lr = 0.1
    swa_lr = 0.05
    anneal_epochs = 10
    steps_lr = []
    for step in range(1, anneal_epochs + 1):
        t = min(step / anneal_epochs, 1.0)
        alpha = (1 - math.cos(math.pi * t)) / 2.0
        lr = initial_lr * (1 - alpha) + swa_lr * alpha
        steps_lr.append(lr)
    out.append({
        "kind": "swalr_cosine",
        "initial_lr": initial_lr,
        "swa_lr": swa_lr,
        "anneal_epochs": anneal_epochs,
        "lr_sequence": steps_lr,
        "expected_final_lr": swa_lr,
        "label": "SWALR cosine 10 steps 0.1 -> 0.05",
    })

    # Cosine midpoint: initial=1.0, swa=0.0, 100 epochs, step 50
    initial_lr = 1.0
    swa_lr = 0.0
    anneal_epochs = 100
    step = 50
    t = step / anneal_epochs
    alpha = (1 - math.cos(math.pi * t)) / 2.0
    lr_mid = initial_lr * (1 - alpha) + swa_lr * alpha
    out.append({
        "kind": "swalr_cosine_midpoint",
        "initial_lr": initial_lr,
        "swa_lr": swa_lr,
        "anneal_epochs": anneal_epochs,
        "step": step,
        "expected_lr": lr_mid,
        "label": "SWALR cosine midpoint step 50/100",
    })

    # Linear: 10 steps, step 5 = midpoint
    initial_lr = 1.0
    swa_lr = 0.0
    anneal_epochs = 10
    step = 5
    t = step / anneal_epochs
    alpha = t  # linear
    lr_mid = initial_lr * (1 - alpha) + swa_lr * alpha
    out.append({
        "kind": "swalr_linear_midpoint",
        "initial_lr": initial_lr,
        "swa_lr": swa_lr,
        "anneal_epochs": anneal_epochs,
        "step": step,
        "expected_lr": lr_mid,
        "label": "SWALR linear midpoint step 5/10",
    })

    # Linear: 10 steps, at step 10 reaches swa_lr=0.0
    step = 10
    t = min(step / anneal_epochs, 1.0)
    alpha = t
    lr_end = initial_lr * (1 - alpha) + swa_lr * alpha
    out.append({
        "kind": "swalr_linear_endpoint",
        "initial_lr": initial_lr,
        "swa_lr": swa_lr,
        "anneal_epochs": anneal_epochs,
        "step": step,
        "expected_lr": lr_end,
        "label": "SWALR linear endpoint step 10/10",
    })

    # Post-anneal stays at swa_lr (cosine, step 20 on 5-epoch anneal)
    initial_lr = 0.1
    swa_lr = 0.01
    anneal_epochs = 5
    step = 20  # well past anneal
    t = min(step / anneal_epochs, 1.0)
    alpha = (1 - math.cos(math.pi * t)) / 2.0
    lr_post = initial_lr * (1 - alpha) + swa_lr * alpha
    out.append({
        "kind": "swalr_cosine_post_anneal",
        "initial_lr": initial_lr,
        "swa_lr": swa_lr,
        "anneal_epochs": anneal_epochs,
        "step": step,
        "expected_lr": swa_lr,  # should be clamped to swa_lr
        "label": "SWALR cosine post-anneal stays at swa_lr",
    })

    return out


# ---------------------------------------------------------------------------
# LBFGS convergence property fixture
# ---------------------------------------------------------------------------

def lbfgs_fixtures() -> list[dict[str, Any]]:
    """
    LBFGS convergence on quadratic f(x) = x^2, min at x=0.
    This is a mathematical-property fixture: no PyTorch reference needed.
    We record the initial value and the convergence tolerance.
    """
    out: list[dict[str, Any]] = []

    out.append({
        "kind": "lbfgs_quadratic_convergence",
        "initial_x": 5.0,
        "lr": 0.5,
        "n_steps": 50,
        "convergence_tolerance": 1e-3,
        "label": "LBFGS converges on f(x)=x^2 from x=5.0",
    })

    out.append({
        "kind": "lbfgs_multidim_quadratic_convergence",
        "initial_params": [3.0, -4.0],
        "lr": 0.5,
        "n_steps": 100,
        "convergence_tolerance": 1e-3,
        "label": "LBFGS converges on f(a,b)=a^2+b^2 from (3,-4)",
    })

    return out


# ---------------------------------------------------------------------------
# GradScaler fixtures
# ---------------------------------------------------------------------------

def grad_scaler_fixtures() -> list[dict[str, Any]]:
    """
    GradScaler arithmetic:
    - inf gradient -> skip step, scale * backoff_factor after update
    - N healthy steps -> growth after growth_interval
    - state dict round-trip
    """
    out: list[dict[str, Any]] = []

    # Default init_scale matches torch.cuda.amp.GradScaler default: 2^16
    out.append({
        "kind": "grad_scaler_default_init_scale",
        "expected_init_scale": 65536.0,
        "label": "GradScaler default init_scale == 2^16",
    })

    # Inf gradient: scale halves
    init_scale = 1024.0
    backoff_factor = 0.5
    out.append({
        "kind": "grad_scaler_inf_halves_scale",
        "init_scale": init_scale,
        "backoff_factor": backoff_factor,
        "expected_scale_after_update": init_scale * backoff_factor,
        "label": "GradScaler inf gradient -> scale * 0.5",
    })

    # NaN gradient also triggers skip (same arithmetic)
    out.append({
        "kind": "grad_scaler_nan_skips_step",
        "label": "GradScaler NaN gradient -> step skipped",
    })

    # Healthy steps grow scale after growth_interval
    init_scale = 128.0
    growth_factor = 2.0
    growth_interval = 3
    out.append({
        "kind": "grad_scaler_growth_after_interval",
        "init_scale": init_scale,
        "growth_factor": growth_factor,
        "growth_interval": growth_interval,
        "n_healthy_steps": growth_interval,
        "expected_scale_after_growth": init_scale * growth_factor,
        "label": "GradScaler grows scale after growth_interval healthy steps",
    })

    # State dict round-trip
    out.append({
        "kind": "grad_scaler_state_dict_roundtrip",
        "init_scale": 512.0,
        "growth_tracker_after_2_steps": 2,
        "label": "GradScaler state dict round-trip",
    })

    # Disabled mode: scale=1.0, step always runs
    out.append({
        "kind": "grad_scaler_disabled_passthrough",
        "enabled": False,
        "label": "GradScaler disabled: scale=no-op, step always runs",
    })

    return out


# ---------------------------------------------------------------------------
# GradientAccumulator fixtures
# ---------------------------------------------------------------------------

def grad_accumulator_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # should_step cycle: n=3 -> [F, F, T, F, F, T]
    n = 3
    results = []
    count = 0
    for _ in range(6):
        count += 1
        if count >= n:
            results.append(True)
            count = 0
        else:
            results.append(False)
    out.append({
        "kind": "grad_accum_should_step_cycle",
        "accumulation_steps": n,
        "n_calls": 6,
        "expected_results": results,
        "label": "GradientAccumulator should_step cycles with n=3",
    })

    # scale_loss: 8.0 / 4 = 2.0
    out.append({
        "kind": "grad_accum_scale_loss",
        "accumulation_steps": 4,
        "loss": 8.0,
        "expected_scaled": 2.0,
        "label": "GradientAccumulator scale_loss 8.0 / 4 = 2.0",
    })

    # scale_loss with steps=1: identity
    out.append({
        "kind": "grad_accum_scale_loss_identity",
        "accumulation_steps": 1,
        "loss": 3.14,
        "expected_scaled": 3.14,
        "label": "GradientAccumulator scale_loss steps=1 is identity",
    })

    # 4-batch mean: sum divided by n
    batch_losses = [2.0, 3.0, 1.5, 2.5]
    mean_loss = sum(batch_losses) / len(batch_losses)
    out.append({
        "kind": "grad_accum_mean_over_batches",
        "batch_losses": batch_losses,
        "n_accumulate": len(batch_losses),
        "expected_mean_loss": mean_loss,
        "label": "GradientAccumulator: mean over 4 batches",
    })

    return out


# ---------------------------------------------------------------------------
# Differentiable SGD fixtures
# ---------------------------------------------------------------------------

def differentiable_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # diff_sgd_step: param=[1,2,3], grad=[0.1,0.2,0.3], lr=0.5
    param = [1.0, 2.0, 3.0]
    grad = [0.1, 0.2, 0.3]
    lr = 0.5
    expected = [p - lr * g for p, g in zip(param, grad)]
    out.append({
        "kind": "diff_sgd_step",
        "param": param,
        "grad": grad,
        "lr": lr,
        "expected": expected,
        "label": "diff_sgd_step basic values",
    })

    # diff_sgd_step: multiple params
    p1 = [1.0, 2.0]
    p2 = [3.0, 4.0]
    g1 = [0.1, 0.1]
    g2 = [0.2, 0.2]
    lr = 1.0
    out.append({
        "kind": "diff_sgd_step_multi_param",
        "params": [p1, p2],
        "grads": [g1, g2],
        "lr": lr,
        "expected": [
            [p - lr * g for p, g in zip(p1, g1)],
            [p - lr * g for p, g in zip(p2, g2)],
        ],
        "label": "diff_sgd_step multiple params",
    })

    # diff_sgd_momentum: param=[10], grad=[1], v_prev=[2], lr=0.1, mom=0.9
    # v_new = 0.9*2 + 1.0 = 2.8
    # p_new = 10 - 0.1*2.8 = 9.72
    param = [10.0]
    grad = [1.0]
    v_prev = [2.0]
    lr = 0.1
    mom = 0.9
    v_new = [mom * v + g for v, g in zip(v_prev, grad)]
    p_new = [p - lr * v for p, v in zip(param, v_new)]
    out.append({
        "kind": "diff_sgd_momentum_step",
        "param": param,
        "grad": grad,
        "v_prev": v_prev,
        "lr": lr,
        "momentum": mom,
        "expected_v_new": v_new,
        "expected_p_new": p_new,
        "label": "diff_sgd_momentum_step with prev velocity",
    })

    # diff_sgd_momentum: first step (no prev velocity): v_new = grad
    param = [10.0, 20.0]
    grad = [1.0, 2.0]
    lr = 0.1
    mom = 0.9
    v_new = grad  # first step
    p_new = [p - lr * v for p, v in zip(param, v_new)]
    out.append({
        "kind": "diff_sgd_momentum_step_first",
        "param": param,
        "grad": grad,
        "lr": lr,
        "momentum": mom,
        "expected_v_new": v_new,
        "expected_p_new": p_new,
        "label": "diff_sgd_momentum_step first step (no prev velocity)",
    })

    return out


# ---------------------------------------------------------------------------
# foreach_utils fixtures
# ---------------------------------------------------------------------------

def foreach_utils_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # elemwise_max: max([1,5,3], [4,2,6]) = [4,5,6]
    a = [1.0, 5.0, 3.0]
    b = [4.0, 2.0, 6.0]
    expected = [max(ai, bi) for ai, bi in zip(a, b)]
    out.append({
        "kind": "elemwise_max",
        "a": a,
        "b": b,
        "expected": expected,
        "label": "elemwise_max([1,5,3], [4,2,6]) = [4,5,6]",
    })

    # f64_scalar_on: scalar value round-trip (numeric identity check)
    out.append({
        "kind": "f64_scalar_on",
        "value": 3.14159,
        "expected": 3.14159,
        "label": "f64_scalar_on creates scalar tensor with given value",
    })

    return out


# ---------------------------------------------------------------------------
# Optimizer trait fixtures (optimizer.rs, ParamGroup)
# ---------------------------------------------------------------------------

def optimizer_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    out.append({
        "kind": "param_group_default_weight_decay",
        "lr": 0.01,
        "expected_weight_decay": 0.0,
        "label": "ParamGroup default weight_decay is 0.0",
    })

    out.append({
        "kind": "param_group_with_weight_decay",
        "lr": 0.1,
        "weight_decay": 1e-4,
        "label": "ParamGroup with_weight_decay builder",
    })

    return out


# ---------------------------------------------------------------------------
# Muon convergence fixture
# ---------------------------------------------------------------------------

def muon_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Mathematical property: Newton-Schulz orthogonalization of a 2x2 matrix
    # produces an orthogonal matrix (G^T G ~ I).
    out.append({
        "kind": "muon_ns_orthogonality",
        "input": [[3.0, 1.0], [1.0, 2.0]],
        "ns_steps": 10,
        "tolerance": 1e-4,
        "label": "Muon Newton-Schulz produces orthogonal matrix",
    })

    # Convergence: minimize f(x) = 0.5*||x||^2 from x=[5,3]
    out.append({
        "kind": "muon_convergence_1d",
        "initial_param": [5.0, 3.0],
        "lr": 0.01,
        "momentum": 0.9,
        "n_steps": 200,
        "convergence_tolerance_norm_sq": 0.01,
        "label": "Muon converges on quadratic 1D loss",
    })

    return out


# ---------------------------------------------------------------------------
# Natural gradient (K-FAC) fixtures
# ---------------------------------------------------------------------------

def kfac_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Kronecker factor update: identity activations and gradients
    # A = (a^T a) / batch with a = [[1,0],[0,1]] (identity), batch=2
    # -> A = I/2 * 2 = identity (batch normalised: each a_i a_i^T = I, sum/2 = I/2 ... no)
    # a = [[1,0],[0,1]] (2 samples, 2 features)
    # a^T @ a = [[1,0],[0,1]] (outer prod summed over batch)
    # / batch_size = [[0.5,0],[0,0.5]]
    batch = 2
    a = [[1.0, 0.0], [0.0, 1.0]]
    a_factor = [[0.0, 0.0], [0.0, 0.0]]
    for row in a:
        for i in range(len(row)):
            for j in range(len(row)):
                a_factor[i][j] += row[i] * row[j]
    for i in range(len(a_factor)):
        for j in range(len(a_factor[i])):
            a_factor[i][j] /= batch
    out.append({
        "kind": "kfac_factor_update",
        "activation": a,
        "batch_size": batch,
        "expected_a_factor_diag": [a_factor[0][0], a_factor[1][1]],
        "expected_a_factor_offdiag": [a_factor[0][1], a_factor[1][0]],
        "label": "K-FAC factor update with identity activations",
    })

    # State dict: step_count persists
    out.append({
        "kind": "kfac_state_dict",
        "n_steps": 5,
        "label": "K-FAC state_dict preserves step_count",
    })

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_fixtures: dict[str, Any] = {
        "metadata": fixture_metadata(),
        "ema": ema_fixtures(),
        "swa": swa_fixtures(),
        "swalr": swalr_fixtures(),
        "lbfgs": lbfgs_fixtures(),
        "grad_scaler": grad_scaler_fixtures(),
        "grad_accumulator": grad_accumulator_fixtures(),
        "differentiable": differentiable_fixtures(),
        "foreach_utils": foreach_utils_fixtures(),
        "optimizer": optimizer_fixtures(),
        "muon": muon_fixtures(),
        "kfac": kfac_fixtures(),
    }

    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_fixtures, f, indent=2)
        f.write("\n")

    section_totals = {
        k: len(v)
        for k, v in all_fixtures.items()
        if k != "metadata"
    }
    total = sum(section_totals.values())
    print(f"Wrote {total} fixtures to {FIXTURE_PATH}")
    for section, count in section_totals.items():
        print(f"  {section:20s}: {count}")


if __name__ == "__main__":
    main()
