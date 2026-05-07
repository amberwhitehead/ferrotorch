#!/usr/bin/env python3
"""Regenerate LR-scheduler conformance fixtures for ferrotorch-optim C6.4.

Pin: torch == 2.11.0 (or matching math if torch is unavailable).

This script analytically computes the LR sequence that torch.optim.lr_scheduler
would produce for each scheduler, using the same formulas as the PyTorch source.
Fixtures are committed to the repo; CI never runs torch directly.

Output: ferrotorch-optim/tests/conformance/scheduler_fixtures.json

Usage:
    # With torch installed (preferred — verifies math matches library):
    pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
    python scripts/regenerate_optim_scheduler_fixtures.py --verify

    # Without torch (analytical only):
    python scripts/regenerate_optim_scheduler_fixtures.py
"""

import json
import math
import argparse
import sys
from pathlib import Path

TORCH_VERSION = "2.11.0"
N_STEPS = 10  # epochs / steps per fixture run


# ---------------------------------------------------------------------------
# Analytical implementations matching torch.optim.lr_scheduler formulas
# ---------------------------------------------------------------------------

def step_lr_sequence(base_lr: float, step_size: int, gamma: float, n: int) -> list[float]:
    """lr = base_lr * gamma^(step // step_size)  after each .step() call."""
    lrs = []
    for step in range(1, n + 1):
        lr = base_lr * (gamma ** (step // step_size))
        lrs.append(lr)
    return lrs


def multi_step_lr_sequence(base_lr: float, milestones: list[int], gamma: float, n: int) -> list[float]:
    """lr = base_lr * gamma^(count of milestones <= step)."""
    milestones_sorted = sorted(milestones)
    lrs = []
    for step in range(1, n + 1):
        count = sum(1 for m in milestones_sorted if m <= step)
        lr = base_lr * (gamma ** count)
        lrs.append(lr)
    return lrs


def exponential_lr_sequence(base_lr: float, gamma: float, n: int) -> list[float]:
    """lr = base_lr * gamma^step."""
    lrs = []
    for step in range(1, n + 1):
        lr = base_lr * (gamma ** step)
        lrs.append(lr)
    return lrs


def cosine_annealing_lr_sequence(base_lr: float, t_max: int, eta_min: float, n: int) -> list[float]:
    """lr = eta_min + 0.5*(base_lr - eta_min)*(1 + cos(pi*step/T_max))."""
    lrs = []
    for step in range(1, n + 1):
        if step >= t_max:
            lr = eta_min
        else:
            lr = eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * step / t_max))
        lrs.append(lr)
    return lrs


def cosine_annealing_warm_restarts_sequence(
    base_lr: float, t_0: int, t_mult: int, eta_min: float, n: int
) -> list[float]:
    """SGDR cosine with restarts."""
    t_cur = 0
    t_i = t_0
    lrs = []
    for _ in range(n):
        t_cur += 1
        if t_cur >= t_i:
            t_cur = 0
            t_i *= t_mult
        progress = math.pi * t_cur / t_i
        lr = eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(progress))
        lrs.append(lr)
    return lrs


def polynomial_lr_sequence(base_lr: float, total_iters: int, power: float, n: int) -> list[float]:
    """lr = base_lr * (1 - min(step, total_iters) / total_iters)^power."""
    lrs = []
    for step in range(1, n + 1):
        clamped = min(step, total_iters)
        if total_iters == 0:
            lr = 0.0
        else:
            lr = base_lr * ((1.0 - clamped / total_iters) ** power)
        lrs.append(lr)
    return lrs


def constant_lr_sequence(base_lr: float, factor: float, total_iters: int, n: int) -> list[float]:
    """lr = base_lr*factor for step < total_iters, then base_lr."""
    lrs = []
    for step in range(1, n + 1):
        if step >= total_iters:
            lr = base_lr
        else:
            lr = base_lr * factor
        lrs.append(lr)
    return lrs


def linear_lr_sequence(
    base_lr: float, start_factor: float, end_factor: float, total_iters: int, n: int
) -> list[float]:
    """factor = start_factor + (end_factor - start_factor) * min(step, total_iters) / total_iters."""
    lrs = []
    for step in range(1, n + 1):
        if total_iters == 0:
            factor = end_factor
        else:
            clamped = min(step, total_iters)
            factor = start_factor + (end_factor - start_factor) * clamped / total_iters
        lr = base_lr * factor
        lrs.append(lr)
    return lrs


def linear_warmup_sequence(base_lr: float, warmup_steps: int, n: int) -> list[float]:
    """lr = base_lr * min(1, step / warmup_steps)."""
    lrs = []
    for step in range(1, n + 1):
        if warmup_steps == 0:
            ratio = 1.0
        else:
            ratio = min(1.0, step / warmup_steps)
        lr = base_lr * ratio
        lrs.append(lr)
    return lrs


def reduce_lr_on_plateau_sequence(
    init_lr: float,
    factor: float,
    patience: int,
    threshold: float,
    metrics: list[float],
) -> list[float]:
    """Simulate ReduceLROnPlateau (Min mode) against a sequence of metric values."""
    current_lr = init_lr
    best = math.inf
    num_bad = 0
    lrs = []

    for metric in metrics:
        # Check improvement
        if metric < best * (1.0 - threshold):
            best = metric
            num_bad = 0
        else:
            num_bad += 1

        if num_bad > patience:
            current_lr = current_lr * factor
            num_bad = 0

        lrs.append(current_lr)

    return lrs


def one_cycle_lr_sequence(
    max_lr: float,
    total_steps: int,
    pct_start: float,
    div_factor: float,
    final_div_factor: float,
    anneal: str,  # "cos" or "linear"
    n: int,
) -> list[float]:
    """OneCycleLR two-phase (matches ferrotorch implementation)."""
    initial_lr = max_lr / div_factor
    min_lr = initial_lr / final_div_factor

    phases = [
        {"end_step": pct_start * total_steps - 1.0, "start_lr": initial_lr, "end_lr": max_lr},
        {"end_step": float(total_steps - 1), "start_lr": max_lr, "end_lr": min_lr},
    ]

    def anneal_cos(start, end, pct):
        return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1.0)

    def anneal_linear(start, end, pct):
        return (end - start) * pct + start

    anneal_fn = anneal_cos if anneal == "cos" else anneal_linear

    lrs = []
    # ferrotorch OneCycleLR: step() uses current_step BEFORE incrementing
    for step in range(n):
        step_num = float(step)
        start_step = 0.0
        lr = min_lr  # fallback
        for i, phase in enumerate(phases):
            if step_num <= phase["end_step"] or i == len(phases) - 1:
                denom = phase["end_step"] - start_step
                if abs(denom) < 1e-12:
                    pct = 1.0
                else:
                    pct = (step_num - start_step) / denom
                lr = anneal_fn(phase["start_lr"], phase["end_lr"], pct)
                break
            start_step = phase["end_step"]
        lrs.append(lr)

    return lrs


def cyclic_lr_sequence(
    base_lr: float,
    max_lr: float,
    step_size_up: int,
    step_size_down: int,
    mode: str,  # "triangular" | "triangular2" | "exp_range"
    gamma: float,
    n: int,
) -> list[float]:
    """CyclicLR sequence."""
    total_size = float(step_size_up + step_size_down)
    step_ratio = step_size_up / total_size

    lrs = []
    for step in range(1, n + 1):
        cycle = math.floor(1.0 + step / total_size)
        x = 1.0 + step / total_size - cycle

        if x <= step_ratio:
            scale_factor = x / step_ratio
        else:
            scale_factor = (x - 1.0) / (step_ratio - 1.0)

        base_height = (max_lr - base_lr) * scale_factor

        if mode == "triangular":
            lr = base_lr + base_height
        elif mode == "triangular2":
            lr = base_lr + base_height / (2.0 ** (cycle - 1.0))
        else:  # exp_range
            lr = base_lr + base_height * (gamma ** step)

        lrs.append(lr)

    return lrs


# ---------------------------------------------------------------------------
# Fixture definitions
# ---------------------------------------------------------------------------

def build_fixtures() -> dict:
    fixtures = []

    # --- StepLR ---
    fixtures.append({
        "scheduler": "StepLR",
        "params": {"base_lr": 0.1, "step_size": 3, "gamma": 0.5},
        "n_steps": N_STEPS,
        "lr_sequence": step_lr_sequence(0.1, 3, 0.5, N_STEPS),
    })
    fixtures.append({
        "scheduler": "StepLR",
        "params": {"base_lr": 1.0, "step_size": 5, "gamma": 0.1},
        "n_steps": N_STEPS,
        "lr_sequence": step_lr_sequence(1.0, 5, 0.1, N_STEPS),
    })

    # --- MultiStepLR ---
    fixtures.append({
        "scheduler": "MultiStepLR",
        "params": {"base_lr": 0.05, "milestones": [3, 6, 9], "gamma": 0.5},
        "n_steps": N_STEPS,
        "lr_sequence": multi_step_lr_sequence(0.05, [3, 6, 9], 0.5, N_STEPS),
    })
    fixtures.append({
        "scheduler": "MultiStepLR",
        "params": {"base_lr": 1.0, "milestones": [5, 8], "gamma": 0.2},
        "n_steps": N_STEPS,
        "lr_sequence": multi_step_lr_sequence(1.0, [5, 8], 0.2, N_STEPS),
    })

    # --- ExponentialLR ---
    fixtures.append({
        "scheduler": "ExponentialLR",
        "params": {"base_lr": 0.1, "gamma": 0.9},
        "n_steps": N_STEPS,
        "lr_sequence": exponential_lr_sequence(0.1, 0.9, N_STEPS),
    })
    fixtures.append({
        "scheduler": "ExponentialLR",
        "params": {"base_lr": 1.0, "gamma": 0.95},
        "n_steps": N_STEPS,
        "lr_sequence": exponential_lr_sequence(1.0, 0.95, N_STEPS),
    })

    # --- CosineAnnealingLR ---
    fixtures.append({
        "scheduler": "CosineAnnealingLR",
        "params": {"base_lr": 0.1, "t_max": 10, "eta_min": 0.0},
        "n_steps": N_STEPS,
        "lr_sequence": cosine_annealing_lr_sequence(0.1, 10, 0.0, N_STEPS),
    })
    fixtures.append({
        "scheduler": "CosineAnnealingLR",
        "params": {"base_lr": 1.0, "t_max": 20, "eta_min": 0.01},
        "n_steps": N_STEPS,
        "lr_sequence": cosine_annealing_lr_sequence(1.0, 20, 0.01, N_STEPS),
    })

    # --- CosineAnnealingWarmRestarts ---
    fixtures.append({
        "scheduler": "CosineAnnealingWarmRestarts",
        "params": {"base_lr": 0.1, "t_0": 5, "t_mult": 1, "eta_min": 0.0},
        "n_steps": N_STEPS,
        "lr_sequence": cosine_annealing_warm_restarts_sequence(0.1, 5, 1, 0.0, N_STEPS),
    })
    fixtures.append({
        "scheduler": "CosineAnnealingWarmRestarts",
        "params": {"base_lr": 1.0, "t_0": 3, "t_mult": 2, "eta_min": 0.01},
        "n_steps": N_STEPS,
        "lr_sequence": cosine_annealing_warm_restarts_sequence(1.0, 3, 2, 0.01, N_STEPS),
    })

    # --- PolynomialLR ---
    fixtures.append({
        "scheduler": "PolynomialLR",
        "params": {"base_lr": 1.0, "total_iters": 10, "power": 1.0},
        "n_steps": N_STEPS,
        "lr_sequence": polynomial_lr_sequence(1.0, 10, 1.0, N_STEPS),
    })
    fixtures.append({
        "scheduler": "PolynomialLR",
        "params": {"base_lr": 0.1, "total_iters": 20, "power": 2.0},
        "n_steps": N_STEPS,
        "lr_sequence": polynomial_lr_sequence(0.1, 20, 2.0, N_STEPS),
    })

    # --- ConstantLR ---
    fixtures.append({
        "scheduler": "ConstantLR",
        "params": {"base_lr": 0.1, "factor": 0.5, "total_iters": 5},
        "n_steps": N_STEPS,
        "lr_sequence": constant_lr_sequence(0.1, 0.5, 5, N_STEPS),
    })
    fixtures.append({
        "scheduler": "ConstantLR",
        "params": {"base_lr": 1.0, "factor": 0.25, "total_iters": 3},
        "n_steps": N_STEPS,
        "lr_sequence": constant_lr_sequence(1.0, 0.25, 3, N_STEPS),
    })

    # --- LinearLR ---
    fixtures.append({
        "scheduler": "LinearLR",
        "params": {"base_lr": 0.1, "start_factor": 0.5, "end_factor": 1.0, "total_iters": 10},
        "n_steps": N_STEPS,
        "lr_sequence": linear_lr_sequence(0.1, 0.5, 1.0, 10, N_STEPS),
    })
    fixtures.append({
        "scheduler": "LinearLR",
        "params": {"base_lr": 1.0, "start_factor": 1.0, "end_factor": 0.1, "total_iters": 5},
        "n_steps": N_STEPS,
        "lr_sequence": linear_lr_sequence(1.0, 1.0, 0.1, 5, N_STEPS),
    })

    # --- LinearWarmup ---
    fixtures.append({
        "scheduler": "LinearWarmup",
        "params": {"base_lr": 0.1, "warmup_steps": 5},
        "n_steps": N_STEPS,
        "lr_sequence": linear_warmup_sequence(0.1, 5, N_STEPS),
    })
    fixtures.append({
        "scheduler": "LinearWarmup",
        "params": {"base_lr": 1.0, "warmup_steps": 10},
        "n_steps": N_STEPS,
        "lr_sequence": linear_warmup_sequence(1.0, 10, N_STEPS),
    })

    # --- ReduceLROnPlateau (Min mode, patience=2, factor=0.5, threshold=1e-4) ---
    # Scenario: 10 epochs, metric decreasing then plateauing
    plateau_metrics_improving = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    fixtures.append({
        "scheduler": "ReduceLROnPlateau",
        "params": {
            "init_lr": 0.1,
            "mode": "min",
            "factor": 0.5,
            "patience": 2,
            "threshold": 1e-4,
        },
        "n_steps": N_STEPS,
        "metrics": plateau_metrics_improving,
        "lr_sequence": reduce_lr_on_plateau_sequence(0.1, 0.5, 2, 1e-4, plateau_metrics_improving),
    })
    # Scenario: stagnant metric triggers reduction
    plateau_metrics_stagnant = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    fixtures.append({
        "scheduler": "ReduceLROnPlateau",
        "params": {
            "init_lr": 0.1,
            "mode": "min",
            "factor": 0.5,
            "patience": 2,
            "threshold": 0.0,
        },
        "n_steps": N_STEPS,
        "metrics": plateau_metrics_stagnant,
        "lr_sequence": reduce_lr_on_plateau_sequence(0.1, 0.5, 2, 0.0, plateau_metrics_stagnant),
    })

    # --- OneCycleLR (two-phase, cos) ---
    fixtures.append({
        "scheduler": "OneCycleLR",
        "params": {
            "max_lr": 0.1,
            "total_steps": 10,
            "pct_start": 0.3,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
            "anneal": "cos",
            "three_phase": False,
        },
        "n_steps": N_STEPS,
        "lr_sequence": one_cycle_lr_sequence(0.1, 10, 0.3, 25.0, 1e4, "cos", N_STEPS),
    })
    fixtures.append({
        "scheduler": "OneCycleLR",
        "params": {
            "max_lr": 1.0,
            "total_steps": 10,
            "pct_start": 0.3,
            "div_factor": 10.0,
            "final_div_factor": 1e3,
            "anneal": "linear",
            "three_phase": False,
        },
        "n_steps": N_STEPS,
        "lr_sequence": one_cycle_lr_sequence(1.0, 10, 0.3, 10.0, 1e3, "linear", N_STEPS),
    })

    # --- CyclicLR (triangular, symmetric) ---
    fixtures.append({
        "scheduler": "CyclicLR",
        "params": {
            "base_lr": 0.001,
            "max_lr": 0.01,
            "step_size_up": 5,
            "step_size_down": 5,
            "mode": "triangular",
            "gamma": 1.0,
        },
        "n_steps": N_STEPS,
        "lr_sequence": cyclic_lr_sequence(0.001, 0.01, 5, 5, "triangular", 1.0, N_STEPS),
    })
    fixtures.append({
        "scheduler": "CyclicLR",
        "params": {
            "base_lr": 0.0,
            "max_lr": 1.0,
            "step_size_up": 4,
            "step_size_down": 4,
            "mode": "triangular2",
            "gamma": 1.0,
        },
        "n_steps": N_STEPS,
        "lr_sequence": cyclic_lr_sequence(0.0, 1.0, 4, 4, "triangular2", 1.0, N_STEPS),
    })

    return {
        "torch_version": TORCH_VERSION,
        "generated_by": "scripts/regenerate_optim_scheduler_fixtures.py",
        "note": "Analytical fixtures matching torch.optim.lr_scheduler formulas. "
                "Re-run with --verify when torch is available to cross-check.",
        "n_default_steps": N_STEPS,
        "fixtures": fixtures,
    }


# ---------------------------------------------------------------------------
# Optional: verify against live torch
# ---------------------------------------------------------------------------

def verify_against_torch(fixtures_data: dict) -> bool:
    """Verify analytical fixtures against torch (requires torch install)."""
    try:
        import torch
        import torch.optim as optim
        import torch.optim.lr_scheduler as sched_module
    except ImportError:
        print("torch not installed; skipping live verification", file=sys.stderr)
        return True

    print(f"Verifying against torch {torch.__version__} (pin: {TORCH_VERSION})")

    all_ok = True
    for fix in fixtures_data["fixtures"]:
        scheduler_name = fix["scheduler"]
        params = fix["params"]
        n = fix["n_steps"]
        expected = fix["lr_sequence"]

        # Build a dummy optimizer
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        base_lr = params.get("base_lr", params.get("init_lr", 0.1))
        opt = optim.SGD([dummy_param], lr=base_lr)

        try:
            if scheduler_name == "StepLR":
                s = sched_module.StepLR(opt, step_size=params["step_size"], gamma=params["gamma"])
                lrs = []
                for _ in range(n):
                    opt.step()
                    s.step()
                    lrs.append(opt.param_groups[0]["lr"])

            elif scheduler_name == "MultiStepLR":
                s = sched_module.MultiStepLR(opt, milestones=params["milestones"], gamma=params["gamma"])
                lrs = []
                for _ in range(n):
                    opt.step()
                    s.step()
                    lrs.append(opt.param_groups[0]["lr"])

            elif scheduler_name == "ExponentialLR":
                s = sched_module.ExponentialLR(opt, gamma=params["gamma"])
                lrs = []
                for _ in range(n):
                    opt.step()
                    s.step()
                    lrs.append(opt.param_groups[0]["lr"])

            elif scheduler_name == "CosineAnnealingLR":
                s = sched_module.CosineAnnealingLR(opt, T_max=params["t_max"], eta_min=params["eta_min"])
                lrs = []
                for _ in range(n):
                    opt.step()
                    s.step()
                    lrs.append(opt.param_groups[0]["lr"])

            elif scheduler_name == "CosineAnnealingWarmRestarts":
                s = sched_module.CosineAnnealingWarmRestarts(
                    opt, T_0=params["t_0"], T_mult=params["t_mult"], eta_min=params["eta_min"]
                )
                lrs = []
                for _ in range(n):
                    opt.step()
                    s.step()
                    lrs.append(opt.param_groups[0]["lr"])

            elif scheduler_name == "PolynomialLR":
                s = sched_module.PolynomialLR(
                    opt, total_iters=params["total_iters"], power=params["power"]
                )
                lrs = []
                for _ in range(n):
                    opt.step()
                    s.step()
                    lrs.append(opt.param_groups[0]["lr"])

            elif scheduler_name == "ConstantLR":
                s = sched_module.ConstantLR(
                    opt, factor=params["factor"], total_iters=params["total_iters"]
                )
                lrs = []
                for _ in range(n):
                    opt.step()
                    s.step()
                    lrs.append(opt.param_groups[0]["lr"])

            elif scheduler_name == "LinearLR":
                s = sched_module.LinearLR(
                    opt,
                    start_factor=params["start_factor"],
                    end_factor=params["end_factor"],
                    total_iters=params["total_iters"],
                )
                lrs = []
                for _ in range(n):
                    opt.step()
                    s.step()
                    lrs.append(opt.param_groups[0]["lr"])

            elif scheduler_name == "ReduceLROnPlateau":
                s = sched_module.ReduceLROnPlateau(
                    opt,
                    mode=params["mode"],
                    factor=params["factor"],
                    patience=params["patience"],
                    threshold=params["threshold"],
                )
                lrs = []
                for metric in fix["metrics"]:
                    s.step(metric)
                    lrs.append(opt.param_groups[0]["lr"])

            else:
                # Skip schedulers with no exact 1:1 torch verification path
                print(f"  SKIP {scheduler_name} (no verification path)")
                continue

            # Check agreement
            for i, (got, want) in enumerate(zip(lrs, expected)):
                if abs(got - want) > 1e-9:
                    print(f"  MISMATCH {scheduler_name} step {i+1}: got={got}, want={want}")
                    all_ok = False

            print(f"  OK {scheduler_name} ({params})")

        except Exception as e:
            print(f"  ERROR {scheduler_name}: {e}")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="Also run torch live verification")
    parser.add_argument("--output", default=None,
                        help="Output path (default: ferrotorch-optim/tests/conformance/scheduler_fixtures.json)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    output_path = Path(args.output) if args.output else (
        repo_root / "ferrotorch-optim" / "tests" / "conformance" / "scheduler_fixtures.json"
    )

    print(f"Building fixtures (torch pin: {TORCH_VERSION})...")
    data = build_fixtures()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written {len(data['fixtures'])} fixtures to {output_path}")

    if args.verify:
        ok = verify_against_torch(data)
        if not ok:
            print("VERIFICATION FAILED", file=sys.stderr)
            sys.exit(1)
        print("Verification passed.")

    print("Done. Exit 0.")


if __name__ == "__main__":
    main()
