#!/usr/bin/env python3
"""DETERMINISTIC live-torch BACKWARD reference generator for the FINAL
negative-pad close-audit (acto-critic, #1611..#1631). Drives
`tests/divergence_negpad_backward_grid.rs`.

This is the BACKWARD twin of `fixtures_pad_grid_det_gen.py`. It emits, for the
SAME deterministic grid (1-D sizes 1..6 with all lo/hi in -(size+2)..=(size+2);
2-D sizes 1..4 broad per-axis mixes), for EACH ACCEPTED forward case:

  - the FORWARD acceptance + (deterministically-classified) garbage flag (reused
    verbatim from the forward oracle's cold-fork + additive-shift gather
    consistency classifier, so backward grads are only demanded on DEFINED
    cases),
  - torch's `x.grad` from `sum(F.pad(x)).backward()` (the all-ones seed VJP), and
  - torch's `x.grad` from a NON-UNIFORM seeded grad_output VJP, where the seed is
    a deterministic, distinct-per-cell ramp computed on the OUTPUT shape. A
    non-uniform seed makes a mis-weighted / mis-routed scatter detectable (under
    the all-ones `sum` seed, two distinct sources both read the same number of
    times look identical; under a distinct-per-cell ramp the exact set of output
    cells routing into each input cell is pinned).

WHY a backward oracle and not a forward-derived analytic grad. torch's
`_pad_circular` (`PadNd.cpp:148-187`) is a differentiable composition of
`new_empty` + `slice` + `copy_`; the backward we must match is torch autograd's
own transpose of that composition, INCLUDING the `copy_`-overwrites-destination
adjoint (`grad_src += grad_dst; grad_dst = 0`) for the LIVE-`out` wrap reads. We
do not re-derive it; we read it from live torch (R-CHAR-3).

Garbage handling: an over-cropped / net-zero circular forward that reads
uninitialized `new_empty` memory has NO defined forward contract, hence no
defined backward; those rows carry `garbage_det=True` and the Rust harness skips
the grad assertion (it only requires no-panic on them — but they are forward-Err
in ferro anyway, so backward is never reached). reflect/replicate/constant never
garbage.

Record schema (one JSON object per line):
  {"rank":1|2, "mode":..., "in_shape":[...], "in_data":[...], "pads":[...],
   "ok":bool,
   "garbage_det":bool,                 # iff ok and circular
   "out_shape":[...],                  # iff ok
   "grad_sum":[...],                   # iff ok and not garbage: sum() VJP grad
   "seed":[...],                       # iff ok and not garbage: the ramp seed
                                       #   (flattened, output-shape order)
   "grad_seed":[...]}                  # iff ok and not garbage: ramp-seed VJP grad
"""
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(0)

SHIFTS = [1000.0, 7.0, 0.5, -333.0, 1e6, 2.71828]
TOL = 1e-6


def _leaf(in_shape, in_data):
    """A fresh leaf tensor of `in_shape` (reshape-before-requires_grad so the
    leaf IS the in-shape tensor and its .grad is populated)."""
    return (
        torch.tensor(in_data, dtype=torch.float64)
        .reshape(in_shape)
        .detach()
        .clone()
        .requires_grad_(True)
    )


def _fwd(in_shape, in_data, pads, mode):
    x = torch.tensor(in_data, dtype=torch.float64).reshape(in_shape)
    try:
        y = F.pad(x, pads, mode=mode)
        return True, list(y.shape), [float(v) for v in y.detach().reshape(-1).tolist()]
    except Exception:  # noqa: BLE001
        return False, None, None


def _classify_in_child(in_shape, pads):
    n = 1
    for s in in_shape:
        n *= s
    base = [float(i + 1) for i in range(n)]
    ok, _shp, oa = _fwd(in_shape, base, pads, "circular")
    if not ok:
        return True
    if len(oa) == 0:
        return False
    for v in oa:
        if not math.isfinite(v):
            return True
    for s in SHIFTS:
        shifted = [b + s for b in base]
        ok2, _shp2, ob = _fwd(in_shape, shifted, pads, "circular")
        if not ok2:
            return True
        for a, b in zip(oa, ob):
            if (not math.isfinite(b)) or abs((b - a) - s) > TOL * (1.0 + abs(s)):
                return True
    return False


def classify_circular_forked(in_shape, pads):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            garbage = _classify_in_child(in_shape, pads)
            payload = json.dumps(garbage)
        except Exception:  # noqa: BLE001
            payload = json.dumps(True)
        os.write(w, payload.encode())
        os.close(w)
        os._exit(0)
    os.close(w)
    buf = b""
    while True:
        chunk = os.read(r, 65536)
        if not chunk:
            break
        buf += chunk
    os.close(r)
    os.waitpid(pid, 0)
    return json.loads(buf.decode())


def _grad_sum(in_shape, in_data, pads, mode):
    """torch x.grad from sum(F.pad(x)).backward() — the all-ones seed VJP."""
    x = _leaf(in_shape, in_data)
    y = F.pad(x, pads, mode=mode)
    if y.numel() == 0:
        return [0.0] * len(in_data)
    y.sum().backward()
    return [float(v) for v in x.grad.detach().reshape(-1).tolist()]


def _grad_seeded(in_shape, in_data, pads, mode):
    """torch x.grad from a NON-UNIFORM seeded grad_output VJP. Seed is a
    deterministic distinct-per-cell ramp on the OUTPUT shape. Returns
    (seed_flat, grad_flat)."""
    x = _leaf(in_shape, in_data)
    y = F.pad(x, pads, mode=mode)
    if y.numel() == 0:
        return [], [0.0] * len(in_data)
    seed_flat = [1.0 + 0.013 * i + 0.000001 * (i * i) for i in range(y.numel())]
    seed = torch.tensor(seed_flat, dtype=torch.float64).reshape(y.shape)
    y.backward(seed)
    return seed_flat, [float(v) for v in x.grad.detach().reshape(-1).tolist()]


def run_case(rank, mode, in_shape, in_data, pads):
    rec = {
        "rank": rank,
        "mode": mode,
        "in_shape": list(in_shape),
        "in_data": list(in_data),
        "pads": list(pads),
    }
    ok, shp, _data = _fwd(in_shape, in_data, pads, mode)
    if not ok:
        rec["ok"] = False
        return rec
    rec["ok"] = True
    rec["out_shape"] = shp

    if mode == "circular":
        garbage = classify_circular_forked(in_shape, pads)
        rec["garbage_det"] = garbage
    else:
        rec["garbage_det"] = False

    if not rec["garbage_det"]:
        rec["grad_sum"] = _grad_sum(in_shape, in_data, pads, mode)
        seed, gseed = _grad_seeded(in_shape, in_data, pads, mode)
        rec["seed"] = seed
        rec["grad_seed"] = gseed
    return rec


def gen_1d(emit):
    modes = ["constant", "reflect", "replicate", "circular"]
    for size in range(1, 7):
        data = [float(i + 1) for i in range(size)]
        lim = size + 2
        for lo in range(-lim, lim + 1):
            for hi in range(-lim, lim + 1):
                for mode in modes:
                    emit(run_case(1, mode, [1, size], data, [lo, hi]))


def gen_2d(emit):
    modes = ["constant", "reflect", "replicate", "circular"]

    def per_axis_choices(s):
        c = set()
        for v in [-(s + 1), -s, -(s - 1) if s >= 1 else 0, -1, 0, 1,
                  s - 1 if s >= 1 else 0, s, s + 1]:
            c.add(v)
        return sorted(c)

    for h in [1, 2, 3, 4]:
        for w in [1, 2, 3, 4]:
            data = [float(i + 1) for i in range(h * w)]
            wc = per_axis_choices(w)
            hc = per_axis_choices(h)
            for lo_w in wc:
                for hi_w in wc:
                    for lo_h in hc:
                        for hi_h in hc:
                            net_w = w + lo_w + hi_w
                            net_h = h + lo_h + hi_h
                            interesting = (
                                lo_w < 0 or hi_w < 0 or lo_h < 0 or hi_h < 0
                                or net_w == 0 or net_h == 0
                                or abs(lo_w) >= w or abs(hi_w) >= w
                                or abs(lo_h) >= h or abs(hi_h) >= h
                            )
                            if not interesting:
                                continue
                            for mode in modes:
                                emit(run_case(
                                    2, mode, [1, h, w], data,
                                    [lo_w, hi_w, lo_h, hi_h]))


def main():
    out = sys.stdout
    write = out.write

    def emit(rec):
        write(json.dumps(rec))
        write("\n")

    gen_1d(emit)
    gen_2d(emit)
    out.flush()


if __name__ == "__main__":
    main()
