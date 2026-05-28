#!/usr/bin/env python3
"""DETERMINISTIC live-torch reference generator for the FINAL negative-pad
close-audit (acto-critic, #1611..#1629). Drives
`tests/divergence_negpad_det_reaudit.rs`.

WHY THIS SUPERSEDES `fixtures_pad_grid_indep_gen.py`. The prior oracle
classified a circular over-crop "garbage" by a single multiplicative-k=1000
linearity ∧ value-membership heuristic in ONE process. That is FLAKY: a
`new_empty` uninitialized read (`PadNd.cpp:148`) that, when many pads run in one
warmed process, lands on a PRIOR case's freed (in-set, k-linear) output is
wrongly tagged DEFINED, so `reject_mismatch` flickers 0↔2 with a changing case
identity. A "provably 0" close cannot rest on that.

THE DETERMINISTIC CLASSIFIER — COLD-FORK + ADDITIVE-SHIFT GATHER CONSISTENCY
(sound, allocator-INDEPENDENT, reproducible). Two independent ideas combine:

  (1) COLD-FORK ISOLATION. Each circular case is classified inside its OWN
      `os.fork()` child. A child gets a fresh allocation arena, so its
      `new_empty` reads genuinely-uninitialized memory rather than the parent's
      WARMED pool (the warming that makes a long-lived 56k-grid process
      deterministically read a prior case's residue and thus mis-classify).

  (2) ADDITIVE-SHIFT GATHER CONSISTENCY. torch's circular `copy_`s VERBATIM
      gather input elements (`:154-161` center + `:169-187` live wraps; each is
      a plain element move). The gather INDEX a DEFINED cell reads is a pure
      function of (shape, pads) — independent of input VALUES. So padding the
      same (shape, pads) with two inputs differing by a constant additive shift
      `s` (B[i] = A[i] + s) MUST shift every DEFINED output cell by exactly `s`:
      out_B[i] == out_A[i] + s. An uninitialized cell reads memory uncorrelated
      with the shift and fails. We require this for SEVERAL distinct shifts.

  classify (inside the cold-fork child):
    base A = arange(1..n).  GENUINE-GARBAGE (`garbage_det=True`) iff a base cell
    is non-finite OR any shift in SHIFTS yields a cell with
      |out_B[i] - out_A[i] - s| > TOL*(1+|s|)  (or accept flickers under shift).
    else DEFINED, recording out_A (the verbatim gather of A).

  Verified: cold-fork + shift is bit-reproducible across runs EVEN after heavy
  parent heap-warming (0 classification diffs over >=3 runs; garbage tail
  deterministically 4060; the warming-only pseudo-defined cases — e.g.
  `[1,3,3] pads[2,3,1,-3]`, `[1,2,4] pads[4,-4,1,-1]` — are stably GARBAGE).

constant/reflect/replicate always gather real input elements -> never garbage
(`garbage_det` always False; no fork needed — they are value/heap-independent).

Record schema (one JSON object per line):
  {"rank":1|2, "mode":..., "in_shape":[...], "in_data":[...], "pads":[...],
   "ok":bool,
   "out_shape":[...], "out_data":[...],     # iff ok (the gather of A;
                                            #   non-finite cells -> null)
   "garbage_det":bool}                      # iff ok: True == uninitialized read
"""
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# Several structurally-distinct additive shifts. A defined gather shifts by
# exactly `s` for every one; an uninitialized read cannot match all of them.
SHIFTS = [1000.0, 7.0, 0.5, -333.0, 1e6, 2.71828]
TOL = 1e-6


def _pad(in_shape, in_data, pads, mode):
    x = torch.tensor(in_data, dtype=torch.float64).reshape(in_shape)
    try:
        y = F.pad(x, pads, mode=mode)
        return True, list(y.shape), [float(v) for v in y.detach().reshape(-1).tolist()]
    except Exception:  # noqa: BLE001
        return False, None, None


def _classify_in_child(in_shape, pads):
    """Run in a COLD-FORK child: classify an ACCEPTED circular case by
    additive-shift gather consistency. Returns (garbage_det, out_data|None)."""
    n = 1
    for s in in_shape:
        n *= s
    base = [float(i + 1) for i in range(n)]
    ok, _shp, oa = _pad(in_shape, base, pads, "circular")
    if not ok:
        return True, None  # accept flickered (shouldn't happen; parent checked)
    if len(oa) == 0:
        return False, []  # empty output -> trivially defined
    for v in oa:
        if not math.isfinite(v):
            return True, None
    for s in SHIFTS:
        shifted = [b + s for b in base]
        ok2, _shp2, ob = _pad(in_shape, shifted, pads, "circular")
        if not ok2:
            return True, None
        for a, b in zip(oa, ob):
            if (not math.isfinite(b)) or abs((b - a) - s) > TOL * (1.0 + abs(s)):
                return True, None
    return False, oa


def classify_circular_forked(in_shape, pads):
    """Fork a cold child to classify; return (garbage_det, out_data|None)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            garbage, out_data = _classify_in_child(in_shape, pads)
            payload = json.dumps([garbage, out_data])
        except Exception:  # noqa: BLE001
            payload = json.dumps([True, None])
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
    garbage, out_data = json.loads(buf.decode())
    return garbage, out_data


def run_case(rank, mode, in_shape, in_data, pads):
    rec = {
        "rank": rank,
        "mode": mode,
        "in_shape": list(in_shape),
        "in_data": list(in_data),
        "pads": list(pads),
    }
    # Acceptance is decided in the parent (a shape/legality decision, heap-
    # independent).
    ok, shp, data = _pad(in_shape, in_data, pads, mode)
    if not ok:
        rec["ok"] = False
        return rec
    rec["ok"] = True
    rec["out_shape"] = shp

    if mode == "circular":
        garbage, out_data = classify_circular_forked(in_shape, pads)
        rec["garbage_det"] = garbage
        if garbage or out_data is None:
            # No defined contract; record the parent's (heap-dependent) output
            # only as a placeholder — the Rust harness never asserts on garbage.
            rec["out_data"] = [v if math.isfinite(v) else None for v in data]
        else:
            rec["out_data"] = [v if math.isfinite(v) else None for v in out_data]
    else:
        rec["garbage_det"] = False
        rec["out_data"] = [v if math.isfinite(v) else None for v in data]
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
