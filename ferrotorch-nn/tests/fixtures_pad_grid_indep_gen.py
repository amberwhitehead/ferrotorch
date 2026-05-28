#!/usr/bin/env python3
"""INDEPENDENT live-torch reference generator for the definitive negative-pad
re-audit (acto-critic, #1628 close). Drives
`tests/divergence_negpad_indep_reaudit.rs`.

GARBAGE ORACLE (ferrotorch-INDEPENDENT, allocator-independent, sound):
LINEARITY ∧ VALUE-MEMBERSHIP.

torch's circular kernel (`aten/src/ATen/native/PadNd.cpp:148-187`) `new_empty`s
the output and then the center copy (`:154-161`) + wrap copies (`:169-187`) only
ever MOVE REAL INPUT ELEMENTS (a gather / permutation with repetition). Hence a
DEFINED circular output is:
  (L) a LINEAR (degree-1, no constant) function of the input — scaling the input
      by k scales every output cell by exactly k; AND
  (M) every output cell is a MEMBER of the input value set.

An uninitialized read of `new_empty` memory satisfies NEITHER in general: it does
not scale with k (catches uninit that coincidentally lands on an input value AND
is allocator-stable), and it is usually outside the input value set (catches
denormals / huge / spurious values). We require BOTH for `garbage_indep=False`;
any failure -> `garbage_indep=True`. (The M check additionally rejects a uninit
read that happens to be 0.0 — which would trivially pass linearity since k*0==0 —
because the arange(1..n) inputs contain no 0.)

  out   = F.pad(x,     pads, 'circular')   (clean run)
  out_k = F.pad(k*x,   pads, 'circular')   (same process, k = 1000)
  DEFINED iff  out_k[i] == k*out[i]  AND  round(out[i],9) in input_set  forall i

This is decisive and needs NO cross-process / allocator pollution. (Earlier
value-set and single-cross-process oracles UNDER-flagged garbage that
coincidentally matched an input value and was allocator-stable — e.g.
`F.pad(arange(1..2),[2,-1,1,0],'circular')` returns all-in-set values stably
across two processes yet is non-linear, hence uninit; while
`F.pad(arange(1..2),[-1,2,0,1],'circular')` returns all-input[1] AND scales,
hence a genuinely DEFINED result ferrotorch must reproduce.)

constant/reflect/replicate always gather real input elements -> never garbage.

Record schema (one JSON object per line):
  {"rank":1|2, "mode":..., "in_shape":[...], "in_data":[...], "pads":[...],
   "ok":bool,
   "out_shape":[...], "out_data":[...],   # iff ok (finite -> null)
   "garbage_indep":bool}                  # iff ok: True == uninitialized read
"""
import json
import math
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(0)

LIN_K = 1000.0
LIN_TOL = 1e-6


def _pad(in_shape, in_data, pads, mode):
    x = torch.tensor(in_data, dtype=torch.float64).reshape(in_shape)
    try:
        y = F.pad(x, pads, mode=mode)
        return True, list(y.shape), [float(v) for v in y.detach().reshape(-1).tolist()]
    except Exception:  # noqa: BLE001
        return False, None, None


def run_case(rank, mode, in_shape, in_data, pads):
    rec = {
        "rank": rank,
        "mode": mode,
        "in_shape": list(in_shape),
        "in_data": list(in_data),
        "pads": list(pads),
    }
    ok, shp, data = _pad(in_shape, in_data, pads, mode)
    if not ok:
        rec["ok"] = False
        return rec
    rec["ok"] = True
    rec["out_shape"] = shp
    rec["out_data"] = [v if math.isfinite(v) else None for v in data]

    garbage = False
    if mode == "circular":
        numel = 1
        for s in shp:
            numel *= s
        if numel > 0:
            in_set = set(round(v, 9) for v in in_data)
            scaled_in = [v * LIN_K for v in in_data]
            ok2, _shp2, data2 = _pad(in_shape, scaled_in, pads, mode)
            if not ok2:
                garbage = True
            else:
                for a, b in zip(data, data2):
                    # (M) membership
                    if (not math.isfinite(a)) or (round(a, 9) not in in_set):
                        garbage = True
                        break
                    # (L) linearity
                    if (not math.isfinite(b)) or \
                       abs(b - LIN_K * a) > LIN_TOL * (1.0 + abs(LIN_K * a)):
                        garbage = True
                        break
    rec["garbage_indep"] = garbage
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
