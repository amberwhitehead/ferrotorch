#!/usr/bin/env python3
"""LIVE torch 2.11 reference generator for the exhaustive negative-pad grid.

Invoked by `ferrotorch-nn/tests/divergence_negpad_chain_close.rs` at test time.
Prints newline-delimited JSON to stdout, one record per grid point. This is the
R-CHAR-3 live oracle — every expected value/acceptance the Rust harness asserts
comes from `torch.nn.functional.pad` here, never copied from ferrotorch.

Record schema (one JSON object per line):
  {"rank":1|2, "mode":"constant|reflect|replicate|circular",
   "in_shape":[...], "in_data":[...], "pads":[lo,hi(,lo,hi)],
   "ok":bool,
   "out_shape":[...], "out_data":[...],   # iff ok
   "garbage":bool,                         # iff ok (R-DEV-6 uninitialized-read)
   "grad":[...]}                           # iff ok and not garbage and grad ran

R-DEV-6 garbage detection: torch's circular kernel `new_empty`s the output then
slice-copies a (possibly degenerate) center; a mixed-sign over-crop leaves the
wrap region reading memory the center copy never wrote. We flag a record garbage
iff any output element is non-finite OR is not (approximately) a member of the
input value set (constant-fill zeros excepted). Those cases have NO reproducible
torch contract, so ferrotorch is permitted to reject them (counted separately).
"""
import json
import math
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def run_case(rank, mode, in_shape, in_data, pads):
    rec = {
        "rank": rank,
        "mode": mode,
        "in_shape": list(in_shape),
        "in_data": list(in_data),
        "pads": list(pads),
    }
    x = torch.tensor(in_data, dtype=torch.float64).reshape(in_shape)
    try:
        y = F.pad(x, pads, mode=mode)
    except Exception as e:  # noqa: BLE001
        rec["ok"] = False
        rec["err"] = type(e).__name__
        return rec
    rec["ok"] = True
    rec["out_shape"] = list(y.shape)
    yd = y.detach().reshape(-1).tolist()
    # Sanitize for strict-JSON: bare NaN/Infinity are not valid JSON. Non-finite
    # output only ever occurs in the circular R-DEV-6 garbage cases (uninitialized
    # reads); emit `null` so the line parses, and the `garbage` flag below marks
    # the record so the Rust harness treats it as a carve-out, not a value.
    rec["out_data"] = [v if math.isfinite(v) else None for v in yd]

    in_set = set(round(v, 9) for v in in_data)
    garbage = False
    for v in yd:
        if not math.isfinite(v):
            garbage = True
            break
        if mode == "constant":
            continue
        if round(v, 9) not in in_set and abs(v) > 1e-12:
            garbage = True
            break
        if abs(v) <= 1e-12 and 0.0 not in in_set:
            # A (near-)zero appearing where the input has no zero and the mode
            # has no fill is an uninitialized read of freed memory.
            garbage = True
            break
    rec["garbage"] = garbage

    if not garbage:
        try:
            xg = x.clone().requires_grad_(True)
            yg = F.pad(xg, pads, mode=mode)
            if yg.numel() > 0:
                yg.sum().backward()
                rec["grad"] = xg.grad.detach().reshape(-1).tolist()
            else:
                rec["grad"] = [0.0] * x.numel()
        except Exception as e:  # noqa: BLE001
            rec["grad_err"] = type(e).__name__
    return rec


def gen_1d(emit):
    modes = ["constant", "reflect", "replicate", "circular"]
    for size in range(1, 7):
        data = [float(i + 1) for i in range(size)]
        in_shape = [1, size]
        lim = size + 2
        for lo in range(-lim, lim + 1):
            for hi in range(-lim, lim + 1):
                for mode in modes:
                    emit(run_case(1, mode, in_shape, data, [lo, hi]))


def gen_2d(emit):
    modes = ["constant", "reflect", "replicate", "circular"]
    sizes = [1, 2, 3, 4]

    def per_axis_choices(s):
        c = set()
        for v in [-(s + 1), -s, -(s - 1) if s >= 1 else 0, -1, 0, 1,
                  s - 1 if s >= 1 else 0, s, s + 1]:
            c.add(v)
        return sorted(c)

    for h in sizes:
        for w in sizes:
            data = [float(i + 1) for i in range(h * w)]
            in_shape = [1, h, w]
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
                                    2, mode, in_shape, data,
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
