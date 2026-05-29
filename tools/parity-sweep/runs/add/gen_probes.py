#!/usr/bin/env python3
"""
Generate adversarial probes for the `add` op discriminator pass.

Schema (one JSON object per line in discriminator_probes.jsonl):
    {
        "id": "<unique slug>",
        "category": "<category name>",
        "rationale": "...",
        "args_spec": [<arg_spec>, ...],
        "kwargs": {"alpha": <number>?, "requires_grad": [bool,bool]?, "out_spec": <arg_spec>?, "inplace": true?},
        "autograd_check": bool   # if true, additionally backprop and compare grads
    }

arg_spec is one of:
    {"kind":"tensor", "shape":[...], "dtype":"float32"|"float64"|"int32"|"int64"|"bool",
     "data": [<numbers>] | null,                # explicit per-element values
     "fill": <number> | null,                   # uniform fill (used if data is null)
     "transform": "none"|"transpose"|"expand"|"slice_step",
     "transform_args": {...}}
    {"kind":"scalar", "value": <number>, "dtype":"int"|"float"}

Special float-data tokens (since JSON has no NaN/Inf literals):
    "NaN", "+Inf", "-Inf", "+0", "-0", "DENORM" (= f32::MIN_POSITIVE / 2)
"""
import json
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "discriminator_probes.jsonl")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tensor(shape, *, dtype="float32", data=None, fill=None,
           transform="none", transform_args=None):
    return {
        "kind": "tensor",
        "shape": list(shape),
        "dtype": dtype,
        "data": data,
        "fill": fill,
        "transform": transform,
        "transform_args": transform_args or {},
    }

def scalar(value, dtype="float"):
    return {"kind": "scalar", "value": value, "dtype": dtype}

def probe(pid, category, rationale, args_spec, kwargs=None, autograd_check=False):
    return {
        "id": pid,
        "category": category,
        "rationale": rationale,
        "args_spec": args_spec,
        "kwargs": kwargs or {},
        "autograd_check": autograd_check,
    }

probes = []

# ---------------------------------------------------------------------------
# 1. NaN propagation (>=5)
# ---------------------------------------------------------------------------
probes += [
    probe("nan_a_scalar", "nan_propagation",
          "input has a single NaN at index 0",
          [tensor([4], data=["NaN", 1.0, 2.0, 3.0]),
           tensor([4], data=[0.0, 0.0, 0.0, 0.0])]),
    probe("nan_b_scalar", "nan_propagation",
          "other has NaN, input is finite — torch yields NaN at index",
          [tensor([4], data=[1.0, 2.0, 3.0, 4.0]),
           tensor([4], data=[0.0, "NaN", 0.0, "NaN"])]),
    probe("nan_both_diff_positions", "nan_propagation",
          "input NaN at 0, other NaN at 3 — both positions must be NaN",
          [tensor([4], data=["NaN", 0.0, 0.0, 0.0]),
           tensor([4], data=[0.0, 0.0, 0.0, "NaN"])]),
    probe("nan_all_in_a", "nan_propagation",
          "input is all NaN; output must be all NaN regardless of other",
          [tensor([6], fill="NaN"),
           tensor([6], fill=1.0)]),
    probe("nan_alpha_zero", "nan_propagation",
          "alpha=0 with NaN in other — torch: input + 0*NaN = NaN (0*NaN=NaN)",
          [tensor([4], data=[1.0, 2.0, 3.0, 4.0]),
           tensor([4], data=["NaN", "NaN", "NaN", "NaN"])],
          kwargs={"alpha": 0.0}),
    probe("nan_alpha_nan", "nan_propagation",
          "alpha=NaN with finite other; entire output must be NaN",
          [tensor([4], fill=1.0),
           tensor([4], fill=2.0)],
          kwargs={"alpha": "NaN"}),
    probe("nan_broadcast", "nan_propagation",
          "NaN preserved through broadcasting",
          [tensor([3, 1], data=["NaN", 1.0, 2.0]),
           tensor([1, 3], data=[10.0, 20.0, 30.0])]),
]

# ---------------------------------------------------------------------------
# 2. +/- Inf and overflow (>=5)
# ---------------------------------------------------------------------------
F32_MAX = "F32_MAX"
NEG_F32_MAX = "-F32_MAX"
probes += [
    probe("inf_plus_inf", "inf_overflow",
          "+inf + +inf = +inf",
          [tensor([2], fill="+Inf"), tensor([2], fill="+Inf")]),
    probe("inf_minus_inf", "inf_overflow",
          "+inf + -inf = NaN (indeterminate)",
          [tensor([2], fill="+Inf"), tensor([2], fill="-Inf")]),
    probe("neg_inf_plus_neg_inf", "inf_overflow",
          "-inf + -inf = -inf",
          [tensor([2], fill="-Inf"), tensor([2], fill="-Inf")]),
    probe("max_plus_max", "inf_overflow",
          "f32::MAX + f32::MAX overflows to +inf",
          [tensor([3], fill=F32_MAX), tensor([3], fill=F32_MAX)]),
    probe("neg_max_minus_max", "inf_overflow",
          "(-f32::MAX) + (-f32::MAX) overflows to -inf",
          [tensor([3], fill=NEG_F32_MAX), tensor([3], fill=NEG_F32_MAX)]),
    probe("inf_with_alpha", "inf_overflow",
          "input=+inf, other=1.0, alpha=+inf -> 0*inf? Actually input + inf*1 = +inf",
          [tensor([2], fill="+Inf"), tensor([2], fill=1.0)],
          kwargs={"alpha": "+Inf"}),
    probe("alpha_inf_times_zero", "inf_overflow",
          "alpha=+inf, other=0.0 -> +inf*0 = NaN, then input + NaN = NaN",
          [tensor([2], fill=1.0), tensor([2], fill=0.0)],
          kwargs={"alpha": "+Inf"}),
    probe("max_plus_one", "inf_overflow",
          "f32::MAX + 1.0 — no overflow (ulp is huge here), still f32::MAX",
          [tensor([2], fill=F32_MAX), tensor([2], fill=1.0)]),
]

# ---------------------------------------------------------------------------
# 3. Denormals (>=5)
# ---------------------------------------------------------------------------
probes += [
    probe("denorm_plus_zero", "denormals",
          "denormal + 0 must preserve denormal (no flush-to-zero)",
          [tensor([4], fill="DENORM"), tensor([4], fill=0.0)]),
    probe("denorm_plus_denorm", "denormals",
          "denormal + denormal still subnormal (or zero) — torch: 2*denorm",
          [tensor([4], fill="DENORM"), tensor([4], fill="DENORM")]),
    probe("denorm_minus_denorm", "denormals",
          "denormal + (-denormal) via alpha=-1 = +0.0",
          [tensor([4], fill="DENORM"), tensor([4], fill="DENORM")],
          kwargs={"alpha": -1.0}),
    probe("denorm_alpha_tiny", "denormals",
          "small * denorm — likely flushes to zero on broken backends",
          [tensor([4], fill=1.0e-30), tensor([4], fill="DENORM")],
          kwargs={"alpha": 1.0e-15}),
    probe("denorm_mixed_finite", "denormals",
          "denormal mixed with normals in a single tensor",
          [tensor([4], data=["DENORM", 1.0, "DENORM", -1.0]),
           tensor([4], data=[0.0, "DENORM", 0.0, "DENORM"])]),
    probe("denorm_broadcast", "denormals",
          "denormal broadcast against finite",
          [tensor([1], fill="DENORM"), tensor([5], fill=1.0e-30)]),
]

# ---------------------------------------------------------------------------
# 4. Empty tensors (>=5)
# ---------------------------------------------------------------------------
probes += [
    probe("empty_1d", "empty_tensors",
          "shape [0] + shape [0]",
          [tensor([0], fill=0.0), tensor([0], fill=0.0)]),
    probe("empty_2d_first", "empty_tensors",
          "shape [0,5] + shape [0,5]",
          [tensor([0, 5], fill=0.0), tensor([0, 5], fill=0.0)]),
    probe("empty_2d_second", "empty_tensors",
          "shape [5,0] + shape [5,0]",
          [tensor([5, 0], fill=0.0), tensor([5, 0], fill=0.0)]),
    probe("empty_broadcast_to_nonempty", "empty_tensors",
          "shape [0,5] + shape [1,5] -> [0,5] (broadcast with empty)",
          [tensor([0, 5], fill=0.0), tensor([1, 5], fill=1.0)]),
    probe("empty_3d", "empty_tensors",
          "shape [2,0,3] + shape [2,0,3]",
          [tensor([2, 0, 3], fill=0.0), tensor([2, 0, 3], fill=0.0)]),
    probe("empty_with_alpha", "empty_tensors",
          "empty + empty with alpha=2.5 — kwargs path on empty",
          [tensor([0], fill=0.0), tensor([0], fill=0.0)],
          kwargs={"alpha": 2.5}),
]

# ---------------------------------------------------------------------------
# 5. Scalar (0-dim) tensors (>=5)
# ---------------------------------------------------------------------------
probes += [
    probe("scalar_0d_plus_0d", "scalar_zerodim",
          "shape [] + shape []",
          [tensor([], data=[3.0]), tensor([], data=[4.0])]),
    probe("scalar_0d_vs_1d", "scalar_zerodim",
          "shape [] + shape [1] — broadcast: torch returns shape [1]",
          [tensor([], data=[3.0]), tensor([1], data=[4.0])]),
    probe("scalar_0d_broadcast_to_nd", "scalar_zerodim",
          "shape [] + shape [3,4] -> [3,4]",
          [tensor([], data=[7.0]), tensor([3, 4], fill=1.0)]),
    probe("scalar_0d_alpha", "scalar_zerodim",
          "shape [] + shape [] with alpha=-2",
          [tensor([], data=[10.0]), tensor([], data=[3.0])],
          kwargs={"alpha": -2.0}),
    probe("scalar_0d_nan", "scalar_zerodim",
          "0-dim NaN — output must be 0-dim NaN, not promoted to [1]",
          [tensor([], data=["NaN"]), tensor([], data=[1.0])]),
    probe("scalar_0d_inf", "scalar_zerodim",
          "0-dim inf - inf = NaN",
          [tensor([], data=["+Inf"]), tensor([], data=["-Inf"])]),
]

# ---------------------------------------------------------------------------
# 6. Non-contiguous strides (>=5)
# ---------------------------------------------------------------------------
probes += [
    probe("transpose_2d", "noncontig_stride",
          "transpose(0,1) of [2,3] -> non-contiguous [3,2]",
          [tensor([2, 3], data=[1, 2, 3, 4, 5, 6], transform="transpose",
                  transform_args={"dim0": 0, "dim1": 1}),
           tensor([3, 2], fill=10.0)]),
    probe("transpose_both", "noncontig_stride",
          "both transposed — torch should still produce the contig logical view",
          [tensor([3, 4], fill=1.0, transform="transpose",
                  transform_args={"dim0": 0, "dim1": 1}),
           tensor([3, 4], fill=2.0, transform="transpose",
                  transform_args={"dim0": 0, "dim1": 1})]),
    probe("transpose_3d", "noncontig_stride",
          "3d transpose(1,2)",
          [tensor([2, 3, 4], fill=1.0, transform="transpose",
                  transform_args={"dim0": 1, "dim1": 2}),
           tensor([2, 4, 3], fill=0.5)]),
    probe("slice_step_2", "noncontig_stride",
          "every-other element via step-2 slice -> stride 2",
          [tensor([8], data=[0, 1, 2, 3, 4, 5, 6, 7], transform="slice_step",
                  transform_args={"start": 0, "stop": 8, "step": 2}),
           tensor([4], fill=100.0)]),
    probe("transpose_with_alpha", "noncontig_stride",
          "transpose + alpha=3 — combines two adversarial axes",
          [tensor([2, 2], data=[1, 2, 3, 4], transform="transpose",
                  transform_args={"dim0": 0, "dim1": 1}),
           tensor([2, 2], data=[10, 20, 30, 40])],
          kwargs={"alpha": 3.0}),
    probe("transpose_with_nan", "noncontig_stride",
          "non-contig view containing NaN",
          [tensor([2, 2], data=["NaN", 1.0, 2.0, 3.0], transform="transpose",
                  transform_args={"dim0": 0, "dim1": 1}),
           tensor([2, 2], fill=0.0)]),
]

# ---------------------------------------------------------------------------
# 7. 0-stride broadcasting (>=5)
# ---------------------------------------------------------------------------
probes += [
    probe("expand_1_to_5", "stride0_broadcast",
          "shape [1] expanded to [5,5] vs shape [5,5]",
          [tensor([1], data=[7.0], transform="expand",
                  transform_args={"shape": [5, 5]}),
           tensor([5, 5], fill=1.0)]),
    probe("expand_1_to_3d", "stride0_broadcast",
          "shape [1] expanded to [2,3,4]",
          [tensor([1], data=[2.0], transform="expand",
                  transform_args={"shape": [2, 3, 4]}),
           tensor([2, 3, 4], fill=0.5)]),
    probe("expand_row", "stride0_broadcast",
          "shape [1,4] expanded to [3,4] (stride 0 on dim 0)",
          [tensor([1, 4], data=[1.0, 2.0, 3.0, 4.0], transform="expand",
                  transform_args={"shape": [3, 4]}),
           tensor([3, 4], fill=10.0)]),
    probe("expand_with_alpha", "stride0_broadcast",
          "expanded tensor + alpha != 1",
          [tensor([1], data=[5.0], transform="expand",
                  transform_args={"shape": [4]}),
           tensor([4], fill=1.0)],
          kwargs={"alpha": -2.0}),
    probe("expand_nan", "stride0_broadcast",
          "stride-0 NaN propagation — every output element must be NaN",
          [tensor([1], data=["NaN"], transform="expand",
                  transform_args={"shape": [3, 3]}),
           tensor([3, 3], fill=0.0)]),
    probe("expand_both_operands", "stride0_broadcast",
          "both operands are stride-0 expanded — degenerate case",
          [tensor([1], data=[3.0], transform="expand",
                  transform_args={"shape": [4, 4]}),
           tensor([1], data=[2.0], transform="expand",
                  transform_args={"shape": [4, 4]})]),
]

# ---------------------------------------------------------------------------
# 8. Dtype promotion edges (>=5)
# ---------------------------------------------------------------------------
probes += [
    probe("f32_plus_f64", "dtype_promotion",
          "f32 + f64 — torch promotes to f64; ferrotorch dispatch is f32-only",
          [tensor([3], fill=1.0, dtype="float32"),
           tensor([3], fill=2.0, dtype="float64")]),
    probe("int32_plus_float32", "dtype_promotion",
          "int + float — torch promotes to float",
          [tensor([3], data=[1, 2, 3], dtype="int32"),
           tensor([3], fill=0.5, dtype="float32")]),
    probe("bool_plus_float", "dtype_promotion",
          "bool + float — torch promotes",
          [tensor([3], data=[True, False, True], dtype="bool"),
           tensor([3], fill=2.0, dtype="float32")]),
    probe("int64_plus_int32", "dtype_promotion",
          "i64 + i32 — promote to i64",
          [tensor([3], data=[1, 2, 3], dtype="int64"),
           tensor([3], data=[10, 20, 30], dtype="int32")]),
    probe("f32_plus_int_alpha", "dtype_promotion",
          "f32 + int + alpha=2 — alpha-aware promotion",
          [tensor([3], fill=1.5, dtype="float32"),
           tensor([3], data=[1, 1, 1], dtype="int64")],
          kwargs={"alpha": 2.0}),
    probe("int_plus_int_alpha_float", "dtype_promotion",
          "int + int + alpha=2.5 — torch rejects float alpha on int tensors",
          [tensor([3], data=[1, 2, 3], dtype="int32"),
           tensor([3], data=[4, 5, 6], dtype="int32")],
          kwargs={"alpha": 2.5}),
]

# ---------------------------------------------------------------------------
# 9. Alpha edges (>=5)
# ---------------------------------------------------------------------------
probes += [
    probe("alpha_zero", "alpha_edges",
          "alpha=0 -> input + 0*other = input (NaN in other still kills it)",
          [tensor([4], fill=3.0), tensor([4], fill=5.0)],
          kwargs={"alpha": 0.0}),
    probe("alpha_neg_zero", "alpha_edges",
          "alpha=-0.0 -> input + (-0)*other; sign of zero may matter",
          [tensor([4], fill=3.0), tensor([4], fill=5.0)],
          kwargs={"alpha": -0.0}),
    probe("alpha_one_exact", "alpha_edges",
          "alpha=1.0 — should match plain add bit-exact",
          [tensor([4], data=[0.1, 0.2, 0.3, 0.4]),
           tensor([4], data=[1e-7, 1e-8, 1e-9, 1e-10])],
          kwargs={"alpha": 1.0}),
    probe("alpha_int_two", "alpha_edges",
          "alpha=2 (int) — JSON integer, ferrotorch must accept as f64",
          [tensor([4], data=[1.0, 2.0, 3.0, 4.0]),
           tensor([4], data=[10.0, 20.0, 30.0, 40.0])],
          kwargs={"alpha": 2}),
    probe("alpha_float_two", "alpha_edges",
          "alpha=2.0 (float) — must match alpha=2 (int) bit-exact",
          [tensor([4], data=[1.0, 2.0, 3.0, 4.0]),
           tensor([4], data=[10.0, 20.0, 30.0, 40.0])],
          kwargs={"alpha": 2.0}),
    probe("alpha_tiny", "alpha_edges",
          "alpha=f32::MIN_POSITIVE -> result essentially equals input",
          [tensor([4], fill=1.0), tensor([4], fill=1.0e10)],
          kwargs={"alpha": 1.1754943508222875e-38}),
    probe("alpha_huge", "alpha_edges",
          "alpha=1e30 with other=1e10 -> overflow to inf",
          [tensor([4], fill=1.0), tensor([4], fill=1.0e10)],
          kwargs={"alpha": 1.0e30}),
    probe("alpha_neg_huge", "alpha_edges",
          "alpha=-1e30 — sign flip + overflow",
          [tensor([4], fill=1.0), tensor([4], fill=1.0e10)],
          kwargs={"alpha": -1.0e30}),
    probe("alpha_neg_one", "alpha_edges",
          "alpha=-1 — equivalent to sub",
          [tensor([5], data=[1.0, 2.0, 3.0, 4.0, 5.0]),
           tensor([5], data=[0.5, 0.5, 0.5, 0.5, 0.5])],
          kwargs={"alpha": -1.0}),
]

# ---------------------------------------------------------------------------
# 10. Autograd graph identity (>=5)
# ---------------------------------------------------------------------------
probes += [
    probe("autograd_basic", "autograd_identity",
          "input requires_grad; loss.sum().backward() — grad_a == ones",
          [tensor([4], data=[1.0, 2.0, 3.0, 4.0]),
           tensor([4], data=[10.0, 20.0, 30.0, 40.0])],
          kwargs={"requires_grad": [True, True]},
          autograd_check=True),
    probe("autograd_alpha_two", "autograd_identity",
          "alpha=2; grad_b should be 2 * grad_output",
          [tensor([3], fill=1.0), tensor([3], fill=2.0)],
          kwargs={"alpha": 2.0, "requires_grad": [True, True]},
          autograd_check=True),
    probe("autograd_alpha_neg_half", "autograd_identity",
          "alpha=-0.5; grad_b = -0.5 * grad_output",
          [tensor([4], fill=1.0), tensor([4], fill=1.0)],
          kwargs={"alpha": -0.5, "requires_grad": [True, True]},
          autograd_check=True),
    probe("autograd_broadcast", "autograd_identity",
          "broadcast + requires_grad — grad must reduce back to input shape",
          [tensor([3, 1], fill=1.0),
           tensor([1, 4], fill=2.0)],
          kwargs={"requires_grad": [True, True]},
          autograd_check=True),
    probe("autograd_alpha_zero", "autograd_identity",
          "alpha=0 — grad_b should still be 0 * grad_output = zeros",
          [tensor([3], fill=1.0), tensor([3], fill=2.0)],
          kwargs={"alpha": 0.0, "requires_grad": [True, True]},
          autograd_check=True),
    probe("autograd_only_a", "autograd_identity",
          "only input requires grad; grad_b is None in torch",
          [tensor([3], fill=1.0), tensor([3], fill=2.0)],
          kwargs={"requires_grad": [True, False]},
          autograd_check=True),
]

# ---------------------------------------------------------------------------
# 11. In-place variant (>=5)
# ---------------------------------------------------------------------------
probes += [
    probe("inplace_basic", "inplace",
          "torch.Tensor.add_ mutates input — ferrotorch must have an equivalent",
          [tensor([4], data=[1.0, 2.0, 3.0, 4.0]),
           tensor([4], data=[10.0, 20.0, 30.0, 40.0])],
          kwargs={"inplace": True}),
    probe("inplace_alpha", "inplace",
          "add_ with alpha=3",
          [tensor([4], fill=1.0), tensor([4], fill=2.0)],
          kwargs={"inplace": True, "alpha": 3.0}),
    probe("inplace_broadcast", "inplace",
          "add_ broadcasting other -> input shape unchanged",
          [tensor([4, 3], fill=1.0), tensor([3], fill=2.0)],
          kwargs={"inplace": True}),
    probe("inplace_nan_propagation", "inplace",
          "in-place add with NaN in other",
          [tensor([4], data=[1.0, 2.0, 3.0, 4.0]),
           tensor([4], data=["NaN", "NaN", 0.0, 0.0])],
          kwargs={"inplace": True}),
    probe("inplace_self_reference", "inplace",
          "x.add_(x, alpha=1) — using same tensor as both operands",
          [tensor([4], data=[1.0, 2.0, 3.0, 4.0]),
           "ALIAS_A"],
          kwargs={"inplace": True}),
]

# ---------------------------------------------------------------------------
# 12. out= kwarg (>=5)
# ---------------------------------------------------------------------------
probes += [
    probe("out_basic", "out_kwarg",
          "out= preallocated tensor receives result",
          [tensor([4], fill=1.0), tensor([4], fill=2.0)],
          kwargs={"out_spec": tensor([4], fill=0.0)}),
    probe("out_with_alpha", "out_kwarg",
          "out= with alpha=2",
          [tensor([4], fill=1.0), tensor([4], fill=2.0)],
          kwargs={"alpha": 2.0, "out_spec": tensor([4], fill=0.0)}),
    probe("out_broadcast", "out_kwarg",
          "out= must match broadcast shape",
          [tensor([3, 1], fill=1.0), tensor([1, 4], fill=2.0)],
          kwargs={"out_spec": tensor([3, 4], fill=0.0)}),
    probe("out_wrong_shape", "out_kwarg",
          "out= tensor with wrong shape — torch errors; what does ferrotorch do?",
          [tensor([4], fill=1.0), tensor([4], fill=2.0)],
          kwargs={"out_spec": tensor([3], fill=0.0)}),
    probe("out_nan", "out_kwarg",
          "out= filled with NaN, then overwritten — must not leak NaN",
          [tensor([4], fill=1.0), tensor([4], fill=2.0)],
          kwargs={"out_spec": tensor([4], fill="NaN")}),
]

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
with open(OUT_PATH, "w") as f:
    for p in probes:
        f.write(json.dumps(p) + "\n")

# Sanity print
from collections import Counter
counts = Counter(p["category"] for p in probes)
print(f"Wrote {len(probes)} probes to {OUT_PATH}")
for cat, n in sorted(counts.items()):
    print(f"  {cat:25s} {n}")
