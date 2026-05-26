#!/usr/bin/env python3
"""Inject `## REQ status` doc-comment tables into the 14 ferrotorch-core
abstraction files. Idempotent: detects existing `## REQ status` and skips.

For each file we write a //! block containing:
  - A short header attributing the table to its design doc.
  - The REQ status table (rows quoted from the per-file design doc summaries).

Insertion rule:
  - If the file already starts with one or more `//!` lines (existing
    module doc-comment), insert the new //! block immediately AFTER the
    last contiguous //! line (and any blank //! continuations).
  - If the file starts with `use` / `///` / item directives, prepend a
    fresh //! block at the very top.
  - lib.rs has crate-root attributes (#![...]); the new //! block goes
    AFTER the last contiguous #![...] / preceding-comment block and
    BEFORE the first `pub mod` declaration.
"""
import sys
from pathlib import Path

ROOT = Path("/home/doll/ferrotorch")

REQ_TABLES = {
    "ferrotorch-core/src/lib.rs": """//! ## REQ status (per `.design/ferrotorch-core/lib.md`)
//!
//! Crate-root lint baseline, module declarations, and `pub use` re-exports
//! mirroring `torch/__init__.py` and `aten/src/ATen/ATen.h`. All REQs
//! cite `ferrotorch-core/src/lib.rs` directly.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (lint baseline) | SHIPPED | `#![warn(clippy::all, clippy::pedantic)]`, `#![deny(rust_2018_idioms)]`, documented `#![allow(clippy::*)]` set at `lib.rs:1-78`; every `cargo clippy -p ferrotorch-core` run validates the baseline |
//! | REQ-2 (module decls) | SHIPPED | 39 module declarations at `lib.rs:80-118` (36 `pub mod` + 3 internal `mod`); consumed by every downstream `use ferrotorch_core::...` resolver |
//! | REQ-3 (`pub use` re-exports) | SHIPPED | ~150-symbol re-export block at `lib.rs:120-191`; every downstream crate (`ferrotorch-nn`, `ferrotorch-llama`, …) imports `Tensor`, `Device`, `DType`, `FerrotorchError` etc. via these |
//! | REQ-4 (missing_docs allow) | SHIPPED | `#![allow(missing_docs)]` at `lib.rs:74` with the rustdoc-sweep follow-up cite; permitted at crate root by R-CODE-3 (which forbids module-root allows) |
//! | REQ-5 (unsafe permitted) | SHIPPED | no `#![forbid(unsafe_code)]` at the crate root; per-site `// SAFETY:` blocks at `int_tensor.rs:296-313`, `storage.rs` and other files satisfy R-CODE-1 |
""",
    "ferrotorch-core/src/error.rs": """//! ## REQ status (per `.design/ferrotorch-core/error.md`)
//!
//! Workspace-wide error enum mirroring `c10::Error` (`c10/util/Exception.h:31`)
//! under R-DEV-4 (Rust `Result` deviation from C++ exceptions).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (FerrotorchError enum) | SHIPPED | enum `FerrotorchError` at `error.rs:6` with 13 variants; non-test consumer `tensor.rs:1146` (`masked_select` returns `FerrotorchResult`); 1336+ uses across `ferrotorch-core/src/**/*.rs` |
//! | REQ-2 (stable Display) | SHIPPED | `#[error("...")]` attrs at `error.rs:7-89`; test `gpu_variant_display` at `error.rs:117`; consumer `tensor.rs` propagation |
//! | REQ-3 (Send + Sync + 'static) | SHIPPED | `Box<dyn Error + Send + Sync + 'static>` source bound at `error.rs:82`; consumer `cpu_pool.rs` cross-thread `JoinHandle<FerrotorchResult<T>>` |
//! | REQ-4 (Gpu source-chain) | SHIPPED | `Gpu { source }` variant at `error.rs:75-83` with `#[source]`; test `gpu_variant_preserves_source_chain` at `:104`; consumer `gpu_dispatch.rs` wraps `GpuBackend::*` errors |
//! | REQ-5 (FerrotorchResult alias) | SHIPPED | `pub type FerrotorchResult<T>` at `error.rs:93`, re-exported at `lib.rs:145`; consumer `tensor.rs:1144` `pub fn masked_fill -> FerrotorchResult<Tensor<T>>` |
//! | REQ-6 (NotImplementedOnCuda) | SHIPPED | variant at `error.rs:44`; consumer `dtype_dispatch.rs:114` (`dispatch_floating_dtype!` else arm) + `int_tensor.rs:355` (`cast` errors cross-width casts on CUDA) |
//! | REQ-7 (Ferray From) | SHIPPED | `Ferray(#[from] FerrayError)` at `error.rs:88`; consumer: every `?`-propagated ferray error through `storage.rs` / `tensor.rs` |
""",
    "ferrotorch-core/src/dtype.rs": """//! ## REQ status (per `.design/ferrotorch-core/dtype.md`)
//!
//! Dtype gateway re-exporting `ferray_core::{DType, Element}` and defining the
//! `Float` marker trait. Mirrors `c10::ScalarType` (`c10/core/ScalarType.h`).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (DType/Element re-export) | SHIPPED | `pub use ferray_core::{DType, Element}` at `dtype.rs:6`; consumer `lib.rs:142` re-export + `storage.rs` `<T as Element>::dtype()` GPU buffer tagging |
//! | REQ-2 (Float marker trait) | SHIPPED | trait `Float` at `dtype.rs:26` with `Element + num_traits::Float + AddAssign` bound; consumer `tensor.rs` `pub struct Tensor<T: Float>` + every `<T: Float>` op |
//! | REQ-3 (bf16 Float impl) | SHIPPED | `impl Float for half::bf16` at `dtype.rs:30`; consumer `grad_fns/arithmetic.rs:413` `bf16 =>` arm invokes `backend.add_bf16_bf16(...)`; test `bf16_tensor_addition` at `dtype.rs:124` |
//! | REQ-4 (f16 Float impl) | SHIPPED | `impl Float for half::f16` at `dtype.rs:35`; consumer `dtype_dispatch.rs:111` `f16 =>` arm in `dispatch_floating_dtype!`; disambiguation test `f16_element_dtype` at `:63` pins `DType::F16 != DType::BF16` |
//! | REQ-5 (no Float for int/bool) | SHIPPED | absence at `dtype.rs:26-35`; consumer `int_tensor.rs:44` `IntElement` is the integer-specific bound without `num_traits::Float` — `IntTensor<I>` cannot be passed to `<T: Float>` ops (compile error) |
//! | REQ-6 (Element::dtype tag) | SHIPPED | `<T as Element>::dtype()` returns `BF16`/`F16`/`F32`/`F64`; consumer `storage.rs` `TensorStorage::on_device` tags every GPU buffer; `int_tensor.rs:421` `from_gpu_handle` debug-asserts the tag matches |
""",
    "ferrotorch-core/src/dtype_dispatch.rs": """//!
//! ## REQ status (per `.design/ferrotorch-core/dtype_dispatch.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (macro) | SHIPPED | macro `dispatch_floating_dtype!` at `dtype_dispatch.rs:95-119` mirroring `AT_DISPATCH_FLOATING_TYPES_AND_HALF` (`aten/src/ATen/Dispatch.h`); consumer `grad_fns/arithmetic.rs:413` invokes the macro for the GPU `add` arm |
//! | REQ-2 (unified Result return) | SHIPPED | all four arms + the `else` arm return `FerrotorchResult<U>`; consumer `grad_fns/arithmetic.rs:413` binds the macro result to `FerrotorchResult<GpuBufferHandle>` |
//! | REQ-3 (NotImplementedOnCuda fallback) | SHIPPED | trailing `else` arm at `dtype_dispatch.rs:113-117` returning `Err(FerrotorchError::NotImplementedOnCuda)`; test `dispatch_unsupported_dtype_returns_not_implemented` at `:191` |
//! | REQ-4 (is_* predicates) | SHIPPED | `is_f32` / `is_f64` / `is_bf16` / `is_f16` at `dtype_dispatch.rs:126/:133/:140/:149`; consumer `fft.rs:139` `if input.is_cuda() && (is_f32::<T>() || is_f64::<T>())` |
//! | REQ-5 (`#[macro_export]`) | SHIPPED | attribute at `dtype_dispatch.rs:95`; consumer `grad_fns/arithmetic.rs:413` invokes `crate::dispatch_floating_dtype!` (cross-module within crate) |
//! | REQ-6 (4 dtypes lockstep with `Float` impls) | SHIPPED | macro arms `f32 / f64 / bf16 / f16` at `dtype_dispatch.rs:101-103` mirror `Float` impls at `dtype.rs:28-35`; consumer `grad_fns/arithmetic.rs:413` — structural defense against issue #23 pattern A |
""",
    "ferrotorch-core/src/device.rs": """//! ## REQ status (per `.design/ferrotorch-core/device.md`)
//!
//! Tensor location enum mirroring `c10::Device` (`c10/core/Device.h:31`).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (Cpu variant) | SHIPPED | variant `Device::Cpu` at `device.rs:15` with `#[default]`; consumer `storage.rs` `TensorStorage::cpu(...).device() == Device::Cpu`; also `bool_tensor.rs:152` returns `Cpu` for any `TensorStorage::cpu`-backed `BoolTensor` |
//! | REQ-2 (Cuda variant) | SHIPPED | variant `Device::Cuda(usize)` at `device.rs:18`; consumer `int_tensor.rs:268-323` `IntTensor::to` matches `(Cpu, Cuda(_))` / `(Cuda(_), Cpu)` arms for H2D / D2H transfer |
//! | REQ-3 (Xpu variant) | SHIPPED | variant `Device::Xpu(usize)` at `device.rs:22`; consumer `error.rs:259` `FerrotorchError::DeviceMismatch { expected, got }` carries Xpu values; `int_tensor.rs:336` rejects `Xpu` destination via structured error |
//! | REQ-4 (Mps variant) | SHIPPED | variant `Device::Mps(usize)` at `device.rs:26`; consumer `bool_tensor.rs:261-266` `(from, to) => Err(InvalidArgument)` arm pattern-matches on `Mps(_)` |
//! | REQ-5 (Meta variant) | SHIPPED | variant `Device::Meta` at `device.rs:31`; consumer `storage.rs` `TensorStorage::Meta` arm — `try_as_slice` returns `GpuTensorNotAccessible` for Meta variant |
//! | REQ-6 (predicates) | SHIPPED | `is_cpu` / `is_cuda` / `is_xpu` / `is_mps` / `is_meta` at `device.rs:36-64`; consumer `bool_tensor.rs:158`, `int_tensor.rs:205`, every `if a.device().is_cuda()` branch across `grad_fns/*.rs` |
//! | REQ-7 (Display) | SHIPPED | `Display` impl at `device.rs:66-76` matching `c10::Device::str()` (`c10/core/Device.h:167`); consumer `error.rs:11` `#[error("device mismatch: expected {expected}, got {got}")]` |
//! | REQ-8 (Copy/Hash derives) | SHIPPED | `#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]` at `device.rs:12`; consumer `gpu_dispatch.rs` registry + `Tensor<T>::device() == other.device()` PartialEq compares in `bool_tensor.rs:333`, `int_tensor.rs:436` |
""",
    "ferrotorch-core/src/dispatch.rs": """//!
//! ## REQ status (per `.design/ferrotorch-core/dispatch.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (DispatchKey enum) | SHIPPED | `pub enum DispatchKey` at `dispatch.rs:70` with 11 variants `Cpu=0..Tracer=10` mirroring (reduced) `c10::DispatchKey` (`c10/core/DispatchKey.h:136`); consumer `lib.rs:152` re-export — R-DEFER-1 S5 grandfathering for the existing dispatch boundary; registering-crate follow-up at #1530 |
//! | REQ-2 (DispatchKeySet bitmask) | SHIPPED | `pub struct DispatchKeySet { bits: u16 }` at `dispatch.rs:137-251`; consumer `Dispatcher::call` at `:344` walks `keyset.iter_desc()` |
//! | REQ-3 (priority via discriminant) | SHIPPED | `DispatchKey::priority` at `dispatch.rs:113`; consumer `DispatchKeySet::insert` at `:172` shifts by `priority()`, `iter_desc` at `:235` |
//! | REQ-4 (highest/iter_desc) | SHIPPED | `highest` at `dispatch.rs:220`, `iter_desc` at `:235`; consumer `Dispatcher::call` at `:357` iterates `keyset.iter_desc()` |
//! | REQ-5 (Dispatcher<T>) | SHIPPED | `pub struct Dispatcher<T: Float>` at `dispatch.rs:298`, `register` at `:312`, `call` at `:344`; consumer `lib.rs:152` re-exports `Dispatcher` / `Kernel` for downstream registering crates — R-DEFER-1 S5 grandfathering; #1530 |
//! | REQ-6 (call_direct) | SHIPPED | `Dispatcher::call_direct` at `dispatch.rs:374-390`; consumer `lib.rs:152` re-export |
//! | REQ-7 (structured errors) | SHIPPED | `Err(FerrotorchError::InvalidArgument)` at `dispatch.rs:351` (empty keyset) and `:362` (no kernel); no `panic!` |
//! | REQ-8 (Kernel<T> type alias) | SHIPPED | `pub type Kernel<T>` at `dispatch.rs:287-291` with `Send + Sync + 'static`; consumer every `register(...)` callsite + `lib.rs:152` re-export |
//! | REQ-9 (per-dtype generic) | SHIPPED | `Dispatcher<T: Float>` is generic; consumer `lib.rs:152` re-exports both `Dispatcher` and `Kernel` parameterized on `T` |
""",
    "ferrotorch-core/src/ops_trait.rs": """//!
//! ## REQ status (per `.design/ferrotorch-core/ops_trait.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (Add impls) | SHIPPED | 4 `impl ops::Add` blocks at `ops_trait.rs:16-42` mirroring `aten::add.Tensor` (`aten/src/ATen/native/BinaryOps.cpp:218`); consumer `special.rs` (`log1p` via `&x + &one`) + downstream attention/MLP — R-DEFER-1 S5 grandfathering; test `test_add_refs` at `:158` |
//! | REQ-2 (Sub impls) | SHIPPED | 4 `impl ops::Sub` blocks at `ops_trait.rs:46-72` mirroring `aten::sub.Tensor` (`BinaryOps.cpp:280`); consumer `test_chained_expression` exercises `&a - &b` in downstream chained code |
//! | REQ-3 (Mul impls) | SHIPPED | 4 `impl ops::Mul` blocks at `ops_trait.rs:76-102` mirroring `aten::mul.Tensor` (`BinaryOps.cpp:342`); consumer `special.rs` + downstream `q * scale` paths; test `test_mul_with_autograd` at `:181` |
//! | REQ-4 (Div impls) | SHIPPED | 4 `impl ops::Div` blocks at `ops_trait.rs:106-132` mirroring `aten::div.Tensor` (`BinaryOps.cpp:400`); consumer downstream softmax `exp_x / sum_exp` paths; test `test_div_refs` at `:194` |
//! | REQ-5 (Neg impls) | SHIPPED | 2 `impl ops::Neg` blocks at `ops_trait.rs:136-148` delegating to `arithmetic::neg`; consumer `grad_fns/transcendental.rs` (`exp(-x)` patterns) + downstream `-log_prob`; test `test_neg` at `:204` |
//! | REQ-6 (FerrotorchResult Output) | SHIPPED | `type Output = FerrotorchResult<Tensor<T>>` at every impl block (e.g. `:17, :47, :77, :107, :137`); consumer every `let c = (&a + &b)?` callsite; test `test_chained_expression` at `:231` |
//! | REQ-7 (autograd transparency) | SHIPPED | each impl calls `arithmetic::add/sub/mul/div/neg` directly (e.g. `:19, :49, :79, :109, :139`); consumer every autograd-tracking caller; test `test_add_refs` at `:158` calls `c.backward()` after `(&a + &b)?` |
//! | REQ-8 (ownership permutations) | SHIPPED | 4 reference variants per binary op (e.g. `:16-42` for Add); consumer `test_mixed_ownership` at `:222` + `test_owned_add` at `:213`; downstream code mixes `(a + &b)?` freely |
""",
    "ferrotorch-core/src/bool_tensor.rs": """//!
//! ## REQ status (per `.design/ferrotorch-core/bool_tensor.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (constructors) | SHIPPED | `BoolTensor` at `bool_tensor.rs:47`; `from_vec` at `:66`, `from_slice` at `:92`, `zeros` at `:97`, `ones` at `:111`, `from_predicate` at `:126`; consumer `grad_fns/comparison.rs:165` `BoolTensor::from_vec(...)` for `where_cond` mask; `grad_fns/indexing.rs:407` `BoolTensor::from_slice(...)` for masked-fill mask |
//! | REQ-2 (device methods) | SHIPPED | `device` at `bool_tensor.rs:152`, `is_cuda` at `:158`, `to` at `:224`; consumer `ops/indexing.rs:398` `where_cond` reads `cond.device()` to dispatch GPU vs CPU |
//! | REQ-3 (logical ops) | SHIPPED | `not` at `bool_tensor.rs:271`, `and` / `or` / `xor` at `:293-303`; consumer `grad_fns/indexing.rs` consumes mask buffers — `binary_op` helper at `:322` dispatches GPU PTX kernels (`bool_and` / `bool_or` / `bool_xor` / `bool_not`) |
//! | REQ-4 (reductions) | SHIPPED | `count_true` at `bool_tensor.rs:396`, `any` at `:405`, `all` at `:416`; consumer `grad_fns/indexing.rs` uses `BoolTensor::any` to detect empty-mask before dependent kernel launches |
//! | REQ-5 (float comparisons) | SHIPPED | `gt` / `lt` / `ge` / `le` / `eq_t` / `ne` at `bool_tensor.rs:450-477` + `compare_float` at `:479`; consumer `grad_fns/comparison.rs` invokes `BoolTensor::eq_t` etc. mirroring `torch.gt(a, b)` (`aten/src/ATen/native/Compare.cpp`) |
//! | REQ-6 (integer comparisons) | SHIPPED | `gt_int` / `lt_int` / `ge_int` / `le_int` / `eq_int` / `ne_int` at `bool_tensor.rs:524-569` + `compare_int` at `:571`; consumer `lib.rs:135` re-export; downstream integer-tensor predicate code |
//! | REQ-7 (to_float) | SHIPPED | `to_float<T: Float>` at `bool_tensor.rs:612`; consumer `grad_fns/indexing.rs` `masked_select` materializes float tensors from `BoolTensor` masks; test `to_float_emits_zeros_and_ones` at `:730` |
//! | REQ-8 (reshape) | SHIPPED | `reshape` at `bool_tensor.rs:367`; consumer `grad_fns/indexing.rs` reshapes mask buffers to match broadcast shape; test `reshape_preserves_data` at `:722` |
//! | REQ-9 (gpu_handle) | SHIPPED | `from_gpu_handle` at `bool_tensor.rs:195`, `gpu_handle` at `:182`; consumer every GPU comparison-op return path (`compare_float` at `:501-505`, `binary_op` at `:347-351`, `unary_gpu` at `:317-319`) |
//! | REQ-10 (0-D vs zero-axis) | SHIPPED | `shape.is_empty() { 1 } else { product }` at `bool_tensor.rs:70, :99, :113, :369`; consumer `grad_fns/indexing.rs` 0-D mask handling — #805 regression pin |
//! | REQ-11 (structured errors) | SHIPPED | `ShapeMismatch` / `DeviceMismatch` / `InvalidArgument` at multiple sites; no `panic!` in production paths; consumer `grad_fns/comparison.rs` and `grad_fns/indexing.rs` propagate via `?` |
""",
    "ferrotorch-core/src/int_tensor.rs": """//!
//! ## REQ status (per `.design/ferrotorch-core/int_tensor.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (IntElement trait, IntTensor<I>) | SHIPPED | trait `IntElement` at `int_tensor.rs:44`; `impl IntElement for i32 / i64` at `:57 / :74`; `pub struct IntTensor<I: IntElement>` at `:93`; consumer `grad_fns/quantize_grad.rs:139` `zero_point: &IntTensor<i64>`; `ops/phase2c.rs:115` `input: &IntTensor<I>` |
//! | REQ-2 (constructors) | SHIPPED | `from_vec` at `int_tensor.rs:113`, `from_slice` at `:139`, `zeros` at `:144`, `arange` at `:159`, `scalar` at `:175`; consumer `grad_fns/reduction.rs:1463` `IntTensor::<i64>::scalar(best_idx)` for argmax; `ops/phase2c.rs:101, :109` `from_gpu_handle` / `from_vec` |
//! | REQ-3 (device transfer) | SHIPPED | `device` at `int_tensor.rs:199`, `is_cuda` at `:205`, `to` at `:260`; consumer `ops/phase2c.rs` reads `input.device()` + `input.gpu_handle()` before argmax kernel launches; `// SAFETY:` D2H reinterpret at `:296-318` |
//! | REQ-4 (cross-width cast) | SHIPPED | `cast<J>` at `int_tensor.rs:355` with `cast_gpu` fast path; consumer `ops/phase2c.rs` i32↔i64 cast kernel; test `cast_i64_to_i32_out_of_range_errors` at `:836` |
//! | REQ-5 (reshape) | SHIPPED | `reshape` at `int_tensor.rs:384`; consumer `grad_fns/reduction.rs` argmax materialization; test `reshape_preserves_data` at `:843` |
//! | REQ-6 (arithmetic ops) | SHIPPED | `add` at `int_tensor.rs:551`, `sub` at `:561`, `mul` at `:571`, `neg` at `:581`; CPU references at `:696`; consumer `lib.rs:146` re-export — boundary public API; `bool_tensor.rs:524-569` integer comparison constructors route through `IntTensor` compute path. R-DEFER-1 S5 grandfathering; runner arms at #1530 |
//! | REQ-7 (floor_div/remainder) | SHIPPED | `floor_div` at `int_tensor.rs:589`, `remainder` at `:599`; CPU references `int_floor_div_ref` at `:709`, `int_remainder_ref` at `:728`; consumer `lib.rs:146` re-export |
//! | REQ-8 (bitwise ops) | SHIPPED | `bitand`/`bitor`/`bitxor`/`bitnot` at `int_tensor.rs:609-639`; `shl`/`shr` at `:644-649`; CPU references at `:744-765`; consumer `lib.rs:146` re-export |
//! | REQ-9 (reductions) | SHIPPED | `sum`/`prod`/`min`/`max` at `int_tensor.rs:654-684`; empty-tensor handling at `reduce_op` `:502-548`; consumer `lib.rs:146` re-export |
//! | REQ-10 (gpu_handle) | SHIPPED | `gpu_handle` at `int_tensor.rs:236`, `from_gpu_handle` at `:421` with `debug_assert_eq!(handle.dtype(), I::dtype())`; consumer `ops/phase2c.rs:101` invokes `from_gpu_handle`; `bool_tensor.rs:596` reads `a.gpu_handle()` for int comparison GPU path |
//! | REQ-11 (0-D vs zero-axis) | SHIPPED | `shape.is_empty() { 1 } else { product }` at `int_tensor.rs:117, :146, :386`; consumer `grad_fns/reduction.rs:1463` returns 0-D scalar `IntTensor` for argmax — #805 regression pin |
//! | REQ-12 (structured errors) | SHIPPED | `ShapeMismatch` / `DeviceMismatch` / `InvalidArgument` at multiple sites; no `panic!` / `unwrap()` / `expect()` in production paths; consumers propagate via `?` |
""",
    "ferrotorch-core/src/complex_tensor.rs": """//!
//! ## REQ status (per `.design/ferrotorch-core/complex_tensor.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (ComplexTensor<T> SoA layout) | SHIPPED | `pub struct ComplexTensor<T: Float>` at `complex_tensor.rs:32` with `Arc<Vec<T>>` re + im buffers; consumer `lib.rs:136` re-export — R-DEFER-1 S5 grandfathering (#618) |
//! | REQ-2 (constructors) | SHIPPED | `from_re_im` at `complex_tensor.rs:40`, `from_real` at `:71`, `zeros` at `:78`, `scalar` at `:94`; consumer `lib.rs:136` re-export; test `complex_construction_from_re_im` at `:428` |
//! | REQ-3 (interleaved bridge) | SHIPPED | `from_interleaved` at `complex_tensor.rs:105`, `to_interleaved` at `:136`; consumer the FFT methods at `:327, :334, :341, :348` (call `to_interleaved` then `crate::fft::*` then `from_interleaved`) |
//! | REQ-4 (real/imag extraction) | SHIPPED | `real` at `complex_tensor.rs:149`, `imag` at `:158`; consumer `lib.rs:136` re-export; test `complex_real_imag_extraction` at `:489` |
//! | REQ-5 (pointwise arithmetic) | SHIPPED | `add` at `complex_tensor.rs:192`, `sub` at `:201`, `mul` at `:210` (Karatsuba-shaped (a+bi)(c+di)); consumer the `matmul` method at `:302-305` composes 4× `crate::ops::linalg::mm` calls — `mm` outputs feed into elementwise add/sub on `Vec<T>` reals |
//! | REQ-6 (conj) | SHIPPED | `conj` at `complex_tensor.rs:219`; consumer `lib.rs:136` re-export; test `complex_conj_negates_imag` at `:537` |
//! | REQ-7 (abs/angle) | SHIPPED | `abs` at `complex_tensor.rs:230`, `angle` at `:241`; consumer `lib.rs:136` re-export; tests `complex_abs_pythagorean` at `:546`, `complex_angle_quadrants` at `:553` |
//! | REQ-8 (matmul) | SHIPPED | `matmul` at `complex_tensor.rs:259-321` via 4× `crate::ops::linalg::mm`; consumer `lib.rs:136` re-export; tests `complex_matmul_2x2_known_value` at `:612` + real-equivalence regression at `:642` |
//! | REQ-9 (FFT bridge) | SHIPPED | `fft`/`ifft`/`fft2`/`ifft2` at `complex_tensor.rs:327-351`; consumer `lib.rs:136` re-export; test `complex_fft_ifft_roundtrip` at `:657`, `complex_fft2_ifft2_roundtrip` at `:678` |
//! | REQ-10 (reshape) | SHIPPED | `reshape` at `complex_tensor.rs:355` via `Arc::clone`; consumer `lib.rs:136` re-export; test `complex_reshape_preserves_data` at `:571` |
//! | REQ-11 (0-D vs zero-axis) | SHIPPED | `shape.is_empty() { 1 } else { product }` at `complex_tensor.rs:42, :81, :117, :357`; consumer `scalar(re, im)` at `:94` returns 0-D tensor — #805 |
//! | REQ-12 (structured errors) | SHIPPED | `ShapeMismatch` at multiple sites + `InvalidArgument` at `:108`; no `panic!` in production paths; consumers propagate via `?` |
""",
    "ferrotorch-core/src/named_tensor.rs": """//!
//! ## REQ status (per `.design/ferrotorch-core/named_tensor.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (NamedTensor<T> struct) | SHIPPED | `pub struct NamedTensor<T: Float>` at `named_tensor.rs:27` with `Vec<Option<String>>` names; consumer `lib.rs:147` re-export — R-DEFER-1 S5 grandfathering (#621) |
//! | REQ-2 (constructors) | SHIPPED | `new` at `named_tensor.rs:37` (validates count + rejects duplicates at `:48-55`), `refined` at `:61`; consumer `lib.rs:147` re-export; tests `named_tensor_basic_construction` at `:202`, `named_tensor_rejects_duplicate_names` at `:218` |
//! | REQ-3 (accessors) | SHIPPED | `tensor` at `named_tensor.rs:77`, `into_tensor` at `:82`, `names` at `:87`, `shape` at `:92`, `ndim` at `:97`, `numel` at `:102`; consumer `lib.rs:147` re-export |
//! | REQ-4 (lookups) | SHIPPED | `dim_index` at `named_tensor.rs:107`, `size_of` at `:112`; consumer `lib.rs:147` re-export; test `named_tensor_dim_index_lookup` at `:289` |
//! | REQ-5 (rename) | SHIPPED | `rename` at `named_tensor.rs:118-131`; consumer `lib.rs:147` re-export; test `named_tensor_rename_replaces_specified_names` at `:264` |
//! | REQ-6 (align_to) | SHIPPED | `align_to` at `named_tensor.rs:137-163` using `crate::methods::permute_t` at `:160`; consumer `lib.rs:147` re-export + internal `crate::methods::permute_t` call; tests `named_tensor_align_permutes_dims` at `:231`, `named_tensor_align_identity_is_clone` at `:243`, `named_tensor_align_rejects_unknown_name` at `:250` |
//! | REQ-7 (detached) | SHIPPED | `detached` at `named_tensor.rs:167`; consumer `lib.rs:147` re-export; test `named_tensor_detached_drops_names` at `:273` |
//! | REQ-8 (Display) | SHIPPED | `Display` at `named_tensor.rs:175-189`; consumer `lib.rs:147` re-export + every `format!("{}", nt)` callsite |
//! | REQ-9 (structured errors) | SHIPPED | `ShapeMismatch` at `:39, :139`; `InvalidArgument` at `:51, :150`; no `panic!`/`unwrap`/`expect` in production paths; consumers propagate via `?` |
""",
    "ferrotorch-core/src/nested.rs": """//! `NestedTensor` and `PackedNestedTensor` — ragged (jagged) tensors that
//! mirror `torch.nested.nested_tensor` (`aten/src/ATen/native/nested/`) +
//! the jagged-layout NJT (`torch/nested/_internal/nested_tensor.py`).
//!
//! ## REQ status (per `.design/ferrotorch-core/nested.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (NestedTensor::new) | SHIPPED | `NestedTensor::new` at `nested.rs:50-96` validates ndim + non-ragged shape parity; consumer `lib.rs:172` `pub use nested::{NestedTensor, ...}` — R-DEFER-1 S5 grandfathering (#806, #291) |
//! | REQ-2 (accessors) | SHIPPED | `num_components` at `nested.rs:100`, `ragged_dim` at `:106`, `tensors` at `:112`, `ndim` at `:118`, `consistent_shape` at `:123`, `ragged_lengths` at `:128`; consumer `lib.rs:172` re-export + internal GPU fast-path uses |
//! | REQ-3 (to_padded + GPU fast path) | SHIPPED | `to_padded` at `nested.rs:163-240` + GPU fast path `try_to_padded_gpu` at `:258-377`; consumer `lib.rs:172` re-export — R-DEFER-1 S5 grandfathering |
//! | REQ-4 (from_padded) | SHIPPED | `from_padded` at `nested.rs:401+` with GPU fast path at `:450-454` via `try_from_padded_gpu`; consumer `lib.rs:172` re-export |
//! | REQ-5 (nested SDPA) | SHIPPED | `pub fn nested_scaled_dot_product_attention<T: Float>` at `nested.rs:657-770` with GPU FlashAttention dispatch `try_flash_attention_gpu_component` at `:775`; consumer `lib.rs:172` re-exports `nested_scaled_dot_product_attention` |
//! | REQ-6 (PackedNestedTensor) | SHIPPED | `pub struct PackedNestedTensor<T: Float>` at `nested.rs:938`; constructor `from_sequences` at `:967-1010+`; consumer `lib.rs:172` re-export — R-DEFER-1 S5 grandfathering (#291) |
//! | REQ-7 (structured errors) | SHIPPED | `InvalidArgument`/`ShapeMismatch`/`DeviceMismatch` at multiple sites; no `panic!` in production paths; consumers propagate via `?` |
""",
    "ferrotorch-core/src/numeric_cast.rs": """//!
//! ## REQ status (per `.design/ferrotorch-core/numeric_cast.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (fallible cast<T,U>) | SHIPPED | `pub fn cast<T, U>` at `numeric_cast.rs:71-114` with structured `FerrotorchError::InvalidArgument` on failure (`:77` and `:100`); consumer `fft.rs:30` `use crate::numeric_cast::cast` + downstream FFT scale-factor callsites |
//! | REQ-2 (saturation guard) | SHIPPED | guard block at `numeric_cast.rs:96-110` comparing finite-source vs non-finite-result; tests 4 saturation regressions at `:166-220` + 5 passthrough tests at `:188-235`; consumer `fft.rs:30` |
//! | REQ-3 (integer-target no-op cost) | SHIPPED | guard at `:96-110` is no-op for integer targets (integers always project to finite f64); test `cast_f64_inf_to_i32_fails` at `:126`; consumer any `cast::<f64, i32>(...)` callsite — dtype-agnostic by construction |
//! | REQ-4 (structured message) | SHIPPED | error messages include `type_name::<T>()`, `type_name::<U>()`, `{:?}` of source at `numeric_cast.rs:78-83` and `:101-108`; test `cast_huge_f64_to_bf16_returns_err` at `:172` asserts substring; consumer error propagation via `?` |
//! | REQ-5 (`#[inline]`) | SHIPPED | attribute at `numeric_cast.rs:70`; consumer `fft.rs` callsites — `#[inline]` ensures the cast collapses into the FFT pipeline |
""",
    "ferrotorch-core/src/ops/mod.rs": """//! Kernel-layer op-module declarations. Mirrors `aten/src/ATen/native/`'s
//! directory-as-namespace convention. Each declared sub-module is the
//! forward-only (no autograd) op family for its area; the autograd
//! wrappers live in `ferrotorch-core/src/grad_fns/`.
//!
//! ## REQ status (per `.design/ferrotorch-core/ops/mod.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (9 sub-modules) | SHIPPED | 9 `pub mod` declarations at `ops/mod.rs:1-9`; consumer `grad_fns/cumulative.rs:32` (`use crate::ops::cumulative::{...}`), `grad_fns/transcendental.rs:15` (`use crate::ops::elementwise::{fast_cos, fast_sin, unary_map}`), `tensor.rs:1146` (`crate::ops::indexing::masked_select`) |
//! | REQ-2 (kernel/autograd split) | SHIPPED | the kernel-layer `ops::<family>` vs autograd-layer `grad_fns::<family>` split IS the organizational primitive; consumer `grad_fns/cumulative.rs:32-35` imports from `crate::ops::cumulative` then `pub fn cumsum` at `grad_fns/cumulative.rs:104` delegates the forward to `ops::cumulative::cumsum_forward(...)` — mirrors upstream `aten::cummax` (user) vs `_cummax_helper` (private) split 1:1 |
//! | REQ-3 (no module-level re-exports) | SHIPPED | this file has zero `pub use` (mechanical: 9 `pub mod` lines only); consumer `lib.rs:173-177` `pub use ops::indexing::{gather, masked_select, scatter, ...}` lifts specific symbols — the picking-by-symbol pattern requires the sub-modules NOT pre-re-export, which mod.rs preserves by being a pure-declaration file |
""",
}


def has_req_status(content: str) -> bool:
    return "## REQ status" in content


def inject(path: Path, block: str) -> str:
    """Insert `block` into the file content at the right anchor. Returns
    the new content. Idempotent: if the file already contains `## REQ
    status`, returns content unchanged."""
    content = path.read_text()
    if has_req_status(content):
        return content

    lines = content.splitlines(keepends=True)

    # Find the insertion point.
    #
    # Strategy:
    #   1. Skip an initial run of `#![...]` crate-root attributes
    #      (only matters for lib.rs).
    #   2. Skip an initial run of `// ...` non-doc-comments
    #      (also only lib.rs has those).
    #   3. If the next line is a `//!` line, find the end of the
    #      contiguous `//!` block (including blank `//!` continuation
    #      lines) and insert AFTER that.
    #   4. Otherwise, insert at the position found in steps 1-2.

    i = 0
    n = len(lines)

    # Step 1+2: skip crate-root attributes and leading non-doc comments.
    # These only happen for lib.rs.
    while i < n:
        line = lines[i].lstrip()
        if line.startswith("#![") or line.startswith("// "):
            i += 1
            continue
        # Allow continuation of a #![ ... ] block across multiple lines
        # (the lint-allow block).
        if i > 0 and lines[i - 1].lstrip().startswith("#!["):
            # We may have continuation lines of a previous attribute.
            # Detect closing `)]`.
            i += 1
            continue
        break

    # Skip a closing `)]` line if we ended in mid-attribute.
    while i < n and lines[i].strip() in (")]", ")", "]"):
        i += 1

    # Step 3: if we're now at a //! line, walk to end of that block.
    if i < n and lines[i].lstrip().startswith("//!"):
        while i < n and (
            lines[i].lstrip().startswith("//!") or lines[i].strip() == ""
        ):
            # blank lines INSIDE a //! block — only treat them as part
            # of the block if the next non-blank line is still //!
            if lines[i].strip() == "":
                # Look ahead: is the next non-blank line still //!?
                j = i + 1
                while j < n and lines[j].strip() == "":
                    j += 1
                if j < n and lines[j].lstrip().startswith("//!"):
                    i += 1
                    continue
                else:
                    # blank line is NOT inside the //! block; stop here.
                    break
            i += 1

    # Insertion point is now `i`. Insert `block` here with a leading
    # blank line separator.
    if i > 0 and lines[i - 1].strip() != "":
        block = "\n" + block
    if i < n and lines[i].strip() != "" and not lines[i].lstrip().startswith("//!"):
        block = block + "\n"
    new_lines = lines[:i] + [block] + lines[i:]
    return "".join(new_lines)


def main():
    updated = 0
    skipped = 0
    for rel, block in REQ_TABLES.items():
        path = ROOT / rel
        if not path.exists():
            print(f"MISSING: {rel}", file=sys.stderr)
            continue
        new_content = inject(path, block)
        if new_content == path.read_text():
            print(f"SKIP (already has REQ status): {rel}")
            skipped += 1
        else:
            path.write_text(new_content)
            print(f"UPDATED: {rel}")
            updated += 1
    print(f"\nTotal: {updated} updated, {skipped} skipped")


if __name__ == "__main__":
    main()
