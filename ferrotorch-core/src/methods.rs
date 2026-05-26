//! Method-style API for Tensor operations.
//!
//! Enables `a.matmul(&b)`, `a.relu()`, `a.sum()`, `a.reshape(&[2, 3])` etc.
//! All methods delegate to the corresponding grad_fns or ops functions.

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

impl<T: Float> Tensor<T> {
    // --- Arithmetic ---

    pub fn add_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::add(self, other)
    }

    pub fn sub_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::sub(self, other)
    }

    /// `torch.Tensor.rsub(other, *, alpha=1)` — reverse subtract:
    /// `self - alpha * other` is the `sub_t` semantic; rsub is the
    /// operand-swapped variant returning `other - alpha * self`.
    ///
    /// Per upstream `aten/src/ATen/native/BinaryOps.cpp:1169 Tensor rsub(
    /// const Tensor& self, const Tensor& other, const Scalar& alpha) {
    /// return at::sub(other, self, alpha); }` — a literal operand-swap
    /// delegation. The non-test production consumer wiring for
    /// `arithmetic::rsub` per R-DEFER-1: this method is the public,
    /// chainable surface that closes the consumer requirement.
    pub fn rsub_t(&self, other: &Tensor<T>, alpha: f64) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::rsub(self, other, alpha)
    }

    pub fn mul_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::mul(self, other)
    }

    pub fn div_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::div(self, other)
    }

    pub fn neg_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::neg(self)
    }

    pub fn pow_t(&self, exponent: f64) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::pow(self, exponent)
    }

    pub fn sqrt_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::sqrt(self)
    }

    /// `torch.Tensor.rsqrt()` — reciprocal square root: `1 / sqrt(self)`.
    ///
    /// Mirrors `torch.rsqrt(input, *, out=None)` per `torch/_torch_docs.py:9656`
    /// and the upstream impl macro at
    /// `aten/src/ATen/native/UnaryOps.cpp:346
    /// CREATE_UNARY_TORCH_IMPL_FUNC(rsqrt_out, rsqrt_stub)`. The non-test
    /// production consumer wiring for `arithmetic::rsqrt` per R-DEFER-1:
    /// this method is the public, chainable surface that closes the
    /// consumer requirement.
    pub fn rsqrt_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::rsqrt(self)
    }

    /// `torch.Tensor.reciprocal()` — elementwise reciprocal: `1 / self`.
    ///
    /// Mirrors `torch.reciprocal(input, *, out=None)` per
    /// `torch/_torch_docs.py:2584` and the upstream impl macro at
    /// `aten/src/ATen/native/UnaryOps.cpp:345
    /// CREATE_UNARY_TORCH_IMPL_FUNC(reciprocal_out, reciprocal_stub)`. The
    /// non-test production consumer wiring for `arithmetic::reciprocal` per
    /// R-DEFER-1: this method is the public, chainable surface that closes
    /// the consumer requirement.
    pub fn reciprocal_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::reciprocal(self)
    }

    pub fn abs_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::abs(self)
    }

    /// `torch.Tensor.remainder(other)` — elementwise remainder with the
    /// **sign of the divisor** (Python `%` / NumPy semantics).
    ///
    /// Mirrors `torch.remainder(input, other, *, out=None)` per
    /// `torch/_torch_docs.py:9453-9472` and the upstream C++ entry at
    /// `aten/src/ATen/native/BinaryOps.cpp:1184 Tensor remainder(const
    /// Tensor& self, const Scalar& other)`. The float-tensor CPU
    /// implementation is at `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:
    /// 391-409 remainder_kernel`. Registration at
    /// `torch/overrides.py:1100 torch.remainder: lambda input, other,
    /// out=None: -1`.
    ///
    /// Distinct from `fmod_t` (dividend-sign / C99 semantics, REQ-14 NOT-
    /// STARTED): for `remainder(-5, 3)` ferrotorch returns `1` (sign
    /// matches divisor `+3`); `fmod(-5, 3)` returns `-2` (sign matches
    /// dividend `-5`).
    ///
    /// The non-test production consumer wiring for `arithmetic::remainder`
    /// per R-DEFER-1: this method is the public, chainable surface that
    /// closes the consumer requirement.
    pub fn remainder_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::remainder(self, other)
    }

    /// `torch.fmod(input, other, *, out=None)` — elementwise remainder
    /// with the sign of the **dividend** (C99 `std::fmod` semantics).
    ///
    /// Mirrors `torch.Tensor.fmod` via the same upstream registration
    /// `torch/overrides.py:666 torch.fmod: lambda input, other, out=None: -1`.
    ///
    /// Distinct from `remainder_t` (divisor-sign, REQ-13 SHIPPED): for
    /// `fmod(-5, 3)` ferrotorch returns `-2` (sign matches dividend
    /// `-5`); `remainder(-5, 3)` returns `1` (sign matches divisor
    /// `+3`). See `arithmetic::fmod` docs for the per-quadrant table.
    ///
    /// The non-test production consumer wiring for `arithmetic::fmod`
    /// per R-DEFER-1: this method is the public, chainable surface that
    /// closes the consumer requirement.
    pub fn fmod_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::fmod(self, other)
    }

    /// `torch.Tensor.floor_divide(other)` — elementwise floor division
    /// (true floor, toward `-infinity`).
    ///
    /// Mirrors `torch.floor_divide(input, other, *, out=None)` per
    /// `torch/_torch_docs.py:4265-4296`:
    ///
    /// > Computes :attr:`input` divided by :attr:`other`, elementwise, and
    /// > floors the result.
    /// >
    /// > .. math::
    /// >     out_i = floor(input_i / other_i)
    ///
    /// Upstream entry at `aten/src/ATen/native/BinaryOps.cpp:979 Tensor
    /// floor_divide(const Tensor& self, const Tensor& other)` dispatching
    /// to `div_floor_stub` -> `div_floor_kernel` at
    /// `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:297-349` ->
    /// `c10::div_floor_floating` at `c10/util/generic_math.h:34-58`.
    /// Registration at `torch/overrides.py:664 torch.floor_divide: lambda
    /// input, other: -1`.
    ///
    /// `torch.floor_divide` was historically broken (performed trunc, NOT
    /// floor) and `torch/_torch_docs.py:4267-4271` explicitly notes:
    ///
    /// > .. note::
    /// >     Before PyTorch 1.13 :func:`torch.floor_divide` incorrectly
    /// >     performed truncation division. To restore the previous
    /// >     behavior use :func:`torch.div` with ``rounding_mode='trunc'``.
    ///
    /// As of PyTorch 1.13+ (and as of the upstream pin this ferrotorch is
    /// translated against), `torch.floor_divide` performs TRUE FLOOR.
    /// Verified live on 2026-05-25:
    /// `torch.floor_divide(-7.0, 3.0).item() == -3.0`.
    ///
    /// Distinct from `remainder_t` and `fmod_t`. The 3-way identity
    /// `a == floor_divide(a,b) * b + remainder(a,b)` holds; the
    /// `fmod` sibling is the trunc-division remainder. For `a=-7, b=3`:
    /// - `floor_divide(-7, 3) = -3` (true floor)
    /// - `remainder(-7, 3) = 2`     (sign of divisor)
    /// - `fmod(-7, 3) = -1`         (sign of dividend / trunc remainder)
    ///
    /// Backward: `torch.floor_divide` has no derivative — verified live
    /// `grad_fn=<NotImplemented object>` raises `derivative for
    /// aten::floor_divide is not implemented`. `FloorDivideBackward`
    /// mirrors that by erroring on `.backward()`.
    ///
    /// The non-test production consumer wiring for
    /// `arithmetic::floor_divide` per R-DEFER-1: this method is the
    /// public, chainable surface that closes the consumer requirement.
    pub fn floor_divide_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::floor_divide(self, other)
    }

    /// `torch.Tensor.addcmul(tensor1, tensor2, *, value=1)` — fused
    /// `self + value * tensor1 * tensor2` (receiver is `input`).
    ///
    /// Mirrors `torch.addcmul(input, tensor1, tensor2, *, value=1, out=None)`
    /// per `torch/_torch_docs.py:510-544`:
    ///
    /// > Performs the element-wise multiplication of :attr:`tensor1` by
    /// > :attr:`tensor2`, multiplies the result by the scalar :attr:`value`
    /// > and adds it to :attr:`input`.
    /// >
    /// > .. math::
    /// >     \text{out}_i = \text{input}_i + \text{value} \times \text{tensor1}_i \times \text{tensor2}_i
    ///
    /// Upstream C++ entry at `aten/src/ATen/native/PointwiseOps.cpp:57-64
    /// TORCH_IMPL_FUNC(addcmul_out)`. Registration at
    /// `torch/overrides.py:462 torch.addcmul: lambda input, tensor1, tensor2,
    /// value=1, out=None: -1`.
    ///
    /// Broadcasting: the 3 input tensors (`self`, `tensor1`, `tensor2`) are
    /// jointly broadcast to a common output shape. Backward: per
    /// `tools/autograd/derivatives.yaml`, `d_input = grad`, `d_tensor1 =
    /// grad * value * tensor2`, `d_tensor2 = grad * value * tensor1` (no
    /// gradient with respect to the scalar `value`).
    ///
    /// The non-test production consumer wiring for `arithmetic::addcmul`
    /// per R-DEFER-1: this method is the public, chainable surface that
    /// closes the consumer requirement.
    pub fn addcmul_t(
        &self,
        tensor1: &Tensor<T>,
        tensor2: &Tensor<T>,
        value: f64,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::addcmul(self, tensor1, tensor2, value)
    }

    /// `torch.Tensor.addcdiv(tensor1, tensor2, *, value=1)` — fused
    /// `self + value * tensor1 / tensor2` (receiver is `input`).
    ///
    /// Mirrors `torch.addcdiv(input, tensor1, tensor2, *, value=1, out=None)`
    /// per `torch/_torch_docs.py:461-473`:
    ///
    /// > Performs the element-wise division of :attr:`tensor1` by
    /// > :attr:`tensor2`, multiplies the result by the scalar :attr:`value`
    /// > and adds it to :attr:`input`.
    /// >
    /// > .. math::
    /// >     \text{out}_i = \text{input}_i + \text{value} \times
    /// >                    \frac{\text{tensor1}_i}{\text{tensor2}_i}
    ///
    /// Upstream C++ entry at `aten/src/ATen/native/PointwiseOps.cpp:66-73
    /// TORCH_IMPL_FUNC(addcdiv_out)`. The integer-dtype deprecation block at
    /// `PointwiseOps.cpp:38-50 TORCH_META_FUNC(addcdiv)` is unreachable for
    /// the `Tensor<T: Float>` family.
    ///
    /// Broadcasting: the 3 input tensors (`self`, `tensor1`, `tensor2`) are
    /// jointly broadcast to a common output shape. Backward: per
    /// `tools/autograd/derivatives.yaml`, `d_input = grad`, `d_tensor1 =
    /// grad * value / tensor2`, `d_tensor2 = -grad * value * tensor1 /
    /// (tensor2 * tensor2)` (no gradient with respect to the scalar
    /// `value`). At `tensor2=0` the d_tensor2 path produces NaN / ±Inf via
    /// IEEE-754 — matches upstream (R-DEV-1).
    ///
    /// The non-test production consumer wiring for `arithmetic::addcdiv`
    /// per R-DEFER-1: this method is the public, chainable surface that
    /// closes the consumer requirement.
    pub fn addcdiv_t(
        &self,
        tensor1: &Tensor<T>,
        tensor2: &Tensor<T>,
        value: f64,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::addcdiv(self, tensor1, tensor2, value)
    }

    // --- Cumulative (scan) ---

    /// `torch.Tensor.cumsum(dim)` — cumulative sum along `dim`.
    ///
    /// Mirrors `torch.cumsum(input, dim, *, dtype=None, out=None)` per
    /// `torch/_torch_docs.py:3429 cumsum(input, dim, *, dtype=None,
    /// out=None) -> Tensor` and the `torch.Tensor` method docstring at
    /// `torch/_tensor_docs.py:1500-1506 add_docstr_all("cumsum", r"""
    /// cumsum(dim, dtype=None) -> Tensor [...] See :func:`torch.cumsum``.
    /// Upstream C++ entry at `aten/src/ATen/native/ReduceOps.cpp:511
    /// TORCH_IMPL_FUNC(cumsum_out)` dispatching `cumsum_stub`. Autograd
    /// VJP per `tools/autograd/derivatives.yaml:529-531 (name: cumsum(
    /// Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor; self:
    /// cumsum_backward(grad.to(self.scalar_type()), dim))` which is the
    /// `reverse_cumsum` (flip → cumsum → flip) upper-triangular
    /// multiplication at `ReduceOps.cpp:527-529 static Tensor
    /// reversed_cumsum(const Tensor& w, int64_t dim)`.
    ///
    /// ferrotorch does NOT accept the `dtype` kwarg (the dtype-promotion
    /// branch at `ReduceOps.cpp:267` is unreachable for the `Tensor<T:
    /// Float>` family — see `.design/ferrotorch-core/grad_fns/
    /// cumulative.md` REQ-1).
    ///
    /// The non-test production consumer wiring for
    /// `grad_fns::cumulative::cumsum` per R-DEFER-1: this method is the
    /// public, chainable surface that closes the consumer requirement
    /// (blocker #1232).
    pub fn cumsum_t(&self, dim: i64) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::cumulative::cumsum(self, dim)
    }

    /// `torch.Tensor.cumprod(dim)` — cumulative product along `dim`.
    ///
    /// Mirrors `torch.cumprod(input, dim, *, dtype=None, out=None)` per
    /// `torch/_torch_docs.py:3390 cumprod(input, dim, *, dtype=None,
    /// out=None) -> Tensor` and the `torch.Tensor` method docstring at
    /// `torch/_tensor_docs.py:1482-1488 add_docstr_all("cumprod", r"""
    /// cumprod(dim, dtype=None) -> Tensor [...] See :func:`torch.cumprod`.
    /// Upstream C++ entry at `aten/src/ATen/native/ReduceOps.cpp:519
    /// TORCH_IMPL_FUNC(cumprod_out)`. Autograd VJP per
    /// `tools/autograd/derivatives.yaml:525-527 (name: cumprod(Tensor
    /// self, int dim, *, ScalarType? dtype=None) -> Tensor; self:
    /// cumprod_backward(grad.to(self.scalar_type()), self, dim, result))`
    /// routing through `cumprod_backward` at `ReduceOps.cpp:531-790`
    /// with the zeros-aware reverse-cumsum-divide algorithm.
    ///
    /// ferrotorch does NOT accept the `dtype` kwarg; the zeros-present
    /// path uses an O(n^3) brute-force backward rather than upstream's
    /// composite-compliance masked-fill (numerically identical, slower,
    /// not second-order-differentiable — see
    /// `.design/ferrotorch-core/grad_fns/cumulative.md` REQ-2).
    ///
    /// The non-test production consumer wiring for
    /// `grad_fns::cumulative::cumprod` per R-DEFER-1: this method is the
    /// public, chainable surface that closes the consumer requirement
    /// (blocker #1232).
    pub fn cumprod_t(&self, dim: i64) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::cumulative::cumprod(self, dim)
    }

    /// `torch.Tensor.logcumsumexp(dim)` — numerically stable
    /// `log(cumsum(exp(self)))` along `dim`.
    ///
    /// Mirrors `torch.logcumsumexp(input, dim, *, out=None)` per
    /// `torch/_torch_docs.py:3298 logcumsumexp(input, dim, *, out=None)
    /// -> Tensor` and the `torch.Tensor` method docstring at
    /// `torch/_tensor_docs.py:1455-1462 add_docstr_all("logcumsumexp",
    /// r""" logcumsumexp(dim) -> Tensor [...] See
    /// :func:`torch.logcumsumexp``. Upstream C++ entry at
    /// `aten/src/ATen/native/ReduceOps.cpp:475 Tensor logcumsumexp(const
    /// Tensor& self, int64_t dim)` dispatching `_logcumsumexp_cpu` at
    /// `:465-468` → `logcumsumexp_stub` at `:471`. Autograd VJP per
    /// `tools/autograd/derivatives.yaml:521-523 (name: logcumsumexp(
    /// Tensor self, int dim) -> Tensor; self: logcumsumexp_backward(grad,
    /// self, result, dim))` factors as `grad_input[i] = exp(input[i]) *
    /// reverse_cumsum(grad_output * exp(-output))` (softmax-weighted
    /// reverse cumsum).
    ///
    /// The numerical-stability invariant (large inputs ~1000.0 stay
    /// finite) is preserved by the two-pass max-rescaling forward
    /// algorithm at `ops/cumulative.rs:378-410`. See
    /// `.design/ferrotorch-core/grad_fns/cumulative.md` REQ-5.
    ///
    /// The non-test production consumer wiring for
    /// `grad_fns::cumulative::logcumsumexp` per R-DEFER-1: this method
    /// is the public, chainable surface that closes the consumer
    /// requirement (blocker #1232).
    pub fn logcumsumexp_t(&self, dim: i64) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::cumulative::logcumsumexp(self, dim)
    }

    // --- Transcendental ---

    pub fn exp_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::transcendental::exp(self)
    }

    pub fn log_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::transcendental::log(self)
    }

    pub fn sin_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::transcendental::sin(self)
    }

    pub fn cos_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::transcendental::cos(self)
    }

    pub fn clamp_t(&self, min: T, max: T) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::transcendental::clamp(self, min, max)
    }

    // --- Activation ---

    pub fn relu(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::relu(self)
    }

    pub fn sigmoid(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::sigmoid(self)
    }

    pub fn tanh_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::tanh(self)
    }

    pub fn gelu(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::gelu(self)
    }

    pub fn gelu_with(
        &self,
        approximate: crate::grad_fns::activation::GeluApproximate,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::gelu_with(self, approximate)
    }

    pub fn silu(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::silu(self)
    }

    pub fn softmax(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::softmax(self)
    }

    pub fn log_softmax(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::log_softmax(self)
    }

    // --- Reduction ---

    pub fn sum_all(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::sum(self)
    }

    pub fn mean_all(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::mean(self)
    }

    pub fn prod_all(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::prod(self)
    }

    /// Global minimum across all elements. Mirrors `torch.amin(self)` with
    /// no `dim` argument. Returns a 0-d tensor. On CUDA f32/f64, dispatches
    /// to the native PTX reduce_min kernel; on CPU walks the buffer. (#627)
    pub fn amin(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::amin(self)
    }

    /// Global maximum across all elements. Mirrors `torch.amax(self)`. (#627)
    pub fn amax(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::amax(self)
    }

    /// LU factorization in cuSOLVER's packed form: returns
    /// `(LU_packed, pivots)`. Mirrors `torch.linalg.lu_factor`. On CUDA
    /// f32/f64, runs natively via cuSOLVER `getrf` with no host bounce
    /// for the matrix; pivots come back as a host `Vec<i32>` (O(n)). (#604)
    pub fn lu_factor(&self) -> FerrotorchResult<(Tensor<T>, Vec<i32>)> {
        crate::linalg::lu_factor(self)
    }

    // --- Linalg ---

    pub fn matmul(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::matmul_differentiable(self, other)
    }

    pub fn mm(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::mm_differentiable(self, other)
    }

    /// Fused A @ B^T — avoids materializing the transpose of B.
    /// A: [M, K], B: [N, K] -> [M, N].
    pub fn mm_bt(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::mm_bt_differentiable(self, other)
    }

    pub fn bmm(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::bmm_differentiable(self, other)
    }

    pub fn mv_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::mv_differentiable(self, other)
    }

    pub fn dot_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::dot_differentiable(self, other)
    }

    pub fn t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::shape::transpose_2d(self)
    }

    /// Einstein summation with this tensor as the first operand.
    ///
    /// `others` contains the remaining input tensors (if any). The equation
    /// must include subscripts for `self` followed by the `others`.
    ///
    /// ```ignore
    /// // Matrix multiply: self @ other
    /// let c = a.einsum("ij,jk->ik", &[&b])?;
    ///
    /// // Trace of self
    /// let t = a.einsum("ii->", &[])?;
    /// ```
    pub fn einsum(&self, equation: &str, others: &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
        let mut inputs: Vec<&Tensor<T>> = vec![self];
        inputs.extend_from_slice(others);
        crate::einsum::einsum_differentiable(equation, &inputs)
    }

    // --- Reduction (dim) ---

    pub fn sum_dim(&self, dim: i64, keepdim: bool) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::sum_dim(self, dim, keepdim)
    }

    pub fn mean_dim(&self, dim: i64, keepdim: bool) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::mean_dim(self, dim, keepdim)
    }

    // --- Shape ---

    pub fn reshape_t(&self, shape: &[isize]) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::shape::reshape(self, shape)
    }

    pub fn flatten_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::shape::flatten(self)
    }

    pub fn squeeze_t(&self, axis: isize) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::shape::squeeze(self, axis)
    }

    pub fn unsqueeze_t(&self, axis: isize) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::shape::unsqueeze(self, axis)
    }

    /// Permute tensor dimensions. Like PyTorch's `tensor.permute(dims)`.
    ///
    /// Zero-copy: returns a view with permuted shape and strides.
    /// `dims` must be a valid permutation of `0..ndim`.
    pub fn permute(&self, dims: &[usize]) -> FerrotorchResult<Tensor<T>> {
        permute_t(self, dims)
    }

    /// Swap two dimensions. Like PyTorch's `tensor.transpose(dim0, dim1)`.
    ///
    /// Zero-copy: returns a view with swapped strides.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> FerrotorchResult<Tensor<T>> {
        let ndim = self.ndim();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(crate::error::FerrotorchError::InvalidArgument {
                message: format!("transpose: dims ({dim0}, {dim1}) out of bounds for ndim {ndim}"),
            });
        }
        if dim0 == dim1 {
            return Ok(self.clone());
        }
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(dim0, dim1);
        permute_t(self, &perm)
    }

    /// Return a narrowed view along `dim` starting at `start` with `length`
    /// elements. Like PyTorch's `tensor.narrow(dim, start, length)`.
    ///
    /// Zero-copy: shares storage with the original tensor.
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> FerrotorchResult<Tensor<T>> {
        narrow_t(self, dim, start, length)
    }

    /// View tensor with new shape. Like PyTorch's `tensor.view(shape)`.
    ///
    /// Exactly one dimension may be `-1`, in which case it is inferred.
    /// Requires the tensor to be contiguous.
    pub fn view(&self, shape: &[i64]) -> FerrotorchResult<Tensor<T>> {
        view_t(self, shape)
    }

    /// Make tensor contiguous — if already contiguous, returns a cheap clone.
    /// Otherwise materializes a new contiguous buffer.
    pub fn contiguous(&self) -> FerrotorchResult<Tensor<T>> {
        contiguous_t(self)
    }

    /// Split tensor into `chunks` roughly equal pieces along `dim`.
    pub fn chunk(&self, chunks: usize, dim: usize) -> FerrotorchResult<Vec<Tensor<T>>> {
        chunk_t(self, chunks, dim)
    }

    /// Split tensor into pieces of given sizes along `dim`.
    pub fn split(&self, split_sizes: &[usize], dim: usize) -> FerrotorchResult<Vec<Tensor<T>>> {
        split_t(self, split_sizes, dim)
    }

    // --- Quantization ---

    /// `torch.Tensor.fake_quantize_per_tensor_affine(scale, zero_point,
    /// quant_min, quant_max)` — per-tensor affine fake quantization with
    /// autograd-tracked clipped STE backward.
    ///
    /// Mirrors `torch.fake_quantize_per_tensor_affine` per
    /// `torch/overrides.py:622 torch.fake_quantize_per_tensor_affine: lambda
    /// input, scale, zero_point, quant_min, quant_max: -1` and the upstream
    /// implementation at `aten/src/ATen/native/quantized/
    /// FakeQuantPerTensorAffine.cpp:31-40 Tensor fake_quantize_per_tensor_affine(
    /// const Tensor& self, double scale, int64_t zero_point, int64_t quant_min,
    /// int64_t quant_max)`. Backward per `tools/autograd/derivatives.yaml:673-674
    /// fake_quantize_per_tensor_affine_cachemask_backward(grad, mask)` returning
    /// `dY * mask` where the mask is `1` iff
    /// `quant_min <= round_ties_even(input/scale) + zero_point <= quant_max`.
    ///
    /// The non-test production consumer wiring for
    /// `grad_fns::quantize_grad::fake_quantize_per_tensor_affine` per
    /// R-DEFER-1: this method is the public, chainable surface that closes
    /// the consumer requirement for the per-tensor variant (blocker #1238).
    pub fn fake_quantize_per_tensor_affine_t(
        &self,
        scale: f64,
        zero_point: i64,
        quant_min: i64,
        quant_max: i64,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::quantize_grad::fake_quantize_per_tensor_affine(
            self, scale, zero_point, quant_min, quant_max,
        )
    }

    /// `torch.Tensor.fake_quantize_per_channel_affine(scale, zero_point, axis,
    /// quant_min, quant_max)` — per-channel affine fake quantization with
    /// autograd-tracked clipped STE backward.
    ///
    /// Mirrors `torch.fake_quantize_per_channel_affine` per
    /// `torch/overrides.py:621 torch.fake_quantize_per_channel_affine: lambda
    /// input, scale, zero_point, axis, quant_min, quant_max: -1` and the
    /// upstream implementation at `aten/src/ATen/native/quantized/
    /// FakeQuantPerChannelAffine.cpp:32-42 Tensor fake_quantize_per_channel_affine(
    /// const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    /// int64_t axis, int64_t quant_min, int64_t quant_max)`. Backward per
    /// `tools/autograd/derivatives.yaml fake_quantize_per_channel_affine_cachemask_backward(
    /// grad, mask)` returning `dY * mask` where the per-channel mask is `1`
    /// iff `quant_min <= round_ties_even(input/scale[c]) + zero_point[c]
    /// <= quant_max` for the channel `c` along `axis`.
    ///
    /// The non-test production consumer wiring for
    /// `grad_fns::quantize_grad::fake_quantize_per_channel_affine` per
    /// R-DEFER-1: this method is the public, chainable surface that closes
    /// the consumer requirement for the per-channel variant (blocker #1239).
    pub fn fake_quantize_per_channel_affine_t(
        &self,
        scale: &Tensor<T>,
        zero_point: &crate::int_tensor::IntTensor<i64>,
        axis: i64,
        quant_min: i64,
        quant_max: i64,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::quantize_grad::fake_quantize_per_channel_affine(
            self, scale, zero_point, axis, quant_min, quant_max,
        )
    }

    // --- Indexing (REQ-8 from `.design/ferrotorch-core/grad_fns/indexing.md`) ---

    /// `torch.Tensor.index_fill(dim, index, value)` — overwrite slices along
    /// `dim` at `index` positions with the scalar `value`.
    ///
    /// Mirrors `torch.index_fill(input, dim, index, value)` per the upstream
    /// docstring at `torch/_torch_docs.py:6563-6567 index_fill(dim, index,
    /// value) -> Tensor [...] Out-of-place version of :meth:`torch.Tensor.
    /// index_fill_`` and `torch/_tensor_docs.py:2489-2509` which gives the
    /// canonical example
    ///
    /// ```text
    /// >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    /// >>> index = torch.tensor([0, 2])
    /// >>> x.index_fill_(1, index, -1)
    /// tensor([[-1.,  2., -1.],
    ///         [-1.,  5., -1.],
    ///         [-1.,  8., -1.]])
    /// ```
    ///
    /// Upstream C++ entry at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:
    /// 1979 Tensor index_fill(const Tensor& self, int64_t dim, const Tensor&
    /// index, const Scalar& source) { return self.clone(at::MemoryFormat::
    /// Preserve).index_fill_(dim, index, source); }`. Registration at
    /// `torch/overrides.py:710 torch.index_fill: lambda input, dim, index,
    /// value: -1`.
    ///
    /// Backward per `tools/autograd/derivatives.yaml:884-887`:
    /// `- name: index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor`
    /// / `self: grad.index_fill(dim, index, 0)` /
    /// `index: non_differentiable` /
    /// `result: self_t.index_fill(dim, index, 0)`
    /// — gradient is zeroed at every position the fill overwrote (those
    /// positions were replaced by a constant and no longer depend on the
    /// input).
    ///
    /// `dim` follows PyTorch's negative-wrapping convention (`at::maybe_wrap_dim`
    /// at `TensorAdvancedIndexing.cpp:1919`). The `index` tensor must be 1-D
    /// or scalar (upstream `TORCH_CHECK(index.dim() <= 1)` at `:1920`).
    /// Negative index values are accepted and wrapped per upstream's
    /// `index_fill_kernel` at `aten/src/ATen/native/cpu/IndexKernel.cpp:
    /// 224-229` (`TORCH_CHECK_INDEX(idx >= -self_dim_size && idx <
    /// self_dim_size, ...); if (idx < 0) { idx += self_dim_size; }`). Indices
    /// strictly outside `[-dim_size, dim_size)` raise `IndexOutOfBounds`
    /// matching upstream's `TORCH_CHECK_INDEX`. A 0-d input is accepted: the
    /// implementation mirrors upstream's `self.unsqueeze(-1)` at
    /// `TensorAdvancedIndexing.cpp:1917` by treating the scalar as a length-1
    /// 1-d tensor for the fill (only `dim ∈ {-1, 0}` and `index ∈ {-1, 0}`
    /// are in range for that case).
    ///
    /// The non-test production consumer wiring for `grad_fns::indexing::
    /// index_fill` per R-DEFER-1: this method is the public, chainable
    /// surface that closes the consumer requirement (blocker #1249).
    pub fn index_fill_t(
        &self,
        dim: i64,
        index: &crate::int_tensor::IntTensor<i64>,
        value: f64,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::indexing::index_fill(self, dim, index, value)
    }

    /// `torch.Tensor.scatter_reduce(dim, index, src, reduce, *, include_self=True)`
    /// — reduce-mode scatter onto a clone of `self`. Mirrors upstream
    /// `Tensor scatter_reduce(...)` at `aten/src/ATen/native/
    /// TensorAdvancedIndexing.cpp:2354 TORCH_IMPL_FUNC(scatter_reduce_two)`.
    /// `reduce` ∈ {`"sum"` SHIPPED, `"prod"`, `"amax"`, `"amin"`}; backward
    /// is implemented only for `"sum"` per `tools/autograd/derivatives.yaml:
    /// 3074-3077` (other modes return a no-grad tensor — the
    /// op_db characterization sweep emits only `"sum"`).
    ///
    /// Non-test production consumer wiring for `grad_fns::indexing::
    /// scatter_reduce` per R-DEFER-1: this method is the chainable surface.
    /// Closes blocker #1245.
    pub fn scatter_reduce_t(
        &self,
        dim: i64,
        index: &[usize],
        index_shape: &[usize],
        src: &Tensor<T>,
        reduce: &str,
        include_self: bool,
    ) -> FerrotorchResult<Tensor<T>> {
        let mode =
            crate::grad_fns::indexing::ScatterReduce::parse_str(reduce).ok_or_else(|| {
                crate::error::FerrotorchError::InvalidArgument {
                    message: format!(
                        "scatter_reduce_t: unknown reduce mode '{reduce}' \
                     (expected sum|prod|amax|amin)"
                    ),
                }
            })?;
        crate::grad_fns::indexing::scatter_reduce(
            self,
            dim,
            index,
            index_shape,
            src,
            mode,
            include_self,
        )
    }

    /// `torch.Tensor.index_add(dim, index, source, *, alpha=1)` —
    /// `out = self.clone(); out[..., index[i], ...] += alpha * source[..., i, ...]`
    /// along `dim`. Mirrors upstream `Tensor index_add(const Tensor& self,
    /// int64_t dim, const Tensor& index, const Tensor& source, const Scalar&
    /// alpha)` at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1153
    /// TORCH_IMPL_FUNC(index_add_cpu_out)`. Backward per
    /// `tools/autograd/derivatives.yaml:862-869 self: grad / source:
    /// maybe_multiply(grad.index_select(dim, index).expand_as(source), alpha)`.
    ///
    /// Non-test production consumer wiring for `grad_fns::indexing::
    /// index_add` per R-DEFER-1: this method is the chainable surface.
    /// Closes blocker #1247.
    pub fn index_add_t(
        &self,
        dim: i64,
        index: &crate::int_tensor::IntTensor<i64>,
        source: &Tensor<T>,
        alpha: f64,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::indexing::index_add(self, dim, index, source, alpha)
    }

    /// `torch.Tensor.index_copy(dim, index, source)` — `out = self.clone();
    /// out[..., index[i], ...] = source[..., i, ...]` along `dim`. Mirrors
    /// upstream `Tensor index_copy(...)` at `aten/src/ATen/native/
    /// TensorAdvancedIndexing.cpp:1082 TORCH_IMPL_FUNC(index_copy_out)`.
    /// Backward per `tools/autograd/derivatives.yaml:875-883
    /// self: grad.index_fill(dim, index, 0) / source:
    /// grad.index_select(dim, index).expand_as(source)`.
    ///
    /// Non-test production consumer wiring for `grad_fns::indexing::
    /// index_copy` per R-DEFER-1: this method is the chainable surface.
    /// Closes blocker #1248.
    pub fn index_copy_t(
        &self,
        dim: i64,
        index: &crate::int_tensor::IntTensor<i64>,
        source: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::indexing::index_copy(self, dim, index, source)
    }

    /// `torch.Tensor.masked_scatter(mask, source)` — copy elements from
    /// `source` into a clone of `self` at positions where `mask` is true,
    /// in C-order. Mirrors upstream `Tensor masked_scatter(const Tensor&
    /// self, const Tensor& mask, const Tensor& source)` at
    /// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2402-2409`.
    /// Backward per `tools/autograd/derivatives.yaml:1105-1108
    /// self: grad.masked_fill(mask, 0) / source: masked_scatter_backward(...)`.
    ///
    /// Non-test production consumer wiring for `grad_fns::indexing::
    /// masked_scatter` per R-DEFER-1: this method is the chainable surface.
    /// Closes blocker #1252.
    pub fn masked_scatter_t(
        &self,
        mask: &crate::bool_tensor::BoolTensor,
        source: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::indexing::masked_scatter(self, mask, source)
    }

    /// `torch.Tensor.take(index)` — `out[i] = self.view(-1)[index[i]]`, a
    /// flat-index gather producing a tensor of shape `index.shape()`.
    /// Mirrors upstream `Tensor take(const Tensor& self, const Tensor& index)`
    /// at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1067-1071`.
    /// Backward per `tools/autograd/derivatives.yaml:1766-1769
    /// self: take_backward(grad, self, index)` — scatter-add grad into a
    /// zeros buffer at the flat index positions.
    ///
    /// Non-test production consumer wiring for `grad_fns::indexing::take`
    /// per R-DEFER-1: this method is the chainable surface.
    /// Closes blocker #1253.
    pub fn take_t(&self, index: &crate::int_tensor::IntTensor<i64>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::indexing::take(self, index)
    }

    /// `torch.Tensor.put(index, source, accumulate=False)` — flat-index
    /// scatter into a clone of `self`: `out.view(-1)[index[i]] = source[i]`
    /// (or `+= source[i]` when `accumulate=true`). Mirrors upstream
    /// `Tensor put(const Tensor& self, const Tensor& index, const Tensor&
    /// source, const bool accumulate)` at `aten/src/ATen/native/
    /// TensorAdvancedIndexing.cpp:928-934`. Backward per
    /// `tools/autograd/derivatives.yaml:1421-1424`.
    ///
    /// Non-test production consumer wiring for `grad_fns::indexing::put`
    /// per R-DEFER-1: this method is the chainable surface.
    /// Closes blocker #1254.
    pub fn put_t(
        &self,
        index: &crate::int_tensor::IntTensor<i64>,
        source: &Tensor<T>,
        accumulate: bool,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::indexing::put(self, index, source, accumulate)
    }

    // --- PyTorch compatibility aliases ---

    /// Alias for `shape()`. Returns the tensor dimensions like PyTorch's `Tensor.size()`.
    #[inline]
    pub fn size(&self) -> &[usize] {
        self.shape()
    }

    /// Alias for `ndim()`. Returns the number of dimensions like PyTorch's `Tensor.dim()`.
    #[inline]
    pub fn dim(&self) -> usize {
        self.ndim()
    }

    // --- Utility ---

    /// Log the tensor's `Display` form and return `self` for chaining.
    ///
    /// Emits a `tracing::info!` event on target `ferrotorch::tensor`. Behaviour
    /// change vs. earlier versions: this no longer writes directly to stdout —
    /// callers must install a `tracing` subscriber (e.g. `tracing_subscriber`)
    /// to see the output. Library code should not write to stdout; downstream
    /// consumers control logging policy.
    pub fn print(&self) -> &Self {
        tracing::info!(target: "ferrotorch::tensor", "{self}");
        self
    }
}

// ---------------------------------------------------------------------------
// Free functions: permute, view, contiguous, chunk, split
// ---------------------------------------------------------------------------

/// Permute tensor dimensions. Like PyTorch's `tensor.permute(dims)`.
///
/// `dims` must be a valid permutation of `0..ndim`.
pub fn permute_t<T: Float>(input: &Tensor<T>, dims: &[usize]) -> FerrotorchResult<Tensor<T>> {
    use crate::error::FerrotorchError;

    let ndim = input.ndim();
    if dims.len() != ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "permute: dims length {} does not match tensor ndim {}",
                dims.len(),
                ndim
            ),
        });
    }

    // Validate that dims is a valid permutation.
    let mut seen = vec![false; ndim];
    for &d in dims {
        if d >= ndim {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("permute: dim {d} is out of bounds for ndim {ndim}"),
            });
        }
        if seen[d] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("permute: duplicate dim {d} in permutation"),
            });
        }
        seen[d] = true;
    }

    // Zero-copy: permute shape and strides without copying data.
    let in_shape = input.shape();
    let in_strides = input.strides();
    let out_shape: Vec<usize> = dims.iter().map(|&d| in_shape[d]).collect();
    let out_strides: Vec<isize> = dims.iter().map(|&d| in_strides[d]).collect();
    let offset = input.storage_offset();

    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        let grad_fn = std::sync::Arc::new(PermuteBackward {
            input: input.clone(),
            dims: dims.to_vec(),
        });
        Ok(input.stride_view_operation(out_shape, out_strides, offset, grad_fn))
    } else {
        Ok(input.stride_view(out_shape, out_strides, offset))
    }
}

/// Backward for permute: apply the inverse permutation to the gradient.
#[derive(Debug)]
struct PermuteBackward<T: Float> {
    input: Tensor<T>,
    dims: Vec<usize>,
}

impl<T: Float> crate::tensor::GradFn<T> for PermuteBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Compute inverse permutation.
        let mut inv_dims = vec![0usize; self.dims.len()];
        for (i, &d) in self.dims.iter().enumerate() {
            inv_dims[d] = i;
        }
        let grad_input = permute_t(grad_output, &inv_dims)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "PermuteBackward"
    }
}

/// Zero-copy narrow (slice) along a dimension.
///
/// Returns a view with the same storage, adjusting offset and shape.
/// Like PyTorch's `tensor.narrow(dim, start, length)`.
pub fn narrow_t<T: Float>(
    input: &Tensor<T>,
    dim: usize,
    start: usize,
    length: usize,
) -> FerrotorchResult<Tensor<T>> {
    use crate::error::FerrotorchError;

    let ndim = input.ndim();
    if dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("narrow: dim {dim} out of bounds for ndim {ndim}"),
        });
    }
    let dim_size = input.shape()[dim];
    if start + length > dim_size {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "narrow: start({}) + length({}) = {} exceeds dim size {}",
                start,
                length,
                start + length,
                dim_size,
            ),
        });
    }

    let strides = input.strides();
    let mut new_shape = input.shape().to_vec();
    new_shape[dim] = length;

    // Advance offset by start * stride[dim] elements.
    let new_offset = input.storage_offset() + start * strides[dim] as usize;

    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        let grad_fn = std::sync::Arc::new(NarrowBackward {
            input: input.clone(),
            dim,
            start,
        });
        Ok(input.stride_view_operation(new_shape, strides.to_vec(), new_offset, grad_fn))
    } else {
        Ok(input.stride_view(new_shape, strides.to_vec(), new_offset))
    }
}

/// Backward for narrow: pad the gradient with zeros in the sliced dimension.
#[derive(Debug)]
struct NarrowBackward<T: Float> {
    input: Tensor<T>,
    dim: usize,
    start: usize,
}

impl<T: Float> crate::tensor::GradFn<T> for NarrowBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Create a zero tensor matching the input shape and scatter the
        // gradient into the narrowed region.
        let mut grad_data = vec![<T as num_traits::Zero>::zero(); self.input.numel()];
        let grad_out_data = grad_output.data_vec()?;
        let in_shape = self.input.shape();
        let dim = self.dim;
        let start = self.start;
        let _length = grad_output.shape()[dim];

        // Walk contiguous output elements and map to input flat indices.
        let out_strides = crate::shape::c_contiguous_strides(grad_output.shape());
        let in_strides = crate::shape::c_contiguous_strides(in_shape);
        let ndim = in_shape.len();
        let out_numel = grad_out_data.len();

        for (flat, &grad_val) in grad_out_data[..out_numel].iter().enumerate() {
            // Decompose flat index to output coords.
            let mut rem = flat;
            let mut in_flat: usize = 0;
            for d in 0..ndim {
                let coord = rem / out_strides[d] as usize;
                rem %= out_strides[d] as usize;
                let in_coord = if d == dim { coord + start } else { coord };
                in_flat += in_coord * in_strides[d] as usize;
            }
            grad_data[in_flat] = grad_val;
        }

        let device = self.input.device();
        let storage = crate::storage::TensorStorage::on_device(grad_data, device)?;
        let grad_input = Tensor::from_storage(storage, in_shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "NarrowBackward"
    }
}

/// View tensor with new shape. Like PyTorch's `tensor.view(shape)`.
///
/// Exactly one dimension may be `-1`, in which case it is inferred.
/// Requires the tensor to be contiguous (currently all tensors are).
pub fn view_t<T: Float>(input: &Tensor<T>, shape: &[i64]) -> FerrotorchResult<Tensor<T>> {
    use crate::error::FerrotorchError;

    if !input.is_contiguous() {
        return Err(FerrotorchError::InvalidArgument {
            message: "view: tensor must be contiguous; call .contiguous() first".into(),
        });
    }

    // Convert i64 shape to isize for reshape (which handles -1 inference).
    let isize_shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
    crate::grad_fns::shape::reshape(input, &isize_shape)
}

/// Make tensor contiguous (copy data if needed).
///
/// If the tensor is already contiguous this returns a cheap clone.
/// Otherwise it gathers the data in C-order and creates a new
/// contiguous tensor, preserving the original device.
///
/// **GPU fast path (CL-496).** For non-contiguous CUDA tensors of rank
/// ≤ 8, this dispatches to the backend's `strided_copy_{f32,f64}`
/// kernel which gathers the view on-device and avoids the CPU
/// roundtrip that `data_vec()` would otherwise incur. Higher ranks
/// or missing GPU backends fall back to the host-memory path.
pub fn contiguous_t<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    use std::any::TypeId;

    if input.is_contiguous() {
        return Ok(input.clone());
    }
    let device = input.device();

    // GPU fast path: dispatch to the backend's strided_copy kernel
    // when the input is a non-contiguous CUDA tensor with rank ≤ 8.
    if device.is_cuda() && input.shape().len() <= 8 {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let in_handle = input.gpu_handle()?;
            let out_shape = input.shape().to_vec();
            let src_strides = input.strides().to_vec();
            let src_offset = input.storage_offset();

            let out_handle = if TypeId::of::<T>() == TypeId::of::<f32>() {
                backend.strided_copy_f32(in_handle, &out_shape, &src_strides, src_offset)
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                backend.strided_copy_f64(in_handle, &out_shape, &src_strides, src_offset)
            } else {
                // Unsupported dtype — fall through to CPU path.
                return contiguous_t_cpu(input);
            };

            if let Ok(handle) = out_handle {
                let storage = TensorStorage::gpu(handle);
                return if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
                    let grad_fn = std::sync::Arc::new(ContiguousBackward {
                        input: input.clone(),
                    });
                    Tensor::from_operation(storage, out_shape, grad_fn)
                } else {
                    Tensor::from_storage(storage, out_shape, false)
                };
            }
            // Kernel failure (negative strides, overflow, etc.) —
            // fall through to the host path which handles any layout.
        }
    }

    contiguous_t_cpu(input)
}

/// CPU path for [`contiguous_t`]. Always valid for any layout; used
/// as a fallback when the GPU fast path declines or errors.
fn contiguous_t_cpu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    let data = input.data_vec()?;
    let storage = TensorStorage::on_device(data, device)?;

    // Preserve the autograd graph: contiguous is a pure data copy, so the
    // backward is the identity (same shape, same semantics). Without this,
    // calling .contiguous() on a non-contiguous view severs the grad_fn chain.
    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        let grad_fn = std::sync::Arc::new(ContiguousBackward {
            input: input.clone(),
        });
        Tensor::from_operation(storage, input.shape().to_vec(), grad_fn)
    } else {
        Tensor::from_storage(storage, input.shape().to_vec(), false)
    }
}

/// Backward for contiguous: gradient passes through unchanged (identity).
#[derive(Debug)]
struct ContiguousBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> crate::tensor::GradFn<T> for ContiguousBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.input.requires_grad() {
            Ok(vec![Some(grad_output.clone())])
        } else {
            Ok(vec![None])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ContiguousBackward"
    }
}

/// Split tensor into `chunks` roughly equal pieces along `dim`.
///
/// If the tensor size along `dim` is not evenly divisible by `chunks`,
/// the last chunk will be smaller.
pub fn chunk_t<T: Float>(
    input: &Tensor<T>,
    chunks: usize,
    dim: usize,
) -> FerrotorchResult<Vec<Tensor<T>>> {
    use crate::error::FerrotorchError;

    if chunks == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "chunk: chunks must be > 0".into(),
        });
    }

    let shape = input.shape();
    if dim >= shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "chunk: dim {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ),
        });
    }

    let dim_size = shape[dim];
    let chunk_size = dim_size.div_ceil(chunks);
    let mut split_sizes = Vec::new();
    let mut remaining = dim_size;
    while remaining > 0 {
        let s = chunk_size.min(remaining);
        split_sizes.push(s);
        remaining -= s;
    }

    split_t(input, &split_sizes, dim)
}

/// Split tensor into pieces of given sizes along `dim`.
///
/// The sum of `split_sizes` must equal the tensor's size along `dim`.
/// When gradient tracking is enabled and the input requires grad, each
/// output chunk is connected to the autograd graph via `SplitBackward`.
pub fn split_t<T: Float>(
    input: &Tensor<T>,
    split_sizes: &[usize],
    dim: usize,
) -> FerrotorchResult<Vec<Tensor<T>>> {
    use crate::autograd::no_grad::is_grad_enabled;
    use crate::error::FerrotorchError;
    use crate::grad_fns::shape::SplitBackward;
    use crate::storage::TensorStorage;
    use std::any::TypeId;
    use std::sync::Arc;

    let shape = input.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("split: dim {dim} is out of bounds for tensor with {ndim} dimensions"),
        });
    }

    let total: usize = split_sizes.iter().sum();
    if total != shape[dim] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "split: split_sizes sum {} does not match dim {} size {}",
                total, dim, shape[dim]
            ),
        });
    }

    let device = input.device();
    let needs_grad = is_grad_enabled() && input.requires_grad();

    // GPU fast path: use strided_split to extract each chunk directly on GPU.
    if device.is_cuda() && TypeId::of::<T>() == TypeId::of::<f32>() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let inner: usize = if dim + 1 < ndim {
                shape[dim + 1..].iter().product()
            } else {
                1
            };
            let total_along_dim = shape[dim];
            let in_handle = input.gpu_handle()?;

            let mut results = Vec::with_capacity(split_sizes.len());
            let mut offset_along_dim = 0usize;

            for &split_size in split_sizes {
                let mut chunk_shape = shape.to_vec();
                chunk_shape[dim] = split_size;
                let chunk_numel: usize = chunk_shape.iter().product();

                let chunk_handle = backend.strided_split_f32(
                    in_handle,
                    total_along_dim,
                    offset_along_dim,
                    split_size,
                    inner,
                    chunk_numel,
                )?;

                let storage = TensorStorage::gpu(chunk_handle);
                let t = if needs_grad {
                    let grad_fn = Arc::new(SplitBackward::new(
                        input.clone(),
                        dim,
                        offset_along_dim,
                        split_size,
                    ));
                    Tensor::from_operation(storage, chunk_shape, grad_fn)?
                } else {
                    Tensor::from_storage(storage, chunk_shape, false)?
                };
                results.push(t);
                offset_along_dim += split_size;
            }

            return Ok(results);
        }
    }

    // CPU path (also serves as fallback for non-f32 or missing backend).
    let in_data = input.data_vec()?;

    let outer: usize = shape[..dim].iter().product();
    let inner: usize = if dim + 1 < ndim {
        shape[dim + 1..].iter().product()
    } else {
        1
    };
    let total_along_dim = shape[dim];

    let mut results = Vec::with_capacity(split_sizes.len());
    let mut offset_along_dim = 0usize;

    for &split_size in split_sizes {
        let mut chunk_shape = shape.to_vec();
        chunk_shape[dim] = split_size;
        let chunk_numel: usize = chunk_shape.iter().product();
        let mut chunk_data = vec![<T as num_traits::Zero>::zero(); chunk_numel];

        for o in 0..outer {
            let src_start = o * total_along_dim * inner + offset_along_dim * inner;
            let dst_start = o * split_size * inner;
            let row_len = split_size * inner;
            chunk_data[dst_start..dst_start + row_len]
                .copy_from_slice(&in_data[src_start..src_start + row_len]);
        }

        let storage = TensorStorage::on_device(chunk_data, device)?;
        let t = if needs_grad {
            let grad_fn = Arc::new(SplitBackward::new(
                input.clone(),
                dim,
                offset_along_dim,
                split_size,
            ));
            Tensor::from_operation(storage, chunk_shape, grad_fn)?
        } else {
            Tensor::from_storage(storage, chunk_shape, false)?
        };
        results.push(t);
        offset_along_dim += split_size;
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    // reason: relu is pure passthrough or hard-zero; both branches preserve
    // the exact bit pattern (no arithmetic), so equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_method_relu() {
        let a = scalar(2.0f32).unwrap();
        assert_eq!(a.relu().unwrap().item().unwrap(), 2.0);

        let b = scalar(-1.0f32).unwrap();
        assert_eq!(b.relu().unwrap().item().unwrap(), 0.0);
    }

    #[test]
    fn test_method_matmul() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    // reason: sum of small integer-valued floats (1+2+3=6) is bit-exact in
    // any deterministic order — the partial sums never lose mantissa bits,
    // so equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_method_sum() {
        let a = tensor(&[1.0f32, 2.0, 3.0]).unwrap();
        let s = a.sum_all().unwrap();
        assert_eq!(s.item().unwrap(), 6.0);
    }

    #[test]
    fn test_method_transpose() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = a.t().unwrap();
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    // reason: 3^2 = 9 in f32 is bit-exact (small integer power of small
    // integer), and relu of a positive integer is passthrough. The whole
    // chain produces exactly 9.0, so equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_method_chain() {
        let a = scalar(3.0f32).unwrap().requires_grad_(true);
        // a.pow(2).relu().sum() = relu(9) = 9
        let c = a.pow_t(2.0).unwrap().relu().unwrap();
        assert_eq!(c.item().unwrap(), 9.0);
    }

    #[test]
    fn test_method_sigmoid() {
        let a = scalar(0.0f32).unwrap();
        let s = a.sigmoid().unwrap();
        assert!((s.item().unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_method_flatten() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let f = a.flatten_t().unwrap();
        assert_eq!(f.shape(), &[6]);
    }

    #[test]
    fn test_method_print_chain() {
        let a = scalar(42.0f32).unwrap();
        // .print() returns &Self for chaining
        let _ = a.print();
    }

    // --- sum_dim / mean_dim method wrappers ---

    #[test]
    fn test_method_sum_dim() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let s = a.sum_dim(1, false).unwrap();
        assert_eq!(s.shape(), &[2]);
        assert!((s.data().unwrap()[0] - 6.0).abs() < 1e-6);
        assert!((s.data().unwrap()[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_method_mean_dim() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let m = a.mean_dim(0, false).unwrap();
        assert_eq!(m.shape(), &[3]);
        assert!((m.data().unwrap()[0] - 2.5).abs() < 1e-6);
    }

    // --- permute ---

    #[test]
    fn test_method_permute_2d() {
        // Transpose via permute — now zero-copy (stride view).
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = a.permute(&[1, 0]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        // Non-contiguous view — use data_vec() to read logical order.
        assert_eq!(b.data_vec().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // Verify it's a view (shares storage).
        assert!(!b.is_contiguous());
    }

    #[test]
    // reason: permute is pure indexing — it rearranges values without any
    // arithmetic, so each output slot holds the exact bit pattern of the
    // corresponding input slot.
    #[allow(clippy::float_cmp)]
    fn test_method_permute_3d() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let a = from_slice(&data, &[2, 3, 4]).unwrap();
        let b = a.permute(&[2, 0, 1]).unwrap();
        assert_eq!(b.shape(), &[4, 2, 3]);
        let bdata = b.data_vec().unwrap();
        // element [0,0,0] of output = element [0,0,0] of input = 1.0
        assert_eq!(bdata[0], 1.0);
        // element [1,0,0] of output = input[0,0,1] = 2.0
        assert_eq!(bdata[2 * 3], 2.0);
    }

    #[test]
    fn test_permute_invalid_dims() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(a.permute(&[0]).is_err()); // wrong length
        assert!(a.permute(&[0, 0]).is_err()); // duplicate
        assert!(a.permute(&[0, 2]).is_err()); // out of bounds
    }

    // --- view ---

    #[test]
    fn test_method_view() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = a.view(&[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_method_view_infer() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
        let b = a.view(&[2, -1]).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
    }

    // --- contiguous ---

    #[test]
    fn test_method_contiguous() {
        let a = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let b = a.contiguous().unwrap();
        assert_eq!(b.shape(), &[3]);
        assert_eq!(b.data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    // --- chunk ---

    #[test]
    fn test_method_chunk_even() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
        let chunks = a.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].data().unwrap(), &[1.0, 2.0]);
        assert_eq!(chunks[1].data().unwrap(), &[3.0, 4.0]);
        assert_eq!(chunks[2].data().unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn test_method_chunk_uneven() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let chunks = a.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape(), &[2]);
        assert_eq!(chunks[1].shape(), &[2]);
        assert_eq!(chunks[2].shape(), &[1]);
    }

    #[test]
    fn test_method_chunk_2d() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let chunks = a.chunk(2, 0).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 2]);
    }

    // --- split ---

    #[test]
    fn test_method_split() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let parts = a.split(&[2, 3], 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].data().unwrap(), &[1.0, 2.0]);
        assert_eq!(parts[1].data().unwrap(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_method_split_2d_axis1() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
        let parts = a.split(&[1, 3], 1).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 1]);
        assert_eq!(parts[0].data().unwrap(), &[1.0, 5.0]);
        assert_eq!(parts[1].shape(), &[2, 3]);
        assert_eq!(parts[1].data().unwrap(), &[2.0, 3.0, 4.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_split_bad_sizes() {
        let a = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert!(a.split(&[1, 1], 0).is_err()); // sum != 3
    }

    // --- split/chunk autograd ---

    #[test]
    fn test_split_preserves_grad() {
        // Split a requires-grad tensor and verify chunks have grad_fn.
        let a = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![6],
            true,
        )
        .unwrap();
        let chunks = a.split(&[2, 4], 0).unwrap();
        assert!(chunks[0].grad_fn().is_some(), "chunk 0 should have grad_fn");
        assert!(chunks[1].grad_fn().is_some(), "chunk 1 should have grad_fn");
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_split_backward_simple() {
        // x = [1, 2, 3, 4, 5, 6], split into [1,2,3] and [4,5,6].
        // loss = sum(chunk0) + 2*sum(chunk1)
        // d_loss/d_x = [1, 1, 1, 2, 2, 2]
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![6],
            true,
        )
        .unwrap();
        let chunks = x.split(&[3, 3], 0).unwrap();

        let sum0 = crate::grad_fns::reduction::sum(&chunks[0]).unwrap();
        let sum1 = crate::grad_fns::reduction::sum(&chunks[1]).unwrap();

        // 2 * sum1
        let two = Tensor::from_storage(TensorStorage::cpu(vec![2.0f64]), vec![], false).unwrap();
        let scaled = crate::grad_fns::arithmetic::mul(&sum1, &two).unwrap();
        let loss = crate::grad_fns::arithmetic::add(&sum0, &scaled).unwrap();

        loss.backward().unwrap();

        let grad = x.grad().unwrap().expect("x should have grad");
        assert_eq!(grad.shape(), &[6]);
        let g = grad.data().unwrap();
        // First 3 elements: grad from sum0 = 1.0 each
        // Last 3 elements: grad from 2*sum1 = 2.0 each
        for i in 0..3 {
            assert!(
                (g[i] - 1.0).abs() < 1e-10,
                "grad[{i}] = {}, expected 1.0",
                g[i]
            );
        }
        for i in 3..6 {
            assert!(
                (g[i] - 2.0).abs() < 1e-10,
                "grad[{i}] = {}, expected 2.0",
                g[i]
            );
        }
    }

    #[test]
    fn test_chunk_backward_2d() {
        // x shape [2, 4], chunk into 2 along dim=1 -> two [2, 2] tensors.
        // loss = sum(chunk0) * 3 + sum(chunk1)
        // grad_x[:, 0:2] = 3, grad_x[:, 2:4] = 1
        let x =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f64; 8]), vec![2, 4], true).unwrap();
        let chunks = x.chunk(2, 1).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[2, 2]);
        assert_eq!(chunks[1].shape(), &[2, 2]);

        let sum0 = crate::grad_fns::reduction::sum(&chunks[0]).unwrap();
        let sum1 = crate::grad_fns::reduction::sum(&chunks[1]).unwrap();

        let three = Tensor::from_storage(TensorStorage::cpu(vec![3.0f64]), vec![], false).unwrap();
        let scaled = crate::grad_fns::arithmetic::mul(&sum0, &three).unwrap();
        let loss = crate::grad_fns::arithmetic::add(&scaled, &sum1).unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().expect("x should have grad");
        assert_eq!(grad.shape(), &[2, 4]);
        let g = grad.data().unwrap();
        // Row 0: [3, 3, 1, 1], Row 1: [3, 3, 1, 1]
        let expected = [3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0];
        for (i, (&actual, &exp)) in g.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-10,
                "grad[{i}] = {actual}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_split_no_grad_when_disabled() {
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]),
            vec![3],
            false, // no grad
        )
        .unwrap();
        let chunks = x.split(&[1, 2], 0).unwrap();
        assert!(chunks[0].grad_fn().is_none());
        assert!(chunks[1].grad_fn().is_none());
    }

    // --- size / dim aliases ---

    #[test]
    fn test_size_alias() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(a.size(), &[2, 3]);
        assert_eq!(a.size(), a.shape());
    }

    #[test]
    fn test_dim_alias() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(a.dim(), 2);
        assert_eq!(a.dim(), a.ndim());
    }

    // --- cumulative (scan) methods ---

    #[test]
    // reason: cumulative sum of small integer-valued floats (1+2+3 = 6 at
    // most) is bit-exact in any deterministic order — the partial sums
    // never lose mantissa bits, so equality is the right check. The
    // expected values [1, 3, 6] are constructed from the named upstream
    // recurrence `out_i = sum_{k=0..=i} input_k` per
    // `aten/src/ATen/native/ReduceOps.cpp:511 TORCH_IMPL_FUNC(cumsum_out)`
    // and the math definition at `torch/_torch_docs.py:3431-3438
    // y_i = x_1 + x_2 + ... + x_i`. The dispatch-correctness assertion
    // (method == free function) protects R-DEFER-1 wiring at the
    // method boundary.
    #[allow(clippy::float_cmp)]
    fn test_method_cumsum_t_1d() {
        let a = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();

        // Expected derived from upstream recurrence:
        //   out[0] = 1
        //   out[1] = 1 + 2 = 3
        //   out[2] = 1 + 2 + 3 = 6
        let expected = [1.0f32, 1.0 + 2.0, 1.0 + 2.0 + 3.0];

        let via_method = a.cumsum_t(0).unwrap();
        assert_eq!(via_method.shape(), &[3]);
        let m = via_method.data_vec().unwrap();
        for i in 0..3 {
            assert_eq!(m[i], expected[i], "method cumsum[{i}] != expected");
        }

        // Dispatch-correctness: method MUST equal the free function on
        // identical input. This is the production-consumer parity check.
        let via_free = crate::grad_fns::cumulative::cumsum(&a, 0).unwrap();
        let f = via_free.data_vec().unwrap();
        for i in 0..3 {
            assert_eq!(m[i], f[i], "cumsum_t and free fn disagree at {i}");
        }
    }

    #[test]
    // reason: cumprod of small ints (1, 2, 6) is bit-exact in f32 (small
    // integer mantissas), so equality is the right check. The expected
    // values [1, 2, 6] are constructed from the named upstream recurrence
    // `out_i = prod_{k=0..=i} input_k` per `aten/src/ATen/native/
    // ReduceOps.cpp:519 TORCH_IMPL_FUNC(cumprod_out)` and the math
    // definition at `torch/_torch_docs.py:3392-3399 y_i = x_1 * x_2 *
    // ... * x_i`.
    #[allow(clippy::float_cmp)]
    fn test_method_cumprod_t_1d() {
        let a = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();

        // Expected derived from upstream recurrence:
        //   out[0] = 1
        //   out[1] = 1 * 2 = 2
        //   out[2] = 1 * 2 * 3 = 6
        let expected = [1.0f32, 1.0 * 2.0, 1.0 * 2.0 * 3.0];

        let via_method = a.cumprod_t(0).unwrap();
        assert_eq!(via_method.shape(), &[3]);
        let m = via_method.data_vec().unwrap();
        for i in 0..3 {
            assert_eq!(m[i], expected[i], "method cumprod[{i}] != expected");
        }

        // Dispatch-correctness check.
        let via_free = crate::grad_fns::cumulative::cumprod(&a, 0).unwrap();
        let f = via_free.data_vec().unwrap();
        for i in 0..3 {
            assert_eq!(m[i], f[i], "cumprod_t and free fn disagree at {i}");
        }
    }

    #[test]
    // reason: logcumsumexp on a single-element vector is the identity:
    // `log(exp(x)) == x` numerically (one term in the sum). The expected
    // value 42.0 is the input value itself, derived from the math
    // definition at `torch/_torch_docs.py:3304-3305
    // logcumsumexp(x)_ij = log(sum_{k=0..=j} exp(x_ik))` evaluated at
    // j=0 (single-element scan). For the 3-element case we also check
    // monotonicity (logcumsumexp is non-decreasing along the scan dim
    // because the running sum-of-exp is non-decreasing and log is
    // monotonic) — verified live 2026-05-25 with torch 2.11.0.
    fn test_method_logcumsumexp_t_1d() {
        // Single-element: the math identity `log(exp(x)) = x` makes the
        // expected value structurally derivable without calling the
        // function on itself.
        let a = from_slice(&[42.0f32], &[1]).unwrap();
        let via_method = a.logcumsumexp_t(0).unwrap();
        assert_eq!(via_method.shape(), &[1]);
        let m = via_method.data_vec().unwrap();
        // logcumsumexp on a single element equals the input. Allow a
        // small fp slop because exp/log round-trip is not bit-exact.
        assert!(
            (m[0] - 42.0_f32).abs() < 1e-3,
            "logcumsumexp single-elt: got {} expected 42.0",
            m[0]
        );

        // Dispatch-correctness check on a 3-element input: method MUST
        // equal the free function bit-exactly (both go through the same
        // forward kernel).
        let b = from_slice(&[0.0f32, 1.0, 2.0], &[3]).unwrap();
        let via_method = b.logcumsumexp_t(0).unwrap();
        let via_free = crate::grad_fns::cumulative::logcumsumexp(&b, 0).unwrap();
        let m = via_method.data_vec().unwrap();
        let f = via_free.data_vec().unwrap();
        for i in 0..3 {
            assert!(
                (m[i] - f[i]).abs() < 1e-6,
                "logcumsumexp_t and free fn disagree at {i}: {} vs {}",
                m[i],
                f[i]
            );
        }
        // Monotonicity: y_0 <= y_1 <= y_2 (running sum of exp is
        // monotonic, log is monotonic).
        assert!(m[0] <= m[1], "logcumsumexp not monotonic: m[0]>m[1]");
        assert!(m[1] <= m[2], "logcumsumexp not monotonic: m[1]>m[2]");
    }
}
