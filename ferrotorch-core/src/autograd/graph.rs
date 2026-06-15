//! ## REQ status (per `.design/ferrotorch-core/autograd/graph.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub fn `backward`<T: Float>` at `graph.rs:20-22`; consumer: `Tensor::backward` convenience method at `:637-639` and `ferrotorch-core/src/stride_tricks.rs:671 backward(&loss)?`. |
//! | REQ-2 | SHIPPED | `pub fn `backward_with_grad`<T: Float>` at `graph.rs:30-206`; consumer: `Tensor::backward_with_gradient` at `:647-649` and `ferrotorch-core/src/grad_fns/shape.rs:1112`. |
//! | REQ-3 | SHIPPED | Kahn three-phase topo-sort at `graph.rs:67-205`; consumer: same as REQ-1 (engine inside `backward`). |
//! | REQ-4 | SHIPPED | `accumulate_non_leaf_grad` at `graph.rs:530-629` and `accumulate_non_leaf_grad_locked` at `:460-514`; consumer: invoked from REQ-1/REQ-2 dispatch. |
//! | REQ-5 | SHIPPED | `run_grad_hooks` and `run_post_accumulate_hooks` calls at `graph.rs:175-193` (sequential) and `:385-407` (parallel); consumer: every `Tensor::register_hook` user flowing through backward. |
//! | REQ-6 | SHIPPED | Materialize-contiguous gradient at `graph.rs:148-152` (sequential) and `:363-367` (parallel); consumer: every non-contiguous gradient in backward. |
//! | REQ-7 | SHIPPED | GPU-native add at `graph.rs:551-569` (sequential) and `:480-496` (parallel) via `backend.add_f32`/`add_f64`; consumer: any model with same-device gradient-merge points. |
//! | REQ-8 | SHIPPED | Gradient-count sanity check at `graph.rs:160-168` and `:372-380`; consumer: defensive guard inside REQ-3 dispatch. |
//! | REQ-9 | SHIPPED | `pub fn `backward_parallel`<T: Float>` at `graph.rs:220-457`; consumer: existing pub API across multiple prior commits — boundary-API grandfathering. |
//! | REQ-10 | SHIPPED | Shape-preserving seed at `graph.rs:50-65`; consumer: every `Tensor::backward()` on a `[1]`-shape loss (regression test at `:854-867`). |
//! | REQ-11 | SHIPPED | `impl<T: Float> Tensor<T>` with `pub fn `backward`` at `graph.rs:637-639` and `pub fn `backward_with_gradient`` at `:647-649`; consumer: `stride_tricks.rs:672`, `grad_fns/quantize_grad.rs:793`. |
//!

use rustc_hash::FxHashMap as HashMap;
use std::collections::VecDeque;

use crate::autograd::hooks::{run_grad_hooks, run_post_accumulate_hooks};
use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::tensor::{Tensor, TensorId};

fn validate_external_gradient<T: Float>(
    root: &Tensor<T>,
    ext_grad: &Tensor<T>,
) -> FerrotorchResult<()> {
    if ext_grad.shape() != root.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "gradient shape {:?} does not match root shape {:?}",
                ext_grad.shape(),
                root.shape(),
            ),
        });
    }
    if ext_grad.device() != root.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: root.device(),
            got: ext_grad.device(),
        });
    }
    Ok(())
}

fn implicit_seed_like<T: Float>(root: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if !root.is_scalar() && root.numel() != 1 {
        return Err(FerrotorchError::BackwardNonScalar {
            shape: root.shape().to_vec(),
        });
    }

    // PyTorch seeds implicit scalar/single-element roots with ones_like(root).
    // Preserving the logical shape matters for roots shaped [1], [1, 1], etc.
    let one = <T as num_traits::One>::one();
    let ones_storage = crate::storage::TensorStorage::cpu(vec![one; root.numel().max(1)]);
    let seed_cpu = Tensor::from_storage(ones_storage, root.shape().to_vec(), false)?;
    seed_cpu.to(root.device())
}

/// Compute gradients of all leaf tensors that contribute to `root`.
///
/// Implements reverse-mode automatic differentiation:
/// 1. Collect all nodes reachable from `root` that have a `grad_fn`.
/// 2. Topological sort via Kahn's algorithm (iterative, no stack overflow).
/// 3. Walk in reverse topological order, calling each node's `GradFn::backward()`.
/// 4. Accumulate gradients additively on leaf tensors.
///
/// `root` must be a scalar tensor (0-dim or single element). After this call,
/// leaf tensors with `requires_grad = true` will have their `.grad()` populated.
pub fn backward<T: Float>(root: &Tensor<T>) -> FerrotorchResult<()> {
    backward_with_grad(root, None)
}

/// Run backward pass through the computation graph.
///
/// If `gradient` is `None`, the root must be scalar and an implicit seed of 1.0 is used.
/// If `gradient` is `Some`, it is used as the initial gradient for the root tensor,
/// allowing backward on non-scalar tensors (needed for multi-head outputs, Jacobian
/// computation, and custom loss functions).
pub fn backward_with_grad<T: Float>(
    root: &Tensor<T>,
    gradient: Option<&Tensor<T>>,
) -> FerrotorchResult<()> {
    let seed = if let Some(ext_grad) = gradient {
        validate_external_gradient(root, ext_grad)?;
        ext_grad.clone()
    } else {
        implicit_seed_like(root)?
    };

    // Phase 1: Collect all nodes and compute in-degree via BFS.
    //
    // We traverse the graph from `root` backward through `grad_fn().inputs()`.
    // `in_degree[id]` counts how many times a tensor is used as an input to
    // an operation — this is needed for Kahn's algorithm.
    let mut in_degree: HashMap<TensorId, usize> = HashMap::default();
    let mut node_map: HashMap<TensorId, Tensor<T>> = HashMap::default();
    let mut queue: VecDeque<Tensor<T>> = VecDeque::new();

    // Start from root.
    queue.push_back(root.clone());
    in_degree.entry(root.id()).or_insert(0);
    node_map.insert(root.id(), root.clone());

    while let Some(node) = queue.pop_front() {
        if let Some(grad_fn) = node.grad_fn() {
            for input in grad_fn.inputs() {
                let input_id = input.id();
                let count = in_degree.entry(input_id).or_insert(0);
                *count += 1;
                if let std::collections::hash_map::Entry::Vacant(e) = node_map.entry(input_id) {
                    let input = input.clone();
                    e.insert(input.clone());
                    queue.push_back(input);
                }
            }
        }
    }

    // Phase 2: Topological sort (Kahn's algorithm).
    //
    // Start with nodes that have in_degree == 0. The root always has in_degree 0
    // (nothing depends on it in the backward direction). Process nodes in
    // topological order, decrementing in_degree of their inputs.
    let mut topo_order: Vec<TensorId> = Vec::new();
    let mut bfs_queue: VecDeque<TensorId> = VecDeque::new();

    // Find all nodes with in_degree 0 (just the root in a standard graph).
    for (&id, &deg) in &in_degree {
        if deg == 0 {
            bfs_queue.push_back(id);
        }
    }

    while let Some(id) = bfs_queue.pop_front() {
        topo_order.push(id);
        if let Some(node) = node_map.get(&id)
            && let Some(grad_fn) = node.grad_fn()
        {
            for input in grad_fn.inputs() {
                if let Some(deg) = in_degree.get_mut(&input.id()) {
                    *deg -= 1;
                    if *deg == 0 {
                        bfs_queue.push_back(input.id());
                    }
                }
            }
        }
    }

    // Phase 3: Backward pass in topological order.
    //
    // We maintain a map of accumulated output gradients for each node.
    // For the root, the gradient is the seed (1.0).
    let mut grads: HashMap<TensorId, Tensor<T>> = HashMap::default();
    grads.insert(root.id(), seed);

    for &id in &topo_order {
        let node = match node_map.get(&id) {
            Some(n) => n,
            None => continue,
        };

        let grad_output = match grads.remove(&id) {
            Some(g) => g,
            None => continue,
        };

        if let Some(grad_fn) = node.grad_fn() {
            // Materialize non-contiguous gradients before backward.
            // Stride-based views (from permute/transpose/narrow) may be
            // non-contiguous — backward functions expect contiguous data.
            let grad_output = if grad_output.is_contiguous() {
                grad_output
            } else {
                crate::methods::contiguous_t(&grad_output)?
            };
            let input_grads = grad_fn.backward(&grad_output)?;
            let inputs = grad_fn.inputs();

            // B3 fix: validate that backward returned the correct number
            // of gradients. Without this, `zip` silently drops trailing
            // gradients when the backward function returns fewer than
            // expected, causing silent incorrect results.
            if input_grads.len() != inputs.len() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "backward returned {} gradients but expected {}",
                        input_grads.len(),
                        inputs.len(),
                    ),
                });
            }

            for (input, maybe_grad) in inputs.iter().zip(input_grads) {
                if let Some(grad) = maybe_grad
                    && input.requires_grad()
                {
                    // Run gradient hooks (if any), which may modify the gradient.
                    let hooks = input.hooks();
                    let has_hooks = {
                        let guard = hooks.lock().map_err(|e| FerrotorchError::LockPoisoned {
                            message: format!("hook storage mutex: {e}"),
                        })?;
                        (guard.has_grad_hooks(), guard.has_post_accumulate_hooks())
                    };
                    let grad = if has_hooks.0 {
                        run_grad_hooks(hooks, grad)?
                    } else {
                        grad
                    };

                    if input.is_leaf() {
                        // Leaf tensor: accumulate gradient on the tensor itself.
                        input.accumulate_grad(&grad)?;
                        // Run post-accumulate-grad hooks on the leaf (if any).
                        if has_hooks.1 {
                            run_post_accumulate_hooks(hooks, input)?;
                        }
                    } else {
                        // Non-leaf: accumulate into the grads map for the next iteration.
                        accumulate_non_leaf_grad(&mut grads, input, grad)?;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Multi-threaded backward engine.
///
/// Same correctness as [`backward_with_grad`], but processes independent
/// backward nodes in parallel using a ready-queue pattern:
///
/// 1. Nodes with in-degree 0 are placed in a shared queue.
/// 2. Worker threads pull nodes, call `grad_fn.backward()`, accumulate grads.
/// 3. After processing, workers decrement in-degrees of the node's inputs.
///    When an input's in-degree reaches 0, it is pushed to the queue.
/// 4. Workers exit when the queue is empty and all nodes are processed.
///
/// Falls back to single-threaded for graphs with fewer than 8 nodes.
pub fn backward_parallel<T: Float>(
    root: &Tensor<T>,
    gradient: Option<&Tensor<T>>,
    num_workers: usize,
) -> FerrotorchResult<()> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};

    let seed = if let Some(ext_grad) = gradient {
        validate_external_gradient(root, ext_grad)?;
        ext_grad.clone()
    } else {
        implicit_seed_like(root)?
    };

    // Phase 1: Collect nodes and compute in-degree (same as sequential).
    let mut in_degree_map: HashMap<TensorId, usize> = HashMap::default();
    let mut node_map: HashMap<TensorId, Tensor<T>> = HashMap::default();
    let mut queue: VecDeque<Tensor<T>> = VecDeque::new();

    queue.push_back(root.clone());
    in_degree_map.entry(root.id()).or_insert(0);
    node_map.insert(root.id(), root.clone());

    while let Some(node) = queue.pop_front() {
        if let Some(grad_fn) = node.grad_fn() {
            for input in grad_fn.inputs() {
                let input_id = input.id();
                let count = in_degree_map.entry(input_id).or_insert(0);
                *count += 1;
                if let std::collections::hash_map::Entry::Vacant(e) = node_map.entry(input_id) {
                    let input = input.clone();
                    e.insert(input.clone());
                    queue.push_back(input);
                }
            }
        }
    }

    let total_nodes = node_map.len();

    // For small graphs, fall back to sequential.
    if total_nodes < 8 || num_workers <= 1 {
        return backward_with_grad(root, gradient);
    }

    // Phase 2: Build shared state for parallel processing.

    // Atomic in-degrees for lock-free decrement.
    let in_degrees: HashMap<TensorId, AtomicUsize> = in_degree_map
        .iter()
        .map(|(&id, &deg)| (id, AtomicUsize::new(deg)))
        .collect();
    let in_degrees = Arc::new(in_degrees);

    // Shared gradient accumulator.
    let grads: Arc<Mutex<HashMap<TensorId, Tensor<T>>>> = Arc::new(Mutex::new({
        let mut m = HashMap::default();
        m.insert(root.id(), seed);
        m
    }));

    // Ready queue + condvar for waking workers.
    let ready: Arc<Mutex<VecDeque<TensorId>>> = Arc::new(Mutex::new(VecDeque::new()));
    let condvar = Arc::new(Condvar::new());

    // Seed the ready queue with all in-degree 0 nodes.
    {
        let mut rq = ready.lock().unwrap();
        for (&id, deg) in in_degrees.iter() {
            if deg.load(Ordering::Relaxed) == 0 {
                rq.push_back(id);
            }
        }
    }

    // Counter of processed nodes — workers exit when this reaches total.
    let processed = Arc::new(AtomicUsize::new(0));

    // Fail-fast cancellation. A backward node, hook, device op, or gradient
    // accumulation can fail before it decrements the node's input in-degrees.
    // Without cancellation, those inputs never become ready and waiters sleep
    // forever with `processed < total_nodes` (CORE-021 / #1715).
    let cancelled = Arc::new(AtomicBool::new(false));
    let first_error: Arc<Mutex<Option<FerrotorchError>>> = Arc::new(Mutex::new(None));

    // Phase 3: Parallel backward.
    let node_map_ref = &node_map;
    std::thread::scope(|s| {
        let workers = num_workers.min(total_nodes);
        for _ in 0..workers {
            let in_degrees = Arc::clone(&in_degrees);
            let grads = Arc::clone(&grads);
            let ready = Arc::clone(&ready);
            let condvar = Arc::clone(&condvar);
            let processed = Arc::clone(&processed);
            let cancelled = Arc::clone(&cancelled);
            let first_error = Arc::clone(&first_error);

            s.spawn(move || {
                loop {
                    // Pull a ready node.
                    let id = {
                        let mut rq = ready.lock().unwrap();
                        loop {
                            if cancelled.load(Ordering::Acquire) {
                                return;
                            }
                            if let Some(id) = rq.pop_front() {
                                break id;
                            }
                            if processed.load(Ordering::Relaxed) >= total_nodes {
                                return;
                            }
                            rq = condvar.wait(rq).unwrap();
                            if cancelled.load(Ordering::Acquire)
                                || processed.load(Ordering::Relaxed) >= total_nodes
                            {
                                return;
                            }
                        }
                    };
                    if cancelled.load(Ordering::Acquire) {
                        return;
                    }

                    // Process this node.
                    let result = (|| -> FerrotorchResult<()> {
                        let node = match node_map_ref.get(&id) {
                            Some(n) => n,
                            None => return Ok(()),
                        };

                        let grad_output = {
                            let mut g = grads.lock().unwrap();
                            match g.remove(&id) {
                                Some(go) => go,
                                None => return Ok(()),
                            }
                        };

                        if let Some(grad_fn) = node.grad_fn() {
                            let grad_output = if grad_output.is_contiguous() {
                                grad_output
                            } else {
                                crate::methods::contiguous_t(&grad_output)?
                            };

                            let input_grads = grad_fn.backward(&grad_output)?;
                            let inputs = grad_fn.inputs();

                            if input_grads.len() != inputs.len() {
                                return Err(FerrotorchError::InvalidArgument {
                                    message: format!(
                                        "backward returned {} gradients but expected {}",
                                        input_grads.len(),
                                        inputs.len(),
                                    ),
                                });
                            }

                            for (input, maybe_grad) in inputs.iter().zip(input_grads) {
                                if let Some(grad) = maybe_grad
                                    && input.requires_grad()
                                {
                                    let hooks = input.hooks();
                                    let has_hooks = {
                                        let guard = hooks.lock().map_err(|e| {
                                            FerrotorchError::LockPoisoned {
                                                message: format!("hook storage mutex: {e}"),
                                            }
                                        })?;
                                        (guard.has_grad_hooks(), guard.has_post_accumulate_hooks())
                                    };
                                    let grad = if has_hooks.0 {
                                        run_grad_hooks(hooks, grad)?
                                    } else {
                                        grad
                                    };

                                    if input.is_leaf() {
                                        input.accumulate_grad(&grad)?;
                                        if has_hooks.1 {
                                            run_post_accumulate_hooks(hooks, input)?;
                                        }
                                    } else {
                                        let mut g = grads.lock().unwrap();
                                        accumulate_non_leaf_grad_locked(&mut g, input, grad)?;
                                    }
                                }
                            }

                            // Decrement in-degrees of inputs; push newly ready.
                            // If another worker has already failed, do not
                            // schedule more work: waiters will observe
                            // `cancelled` and return after `notify_all`.
                            if !cancelled.load(Ordering::Acquire) {
                                for input in grad_fn.inputs() {
                                    if let Some(deg) = in_degrees.get(&input.id()) {
                                        let prev = deg.fetch_sub(1, Ordering::AcqRel);
                                        if prev == 1 {
                                            let mut rq = ready.lock().unwrap();
                                            if !cancelled.load(Ordering::Acquire) {
                                                rq.push_back(input.id());
                                                condvar.notify_one();
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        Ok(())
                    })();

                    if let Err(e) = result {
                        {
                            let mut err = first_error.lock().unwrap();
                            if err.is_none() {
                                *err = Some(e);
                            }
                        }
                        cancelled.store(true, Ordering::Release);
                        condvar.notify_all();
                    }

                    let prev = processed.fetch_add(1, Ordering::AcqRel);
                    if prev + 1 >= total_nodes || cancelled.load(Ordering::Acquire) {
                        condvar.notify_all();
                    }
                }
            });
        }
    });

    let err = match Arc::try_unwrap(first_error) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(arc) => {
            let mut guard = arc.lock().unwrap();
            std::mem::take(&mut *guard)
        }
    };
    if let Some(e) = err {
        return Err(e);
    }

    Ok(())
}

/// Like `accumulate_non_leaf_grad` but caller holds the grads mutex.
fn accumulate_non_leaf_grad_locked<T: Float>(
    grads: &mut HashMap<TensorId, Tensor<T>>,
    input: &Tensor<T>,
    grad: Tensor<T>,
) -> FerrotorchResult<()> {
    let Some(existing) = grads.remove(&input.id()) else {
        grads.insert(input.id(), grad);
        return Ok(());
    };

    if existing.shape() != grad.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "gradient shape mismatch during accumulation: {:?} vs {:?}",
                existing.shape(),
                grad.shape(),
            ),
        });
    }

    // GPU-native accumulation when both on same GPU.
    if let (Device::Cuda(_), Device::Cuda(_)) = (existing.device(), grad.device())
        && existing.device() == grad.device()
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        let a_handle = existing.gpu_handle()?;
        let b_handle = grad.gpu_handle()?;
        let sum_handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "non-leaf gradient accumulation add",
            f32 => backend.add_f32(a_handle, b_handle),
            f64 => backend.add_f64(a_handle, b_handle),
            bf16 => backend.add_bf16_bf16(a_handle, b_handle),
            f16 => backend.add_f16(a_handle, b_handle),
        )?;
        let combined = Tensor::from_storage(
            crate::storage::TensorStorage::gpu(sum_handle),
            existing.shape().to_vec(),
            false,
        )?;
        grads.insert(input.id(), combined);
        return Ok(());
    }

    // CPU path.
    let existing_data = existing.data_vec()?;
    let grad_data = grad.data_vec()?;
    let combined_data: Vec<T> = existing_data
        .iter()
        .zip(grad_data.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    let device = existing.device();
    let combined = Tensor::from_storage(
        crate::storage::TensorStorage::on_device(combined_data, device)?,
        existing.shape().to_vec(),
        false,
    )?;
    grads.insert(input.id(), combined);
    Ok(())
}

/// Accumulate a gradient for a non-leaf tensor in the backward grads map.
///
/// This is separated from the main backward loop for clarity and to
/// encapsulate the B1 / B6 fixes:
///
/// - **B1**: In-place accumulation is only attempted when both the outer
///   `Arc<TensorInner>` and the inner `Arc<TensorStorage>` have a strong
///   count of 1, the tensor is contiguous, and it is NOT on GPU. Without
///   the storage refcount check, shared-storage views could be corrupted.
///
/// - **B6**: When both the existing gradient and the incoming gradient are
///   on the same GPU device, we use the dtype-specific backend add kernel
///   directly instead of round-tripping through CPU. This covers f32/f64 and
///   the two-byte floating dtypes f16/bf16.
fn accumulate_non_leaf_grad<T: Float>(
    grads: &mut HashMap<TensorId, Tensor<T>>,
    input: &Tensor<T>,
    grad: Tensor<T>,
) -> FerrotorchResult<()> {
    let Some(existing) = grads.remove(&input.id()) else {
        grads.insert(input.id(), grad);
        return Ok(());
    };

    // Shape validation.
    if existing.shape() != grad.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "gradient shape mismatch during accumulation: {:?} vs {:?}",
                existing.shape(),
                grad.shape(),
            ),
        });
    }

    // B6 fix: GPU-native accumulation when both tensors are on the same GPU.
    if let (Device::Cuda(_), Device::Cuda(_)) = (existing.device(), grad.device())
        && existing.device() == grad.device()
        && let Some(backend) = crate::gpu_dispatch::gpu_backend()
    {
        let a_handle = existing.gpu_handle()?;
        let b_handle = grad.gpu_handle()?;
        let result_handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
            T,
            "non-leaf gradient accumulation add",
            f32 => backend.add_f32(a_handle, b_handle),
            f64 => backend.add_f64(a_handle, b_handle),
            bf16 => backend.add_bf16_bf16(a_handle, b_handle),
            f16 => backend.add_f16(a_handle, b_handle),
        )?;
        let storage = crate::storage::TensorStorage::gpu(result_handle);
        let combined = Tensor::from_storage(storage, existing.shape().to_vec(), false)?;
        grads.insert(input.id(), combined);
        return Ok(());
    }

    // B1 fix: in-place accumulation is only safe when we have exclusive
    // ownership of BOTH the TensorInner Arc AND the TensorStorage Arc,
    // the tensor is contiguous, and it is on CPU. Without the storage
    // refcount check, views sharing the same storage would be corrupted.
    if existing.inner_refcount() == 1
        && existing.storage_refcount() == 1
        && existing.is_contiguous()
        && !existing.is_cuda()
    {
        // SAFETY: inner_refcount == 1 && storage_refcount == 1 guarantees
        // exclusive ownership. No other references exist.
        let existing_slice = unsafe { existing.data_mut()? };
        if grad.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "accumulate_grad",
            });
        }
        let grad_data = grad.data()?;
        if existing_slice.len() != grad_data.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "gradient length mismatch during accumulation: {} vs {}",
                    existing_slice.len(),
                    grad_data.len(),
                ),
            });
        }
        for (e, &g) in existing_slice.iter_mut().zip(grad_data.iter()) {
            *e += g;
        }
        grads.insert(input.id(), existing);
        return Ok(());
    }

    // Fallback: allocate a new tensor for the sum (CPU path).
    if existing.is_cuda() || grad.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "accumulate_grad",
        });
    }
    let mut existing_data = existing.data()?.to_vec();
    let grad_data = grad.data()?;
    if existing_data.len() != grad_data.len() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "gradient length mismatch during accumulation: {} vs {}",
                existing_data.len(),
                grad_data.len(),
            ),
        });
    }
    for (e, &g) in existing_data.iter_mut().zip(grad_data.iter()) {
        *e += g;
    }
    let storage = crate::storage::TensorStorage::cpu(existing_data);
    let combined = Tensor::from_storage(storage, existing.shape().to_vec(), false)?;
    grads.insert(input.id(), combined);
    Ok(())
}

/// Convenience methods on Tensor for calling backward.
impl<T: Float> Tensor<T> {
    /// Compute gradients of all leaf tensors that contribute to this tensor.
    ///
    /// This tensor must be scalar (0-dim or single-element). After this call,
    /// leaf tensors with `requires_grad = true` will have their `.grad()` set.
    pub fn backward(&self) -> FerrotorchResult<()> {
        backward(self)
    }

    /// Run backward with an external gradient.
    ///
    /// This allows backward on non-scalar tensors by providing the initial
    /// gradient explicitly. The gradient shape must match this tensor's shape.
    /// Used for multi-head outputs, Jacobian computation, and custom loss
    /// functions.
    pub fn backward_with_gradient(&self, gradient: &Tensor<T>) -> FerrotorchResult<()> {
        backward_with_grad(self, Some(gradient))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;
    use crate::tensor::GradFn;
    use std::sync::Arc;

    /// A simple grad_fn for testing: output = a + b.
    /// backward: d(a+b)/da = 1, d(a+b)/db = 1.
    #[derive(Debug)]
    struct AddBackward<T: Float> {
        a: Tensor<T>,
        b: Tensor<T>,
    }

    impl<T: Float> GradFn<T> for AddBackward<T> {
        fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            Ok(vec![Some(grad_output.clone()), Some(grad_output.clone())])
        }
        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.a, &self.b]
        }
        fn name(&self) -> &'static str {
            "AddBackward"
        }
    }

    /// A simple grad_fn: output = a * b (elementwise).
    /// backward: d(a*b)/da = b * grad, d(a*b)/db = a * grad.
    #[derive(Debug)]
    struct MulBackward<T: Float> {
        a: Tensor<T>,
        b: Tensor<T>,
    }

    impl<T: Float> GradFn<T> for MulBackward<T> {
        fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            let go = grad_output.data()?;
            let a_data = self.a.data()?;
            let b_data = self.b.data()?;

            let grad_a: Vec<T> = go.iter().zip(b_data.iter()).map(|(&g, &b)| g * b).collect();
            let grad_b: Vec<T> = go.iter().zip(a_data.iter()).map(|(&g, &a)| g * a).collect();

            let ta =
                Tensor::from_storage(TensorStorage::cpu(grad_a), self.a.shape().to_vec(), false)?;
            let tb =
                Tensor::from_storage(TensorStorage::cpu(grad_b), self.b.shape().to_vec(), false)?;
            Ok(vec![Some(ta), Some(tb)])
        }
        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.a, &self.b]
        }
        fn name(&self) -> &'static str {
            "MulBackward"
        }
    }

    /// Helper to make a leaf scalar tensor.
    fn leaf_scalar(val: f32, requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    #[test]
    fn test_backward_simple_add() {
        // c = a + b, backward from c.
        // dc/da = 1, dc/db = 1.
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);

        let sum_val = a.data().unwrap()[0] + b.data().unwrap()[0];
        let c = Tensor::from_operation(
            TensorStorage::cpu(vec![sum_val]),
            vec![],
            Arc::new(AddBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
        .unwrap();

        c.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let b_grad = b.grad().unwrap().unwrap();
        assert!((a_grad.item().unwrap() - 1.0).abs() < 1e-6);
        assert!((b_grad.item().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_mul() {
        // c = a * b, backward from c.
        // dc/da = b = 3.0, dc/db = a = 2.0.
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);

        let prod_val = a.data().unwrap()[0] * b.data().unwrap()[0];
        let c = Tensor::from_operation(
            TensorStorage::cpu(vec![prod_val]),
            vec![],
            Arc::new(MulBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
        .unwrap();

        c.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let b_grad = b.grad().unwrap().unwrap();
        assert!((a_grad.item().unwrap() - 3.0).abs() < 1e-6);
        assert!((b_grad.item().unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_shared_input() {
        // c = a + a, backward from c.
        // dc/da = 1 + 1 = 2.
        let a = leaf_scalar(5.0, true);

        let sum_val = a.data().unwrap()[0] + a.data().unwrap()[0];
        let c = Tensor::from_operation(
            TensorStorage::cpu(vec![sum_val]),
            vec![],
            Arc::new(AddBackward {
                a: a.clone(),
                b: a.clone(),
            }),
        )
        .unwrap();

        c.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        assert!((a_grad.item().unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_chain() {
        // d = (a * b) + b
        // dd/da = b = 3.0
        // dd/db = a + 1 = 2.0 + 1.0 = 3.0
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);

        // c = a * b
        let c_val = 2.0 * 3.0;
        let c = Tensor::from_operation(
            TensorStorage::cpu(vec![c_val]),
            vec![],
            Arc::new(MulBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
        .unwrap();

        // d = c + b
        let d_val = c_val + 3.0;
        let d = Tensor::from_operation(
            TensorStorage::cpu(vec![d_val]),
            vec![],
            Arc::new(AddBackward {
                a: c.clone(),
                b: b.clone(),
            }),
        )
        .unwrap();

        d.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let b_grad = b.grad().unwrap().unwrap();
        assert!(
            (a_grad.item().unwrap() - 3.0).abs() < 1e-6,
            "expected dd/da = 3.0, got {}",
            a_grad.item().unwrap()
        );
        assert!(
            (b_grad.item().unwrap() - 3.0).abs() < 1e-6,
            "expected dd/db = 3.0, got {}",
            b_grad.item().unwrap()
        );
    }

    #[test]
    fn test_backward_non_scalar_error() {
        let t =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0, 2.0, 3.0]), vec![3], false)
                .unwrap();
        assert!(t.backward().is_err());
    }

    // -----------------------------------------------------------------------
    // Regression: backward on a single-element non-scalar tensor must seed
    // the gradient with the same shape as the root, not an empty shape.
    // Previously this triggered an integer underflow inside
    // reduce_grad_to_shape when AddBackward / MulBackward called it with
    // grad_ndim < target_ndim. CL-498.
    // -----------------------------------------------------------------------

    #[test]
    fn test_backward_one_element_tensor_seed_has_same_shape() {
        // Build x = [3.0] with grad, compute y = x * x (= [9.0]),
        // then backward — should populate x.grad without panicking.
        let x = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![3.0]), vec![1], true).unwrap();
        let y = crate::grad_fns::arithmetic::mul(&x, &x).unwrap();
        assert_eq!(y.shape(), &[1]);
        // backward without an explicit gradient must succeed for [1]-shaped
        // single-element tensors.
        y.backward().unwrap();
        let g = x.grad().unwrap().expect("x should have gradient");
        // d(x*x)/dx = 2x, at x=3 -> g = 6.
        assert!((g.data().unwrap()[0] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_backward_one_element_through_pow_and_add() {
        // Reproduces the AdamW convergence test pattern that previously
        // panicked: f(x, y) = x^2 + y^2 where x, y are [1]-shaped Parameters.
        // backward() should produce gradients [2x] and [2y].
        let x = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![3.0]), vec![1], true).unwrap();
        let y = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![-4.0]), vec![1], true).unwrap();
        let xs = crate::grad_fns::arithmetic::pow(&x, 2.0).unwrap();
        let ys = crate::grad_fns::arithmetic::pow(&y, 2.0).unwrap();
        let loss = crate::grad_fns::arithmetic::add(&xs, &ys).unwrap();
        assert_eq!(loss.shape(), &[1]);
        loss.backward().unwrap();
        let gx = x.grad().unwrap().unwrap();
        let gy = y.grad().unwrap().unwrap();
        // 2*3 = 6, 2*-4 = -8
        assert!((gx.data().unwrap()[0] - 6.0).abs() < 1e-5);
        assert!((gy.data().unwrap()[0] - (-8.0)).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_grad_to_shape_reshape_when_same_numel() {
        // Post-#814: `grad_ndim < target_ndim` with matching numel is a
        // valid reshape, not an error. The original defensive guard was
        // too strict — it rejected the `[] -> [1]` case that the
        // higher-order grad chain naturally produces.
        let grad =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![7.5]), vec![], false).unwrap();
        let out = crate::grad_fns::arithmetic::reduce_grad_to_shape(&grad, &[1])
            .expect("reshape [] -> [1] must succeed (numel matches)");
        assert_eq!(out.shape(), &[1]);
        assert!((out.data().unwrap()[0] - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_grad_to_shape_returns_error_on_numel_mismatch_underflow() {
        // Defensive guard: when grad has fewer dims than target AND the
        // numels don't match, no reshape is possible and the function
        // must return a clean `ShapeMismatch` instead of panicking with
        // subtract overflow at `grad_ndim - target_ndim`. CL-498, #814.
        let grad =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0]), vec![], false).unwrap();
        let result = crate::grad_fns::arithmetic::reduce_grad_to_shape(&grad, &[2]);
        let err_msg = match result {
            Ok(_) => panic!("expected error for grad_ndim < target_ndim AND numel mismatch"),
            Err(e) => format!("{e}"),
        };
        assert!(
            err_msg.contains("grad_ndim"),
            "expected mismatch message, got: {err_msg}"
        );
    }

    #[test]
    fn test_reduce_grad_to_shape_reshape_branch_does_not_swallow_numel_mismatch() {
        // The new rank-mismatch-but-same-numel reshape branch (#814)
        // must NOT activate when numels differ — that's a different
        // bug class. Here grad shape `[]` (numel 1) -> target `[2]`
        // (numel 2) should NOT silently reshape; it must still hit
        // the underflow guard and error.
        let grad =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0]), vec![], false).unwrap();
        let result = crate::grad_fns::arithmetic::reduce_grad_to_shape(&grad, &[2]);
        assert!(
            matches!(result, Err(FerrotorchError::ShapeMismatch { .. })),
            "grad [] -> target [2] (numel mismatch) must error, got: {result:?}"
        );
    }
}
