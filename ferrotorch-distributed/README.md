# ferrotorch-distributed

Distributed training for ferrotorch -- backends, collectives, and DDP.

## What it provides

- **Backends** -- `TcpBackend` for real multi-process training, `SimulatedBackend` for in-process testing, the `Backend` trait, plus optional native-Rust `GlooBackend` / `MpiBackend` (feature-gated) and an `NcclBackend` (requires `nccl` feature)
- **Collectives** -- `allreduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`, `barrier` with `ReduceOp` (Sum, Mean)
- **DDP** -- `DDP` wraps any `Module` and synchronizes gradients across ranks after each backward pass
- **FSDP** -- `FSDP` shards parameters across ranks, all-gathering during forward and reduce-scattering gradients during backward
- **RPC** -- `RpcAgent` / `TcpRpcBackend` for invoking functions on remote ranks
- **Pipeline parallelism** -- `Pipeline` splits a model into sequential stages with GPipe / Interleaved1F1B schedules
- **GPU collectives** (requires `gpu` feature) -- `gpu_allreduce`, `gpu_broadcast` for GPU tensor communication

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `gpu`   | no      | Enable GPU-aware collectives via ferrotorch-gpu |

## Quick start

```rust
use ferrotorch_distributed::{TcpBackend, Backend, allreduce, ReduceOp, DDP};

let backend = TcpBackend::init(rank, world_size, &addr)?;
let mut ddp_model = DDP::new(model, &backend)?;

// Training loop -- gradients are synchronized automatically
let loss = ddp_model.forward(&input)?;
backward(&loss)?;
allreduce(&backend, &mut grad_tensor, ReduceOp::Mean)?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
