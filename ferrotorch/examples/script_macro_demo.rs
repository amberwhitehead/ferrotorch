//! Non-test production consumer of the `#[script]` proc-macro (#1482).
//!
//! `ferrotorch-jit-script` is a `proc-macro = true` crate, so it cannot
//! host examples that use its own attribute macro (Cargo refuses to
//! compile non-proc-macro targets inside a proc-macro crate). The
//! umbrella `ferrotorch` crate re-exports the macro via its
//! `jit-script` default feature, so this is the canonical home for the
//! demo: a 25-line script that takes two `Tensor<f32>` inputs, traces a
//! `mul → sum` graph, and re-executes the captured `TracedModule` on
//! fresh inputs to prove the trace round-trip works.
//!
//! Closes the consumer-wiring blocker filed against `ferrotorch-jit-script`'s
//! REQ-1 / REQ-5: every other REQ in the script-macro design doc was
//! already SHIPPED with test-only consumers — this binary is the
//! non-test caller goal.md R-DEFER-1 requires.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p ferrotorch --example script_macro_demo
//! ```
//!
//! The example exits with status 0 on success and prints the captured
//! result so a wrapping CI harness can grep the line. Determinism: every
//! tensor is built from a fixed `[f32]` slice so the output is
//! bit-identical across runs.

use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{FerrotorchResult, Tensor};
use ferrotorch_jit::TracedModule;
use ferrotorch_jit_script::script;

fn t1d(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("from_storage on the cpu path is infallible for a flat f32 slice")
}

/// Annotated with `#[script]`: the macro rewrites this body to build an
/// `IrGraph` via `ferrotorch_jit::trace` and returns a
/// `TracedModule<f32>`-returning function. The body itself stays
/// readable — it's just the standard `mul + sum` two-step that any
/// inline weighted-sum implementation would use.
#[script]
fn weighted_sum(a: Tensor<f32>, w: Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let prod = mul(&a, &w)?;
    sum(&prod)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build the inputs and produce a `TracedModule` by calling the
    // scripted function. The macro's rewrite captures the graph
    // (mul → sum) once and returns a module that can be re-executed on
    // fresh inputs of matching shape.
    let a = t1d(&[1.0, 2.0, 3.0]);
    let w = t1d(&[4.0, 5.0, 6.0]);
    let module: TracedModule<f32> = weighted_sum(a, w)?;
    println!(
        "[script_macro_demo] captured TracedModule<f32> (graph nodes invisible via public API)"
    );

    // Re-execute the captured graph with fresh inputs.
    let a2 = t1d(&[1.0, 2.0, 3.0]);
    let w2 = t1d(&[4.0, 5.0, 6.0]);
    let result = module.forward_multi(&[a2, w2])?;
    let result_data = result.data_vec()?;
    // mul(a, w) = [4, 10, 18]; sum = 32.0
    let expected = 32.0_f32;
    if (result_data[0] - expected).abs() > 1e-5 {
        return Err(format!(
            "[script_macro_demo] expected {expected}, got {}",
            result_data[0]
        )
        .into());
    }
    println!(
        "[script_macro_demo] forward_multi(weighted_sum) = {} (expected {expected})",
        result_data[0]
    );

    // Save/load round-trip is the other half of the `TracedModule`
    // contract — exercise it so a regression that broke the
    // serialisation surface fails this example.
    let bytes = module.to_bytes();
    let loaded: TracedModule<f32> = TracedModule::<f32>::from_bytes(&bytes)?;
    let r = loaded.forward_multi(&[t1d(&[2.0, 3.0]), t1d(&[4.0, 5.0])])?;
    let r_data = r.data_vec()?;
    // mul([2,3], [4,5]) = [8, 15]; sum = 23
    let expected_rt = 23.0_f32;
    if (r_data[0] - expected_rt).abs() > 1e-5 {
        return Err(format!(
            "[script_macro_demo] roundtrip expected {expected_rt}, got {}",
            r_data[0]
        )
        .into());
    }
    println!(
        "[script_macro_demo] to_bytes/from_bytes/forward_multi = {} (expected {expected_rt})",
        r_data[0]
    );
    Ok(())
}
