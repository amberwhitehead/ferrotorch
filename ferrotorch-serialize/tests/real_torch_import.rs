//! Integration test: `load_pytorch_state_dict` reads a `.pt` produced by a real
//! `torch.save` of an `nn.Module.state_dict()` (PyTorch 2.x). The fixture is
//! committed under `tests/fixtures/real_torch_v2.pt` and was generated from a
//! tiny Conv2d + BatchNorm2d module:
//!
//! ```text
//! torch.save(Tiny().state_dict(), "real_torch_v2.pt")
//! ```
//!
//! This pins three things that previously broke on real torch files (see
//! FERRO.md "load_pytorch_state_dict cannot parse real torch .pt files"):
//!
//! 1. The modern zip layout `<filestem>/data.pkl` + `<filestem>/data/<n>`
//!    (not just the legacy `archive/data.pkl`).
//! 2. The `_metadata` attribute torch attaches to `nn.Module.state_dict()`
//!    (the loader must read the OrderedDict entries, not the `_metadata`).
//! 3. Integer buffers — `BatchNorm.num_batches_tracked` is a `Long`/int64
//!    scalar and must be skipped (not abort the load) when producing a
//!    `StateDict<f32>`.

use ferrotorch_serialize::load_pytorch_state_dict;

#[test]
fn loads_real_torch_module_state_dict() {
    let path = "tests/fixtures/real_torch_v2.pt";
    let sd = load_pytorch_state_dict::<f32>(path).expect("real torch .pt must load");

    // The six float entries load; the Long `num_batches_tracked` is skipped.
    assert_eq!(sd.len(), 6, "expected 6 float tensors (num_batches_tracked skipped)");

    // Float weights/stats are present with exact values.
    let conv_w = sd.get("conv.weight").expect("conv.weight present");
    assert_eq!(conv_w.shape(), &[2, 1, 3, 3]);
    assert_eq!(
        conv_w.data().unwrap(),
        &(1..=18).map(|i| i as f32).collect::<Vec<_>>(),
    );

    let rm = sd.get("bn.running_mean").expect("bn.running_mean present");
    assert_eq!(rm.shape(), &[2]);
    assert_eq!(rm.data().unwrap(), &[4.0_f32, 5.0]);

    let rv = sd.get("bn.running_var").expect("bn.running_var present");
    assert_eq!(rv.data().unwrap(), &[7.0_f32, 8.0]);

    // The integer buffer was skipped, not included as a bogus/zeroed float.
    assert!(
        !sd.contains_key("bn.num_batches_tracked"),
        "Long (num_batches_tracked) must be skipped, not loaded into StateDict<f32>"
    );
}
