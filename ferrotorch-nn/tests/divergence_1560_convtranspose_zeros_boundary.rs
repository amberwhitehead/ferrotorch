//! Divergence pin for #1560: ConvTranspose1d Zeros-mode FORWARD parity FAIL.
//!
//! This is a PRE-EXISTING transpose-conv forward-math bug, NOT introduced by
//! #1443. Commit 69525542c only added the non-zeros padding_mode REJECTION
//! path for ConvTranspose layers; it did not touch any ConvTranspose `forward`
//! body (verified: the diff's ConvTranspose hunks at conv.rs add only
//! `with_padding_mode`). This test exists so the divergence is a runnable,
//! self-contained artifact independent of #1443's scope.
//!
//! Reproduction: this is parity-sweep `nn.functional.conv_transpose1d` case
//! i=6 (`op_db.sample_inputs`), which fails on EVERY seed (it is a fixed
//! fixture, not RNG-dependent):
//!   FAIL i=6 shape=[1,2,8] index 7: ferrotorch=0 vs torch=19.366032
//! Params: input [1,1,4], weight [1,2,3], stride=2, padding=1, output_padding=1.
//! The combination of padding>0 AND output_padding>0 is what trips the
//! stride-insert-zeros / internal-pad boundary handling in conv.rs.
//!
//! R-CHAR-3: input/weight values and `expected` are from the live PyTorch
//! 2.11.0+cu130 oracle (op_db sample i=6 + F.conv_transpose1d), NOT copied
//! from ferrotorch.
//!
//! Upstream: torch/nn/modules/conv.py ConvTranspose1d / aten
//! slow_conv_transpose; output length = (L_in-1)*stride - 2*pad + k + outpad.

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::functional::conv_transpose1d;

fn tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Divergence: ferrotorch ConvTranspose1d (Zeros padding) with padding=1,
/// output_padding=1 diverges from torch across the output. The boundary cell
/// (index 7) is ferrotorch=0 vs torch=19.366032. Tracking: #1560.
#[test]
#[ignore = "divergence: ConvTranspose1d Zeros forward (pad=1,outpad=1) wrong; index7 ferro=0 vs torch=19.366; pre-existing, NOT in #1443 scope; tracking #1560"]
fn divergence_convtranspose1d_zeros_padding_outpad_matches_torch() {
    // op_db conv_transpose1d sample i=6.
    let input = tensor(
        &[
            -8.478373527526855,
            -1.765825867652893,
            -4.3228044509887695,
            -2.400455951690674,
        ],
        &[1, 1, 4],
    );
    // weight [in=1, out=2, k=3]
    let weight = tensor(
        &[
            -7.950586795806885,
            3.611605167388916,
            -8.067646980285645,
            -0.5734938383102417,
            3.1285104751586914,
            -3.033684730529785,
        ],
        &[1, 2, 3],
    );
    // stride=2, padding=1, output_padding=1
    let y = conv_transpose1d(&input, &weight, None, 2, 1, 1).unwrap();
    assert_eq!(y.shape(), &[1, 2, 8], "ConvTranspose1d output shape");

    // Live torch F.conv_transpose1d output (full vector).
    let expected = [
        -30.62053680419922,
        82.43988037109375,
        -6.377465724945068,
        48.614891052246094,
        -15.612262725830078,
        53.95989227294922,
        -8.669499397277832,
        19.366031646728516,
        -26.524681091308594,
        26.733402252197266,
        -5.524404525756836,
        7.836060523986816,
        -13.52393913269043,
        14.490673065185547,
        -7.509851455688477,
        7.2822265625,
    ];
    let actual = y.data().unwrap();
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= 1e-2,
            "ConvTranspose1d(pad=1,outpad=1) elem {i}: ferrotorch={a} torch={e}\n ferro={actual:?}"
        );
    }
}
