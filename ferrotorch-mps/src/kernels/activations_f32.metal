// activations_f32.metal — relu, sigmoid for ferrotorch MPS backend (#626).
//
// Each thread handles one element. Grid: (len,) threads.
//
// Matches PyTorch's torch.relu / torch.sigmoid on MPS device (f32).

#include <metal_stdlib>
using namespace metal;

kernel void relu_f32(
    device const float* a   [[ buffer(0) ]],
    device       float* out [[ buffer(1) ]],
    constant     uint&  n   [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= n) return;
    out[gid] = max(a[gid], 0.0f);
}

// sigmoid(x) = 1 / (1 + exp(-x))
kernel void sigmoid_f32(
    device const float* a   [[ buffer(0) ]],
    device       float* out [[ buffer(1) ]],
    constant     uint&  n   [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= n) return;
    out[gid] = 1.0f / (1.0f + exp(-a[gid]));
}
