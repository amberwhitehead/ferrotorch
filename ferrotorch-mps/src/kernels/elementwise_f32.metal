// elementwise_f32.metal — add, sub, mul, div for ferrotorch MPS backend (#626).
//
// Each thread handles one element. Grid: (len,) threads.
//
// Matches PyTorch's torch.add / torch.sub / torch.mul / torch.div on MPS
// device (single-precision, element-wise, no broadcasting here — caller
// broadcasts before dispatch).

#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* a   [[ buffer(0) ]],
    device const float* b   [[ buffer(1) ]],
    device       float* out [[ buffer(2) ]],
    constant     uint&  n   [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] + b[gid];
}

kernel void sub_f32(
    device const float* a   [[ buffer(0) ]],
    device const float* b   [[ buffer(1) ]],
    device       float* out [[ buffer(2) ]],
    constant     uint&  n   [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] - b[gid];
}

kernel void mul_f32(
    device const float* a   [[ buffer(0) ]],
    device const float* b   [[ buffer(1) ]],
    device       float* out [[ buffer(2) ]],
    constant     uint&  n   [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] * b[gid];
}

kernel void div_f32(
    device const float* a   [[ buffer(0) ]],
    device const float* b   [[ buffer(1) ]],
    device       float* out [[ buffer(2) ]],
    constant     uint&  n   [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] / b[gid];
}
