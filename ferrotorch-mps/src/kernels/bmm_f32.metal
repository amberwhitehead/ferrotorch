// bmm_f32.metal — batched matrix multiply for ferrotorch MPS backend (#626).
//
// C[b, m, n] = A[b, m, k] * B[b, k, n]  for b in 0..batch
//
// Grid: (ceil(N/16), ceil(M/16), batch) threadgroups of 16x16x1 threads.
//
// Matches PyTorch's torch.bmm on MPS device (single-precision).

#include <metal_stdlib>
using namespace metal;

kernel void bmm_f32(
    device const float* A     [[ buffer(0) ]],
    device const float* B     [[ buffer(1) ]],
    device       float* C     [[ buffer(2) ]],
    constant     uint&  batch [[ buffer(3) ]],
    constant     uint&  M     [[ buffer(4) ]],
    constant     uint&  K     [[ buffer(5) ]],
    constant     uint&  N     [[ buffer(6) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    uint col   = gid.x;
    uint row   = gid.y;
    uint b     = gid.z;

    if (b >= batch || row >= M || col >= N) return;

    uint a_off = b * M * K;
    uint b_off = b * K * N;
    uint c_off = b * M * N;

    float acc = 0.0f;
    for (uint i = 0; i < K; ++i) {
        acc += A[a_off + row * K + i] * B[b_off + i * N + col];
    }
    C[c_off + row * N + col] = acc;
}
