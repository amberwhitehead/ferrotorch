// matmul_f32.metal — single-stream GEMM for ferrotorch MPS backend (#626).
//
// C[m, n] = A[m, k] * B[k, n]
//
// Each thread computes one output element C[row, col].
// Grid: (ceil(N/16), ceil(M/16)) threadgroups of 16x16 threads.
//
// Matches PyTorch's torch.matmul / torch.mm on MPS device (single-precision).

#include <metal_stdlib>
using namespace metal;

kernel void matmul_f32(
    device const float* A      [[ buffer(0) ]],
    device const float* B      [[ buffer(1) ]],
    device       float* C      [[ buffer(2) ]],
    constant     uint&  M      [[ buffer(3) ]],
    constant     uint&  K      [[ buffer(4) ]],
    constant     uint&  N      [[ buffer(5) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (uint i = 0; i < K; ++i) {
        acc += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = acc;
}
