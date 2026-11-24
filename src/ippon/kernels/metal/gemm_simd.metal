// x [M, K] @ w [N, K].T through MLX's own SIMD-group tile library (steel/gemm/gemm.h, M1 to M4: no neural
// accelerators), its tile a template parameter: MLX's GEMMKernel loaders, multiply-accumulate and loop, in MLX's
// order, so a product equals MLX's matmul at the same tile bit for bit.
using namespace mlx::steel;
using kernel_t = GEMMKernel<T, T, BM, BN, BK, WM, WN, false, true, true, true>;
threadgroup T As[kernel_t::tgp_mem_size_a];
threadgroup T Bs[kernel_t::tgp_mem_size_b];
const int M = dims[0];
const int tiles_m = (M + BM - 1) / BM;
const int swizzled = int(threadgroup_position_in_grid.x) & ((1 << SWZ) - 1);
const int tid_y = (int(threadgroup_position_in_grid.y) << SWZ) + swizzled;
const int tid_x = int(threadgroup_position_in_grid.x) >> SWZ;
if (tid_x >= N / BN || tid_y >= tiles_m) { return; }
const ushort sg = ushort(simdgroup_index_in_threadgroup), lane = ushort(thread_index_in_simdgroup);
// SHIFT: the last row block moves back to end at M (M holds a whole block), so every block runs the whole-tile loop;
// the rows it shares with the block before get the same values again
const int row = (SHIFT && tid_y * BM + BM > M) ? M - BM : tid_y * BM;
const int col = tid_x * BN;
thread typename kernel_t::loader_a_t loader_a(x + size_t(row) * K, K, As, sg, lane);
thread typename kernel_t::loader_b_t loader_b(w + size_t(col) * K, K, Bs, sg, lane);
thread typename kernel_t::mma_t mma_op(sg, lane);
device T* D = out + size_t(row) * N + col;
const short rows = short(min(BM, M - row));
const short bn = BN, none = 0;
if (ALIGN_M || rows == BM) {
    kernel_t::template gemm_loop<true, true, true>(As, Bs, K / BK, loader_a, loader_b, mma_op, rows, bn, none);
    mma_op.store_result(D, N);
} else {
    kernel_t::template gemm_loop<false, true, true>(As, Bs, K / BK, loader_a, loader_b, mma_op, rows, bn, none);
    mma_op.store_result_safe(D, N, short2(BN, rows));
}
