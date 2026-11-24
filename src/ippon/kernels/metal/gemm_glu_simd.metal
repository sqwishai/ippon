// act(x @ gate.T) * (x @ up.T) for w = [gate; up] through MLX's SIMD-group tile library (M1 to M4): each K step
// loads x's tile into threadgroup memory once, the gate's and the up's tiles beside it, and multiplies it into both
// accumulators; the product is stored once from them, so the [M, 2I] projection is never written or read back.
using namespace mlx::steel;
using kernel_t = GEMMKernel<T, T, BM, BN, BK, WM, WN, false, true, true, true>;
threadgroup T As[kernel_t::tgp_mem_size_a];
threadgroup T Gs[kernel_t::tgp_mem_size_b];
threadgroup T Us[kernel_t::tgp_mem_size_b];
const int M = dims[0];
const int tiles_m = (M + BM - 1) / BM;
const int swizzled = int(threadgroup_position_in_grid.x) & ((1 << SWZ) - 1);
const int tid_y = (int(threadgroup_position_in_grid.y) << SWZ) + swizzled;
const int tid_x = int(threadgroup_position_in_grid.x) >> SWZ;
if (tid_x >= N / BN || tid_y >= tiles_m) { return; }
const ushort sg = ushort(simdgroup_index_in_threadgroup), lane = ushort(thread_index_in_simdgroup);
const int row = (SHIFT && tid_y * BM + BM > M) ? M - BM : tid_y * BM;
const int col = tid_x * BN;  // the gate's rows col..; the up's are N rows further
thread typename kernel_t::loader_a_t loader_a(x + size_t(row) * K, K, As, sg, lane);
thread typename kernel_t::loader_b_t loader_g(w + size_t(col) * K, K, Gs, sg, lane);
thread typename kernel_t::loader_b_t loader_u(w + size_t(N + col) * K, K, Us, sg, lane);
thread typename kernel_t::mma_t g(sg, lane);
thread typename kernel_t::mma_t u(sg, lane);
const short rows = short(min(BM, M - row));
const bool full = ALIGN_M || rows == BM;
const short2 dims_a = short2(BK, rows);
for (int k = 0; k < K / BK; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (full) { loader_a.load_unsafe(); } else { loader_a.load_safe(dims_a); }
    loader_g.load_unsafe();
    loader_u.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    g.mma(As, Gs);
    u.mma(As, Us);
    loader_a.next();
    loader_g.next();
    loader_u.next();
}
STEEL_PRAGMA_UNROLL
for (short i = 0; i < decltype(g.Ctile)::kElemsPerTile; i++) {
    const float v = g.Ctile.elems()[i];
    ACTIVATION
    g.Ctile.elems()[i] = a * u.Ctile.elems()[i];
}
device T* D = out + size_t(row) * N + col;
if (full) {
    g.store_result(D, N);
} else {
    g.store_result_safe(D, N, short2(BN, rows));
}
