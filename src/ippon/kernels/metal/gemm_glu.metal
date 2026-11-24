
using namespace mlx::steel;
const int M = dims[0];
constexpr short SM = BM / WM, SN = BN / WN;
const int tiles_m = (M + BM - 1) / BM;
const int swizzled = int(threadgroup_position_in_grid.x) & ((1 << SWZ) - 1);
const int tid_y = (int(threadgroup_position_in_grid.y) << SWZ) + swizzled;
const int tid_x = int(threadgroup_position_in_grid.x) >> SWZ;
if (tid_x >= N / BN || tid_y >= tiles_m) { return; }
const short sg = short(simdgroup_index_in_threadgroup);
// SHIFT: M holds a whole block but isn't a multiple of one, so the last row block moves back to end at M and every
// block runs the whole-tile loop (the rows it shares with the block before get the same values again, as in MLX's
// thin kernel); ALIGN_M is then true too, and no partial-tile loop is compiled
const int block = (SHIFT && tid_y * BM + BM > M) ? M - BM : tid_y * BM;
const int row = block + SM * (sg / WN), col = tid_x * BN + SN * (sg % WN);
const device T* A = x + size_t(row) * K;
const device T* G = w + size_t(col) * K;  // the gate's rows col..; the up's are N rows further
const device T* U = w + size_t(N + col) * K;
device T* D = out + size_t(row) * N + col;
const int left = M - row;
const bool full = ALIGN_M || left >= SM;
const short rows = full ? SM : short(max(left, 0));
NAXTile<float, SM / 16, SN / 16> g, u;
if (full) {
    glu_tiles<T, SM, SN, K, true, PIPE>(A, G, U, SM, g, u);
} else {  // a SIMD group past the last row still runs the loop, without loads
    glu_tiles<T, SM, SN, K, false, PIPE>(A, G, U, rows, g, u);
}
using tile_t = decltype(g);
STEEL_PRAGMA_UNROLL
for (short mm = 0; mm < tile_t::kTileRows; mm++) {
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < tile_t::kTileCols; nn++) {
        thread auto& gf = g.frag_at(mm, nn);
        const thread auto& uf = u.frag_at(mm, nn);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < tile_t::kElemsPerFrag; i++) {
            const float v = gf[i];
            ACTIVATION
            gf[i] = a * uf[i];
        }
    }
}
if (full) {
    g.store(D, N);
} else if (rows > 0) {
    g.store_safe(D, N, short2(SN, rows));
}
