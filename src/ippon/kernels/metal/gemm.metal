
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
const device T* B = w + size_t(col) * K;
device T* D = out + size_t(row) * N + col;
const int left = M - row;
if (ALIGN_M || left >= SM) {  // ALIGN_M: every tile whole, and no second copy of the loop compiled
    auto tile = steel_tile<T, SM, SN, BK, K, true, PIPE>(A, B, SM);
    tile.store(D, N);
} else {  // a SIMD group past the last row still runs the loop, without loads, for its barriers
    const short rows = short(max(left, 0));
    auto tile = steel_tile<T, SM, SN, BK, K, false, PIPE>(A, B, rows);
    if (rows > 0) { tile.store_safe(D, N, short2(SN, rows)); }
}
