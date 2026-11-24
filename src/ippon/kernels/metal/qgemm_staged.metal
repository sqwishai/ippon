// Each SIMD group computes SM rows of a BM x 64 block over K steps [0, k_end) into Dtile, the weights' 64 rows from
// `loader_w` (DqLoader's or FqLoader's) (MLX's qmm_t_nax_staged_impl, its store left to the caller): `rows`, this
// SIMD group's rows of x, set
template <typename T, int BM, int SM, typename L>
METAL_FUNC void dq_qmm_staged(
    thread L& loader_w, const device T* x, threadgroup T* Ws, const int K, const int M, const int k_end, uint3 tid,
    uint simd_gid, uint simd_lid, thread NAXTile<float, SM / 16, 4>& Dtile, thread int& rows) {
  constexpr int BK = 64, BN = 64;
  static_assert(SM > 0 && SM % 16 == 0 && BM % SM == 0, "whole row fragments");
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int tile_size = BN * BK_padded;
  using frag_t = typename BaseNAXFrag::template dtype_frag_t<T>;

  const int y_row = tid.y * BM;

  const int tile_rows = min(BM, M - y_row);
  constexpr int TM = SM / 16;
  const int sm = SM * int(simd_gid);
  rows = min(SM, tile_rows - sm);
  const short2 sc = BaseNAXFrag::get_coord();
  const device T* x0[TM];
  const device T* x1[TM];
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < TM; m++) {
    const int r0 = min(sm + m * 16 + int(sc.y), tile_rows - 1);
    const int r1 = min(sm + m * 16 + int(sc.y) + 8, tile_rows - 1);
    x0[m] = x + (y_row + r0) * static_cast<int64_t>(K) + sc.x;
    x1[m] = x + (y_row + r1) * static_cast<int64_t>(K) + sc.x;
  }

  Dtile.clear();
  vec<T, 4> a_cur[BK / 16][TM][2];
  vec<T, 4> a_next[BK / 16][TM][2];

  loader_w.fetch(0);
  loader_w.store(Ws);
  if (rows > 0) {
    STEEL_PRAGMA_UNROLL
    for (short s = 0; s < BK / 16; s++) {
      STEEL_PRAGMA_UNROLL
      for (short m = 0; m < TM; m++) {
        a_cur[s][m][0] = *(const device vec<T, 4>*)(x0[m] + s * 16);
        a_cur[s][m][1] = *(const device vec<T, 4>*)(x1[m] + s * 16);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0, it = 0; k < k_end; k += BK, it++) {
    const bool more = k + BK < k_end;
    if (more) { loader_w.fetch(k + BK); }
    if (rows > 0) {
      const int kn = more ? k + BK : k;
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        STEEL_PRAGMA_UNROLL
        for (short m = 0; m < TM; m++) {
          a_next[s][m][0] = *(const device vec<T, 4>*)(x0[m] + kn + s * 16);
          a_next[s][m][1] = *(const device vec<T, 4>*)(x1[m] + kn + s * 16);
        }
      }
      const threadgroup T* ws = Ws + (it & 1) * tile_size + sc.y * BK_padded + sc.x;
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < BN / 16; j += 2) {
          const threadgroup T* p0 = ws + (j * 16) * BK_padded + s * 16;
          const threadgroup T* p1 = p0 + 16 * BK_padded;
          frag_t b0;
          frag_t b1;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const vec<T, 4> u0 = *(const threadgroup vec<T, 4>*)(p0 + h * 8 * BK_padded);
            const vec<T, 4> u1 = *(const threadgroup vec<T, 4>*)(p1 + h * 8 * BK_padded);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) { b0[h * 4 + c] = u0[c]; b1[h * 4 + c] = u1[c]; }
          }
          STEEL_PRAGMA_UNROLL
          for (short m = 0; m < TM; m++) {
            frag_t a;
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) { a[c] = a_cur[s][m][0][c]; a[4 + c] = a_cur[s][m][1][c]; }
            BaseNAXFrag::mma(Dtile.frag_at(m, j), Dtile.frag_at(m, j + 1), a, metal::bool_constant<false>{}, b0, b1,
                             metal::bool_constant<true>{});
          }
        }
      }
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        STEEL_PRAGMA_UNROLL
        for (short m = 0; m < TM; m++) { a_cur[s][m][0] = a_next[s][m][0]; a_cur[s][m][1] = a_next[s][m][1]; }
      }
    }
    if (more) { loader_w.store(Ws + ((it + 1) & 1) * tile_size); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
