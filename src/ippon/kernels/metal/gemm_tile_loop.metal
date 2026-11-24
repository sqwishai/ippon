
using namespace mlx::steel;
template <typename T, short SM, short SN, int BK, int K, bool ALIGNED, bool PIPE>
METAL_FUNC NAXTile<float, SM / 16, SN / 16> steel_tile(const device T* A, const device T* B, const short rows) {
    if constexpr (!PIPE) {
        return gemm_loop<T, SM, SN, 32, BK, false, true, ALIGNED, true, true, float>(A, B, K, K, K, K / BK, rows, SN);
    } else {
        constexpr short SK = 32;
        NAXTile<float, SM / 16, SN / 16> acc;
        acc.clear();
        NAXTile<T, SM / 16, SK / 16> a0, a1;
        NAXTile<T, SN / 16, SK / 16> b0, b1;
        if (ALIGNED) { a0.load(A, K); } else { a0.load_safe(A, K, short2(SK, rows)); }
        b0.load(B, K);
        for (int k = 0; k < K; k += 2 * SK) {
            if (ALIGNED) { a1.load(A + k + SK, K); } else { a1.load_safe(A + k + SK, K, short2(SK, rows)); }
            b1.load(B + k + SK, K);
            tile_matmad_nax(acc, a0, metal::bool_constant<false>{}, b0, metal::bool_constant<true>{});
            if (k + 2 * SK < K) {
                if (ALIGNED) { a0.load(A + k + 2 * SK, K); }
                else { a0.load_safe(A + k + 2 * SK, K, short2(SK, rows)); }
                b0.load(B + k + 2 * SK, K);
            }
            tile_matmad_nax(acc, a1, metal::bool_constant<false>{}, b1, metal::bool_constant<true>{});
        }
        return acc;
    }
}
// the fused GLU's two accumulators in one pass over K: each 32-deep slice of x loads once and multiplies into the
// gate's tile and the up's, where a loop for each read x twice (M5 Pro, fp16, against MLX's matmul then the GLU:
// d1-3B's 21504x2048 1.07-1.11 times as fast from 160 to 2,048 rows, the two loops 1.03-1.07; Clef-Flash's
// 24576x4096 1.22 and 1.12 at 160 and 333 rows, the two loops 1.15 and 1.02; a ModernBERT-large GeGLU 1.03-1.17, 1.02-1.15)
template <typename T, short SM, short SN, int K, bool ALIGNED, bool PIPE>
METAL_FUNC void glu_tiles(const device T* A, const device T* G, const device T* U, const short rows,
                          thread NAXTile<float, SM / 16, SN / 16>& g, thread NAXTile<float, SM / 16, SN / 16>& u) {
    constexpr short SK = 32;
    g.clear();
    u.clear();
    if constexpr (!PIPE) {
        NAXTile<T, SM / 16, SK / 16> a;
        NAXTile<T, SN / 16, SK / 16> bg, bu;
        for (int k = 0; k < K; k += SK) {
            if (ALIGNED) { a.load(A + k, K); } else { a.load_safe(A + k, K, short2(SK, rows)); }
            bg.load(G + k, K);
            bu.load(U + k, K);
            tile_matmad_nax(g, a, metal::bool_constant<false>{}, bg, metal::bool_constant<true>{});
            tile_matmad_nax(u, a, metal::bool_constant<false>{}, bu, metal::bool_constant<true>{});
        }
    } else {  // the next slices load while the current ones multiply
        NAXTile<T, SM / 16, SK / 16> a0, a1;
        NAXTile<T, SN / 16, SK / 16> g0, g1, u0, u1;
        if (ALIGNED) { a0.load(A, K); } else { a0.load_safe(A, K, short2(SK, rows)); }
        g0.load(G, K);
        u0.load(U, K);
        for (int k = 0; k < K; k += 2 * SK) {
            if (ALIGNED) { a1.load(A + k + SK, K); } else { a1.load_safe(A + k + SK, K, short2(SK, rows)); }
            g1.load(G + k + SK, K);
            u1.load(U + k + SK, K);
            tile_matmad_nax(g, a0, metal::bool_constant<false>{}, g0, metal::bool_constant<true>{});
            tile_matmad_nax(u, a0, metal::bool_constant<false>{}, u0, metal::bool_constant<true>{});
            if (k + 2 * SK < K) {
                if (ALIGNED) { a0.load(A + k + 2 * SK, K); }
                else { a0.load_safe(A + k + 2 * SK, K, short2(SK, rows)); }
                g0.load(G + k + 2 * SK, K);
                u0.load(U + k + 2 * SK, K);
            }
            tile_matmad_nax(g, a1, metal::bool_constant<false>{}, g1, metal::bool_constant<true>{});
            tile_matmad_nax(u, a1, metal::bool_constant<false>{}, u1, metal::bool_constant<true>{});
        }
    }
}
