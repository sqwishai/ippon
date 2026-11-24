
using namespace mlx::steel;
const int M = dims[0];
const uint3 tid = threadgroup_position_in_grid;
threadgroup T Ws[2 * 64 * (64 + 16 / sizeof(T))];
constexpr int part = K / S;
const int k_start = int(tid.z) * part;
NAXTile<float, SM / 16, 4> D;
int rows = 0;
// a fused GLU's block: 32 columns of act(gate) * up from w's gate rows and the same up rows (N: the gate's rows)
const int row0 = GLU ? int(tid.x) * 32 : int(tid.x) * 64;
const ushort sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
LOADER
dq_qmm_staged<T, BM, SM>(loader, x + k_start, Ws, K, M, part, tid, sg, lane, D, rows);
if (rows <= 0) { return; }
const int sm = SM * int(sg);
if constexpr (GLU) {
  NAXTile<float, SM / 16, 2> G;
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < SM / 16; m++) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < 2; j++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 8; i++) {
        const float v = D.frag_at(m, j)[i];
        ACTIVATION
        G.frag_at(m, j)[i] = a * D.frag_at(m, j + 2)[i];
      }
    }
  }
  device T* y = out + (int(tid.y) * BM + sm) * static_cast<int64_t>(N) + row0;
  if (rows == SM) { G.store(y, N); } else { G.store_safe(y, N, short2(32, rows)); }
} else {
  device U* y = out + int64_t(tid.z) * M * N + (int(tid.y) * BM + sm) * static_cast<int64_t>(N) + row0;
  if (rows == SM) { D.store(y, N); } else { D.store_safe(y, N, short2(64, rows)); }
}
