// The fp formats' loader (MLX's fp_quantized_nax.h): values fp8 e4m3 (8 bits) or fp4 e2m1 (4 bits), one uint8
// scale per group, e4m3 for groups of 16 (nvfp4) and e8m0 for 32 (mxfp8, mxfp4), times `mult`, an nvfp4
// checkpoint's multiplier folded into the scale as MLX folds its tensor scale (1 otherwise): the matrix's, or its
// row's (`per_row`: projections joined across different global scales); fetch() and store() as DqLoader's, a
// thread's values spanning one group or several, all in one row
template <typename T, int group_size>
inline float fq_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    return float(T(*(thread fp8_e4m3*)(&s)));
  } else {
    return float(T(*(thread fp8_e8m0*)(&s)));
  }
}

template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size, short group_size, short bits>
struct FqLoader {
  static constant constexpr const short pack_factor = 8 / bits;
  static constant constexpr const short BCOLS_PACKED = BCOLS / pack_factor;
  static constant constexpr const short n_reads = (BCOLS_PACKED * BROWS) / tgp_size;  // bytes per thread
  static constant constexpr const short n_words = n_reads / 4;
  static constant constexpr const short n_vals = n_reads * pack_factor;
  static constant constexpr const short n_scales = n_vals > group_size ? n_vals / group_size : 1;
  static_assert(BCOLS_PACKED * BROWS >= tgp_size && n_words * 4 == n_reads, "whole words");
  static_assert(n_vals <= BCOLS && (n_vals % group_size == 0 || group_size % n_vals == 0), "whole groups");

  const short thread_idx;
  const short bi;
  const short bj;
  const device uint8_t* src;
  const device uint8_t* scales;
  const float mult;
  uint32_t raw[n_words];
  float scale[n_scales];

  template <typename Mult>  // MLX passes a one-element multiplier as `constant`, one per row as `device`
  FqLoader(const device uint8_t* w, const device uint8_t* scales_, const int src_ld, const int row0, const int up,
           Mult mult_, const bool per_row, ushort simd_group_id, ushort simd_lane_id) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        src(w + int64_t(row_of(row0, up)) * src_ld / pack_factor + bj),
        scales(scales_ + int64_t(row_of(row0, up)) * src_ld / group_size),
        mult(mult_[per_row ? row_of(row0, up) : 0]) {}

  int row_of(const int row0, const int up) const thread {
    return up == 0 ? row0 + bi : (bi < BROWS / 2 ? row0 + bi : up + row0 + bi - BROWS / 2);
  }

  void fetch(const int k) thread {
    auto p = (const device uint32_t*)(src + k / pack_factor);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_words; i++) { raw[i] = p[i]; }
    const int g = (k + bj * pack_factor) / group_size;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_scales; i++) { scale[i] = fq_scale<T, group_size>(scales[g + i]) * mult; }
  }

  void store(threadgroup T* dst) const thread {
    static_assert(n_vals % 8 == 0, "8-value stores");
    T v[n_vals];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_reads; i++) {
      uint8_t b = uint8_t(raw[i / 4] >> (8 * (i % 4)));
      const float s = scale[n_scales == 1 ? 0 : (i * pack_factor) / group_size];
      if constexpr (bits == 8) {
        v[i] = static_cast<T>(s * float(*(thread fp8_e4m3*)(&b)));
      } else {
        uint8_t lo = b & 0x0f, hi = b >> 4;
        v[2 * i] = static_cast<T>(s * float(*(thread fp4_e2m1*)(&lo)));
        v[2 * i + 1] = static_cast<T>(s * float(*(thread fp4_e2m1*)(&hi)));
      }
    }
    dst += bi * dst_ld + bj * pack_factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_vals; i += 8) { *(threadgroup vec<T, 8>*)(dst + i) = *(thread vec<T, 8>*)(v + i); }
  }
};

