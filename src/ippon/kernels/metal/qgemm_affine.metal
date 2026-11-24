
template <int bits, int wsize = 8>
inline constexpr short dq_pack_factor() { return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits); }
template <int bits, int wsize = 8>
inline constexpr short dq_bytes_per_pack() {
  constexpr int p2 = (bits & (bits - 1)) == 0;
  return p2 ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

// One pack of `bits`-bit values (pack factor N) into registers: MLX's dequantize, the same arithmetic
template <typename U, int N, int bits>
inline void dq_unpack(const thread uint8_t* w, U scale, U bias, thread U* v) {
  const float s = float(scale);
  const float b = float(bias);
  if (bits == 2) {
    float sc[4] = {s, s / 4.0f, s / 16.0f, s / 64.0f};
    v[0] = static_cast<U>(sc[0] * (w[0] & 0x03) + b);
    v[1] = static_cast<U>(sc[1] * (w[0] & 0x0c) + b);
    v[2] = static_cast<U>(sc[2] * (w[0] & 0x30) + b);
    v[3] = static_cast<U>(sc[3] * (w[0] & 0xc0) + b);
  } else if (bits == 3) {
    v[0] = static_cast<U>((w[0] & 0x7) * s + b);
    v[1] = static_cast<U>(((w[0] & 0x38) >> 3) * s + b);
    v[2] = static_cast<U>((((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * s + b);
    v[3] = static_cast<U>(((w[1] & 0xe) >> 1) * s + b);
    v[4] = static_cast<U>(((w[1] & 0x70) >> 4) * s + b);
    v[5] = static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * s + b);
    v[6] = static_cast<U>(((w[2] & 0x1c) >> 2) * s + b);
    v[7] = static_cast<U>(((w[2] & 0xe0) >> 5) * s + b);
  } else if (bits == 4) {
    float sc[2] = {s, s / 16.0f};
    v[0] = static_cast<U>(sc[0] * (w[0] & 0x0f) + b);
    v[1] = static_cast<U>(sc[1] * (w[0] & 0xf0) + b);
  } else if (bits == 5) {
    v[0] = static_cast<U>((w[0] & 0x1f) * s + b);
    v[1] = static_cast<U>((((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * s + b);
    v[2] = static_cast<U>(((w[1] & 0x7c) >> 2) * s + b);
    v[3] = static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * s + b);
    v[4] = static_cast<U>((((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * s + b);
    v[5] = static_cast<U>(((w[3] & 0x3e) >> 1) * s + b);
    v[6] = static_cast<U>((((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * s + b);
    v[7] = static_cast<U>(((w[4] & 0xf8) >> 3) * s + b);
  } else if (bits == 6) {
    v[0] = static_cast<U>((w[0] & 0x3f) * s + b);
    v[1] = static_cast<U>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
    v[2] = static_cast<U>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
    v[3] = static_cast<U>(((w[2] >> 2) & 0x3f) * s + b);
  } else if (bits == 8) {
    v[0] = static_cast<U>(s * w[0] + b);
  }
}

// fetch() reads one 64-deep K step's packed words and its group's scale and bias into registers; store() unpacks
// them into threadgroup memory (MLX's QuantizedBlockLoaderStaged)
template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size, short group_size, short bits>
struct DqLoader {
  static constant constexpr const short pack_factor = dq_pack_factor<bits, 8>();
  static constant constexpr const short bytes_per_pack = dq_bytes_per_pack<bits>();
  static constant constexpr const short BCOLS_PACKED = BCOLS / pack_factor;
  static constant constexpr const short n_reads = (BCOLS_PACKED * BROWS) / tgp_size;
  static constant constexpr const short n_words = n_reads * bytes_per_pack / 4;
  static_assert(BCOLS_PACKED * BROWS >= tgp_size && n_words * 4 == n_reads * bytes_per_pack, "whole words");
  static_assert(n_reads * pack_factor <= group_size && n_reads * pack_factor <= BCOLS, "one row and group");

  const short thread_idx;
  const short bi;
  const short bj;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;
  uint32_t raw[n_words];
  T scale;
  T bias;

  // the tile's rows: w's from `row0`, or with `up` (a fused GLU's [gate; up], `up` the gate's rows) its first half
  // the gate's from row0 and its second the up's from up + row0
  DqLoader(const device uint8_t* w, const device T* scales_, const device T* biases_, const int src_ld,
           const int row0, const int up, ushort simd_group_id, ushort simd_lane_id) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        src(w + int64_t(row_of(row0, up)) * src_ld * bytes_per_pack / pack_factor + bj * bytes_per_pack),
        scales(scales_ + int64_t(row_of(row0, up)) * src_ld / group_size),
        biases(biases_ + int64_t(row_of(row0, up)) * src_ld / group_size) {}

  int row_of(const int row0, const int up) const thread {
    return up == 0 ? row0 + bi : (bi < BROWS / 2 ? row0 + bi : up + row0 + bi - BROWS / 2);
  }

  void fetch(const int k) thread {
    auto p = (const device uint32_t*)(src + k * bytes_per_pack / pack_factor);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_words; i++) { raw[i] = p[i]; }
    const int g = (k + bj * pack_factor) / group_size;
    scale = scales[g];
    bias = biases[g];
  }

  void store(threadgroup T* dst) const thread {
    constexpr short n_vals = n_reads * pack_factor;
    static_assert(n_vals % 8 == 0, "8-value stores");
    T v[n_vals];
    uint8_t b[n_words * 4];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_words * 4; i++) { b[i] = uint8_t(raw[i / 4] >> (8 * (i % 4))); }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_reads; i++) {
      dq_unpack<T, pack_factor, bits>(b + i * bytes_per_pack, scale, bias, v + i * pack_factor);
    }
    dst += bi * dst_ld + bj * pack_factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_vals; i += 8) { *(threadgroup vec<T, 8>*)(dst + i) = *(thread vec<T, 8>*)(v + i); }
  }
};

