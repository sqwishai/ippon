constexpr int pack_factor = dq_pack_factor<BITS, 8>();
constexpr int bytes_per_pack = dq_bytes_per_pack<BITS>();
const device uint8_t* wk = (const device uint8_t*)w + k_start * bytes_per_pack / pack_factor;
DqLoader<T, 64, 64, 64 + 16 / sizeof(T), (BM / SM) * 32, GS, BITS> loader(
    wk, scales + k_start / GS, biases + k_start / GS, K, row0, GLU ? N : 0, sg, lane);
