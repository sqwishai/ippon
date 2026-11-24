const device uint8_t* wk = (const device uint8_t*)w + k_start / (8 / BITS);
FqLoader<T, 64, 64, 64 + 16 / sizeof(T), (BM / SM) * 32, GS, BITS> loader(
    wk, scales + k_start / GS, K, row0, GLU ? N : 0, mult, PER_ROW, sg, lane);
