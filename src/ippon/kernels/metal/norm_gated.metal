
uint group = threadgroup_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;
uint row = group / HEADS;
uint head = group - row * HEADS;
uint obase = group * WIDTH;
uint zbase = row * ROW + OFFSET + head * WIDTH;
float ss = 0.0f;
for (uint i = lane; i < WIDTH; i += THREADS) {
    float v = static_cast<float>(static_cast<T>(o[obase + i]));
    ss += v * v;
}
threadgroup float partial[THREADS / 32];
ss = simd_sum(ss);
if (thread_index_in_simdgroup == 0) { partial[simdgroup_index_in_threadgroup] = ss; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simdgroup_index_in_threadgroup == 0) {
    ss = simd_sum(thread_index_in_simdgroup < THREADS / 32 ? partial[thread_index_in_simdgroup] : 0.0f);
    if (thread_index_in_simdgroup == 0) { partial[0] = ss; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv = metal::rsqrt(partial[0] / float(WIDTH) + EPS);
for (uint i = lane; i < WIDTH; i += THREADS) {
    float v = static_cast<float>(static_cast<T>(o[obase + i]));
    T normed = static_cast<T>(v * inv);
    float weighted = static_cast<float>(w[i]) * static_cast<float>(normed);
    if (ROUND_W) { weighted = static_cast<float>(static_cast<T>(weighted)); }
    float zz = static_cast<float>(z[zbase + i]);
    y[obase + i] = static_cast<T>(weighted * (zz / (1.0f + metal::exp(-zz))));
}
