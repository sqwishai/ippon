
uint row = threadgroup_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;
uint base = row * WIDTH;
float s = 0.0f;
float ss = 0.0f;
for (uint i = lane; i < WIDTH; i += THREADS) {
    float v = static_cast<float>(x[base + i]);
    ADD_SUM
    s += v;
    ss += v * v;
}
threadgroup float partial_s[THREADS / 32];
threadgroup float partial_ss[THREADS / 32];
s = simd_sum(s);
ss = simd_sum(ss);
if (thread_index_in_simdgroup == 0) {
    partial_s[simdgroup_index_in_threadgroup] = s;
    partial_ss[simdgroup_index_in_threadgroup] = ss;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simdgroup_index_in_threadgroup == 0) {
    bool live = thread_index_in_simdgroup < THREADS / 32;
    s = simd_sum(live ? partial_s[thread_index_in_simdgroup] : 0.0f);
    ss = simd_sum(live ? partial_ss[thread_index_in_simdgroup] : 0.0f);
    if (thread_index_in_simdgroup == 0) { partial_s[0] = s; partial_ss[0] = ss; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean = LAYER ? partial_s[0] / float(WIDTH) : 0.0f;
float var = metal::max(partial_ss[0] / float(WIDTH) - mean * mean, 0.0f);  // never below 0 from rounding
float inv = metal::rsqrt(var + EPS);
for (uint i = lane; i < WIDTH; i += THREADS) {
    float v = static_cast<float>(x[base + i]);
    ADD_STORE
    n[base + i] = static_cast<T>((v - mean) * inv * w[WEIGHT_AT + i]);
}
