
const uint lane = thread_position_in_threadgroup.x;
const uint row = threadgroup_position_in_grid.x;
const uint head = threadgroup_position_in_grid.y;
const int n = meta[0];
const int t = int(row) % n;
const int seq = int(row) / n;
const int pos = POSITIONS ? positions[t] : meta[1] + t;
const device T* src = proj + (size_t)row * ROW + OFFSET + head * STRIDE;
const float a = static_cast<float>(src[lane]);
const float c = static_cast<float>(src[lane + WIDTH / 2]);
float ss = simd_sum(a * a + c * c);
threadgroup float partial[(THREADS + 31) / 32];
threadgroup float normed[WIDTH];
if (thread_index_in_simdgroup == 0) { partial[simdgroup_index_in_threadgroup] = ss; }
threadgroup_barrier(mem_flags::mem_threadgroup);
float total = 0.0f;
for (uint i = 0; i < (THREADS + 31) / 32; i++) { total += partial[i]; }
const float inv = metal::rsqrt(total / float(WIDTH) + EPS);
normed[lane] = static_cast<float>(static_cast<T>(a * inv * static_cast<float>(w[lane])));
normed[lane + WIDTH / 2] = static_cast<float>(static_cast<T>(c * inv * static_cast<float>(w[lane + WIDTH / 2])));
threadgroup_barrier(mem_flags::mem_threadgroup);
device T* dst = y + (((size_t)seq * HEADS + head) * n + t) * WIDTH;
for (uint ch = lane; ch < WIDTH; ch += THREADS) {
    float out = normed[ch];
    if (ch < DIMS) {
        const uint i = ch % (DIMS / 2);
        const float inv_freq = metal::exp2(-(float(i) / float(DIMS / 2)) * LOG2BASE);
        const float theta = (SCALE * float(pos)) * inv_freq;
        const float co = metal::fast::cos(theta);
        const float si = metal::fast::sin(theta);
        const float x1 = normed[i];
        const float x2 = normed[i + DIMS / 2];
        out = ch < DIMS / 2 ? x1 * co - x2 * si : x1 * si + x2 * co;
    }
    dst[ch] = static_cast<T>(out);
}
