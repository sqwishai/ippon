
uint group = threadgroup_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;
uint row = group / (2 * HK + HV);
uint head = group - row * (2 * HK + HV);
uint base = row * WIDTH;
if (head >= 2 * HK) {  // a value head: widened as it is, and its gates
    uint vh = head - 2 * HK;
    for (uint i = lane; i < DV; i += 32) {
        v[row * HV * DV + vh * DV + i] = static_cast<float>(mixed[base + 2 * HK * DK + vh * DV + i]);
    }
    if (lane == 0) {
        T bt = proj[row * PW + BOFF + vh];
        float sig = 1.0f / (1.0f + metal::exp(-static_cast<float>(bt)));  // in fp32, rounded once to T
        beta[row * HV + vh] = static_cast<float>(static_cast<T>(sig));
        float x = static_cast<float>(proj[row * PW + BOFF + HV + vh]) + dt_bias[vh];
        float softplus = metal::max(x, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(x)));
        gamma[row * HV + vh] = metal::exp(neg_a[vh] * softplus);
    }
    return;
}
uint at = base + head * DK;  // q's heads, then k's
float ss = 0.0f;
for (uint i = lane; i < DK; i += 32) {
    float x = static_cast<float>(mixed[at + i]);
    ss += x * x;
}
ss = simd_sum(ss);
float inv = metal::rsqrt(ss + 1e-6f);
bool is_q = head < HK;
uint out = row * HK * DK + (is_q ? head : head - HK) * DK;
for (uint i = lane; i < DK; i += 32) {
    float x = static_cast<float>(mixed[at + i]) * inv;
    if (is_q) { q[out + i] = x * SCALE; } else { k[out + i] = x; }
}
