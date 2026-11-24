
const uint N = sizes[0], L = sizes[1];  // at run time: one compiled kernel for every length
uint i = thread_position_in_grid.x;
if (i >= N) { return; }
uint c = i % C;
uint t = (i / C) % L;
uint b = i / (C * L);
uint wb = (WB == 1) ? 0 : b;
float acc = 0.0f;
for (uint j = 0; j < K; ++j) {
    int s = int(t) + int(j) - int(K - 1);
    float v;
    if (s >= 0) {
        uint base = (b * L + uint(s)) * 3 * C;
        float bx = static_cast<float>(proj[base + c]) * static_cast<float>(proj[base + 2 * C + c]);
        v = static_cast<float>(static_cast<T>(bx));
    } else {
        v = static_cast<float>(window[(wb * (K - 1) + uint(s + int(K - 1))) * C + c]);
    }
    acc += taps[c * K + j] * v;
}
uint row = (b * L + t) * 3 * C;
y[i] = static_cast<T>(static_cast<float>(proj[row + C + c]) * static_cast<float>(static_cast<T>(acc)));
