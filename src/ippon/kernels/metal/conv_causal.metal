
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
    float v = s >= 0 ? static_cast<float>(x[(b * L + uint(s)) * ROW + c])
                     : static_cast<float>(window[(wb * (K - 1) + uint(s + int(K - 1))) * C + c]);
    acc += taps[c * K + j] * v;
}
y[i] = static_cast<T>(SILU ? acc / (1.0f + metal::exp(-acc)) : acc);
