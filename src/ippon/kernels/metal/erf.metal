
inline float erf_f32(float x) {
    float s = x < 0.0f ? -1.0f : 1.0f;
    x = metal::fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float y = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return s * (1.0f - y * metal::exp(-x * x));
}
