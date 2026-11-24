
using namespace mlx::steel;
using extents_t = metal::dextents<int32_t, 2>;
using strides_t = metal::array<int32_t, 2>;
using tensor_t = metal::tensor<device T, extents_t, metal::tensor_inline>;
constexpr int TSM = 16 * UM, TSN = 16 * UN, BKT = 64;
const int M = dims[0];
const int tiles_m = (M + TSM * SM - 1) / (TSM * SM);
const int tgp = int(threadgroup_position_in_grid.x);
const int tile_m = tgp % tiles_m, tile_n = tgp / tiles_m;
const int simd = int(simdgroup_index_in_threadgroup);
const int k_group = simd / (SM * SN), sg = simd % (SM * SN);
int m_off = (tile_m * SM + sg % SM) * TSM;
const int n_off = (tile_n * SN + sg / SM) * TSN;
const bool active = m_off < M;
if (m_off + TSM > M && M >= TSM) { m_off = M - TSM; }
const bool inside = m_off + TSM <= M;
const int rows = active ? M - m_off : 0;
const int cols = N - n_off;
tensor_t tA((device T*)(x + size_t(m_off) * K), extents_t(K, rows), strides_t{1, K});
tensor_t tB((device T*)(w + size_t(n_off) * K), extents_t(K, cols), strides_t{1, K});
tensor_t tU((device T*)(w + size_t((GLU ? N : 0) + n_off) * K), extents_t(K, cols), strides_t{1, K});  // up
auto a_chunk = [&](int k) { return tA.template slice<BKT, TSM>(k, 0); };
auto b_chunk = [&](int k) { return tB.template slice<BKT, TSN>(k, 0); };
auto u_chunk = [&](int k) { return tU.template slice<BKT, TSN>(k, 0); };
using a_chunk_t = decltype(a_chunk(0));
using b_chunk_t = decltype(b_chunk(0));
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    TSM, TSN, BKT, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto Dacc = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
auto Uacc = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
for (uint16_t i = 0; i < Dacc.get_capacity(); i++) { Dacc[i] = 0.0f; }
if constexpr (GLU) { for (uint16_t i = 0; i < Uacc.get_capacity(); i++) { Uacc[i] = 0.0f; } }
auto Areg = op.template get_left_input_cooperative_tensor<T, T, float>();
constexpr int k_iters = (K + BKT * KS - 1) / (BKT * KS);
for (int it = 0; it < k_iters; it++) {
    if ((it & 3) == 0) { threadgroup_barrier(mem_flags::mem_none); }
    const int k = (it * KS + k_group) * BKT;
    if (!active || k >= K) { continue; }
    if (inside) {
        auto a = a_chunk(k);
        Areg.load(a);
        auto b = b_chunk(k);
        op.run(Areg, b, Dacc);
        if constexpr (GLU) { auto u = u_chunk(k); op.run(Areg, u, Uacc); }
    } else {
        tensor_t vA((device T*)(x + size_t(m_off) * K) + k, extents_t(BKT, rows), strides_t{1, K});
        tensor_t vB((device T*)(w + size_t(n_off) * K) + k, extents_t(BKT, cols), strides_t{1, K});
        op.run(vA, vB, Dacc);
        if constexpr (GLU) {
            tensor_t vU((device T*)(w + size_t(N + n_off) * K) + k, extents_t(BKT, cols), strides_t{1, K});
            op.run(vA, vU, Uacc);
        }
    }
}
if constexpr (KS > 1) {
    using tg_tensor_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;
    constexpr int kBlockSize = TSM * TSN, kUp = (KS - 1) * SM * SN * kBlockSize;  // the up's partials after the gate's
    threadgroup float partials[(GLU ? 2 : 1) * kUp];
    if (active && k_group > 0) {
        const int at = ((k_group - 1) * SM * SN + sg) * kBlockSize;
        tg_tensor_t tP(partials + at, extents_t(TSN, TSM), strides_t{1, TSN});
        Dacc.store(tP);
        if constexpr (GLU) {
            tg_tensor_t tQ(partials + kUp + at, extents_t(TSN, TSM), strides_t{1, TSN});
            Uacc.store(tQ);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active && k_group == 0) {
        for (int g = 0; g < KS - 1; g++) {
            const int at = (g * SM * SN + sg) * kBlockSize;
            tg_tensor_t tP(partials + at, extents_t(TSN, TSM), strides_t{1, TSN});
            auto part = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
            part.load(tP);
            for (uint16_t i = 0; i < Dacc.get_capacity(); i++) { Dacc[i] += part[i]; }
            if constexpr (GLU) {
                tg_tensor_t tQ(partials + kUp + at, extents_t(TSN, TSM), strides_t{1, TSN});
                part.load(tQ);
                for (uint16_t i = 0; i < Uacc.get_capacity(); i++) { Uacc[i] += part[i]; }
            }
        }
    }
}
if (k_group != 0 || !active) { return; }
auto Dout = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, T>();
for (uint16_t i = 0; i < Dacc.get_capacity(); i++) {
    if constexpr (GLU) {
        const float v = Dacc[i];
        ACTIVATION
        Dout[i] = static_cast<T>(a * Uacc[i]);
    } else {
        Dout[i] = static_cast<T>(Dacc[i]);
    }
}
tensor_t tD(out + size_t(m_off) * N + n_off, extents_t(cols, rows), strides_t{1, N});
if (inside) { auto d = tD.template slice<TSN, TSM>(0, 0); Dout.store(d); } else { Dout.store(tD); }
