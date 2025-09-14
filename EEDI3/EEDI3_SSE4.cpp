#ifdef EEDI3_X86
#define INSTRSET 5
#include "EEDI3.h"

template<typename vector_t>
static inline void calculateConnectionCosts(const vector_t* srcp, const bool* bmask, float* ccosts, const int width, const int stride,
                                            const EEDI3Data* VS_RESTRICT d) noexcept {
    auto src3p = reinterpret_cast<const __m128i*>(srcp) + MARGIN_H;
    auto src1p = src3p + stride;
    auto src1n = src1p + stride;
    auto src3n = src1n + stride;

    if (d->cost3) {
        for (int x = 0; x < width; x++) {
            if (!bmask || bmask[x]) {
                const int umax = std::min({ x, width - 1 - x, d->mdis });
                for (int u = -umax; u <= umax; u++) {
                    const int u2 = u * 2;
                    const bool s1Flag = (u >= 0 && x >= u2) || (u <= 0 && x < width + u2);
                    const bool s2Flag = (u <= 0 && x >= -u2) || (u >= 0 && x < width + u2);
                    Vec4i s0 = zero_si128(), s1, s2;

                    for (int k = -(d->nrad); k <= d->nrad; k++) {
                        s0 += abs(Vec4i().load_a(src3p + x + u + k) - Vec4i().load_a(src1p + x - u + k)) +
                            abs(Vec4i().load_a(src1p + x + u + k) - Vec4i().load_a(src1n + x - u + k)) +
                            abs(Vec4i().load_a(src1n + x + u + k) - Vec4i().load_a(src3n + x - u + k));
                    }

                    if (s1Flag) {
                        s1 = zero_si128();
                        for (int k = -(d->nrad); k <= d->nrad; k++) {
                            s1 += abs(Vec4i().load_a(src3p + x + k) - Vec4i().load_a(src1p + x - u2 + k)) +
                                abs(Vec4i().load_a(src1p + x + k) - Vec4i().load_a(src1n + x - u2 + k)) +
                                abs(Vec4i().load_a(src1n + x + k) - Vec4i().load_a(src3n + x - u2 + k));
                        }
                    }

                    if (s2Flag) {
                        s2 = zero_si128();
                        for (int k = -(d->nrad); k <= d->nrad; k++) {
                            s2 += abs(Vec4i().load_a(src3p + x + u2 + k) - Vec4i().load_a(src1p + x + k)) +
                                abs(Vec4i().load_a(src1p + x + u2 + k) - Vec4i().load_a(src1n + x + k)) +
                                abs(Vec4i().load_a(src1n + x + u2 + k) - Vec4i().load_a(src3n + x + k));
                        }
                    }

                    s1 = s1Flag ? s1 : (s2Flag ? s2 : s0);
                    s2 = s2Flag ? s2 : (s1Flag ? s1 : s0);

                    const Vec4i ip = (Vec4i().load_a(src1p + x + u) + Vec4i().load_a(src1n + x - u) + 1) >> 1; // should use cubic if ucubic=true
                    const Vec4i v = abs(Vec4i().load_a(src1p + x) - ip) + abs(Vec4i().load_a(src1n + x) - ip);
                    const Vec4f result = mul_add(d->alpha, to_float(s0 + s1 + s2), mul_add(d->remainingWeight, to_float(v), d->beta * std::abs(u)));
                    result.store(ccosts + (d->tpitch * x + d->mdis + u) * d->vectorSize);
                }
            }
        }
    } else {
        for (int x = 0; x < width; x++) {
            if (!bmask || bmask[x]) {
                const int umax = std::min({ x, width - 1 - x, d->mdis });
                for (int u = -umax; u <= umax; u++) {
                    Vec4i s = zero_si128();

                    for (int k = -(d->nrad); k <= d->nrad; k++) {
                        s += abs(Vec4i().load_a(src3p + x + u + k) - Vec4i().load_a(src1p + x - u + k)) +
                            abs(Vec4i().load_a(src1p + x + u + k) - Vec4i().load_a(src1n + x - u + k)) +
                            abs(Vec4i().load_a(src1n + x + u + k) - Vec4i().load_a(src3n + x - u + k));
                    }

                    const Vec4i ip = (Vec4i().load_a(src1p + x + u) + Vec4i().load_a(src1n + x - u) + 1) >> 1; // should use cubic if ucubic=true
                    const Vec4i v = abs(Vec4i().load_a(src1p + x) - ip) + abs(Vec4i().load_a(src1n + x) - ip);
                    const Vec4f result = mul_add(d->alpha, to_float(s), mul_add(d->remainingWeight, to_float(v), d->beta * std::abs(u)));
                    result.store(ccosts + (d->tpitch * x + d->mdis + u) * d->vectorSize);
                }
            }
        }
    }
}

template<>
inline void calculateConnectionCosts(const float* srcp, const bool* bmask, float* ccosts, const int width, const int stride,
                                     const EEDI3Data* VS_RESTRICT d) noexcept {
    auto src3p = reinterpret_cast<const __m128*>(srcp) + MARGIN_H;
    auto src1p = src3p + stride;
    auto src1n = src1p + stride;
    auto src3n = src1n + stride;

    if (d->cost3) {
        for (int x = 0; x < width; x++) {
            if (!bmask || bmask[x]) {
                const int umax = std::min({ x, width - 1 - x, d->mdis });
                for (int u = -umax; u <= umax; u++) {
                    const int u2 = u * 2;
                    const bool s1Flag = (u >= 0 && x >= u2) || (u <= 0 && x < width + u2);
                    const bool s2Flag = (u <= 0 && x >= -u2) || (u >= 0 && x < width + u2);
                    Vec4f s0 = zero_4f(), s1, s2;

                    for (int k = -(d->nrad); k <= d->nrad; k++) {
                        s0 += abs(Vec4f().load_a(src3p + x + u + k) - Vec4f().load_a(src1p + x - u + k)) +
                            abs(Vec4f().load_a(src1p + x + u + k) - Vec4f().load_a(src1n + x - u + k)) +
                            abs(Vec4f().load_a(src1n + x + u + k) - Vec4f().load_a(src3n + x - u + k));
                    }

                    if (s1Flag) {
                        s1 = zero_4f();
                        for (int k = -(d->nrad); k <= d->nrad; k++) {
                            s1 += abs(Vec4f().load_a(src3p + x + k) - Vec4f().load_a(src1p + x - u2 + k)) +
                                abs(Vec4f().load_a(src1p + x + k) - Vec4f().load_a(src1n + x - u2 + k)) +
                                abs(Vec4f().load_a(src1n + x + k) - Vec4f().load_a(src3n + x - u2 + k));
                        }
                    }

                    if (s2Flag) {
                        s2 = zero_4f();
                        for (int k = -(d->nrad); k <= d->nrad; k++) {
                            s2 += abs(Vec4f().load_a(src3p + x + u2 + k) - Vec4f().load_a(src1p + x + k)) +
                                abs(Vec4f().load_a(src1p + x + u2 + k) - Vec4f().load_a(src1n + x + k)) +
                                abs(Vec4f().load_a(src1n + x + u2 + k) - Vec4f().load_a(src3n + x + k));
                        }
                    }

                    s1 = s1Flag ? s1 : (s2Flag ? s2 : s0);
                    s2 = s2Flag ? s2 : (s1Flag ? s1 : s0);

                    const Vec4f ip = (Vec4f().load_a(src1p + x + u) + Vec4f().load_a(src1n + x - u)) * 0.5f; // should use cubic if ucubic=true
                    const Vec4f v = abs(Vec4f().load_a(src1p + x) - ip) + abs(Vec4f().load_a(src1n + x) - ip);
                    const Vec4f result = mul_add(d->alpha, s0 + s1 + s2, mul_add(d->remainingWeight, v, d->beta * std::abs(u)));
                    result.store(ccosts + (d->tpitch * x + d->mdis + u) * d->vectorSize);
                }
            }
        }
    } else {
        for (int x = 0; x < width; x++) {
            if (!bmask || bmask[x]) {
                const int umax = std::min({ x, width - 1 - x, d->mdis });
                for (int u = -umax; u <= umax; u++) {
                    Vec4f s = zero_4f();

                    for (int k = -(d->nrad); k <= d->nrad; k++) {
                        s += abs(Vec4f().load_a(src3p + x + u + k) - Vec4f().load_a(src1p + x - u + k)) +
                            abs(Vec4f().load_a(src1p + x + u + k) - Vec4f().load_a(src1n + x - u + k)) +
                            abs(Vec4f().load_a(src1n + x + u + k) - Vec4f().load_a(src3n + x - u + k));
                    }

                    const Vec4f ip = (Vec4f().load_a(src1p + x + u) + Vec4f().load_a(src1n + x - u)) * 0.5f; // should use cubic if ucubic=true
                    const Vec4f v = abs(Vec4f().load_a(src1p + x) - ip) + abs(Vec4f().load_a(src1n + x) - ip);
                    const Vec4f result = mul_add(d->alpha, s, mul_add(d->remainingWeight, v, d->beta * std::abs(u)));
                    result.store(ccosts + (d->tpitch * x + d->mdis + u) * d->vectorSize);
                }
            }
        }
    }
}

template<typename pixel_t, typename vector_t>
void filter_sse4(const VSFrame* src, const VSFrame* scp, const VSFrame* mclip, VSFrame* mcp, VSFrame** pad, VSFrame* dst, void* _srcVector, uint8_t* _mskVector,
                 bool* VS_RESTRICT bmask, int* _pbackt, int* VS_RESTRICT fpath, int* _dmap, const int field_n, const EEDI3Data* VS_RESTRICT d,
                 const VSAPI* vsapi) noexcept {
    for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
        if (d->process[plane]) {
            const int srcWidth = vsapi->getFrameWidth(pad[plane], 0);
            const int dstWidth = vsapi->getFrameWidth(dst, plane);
            const int srcHeight = vsapi->getFrameHeight(pad[plane], 0);
            const int dstHeight = vsapi->getFrameHeight(dst, plane);
            const ptrdiff_t srcStride = vsapi->getStride(pad[plane], 0) / sizeof(pixel_t);
            const ptrdiff_t dstStride = vsapi->getStride(dst, plane) / sizeof(pixel_t);
            auto _srcp = reinterpret_cast<const pixel_t*>(vsapi->getReadPtr(pad[plane], 0));
            auto _dstp = reinterpret_cast<pixel_t*>(vsapi->getWritePtr(dst, plane));

            auto srcVector = reinterpret_cast<vector_t*>(_srcVector);

            auto _ccosts = std::make_unique<float[]>(dstWidth * d->tpitchVector);
            auto _pcosts = std::make_unique<float[]>(dstWidth * d->tpitchVector);
            auto ccosts = _ccosts.get();
            auto pcosts = _pcosts.get();

            copyPad<pixel_t>(src, pad[plane], plane, d->dh, 1 - field_n, vsapi);

            const uint8_t* maskp = nullptr;
            if (bmask) {
                maskp = vsapi->getReadPtr(mcp, plane);
                copyMask(mclip, mcp, plane, d->dh, field_n, vsapi);
            }

            vsh::bitblt(_dstp + dstStride * (1 - field_n),
                        vsapi->getStride(dst, plane) * 2,
                        _srcp + srcStride * (MARGIN_V + 1 - field_n) + MARGIN_H,
                        vsapi->getStride(pad[plane], 0) * 2,
                        dstWidth * sizeof(pixel_t),
                        dstHeight / 2);

            _srcp += srcStride * MARGIN_V;
            _dstp += dstStride * field_n;

            for (int y = field_n; y < dstHeight; y += 2 * d->vectorSize) {
                const int off = (y - field_n) / 2;

                prepareLines<pixel_t, vector_t>(_srcp + srcStride * (1 - field_n) + MARGIN_H, srcVector, dstWidth, (dstHeight + field_n) / 2, srcStride * 2,
                                                srcWidth, off + field_n, d->vectorSize);

                if (bmask) {
                    prepareMask(maskp, _mskVector, dstWidth, (dstHeight + field_n) / 2, vsapi->getStride(mcp, plane), off, d->vectorSize);

                    auto mskVector = reinterpret_cast<const int32_t*>(_mskVector);
                    const int minmdis = std::min(dstWidth, d->mdis);
                    int last = -666999;

                    for (int x = 0; x < minmdis; x++)
                        if (mskVector[x] != 0)
                            last = x + d->mdis;

                    for (int x = 0; x < dstWidth - minmdis; x++) {
                        if (mskVector[x + d->mdis] != 0)
                            last = x + d->mdis * 2;

                        bmask[x] = (x <= last);
                    }

                    for (int x = dstWidth - minmdis; x < dstWidth; x++)
                        bmask[x] = (x <= last);
                }

                calculateConnectionCosts<vector_t>(srcVector, bmask, ccosts, dstWidth, srcWidth, d);

                // calculate path costs
                Vec4f().load(ccosts + d->mdisVector).store(pcosts + d->mdisVector);
                for (int x = 1; x < dstWidth; x++) {
                    auto tT = ccosts + d->tpitchVector * x;
                    auto ppT = pcosts + d->tpitchVector * (x - 1);
                    auto pT = pcosts + d->tpitchVector * x;
                    auto piT = _pbackt + d->tpitchVector * (x - 1);

                    if (bmask && !bmask[x]) {
                        if (x == 1) {
                            const int umax = std::min({ x, dstWidth - 1 - x, d->mdis });
                            const int p = (d->mdis - umax) * d->vectorSize;
                            memcpy(pT + p, tT + p, (umax * 2 + 1) * d->vectorSize * sizeof(*pT));

                            memset(piT, 0, d->tpitchVector * sizeof(*piT));
                        } else {
                            memcpy(pT, ppT, d->tpitchVector * sizeof(*pT));
                            memcpy(piT, piT - d->tpitchVector, d->tpitchVector * sizeof(*piT));

                            const int pumax = std::min(x - 1, dstWidth - x);
                            if (pumax < d->mdis) {
                                Vec4i(1 - pumax).store_nt(piT + (d->mdis - pumax) * d->vectorSize);
                                Vec4i(pumax - 1).store_nt(piT + (d->mdis + pumax) * d->vectorSize);
                            }
                        }
                    } else {
                        const int umax = std::min({ x, dstWidth - 1 - x, d->mdis });
                        const int umax2 = std::min({ x - 1, dstWidth - x, d->mdis });
                        for (int u = -umax; u <= umax; u++) {
                            Vec4i idx = zero_si128();
                            Vec4f bval = FLT_MAX;

                            for (int v = std::max(-umax2, u - 1); v <= std::min(umax2, u + 1); v++) {
                                const Vec4f z = Vec4f().load(ppT + (d->mdis + v) * d->vectorSize) + d->gamma * std::abs(u - v);
                                const Vec4f ccost = min(z, FLT_MAX * 0.9f);
                                idx = select(Vec4ib(ccost < bval), v, idx);
                                bval = min(ccost, bval);
                            }

                            const int mu = (d->mdis + u) * d->vectorSize;
                            const Vec4f z = bval + Vec4f().load(tT + mu);
                            min(z, FLT_MAX * 0.9f).store(pT + mu);
                            idx.store_nt(piT + mu);
                        }
                    }
                }

                for (int vs = 0; vs < d->vectorSize; vs++) {
                    const int realY = field_n + (off + vs) * 2;
                    if (realY >= dstHeight)
                        break;

                    auto srcp = _srcp + srcStride * realY + MARGIN_H;
                    auto dstp = _dstp + dstStride * (off + vs) * 2;
                    auto dmap = _dmap + dstWidth * (off + vs);

                    auto src3p = srcp - srcStride * 3;
                    auto src1p = srcp - srcStride;
                    auto src1n = srcp + srcStride;
                    auto src3n = srcp + srcStride * 3;

                    auto pbackt = _pbackt + vs;

                    // backtrack
                    fpath[dstWidth - 1] = 0;
                    for (int x = dstWidth - 2; x >= 0; x--)
                        fpath[x] = pbackt[(d->tpitch * x + d->mdis + fpath[x + 1]) * d->vectorSize];

                    interpolate<pixel_t>(src3p, src1p, src1n, src3n, dstp, bmask, fpath, dmap, dstWidth, d->ucubic, d->peak);
                }
            }

            _srcp += srcStride * field_n;

            if (d->vcheck > 0) {
                const pixel_t* scpp = nullptr;
                if (d->sclip)
                    scpp = reinterpret_cast<const pixel_t*>(vsapi->getReadPtr(scp, plane)) + dstStride * field_n;

                vCheck<pixel_t>(_srcp, scpp, _dstp, _dmap, fpath, field_n, dstWidth, srcHeight, srcStride, dstStride, d);
            }
        }
    }
}

template void filter_sse4<uint8_t, int>(const VSFrame* src, const VSFrame* scp, const VSFrame* mclip, VSFrame* mcp, VSFrame** pad, VSFrame* dst,
                                        void* _srcVector, uint8_t* _mskVector, bool* VS_RESTRICT bmask, int* _pbackt, int* VS_RESTRICT fpath, int* _dmap,
                                        const int field_n, const EEDI3Data* VS_RESTRICT d, const VSAPI* vsapi) noexcept;

template void filter_sse4<uint16_t, int>(const VSFrame* src, const VSFrame* scp, const VSFrame* mclip, VSFrame* mcp, VSFrame** pad, VSFrame* dst,
                                         void* _srcVector, uint8_t* _mskVector, bool* VS_RESTRICT bmask, int* _pbackt, int* VS_RESTRICT fpath, int* _dmap,
                                         const int field_n, const EEDI3Data* VS_RESTRICT d, const VSAPI* vsapi) noexcept;

template void filter_sse4<float, float>(const VSFrame* src, const VSFrame* scp, const VSFrame* mclip, VSFrame* mcp, VSFrame** pad, VSFrame* dst,
                                        void* _srcVector, uint8_t* _mskVector, bool* VS_RESTRICT bmask, int* _pbackt, int* VS_RESTRICT fpath, int* _dmap,
                                        const int field_n, const EEDI3Data* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
#endif
