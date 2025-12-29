/*
**   Another VapourSynth port by HolyWu
**
**   eedi3 (enhanced edge directed interpolation 3). Works by finding the
**   best non-decreasing (non-crossing) warping between two lines according to
**   a cost functional. Doesn't really have anything to do with eedi2 aside
**   from doing edge-directed interpolation (they use different techniques).
**
**   Copyright (C) 2010 Kevin Stone - some part by Laurent de Soras, 2013
**
**   This program is free software; you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation; either version 2 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program; if not, write to the Free Software
**   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <string>
#include <vector>

#include "EEDI3.h"

using namespace std::literals;

#ifdef EEDI3_X86
template<typename pixel_t, typename vector_t>
extern void filter_sse4(const VSFrame* src, const VSFrame* scp, const VSFrame* mclip, VSFrame* mcp, VSFrame** pad, VSFrame* dst, void* srcVector,
                        uint8_t* mskVector, bool* VS_RESTRICT bmask, int* pbackt, int* VS_RESTRICT fpath, int* dmap, const int field_n,
                        const EEDI3Data* VS_RESTRICT d, const VSAPI* vsapi) noexcept;

template<typename pixel_t, typename vector_t>
extern void filter_avx2(const VSFrame* src, const VSFrame* scp, const VSFrame* mclip, VSFrame* mcp, VSFrame** pad, VSFrame* dst, void* srcVector,
                        uint8_t* mskVector, bool* VS_RESTRICT bmask, int* pbackt, int* VS_RESTRICT fpath, int* dmap, const int field_n,
                        const EEDI3Data* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
#endif

template<typename pixel_t>
static inline void calculateConnectionCosts(const pixel_t* src3p, const pixel_t* src1p, const pixel_t* src1n, const pixel_t* src3n, const bool* bmask,
                                            float* VS_RESTRICT ccosts, const int width, const EEDI3Data* VS_RESTRICT d) noexcept {
    if (d->cost3) {
        for (int x = 0; x < width; x++) {
            if (!bmask || bmask[x]) {
                const int umax = std::min({ x, width - 1 - x, d->mdis });
                for (int u = -umax; u <= umax; u++) {
                    const int u2 = u * 2;
                    int s0 = 0, s1 = -1, s2 = -1;

                    for (int k = -(d->nrad); k <= d->nrad; k++) {
                        s0 += std::abs(src3p[x + u + k] - src1p[x - u + k]) +
                            std::abs(src1p[x + u + k] - src1n[x - u + k]) +
                            std::abs(src1n[x + u + k] - src3n[x - u + k]);
                    }

                    if ((u >= 0 && x >= u2) || (u <= 0 && x < width + u2)) {
                        s1 = 0;
                        for (int k = -(d->nrad); k <= d->nrad; k++) {
                            s1 += std::abs(src3p[x + k] - src1p[x - u2 + k]) +
                                std::abs(src1p[x + k] - src1n[x - u2 + k]) +
                                std::abs(src1n[x + k] - src3n[x - u2 + k]);
                        }
                    }

                    if ((u <= 0 && x >= -u2) || (u >= 0 && x < width + u2)) {
                        s2 = 0;
                        for (int k = -(d->nrad); k <= d->nrad; k++) {
                            s2 += std::abs(src3p[x + u2 + k] - src1p[x + k]) +
                                std::abs(src1p[x + u2 + k] - src1n[x + k]) +
                                std::abs(src1n[x + u2 + k] - src3n[x + k]);
                        }
                    }

                    s1 = (s1 >= 0) ? s1 : (s2 >= 0 ? s2 : s0);
                    s2 = (s2 >= 0) ? s2 : (s1 >= 0 ? s1 : s0);

                    const int ip = (src1p[x + u] + src1n[x - u] + 1) / 2; // should use cubic if ucubic=true
                    const int v = std::abs(src1p[x] - ip) + std::abs(src1n[x] - ip);
                    ccosts[d->tpitch * x + d->mdis + u] = d->alpha * (s0 + s1 + s2) + d->beta * std::abs(u) + d->remainingWeight * v;
                }
            }
        }
    } else {
        for (int x = 0; x < width; x++) {
            if (!bmask || bmask[x]) {
                const int umax = std::min({ x, width - 1 - x, d->mdis });
                for (int u = -umax; u <= umax; u++) {
                    int s = 0;

                    for (int k = -(d->nrad); k <= d->nrad; k++) {
                        s += std::abs(src3p[x + u + k] - src1p[x - u + k]) +
                            std::abs(src1p[x + u + k] - src1n[x - u + k]) +
                            std::abs(src1n[x + u + k] - src3n[x - u + k]);
                    }

                    const int ip = (src1p[x + u] + src1n[x - u] + 1) / 2; // should use cubic if ucubic=true
                    const int v = std::abs(src1p[x] - ip) + std::abs(src1n[x] - ip);
                    ccosts[d->tpitch * x + d->mdis + u] = d->alpha * s + d->beta * std::abs(u) + d->remainingWeight * v;
                }
            }
        }
    }
}

template<>
inline void calculateConnectionCosts(const float* src3p, const float* src1p, const float* src1n, const float* src3n, const bool* bmask,
                                     float* VS_RESTRICT ccosts, const int width, const EEDI3Data* VS_RESTRICT d) noexcept {
    if (d->cost3) {
        for (int x = 0; x < width; x++) {
            if (!bmask || bmask[x]) {
                const int umax = std::min({ x, width - 1 - x, d->mdis });
                for (int u = -umax; u <= umax; u++) {
                    const int u2 = u * 2;
                    float s0 = 0.0f, s1 = -FLT_MAX, s2 = -FLT_MAX;

                    for (int k = -(d->nrad); k <= d->nrad; k++) {
                        s0 += std::abs(src3p[x + u + k] - src1p[x - u + k]) +
                            std::abs(src1p[x + u + k] - src1n[x - u + k]) +
                            std::abs(src1n[x + u + k] - src3n[x - u + k]);
                    }

                    if ((u >= 0 && x >= u2) || (u <= 0 && x < width + u2)) {
                        s1 = 0.0f;
                        for (int k = -(d->nrad); k <= d->nrad; k++) {
                            s1 += std::abs(src3p[x + k] - src1p[x - u2 + k]) +
                                std::abs(src1p[x + k] - src1n[x - u2 + k]) +
                                std::abs(src1n[x + k] - src3n[x - u2 + k]);
                        }
                    }

                    if ((u <= 0 && x >= -u2) || (u >= 0 && x < width + u2)) {
                        s2 = 0.0f;
                        for (int k = -(d->nrad); k <= d->nrad; k++) {
                            s2 += std::abs(src3p[x + u2 + k] - src1p[x + k]) +
                                std::abs(src1p[x + u2 + k] - src1n[x + k]) +
                                std::abs(src1n[x + u2 + k] - src3n[x + k]);
                        }
                    }

                    s1 = (s1 > -FLT_MAX) ? s1 : (s2 > -FLT_MAX ? s2 : s0);
                    s2 = (s2 > -FLT_MAX) ? s2 : (s1 > -FLT_MAX ? s1 : s0);

                    const float ip = (src1p[x + u] + src1n[x - u]) / 2.0f; // should use cubic if ucubic=true
                    const float v = std::abs(src1p[x] - ip) + std::abs(src1n[x] - ip);
                    ccosts[d->tpitch * x + d->mdis + u] = d->alpha * (s0 + s1 + s2) + d->beta * std::abs(u) + d->remainingWeight * v;
                }
            }
        }
    } else {
        for (int x = 0; x < width; x++) {
            if (!bmask || bmask[x]) {
                const int umax = std::min({ x, width - 1 - x, d->mdis });
                for (int u = -umax; u <= umax; u++) {
                    float s = 0.0f;

                    for (int k = -(d->nrad); k <= d->nrad; k++) {
                        s += std::abs(src3p[x + u + k] - src1p[x - u + k]) +
                            std::abs(src1p[x + u + k] - src1n[x - u + k]) +
                            std::abs(src1n[x + u + k] - src3n[x - u + k]);
                    }

                    const float ip = (src1p[x + u] + src1n[x - u]) / 2.0f; // should use cubic if ucubic=true
                    const float v = std::abs(src1p[x] - ip) + std::abs(src1n[x] - ip);
                    ccosts[d->tpitch * x + d->mdis + u] = d->alpha * s + d->beta * std::abs(u) + d->remainingWeight * v;
                }
            }
        }
    }
}

template<typename pixel_t>
static void filter_c(const VSFrame* src, const VSFrame* scp, const VSFrame* mclip, VSFrame* mcp, VSFrame** pad, VSFrame* dst, [[maybe_unused]] void* srcVector,
                     [[maybe_unused]] uint8_t* mskVector, bool* VS_RESTRICT bmask, int* pbackt, int* VS_RESTRICT fpath, int* _dmap, const int field_n,
                     const EEDI3Data* VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
        if (d->process[plane]) {
            [[maybe_unused]] const int srcWidth = vsapi->getFrameWidth(pad[plane], 0);
            const int dstWidth = vsapi->getFrameWidth(dst, plane);
            const int srcHeight = vsapi->getFrameHeight(pad[plane], 0);
            const int dstHeight = vsapi->getFrameHeight(dst, plane);
            const ptrdiff_t srcStride = vsapi->getStride(pad[plane], 0) / sizeof(pixel_t);
            const ptrdiff_t dstStride = vsapi->getStride(dst, plane) / sizeof(pixel_t);
            auto _srcp = reinterpret_cast<const pixel_t*>(vsapi->getReadPtr(pad[plane], 0));
            auto _dstp = reinterpret_cast<pixel_t*>(vsapi->getWritePtr(dst, plane));

            auto _ccosts = std::make_unique<float[]>(dstWidth * d->tpitch);
            auto _pcosts = std::make_unique<float[]>(dstWidth * d->tpitch);
            auto ccosts = _ccosts.get();
            auto pcosts = _pcosts.get();

            copyPad<pixel_t>(src, pad[plane], plane, d->dh, 1 - field_n, vsapi);

            const uint8_t* _maskp = nullptr;
            if (bmask) {
                _maskp = vsapi->getReadPtr(mcp, plane);
                copyMask(mclip, mcp, plane, d->dh, field_n, vsapi);
            }

            vsh::bitblt(_dstp + dstStride * (1 - field_n),
                        vsapi->getStride(dst, plane) * 2,
                        _srcp + srcStride * (MARGIN_V + 1 - field_n) + MARGIN_H,
                        vsapi->getStride(pad[plane], 0) * 2,
                        dstWidth * sizeof(pixel_t),
                        dstHeight / 2);

            _srcp += srcStride * (MARGIN_V + field_n);
            _dstp += dstStride * field_n;

            for (int y = MARGIN_V + field_n; y < srcHeight - MARGIN_V; y += 2) {
                const int off = (y - MARGIN_V - field_n) / 2;
                auto srcp = _srcp + srcStride * off * 2 + MARGIN_H;
                auto dstp = _dstp + dstStride * off * 2;
                auto dmap = _dmap + dstWidth * off;

                auto src3p = srcp - srcStride * 3;
                auto src1p = srcp - srcStride;
                auto src1n = srcp + srcStride;
                auto src3n = srcp + srcStride * 3;

                if (bmask) {
                    auto maskp = _maskp + vsapi->getStride(mcp, plane) * off;
                    const int minmdis = std::min(dstWidth, d->mdis);
                    int last = -666999;

                    for (int x = 0; x < minmdis; x++)
                        if (maskp[x] != 0)
                            last = x + d->mdis;

                    for (int x = 0; x < dstWidth - minmdis; x++) {
                        if (maskp[x + d->mdis] != 0)
                            last = x + d->mdis * 2;

                        bmask[x] = (x <= last);
                    }

                    for (int x = dstWidth - minmdis; x < dstWidth; x++)
                        bmask[x] = (x <= last);
                }

                calculateConnectionCosts<pixel_t>(src3p, src1p, src1n, src3n, bmask, ccosts, dstWidth, d);

                // calculate path costs
                pcosts[d->mdis] = ccosts[d->mdis];
                for (int x = 1; x < dstWidth; x++) {
                    auto tT = ccosts + d->tpitch * x;
                    auto ppT = pcosts + d->tpitch * (x - 1);
                    auto pT = pcosts + d->tpitch * x;
                    auto piT = pbackt + d->tpitch * (x - 1);

                    if (bmask && !bmask[x]) {
                        if (x == 1) {
                            const int umax = std::min({ x, dstWidth - 1 - x, d->mdis });
                            for (int u = -umax; u <= umax; u++)
                                pT[d->mdis + u] = tT[d->mdis + u];

                            memset(piT, 0, d->tpitch * sizeof(*piT));
                        } else {
                            memcpy(pT, ppT, d->tpitch * sizeof(*ppT));
                            memcpy(piT, piT - d->tpitch, d->tpitch * sizeof(*piT));

                            const int pumax = std::min(x - 1, dstWidth - x);
                            if (pumax < d->mdis) {
                                piT[d->mdis - pumax] = 1 - pumax;
                                piT[d->mdis + pumax] = pumax - 1;
                            }
                        }
                    } else {
                        const int umax = std::min({ x, dstWidth - 1 - x, d->mdis });
                        const int umax2 = std::min({ x - 1, dstWidth - x, d->mdis });
                        for (int u = -umax; u <= umax; u++) {
                            int idx = 0;
                            float bval = FLT_MAX;

                            for (int v = std::max(-umax2, u - 1); v <= std::min(umax2, u + 1); v++) {
                                const double z = ppT[d->mdis + v] + d->gamma * std::abs(u - v);
                                const float ccost = static_cast<float>(std::min(z, FLT_MAX * 0.9));
                                if (ccost < bval) {
                                    bval = ccost;
                                    idx = v;
                                }
                            }

                            const double z = bval + tT[d->mdis + u];
                            pT[d->mdis + u] = static_cast<float>(std::min(z, FLT_MAX * 0.9));
                            piT[d->mdis + u] = idx;
                        }
                    }
                }

                // backtrack
                fpath[dstWidth - 1] = 0;
                for (int x = dstWidth - 2; x >= 0; x--)
                    fpath[x] = pbackt[d->tpitch * x + d->mdis + fpath[x + 1]];

                interpolate<pixel_t>(src3p, src1p, src1n, src3n, dstp, bmask, fpath, dmap, dstWidth, d->ucubic, d->peak);
            }

            if (d->vcheck > 0) {
                const pixel_t* scpp = nullptr;
                if (d->sclip)
                    scpp = reinterpret_cast<const pixel_t*>(vsapi->getReadPtr(scp, plane)) + dstStride * field_n;

                vCheck<pixel_t>(_srcp, scpp, _dstp, _dmap, fpath, field_n, dstWidth, srcHeight, srcStride, dstStride, d);
            }
        }
    }
}

static const VSFrame* VS_CC eedi3GetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData, VSFrameContext* frameCtx,
                                          VSCore* core, const VSAPI* vsapi) {
    auto d = static_cast<EEDI3Data*>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);

        if (d->vcheck > 0 && d->sclip)
            vsapi->requestFrameFilter(n, d->sclip, frameCtx);

        if (d->mclip)
            vsapi->requestFrameFilter(d->field > 1 ? n / 2 : n, d->mclip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        float* srcVector;
        uint8_t* mskVector;
        bool* bmask;
        int* pbackt;
        int* fpath;
        int* dmap;

        try {
            auto threadID = std::this_thread::get_id();
            std::lock_guard<std::mutex> lock(d->mutex);

            if (!d->srcVector.count(threadID)) {
                if (d->vectorSize > 1) {
                    auto _srcVector = vsh::vsh_aligned_malloc<float>((d->vi.width + MARGIN_H * 2) * 4 * d->vectorSize * sizeof(float), d->alignment);
                    if (!_srcVector)
                        throw "malloc failure (srcVector)"s;
                    d->srcVector.emplace(threadID, aligned_float{ _srcVector, &vsh::vsh_aligned_free });

                    if (d->mclip) {
                        auto _mskVector = new (std::nothrow) uint8_t[d->vi.width * d->vectorSize];
                        if (!_mskVector)
                            throw "malloc failure (mskVector)"s;
                        d->mskVector.emplace(threadID, _mskVector);
                    } else {
                        d->mskVector.emplace(threadID, nullptr);
                    }
                } else {
                    d->srcVector.emplace(threadID, aligned_float{ nullptr, &vsh::vsh_aligned_free });
                    d->mskVector.emplace(threadID, nullptr);
                }

                if (d->mclip) {
                    auto _bmask = new (std::nothrow) bool[d->vi.width];
                    if (!_bmask)
                        throw "malloc failure (bmask)"s;
                    d->bmask.emplace(threadID, _bmask);
                } else {
                    d->bmask.emplace(threadID, nullptr);
                }

                auto _pbackt = vsh::vsh_aligned_malloc<int>(d->vi.width * d->tpitchVector * sizeof(int), d->alignment);
                if (!_pbackt)
                    throw "malloc failure (pbackt)"s;
                d->pbackt.emplace(threadID, aligned_int{ _pbackt, &vsh::vsh_aligned_free });

                auto _fpath = new (std::nothrow) int[d->vi.width];
                if (!_fpath)
                    throw "malloc failure (fpath)"s;
                d->fpath.emplace(threadID, _fpath);

                auto _dmap = new (std::nothrow) int[d->vi.width * d->vi.height];
                if (!_dmap)
                    throw "malloc failure (dmap)"s;
                d->dmap.emplace(threadID, _dmap);
            }

            srcVector = d->srcVector.at(threadID).get();
            mskVector = d->mskVector.at(threadID).get();
            bmask = d->bmask.at(threadID).get();
            pbackt = d->pbackt.at(threadID).get();
            fpath = d->fpath.at(threadID).get();
            dmap = d->dmap.at(threadID).get();
        } catch (const std::string& error) {
            vsapi->setFilterError(("EEDI3: " + error).c_str(), frameCtx);
            return nullptr;
        }

        const VSFrame* src = vsapi->getFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);
        VSFrame* dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);

        const VSFrame* scp = nullptr;
        if (d->vcheck > 0 && d->sclip)
            scp = vsapi->getFrameFilter(n, d->sclip, frameCtx);

        const VSFrame* mclip = nullptr;
        VSFrame* mcp = nullptr;
        if (d->mclip) {
            mclip = vsapi->getFrameFilter(d->field > 1 ? n / 2 : n, d->mclip, frameCtx);
            mcp = vsapi->newVideoFrame(vsapi->getVideoFrameFormat(mclip), d->vi.width, d->vi.height / 2, nullptr, core);
        }

        VSFrame* pad[3] = {};
        for (int plane = 0; plane < d->vi.format.numPlanes; plane++)
            if (d->process[plane])
                pad[plane] = vsapi->newVideoFrame(&d->padFormat, vsapi->getFrameWidth(dst, plane) + MARGIN_H * 2,
                                                  vsapi->getFrameHeight(dst, plane) + MARGIN_V * 2, nullptr, core);

        int field = d->field;
        if (field > 1)
            field -= 2;

        int err;
        const int fieldBased = vsapi->mapGetIntSaturated(vsapi->getFramePropertiesRO(src), "_FieldBased", 0, &err);
        if (fieldBased == 1)
            field = 0;
        else if (fieldBased == 2)
            field = 1;

        int field_n;
        if (d->field > 1) {
            if (n & 1)
                field_n = (field == 0);
            else
                field_n = (field == 1);
        } else {
            field_n = field;
        }

        d->filter(src, scp, mclip, mcp, pad, dst, srcVector, mskVector, bmask, pbackt, fpath, dmap, field_n, d, vsapi);

        VSMap* props = vsapi->getFramePropertiesRW(dst);
        vsapi->mapSetInt(props, "_FieldBased", 0, maReplace);

        if (d->field > 1) {
            int errNum, errDen;
            int64_t durationNum = vsapi->mapGetInt(props, "_DurationNum", 0, &errNum);
            int64_t durationDen = vsapi->mapGetInt(props, "_DurationDen", 0, &errDen);
            if (!errNum && !errDen) {
                vsh::muldivRational(&durationNum, &durationDen, 1, 2);
                vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
                vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
            }
        }

        vsapi->freeFrame(src);
        vsapi->freeFrame(scp);
        vsapi->freeFrame(mclip);
        vsapi->freeFrame(mcp);
        for (int plane = 0; plane < d->vi.format.numPlanes; plane++)
            vsapi->freeFrame(pad[plane]);

        return dst;
    }

    return nullptr;
}

static void VS_CC eedi3Free(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d = static_cast<EEDI3Data*>(instanceData);
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->sclip);
    vsapi->freeNode(d->mclip);
    delete d;
}

static void VS_CC eedi3Create(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d = std::make_unique<EEDI3Data>();
    int err;

    try {
        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->sclip = vsapi->mapGetNode(in, "sclip", 0, &err);
        d->mclip = vsapi->mapGetNode(in, "mclip", 0, &err);
        d->vi = *vsapi->getVideoInfo(d->node);

        if (!vsh::isConstantVideoFormat(&d->vi) ||
            (d->vi.format.sampleType == stInteger && d->vi.format.bitsPerSample > 16) ||
            (d->vi.format.sampleType == stFloat && d->vi.format.bitsPerSample != 32))
            throw "only constant format 8-16 bit integer and 32 bit float input supported"s;

        vsapi->queryVideoFormat(&d->padFormat, cfGray, d->vi.format.sampleType, d->vi.format.bitsPerSample, 0, 0, core);

        d->field = vsapi->mapGetIntSaturated(in, "field", 0, nullptr);

        d->dh = !!vsapi->mapGetInt(in, "dh", 0, &err);

        const int m = vsapi->mapNumElements(in, "planes");

        for (int i = 0; i < 3; i++)
            d->process[i] = (m <= 0);

        for (int i = 0; i < m; i++) {
            const int n = vsapi->mapGetIntSaturated(in, "planes", i, nullptr);

            if (n < 0 || n >= d->vi.format.numPlanes)
                throw "plane index out of range"s;

            if (d->process[n])
                throw "plane specified twice"s;

            d->process[n] = true;
        }

        d->alpha = vsapi->mapGetFloatSaturated(in, "alpha", 0, &err);
        if (err)
            d->alpha = 0.2f;

        d->beta = vsapi->mapGetFloatSaturated(in, "beta", 0, &err);
        if (err)
            d->beta = 0.25f;

        d->gamma = vsapi->mapGetFloatSaturated(in, "gamma", 0, &err);
        if (err)
            d->gamma = 20.0f;

        d->nrad = vsapi->mapGetIntSaturated(in, "nrad", 0, &err);
        if (err)
            d->nrad = 2;

        d->mdis = vsapi->mapGetIntSaturated(in, "mdis", 0, &err);
        if (err)
            d->mdis = 20;

        d->ucubic = !!vsapi->mapGetInt(in, "ucubic", 0, &err);
        if (err)
            d->ucubic = true;

        d->cost3 = !!vsapi->mapGetInt(in, "cost3", 0, &err);
        if (err)
            d->cost3 = true;

        d->vcheck = vsapi->mapGetIntSaturated(in, "vcheck", 0, &err);
        if (err)
            d->vcheck = 2;

        float vthresh0 = vsapi->mapGetFloatSaturated(in, "vthresh0", 0, &err);
        if (err)
            vthresh0 = 32.0f;

        float vthresh1 = vsapi->mapGetFloatSaturated(in, "vthresh1", 0, &err);
        if (err)
            vthresh1 = 64.0f;

        d->vthresh2 = vsapi->mapGetFloatSaturated(in, "vthresh2", 0, &err);
        if (err)
            d->vthresh2 = 4.0f;

        int opt = vsapi->mapGetIntSaturated(in, "opt", 0, &err);

        if (d->field < 0 || d->field > 3)
            throw "field must be 0, 1, 2, or 3"s;

        if (!d->dh) {
            auto frame = vsapi->getFrame(0, d->node, nullptr, 0);

            for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
                if (d->process[plane] && (vsapi->getFrameHeight(frame, plane) & 1)) {
                    vsapi->freeFrame(frame);
                    throw "plane's height must be mod 2 when dh=False"s;
                }
            }

            vsapi->freeFrame(frame);
        }

        if (d->dh && d->field > 1)
            throw "field must be 0 or 1 when dh=True"s;

        if (d->alpha < 0.0f || d->alpha > 1.0f)
            throw "alpha must be between 0.0 and 1.0 (inclusive)"s;

        if (d->beta < 0.0f || d->beta > 1.0f)
            throw "beta must be between 0.0 and 1.0 (inclusive)"s;

        if (d->alpha + d->beta > 1.0f)
            throw "alpha+beta must be between 0.0 and 1.0 (inclusive)"s;

        if (d->gamma < 0.0f)
            throw "gamma must be greater than or equal to 0.0"s;

        if (d->nrad < 0 || d->nrad > 3)
            throw "nrad must be between 0 and 3 (inclusive)"s;

        if (d->mdis < 1 || d->mdis > 40)
            throw "mdis must be between 1 and 40 (inclusive)"s;

        if (d->vcheck < 0 || d->vcheck > 3)
            throw "vcheck must be 0, 1, 2, or 3"s;

        if (d->vcheck > 0 && (vthresh0 <= 0.0f || vthresh1 <= 0.0f || d->vthresh2 <= 0.0f))
            throw "vthresh0, vthresh1 and vthresh2 must be greater than 0.0"s;

        if (d->mclip) {
            if (!vsh::isSameVideoInfo(vsapi->getVideoInfo(d->mclip), &d->vi))
                throw "mclip's format and dimensions don't match"s;

            if (vsapi->getVideoInfo(d->mclip)->numFrames != d->vi.numFrames)
                throw "mclip's number of frames doesn't match"s;

            if (d->vi.format.bitsPerSample != 8) {
                VSMap* args = vsapi->createMap();
                VSMap* ret;

                if (d->vi.format.colorFamily == cfYUV && d->vi.format.sampleType == stFloat) {
                    vsapi->mapConsumeNode(args, "clips", d->mclip, maReplace);
                    vsapi->mapSetData(args, "expr", "", -1, dtUtf8, maAppend);
                    vsapi->mapSetData(args, "expr", "x 0.5 -", -1, dtUtf8, maAppend);

                    ret = vsapi->invoke(vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "Expr", args);
                    if (vsapi->mapGetError(ret)) {
                        vsapi->mapSetError(out, vsapi->mapGetError(ret));
                        vsapi->freeNode(d->node);
                        vsapi->freeNode(d->sclip);
                        vsapi->freeMap(args);
                        vsapi->freeMap(ret);
                        return;
                    }

                    d->mclip = vsapi->mapGetNode(ret, "clip", 0, nullptr);
                    vsapi->clearMap(args);
                    vsapi->freeMap(ret);
                }

                vsapi->mapConsumeNode(args, "clip", d->mclip, maReplace);
                vsapi->mapSetInt(args,
                                 "format",
                                 vsapi->queryVideoFormatID(d->vi.format.colorFamily, stInteger, 8, d->vi.format.subSamplingW, d->vi.format.subSamplingH, core),
                                 maReplace);

                ret = vsapi->invoke(vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core), "Point", args);
                if (vsapi->mapGetError(ret)) {
                    vsapi->mapSetError(out, vsapi->mapGetError(ret));
                    vsapi->freeNode(d->node);
                    vsapi->freeNode(d->sclip);
                    vsapi->freeMap(args);
                    vsapi->freeMap(ret);
                    return;
                }

                d->mclip = vsapi->mapGetNode(ret, "clip", 0, nullptr);
                vsapi->freeMap(args);
                vsapi->freeMap(ret);
            }
        }

        if (opt < 0 || opt > 3)
            throw "opt must be 0, 1, 2, or 3"s;

        if (d->field > 1) {
            if (d->vi.numFrames > INT_MAX / 2)
                throw "resulting clip is too long"s;
            d->vi.numFrames *= 2;

            vsh::muldivRational(&d->vi.fpsNum, &d->vi.fpsDen, 2, 1);
        }

        if (d->dh)
            d->vi.height *= 2;

        d->remainingWeight = 1.0f - d->alpha - d->beta;

        if (d->cost3)
            d->alpha /= 3.0f;

        if (d->vcheck > 0 && d->sclip) {
            if (!vsh::isSameVideoInfo(vsapi->getVideoInfo(d->sclip), &d->vi))
                throw "sclip's format and dimensions don't match"s;

            if (vsapi->getVideoInfo(d->sclip)->numFrames != d->vi.numFrames)
                throw "sclip's number of frames doesn't match"s;
        }

        {
            d->vectorSize = 1;
            d->alignment = alignof(std::max_align_t);

#ifdef EEDI3_X86
            const int iset = instrset_detect();

            if (d->mclip && opt == 0 && iset >= 5)
                opt = 2;

            if ((opt == 0 && iset >= 8) || opt == 3) {
                d->vectorSize = 8;
                d->alignment = 32;
            } else if ((opt == 0 && iset >= 5) || opt == 2) {
                d->vectorSize = 4;
                d->alignment = 16;
            }
#endif

            if (d->vi.format.bytesPerSample == 1) {
                d->filter = filter_c<uint8_t>;

#ifdef EEDI3_X86
                if ((opt == 0 && iset >= 8) || opt == 3)
                    d->filter = filter_avx2<uint8_t, int>;
                else if ((opt == 0 && iset >= 5) || opt == 2)
                    d->filter = filter_sse4<uint8_t, int>;
#endif
            } else if (d->vi.format.bytesPerSample == 2) {
                d->filter = filter_c<uint16_t>;

#ifdef EEDI3_X86
                if ((opt == 0 && iset >= 8) || opt == 3)
                    d->filter = filter_avx2<uint16_t, int>;
                else if ((opt == 0 && iset >= 5) || opt == 2)
                    d->filter = filter_sse4<uint16_t, int>;
#endif
            } else {
                d->filter = filter_c<float>;

#ifdef EEDI3_X86
                if ((opt == 0 && iset >= 8) || opt == 3)
                    d->filter = filter_avx2<float, float>;
                else if ((opt == 0 && iset >= 5) || opt == 2)
                    d->filter = filter_sse4<float, float>;
#endif
            }
        }

        if (d->vi.format.sampleType == stInteger) {
            d->peak = (1 << d->vi.format.bitsPerSample) - 1;
            const int scale = 1 << (d->vi.format.bitsPerSample - 8);
            d->beta *= scale;
            d->gamma *= scale;
            vthresh0 *= scale;
            vthresh1 *= scale;
        } else {
            d->beta /= 255.0f;
            d->gamma /= 255.0f;
            vthresh0 /= 255.0f;
            vthresh1 /= 255.0f;
        }

        d->tpitch = d->mdis * 2 + 1;
        d->tpitchVector = d->tpitch * d->vectorSize;
        d->mdisVector = d->mdis * d->vectorSize;

        d->rcpVthresh0 = 1.0f / vthresh0;
        d->rcpVthresh1 = 1.0f / vthresh1;
        d->rcpVthresh2 = 1.0f / d->vthresh2;

        VSCoreInfo info;
        vsapi->getCoreInfo(core, &info);
        d->srcVector.reserve(info.numThreads);
        d->mskVector.reserve(info.numThreads);
        d->bmask.reserve(info.numThreads);
        d->pbackt.reserve(info.numThreads);
        d->fpath.reserve(info.numThreads);
        d->dmap.reserve(info.numThreads);
    } catch (const std::string& error) {
        vsapi->mapSetError(out, ("EEDI3: " + error).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->sclip);
        vsapi->freeNode(d->mclip);
        return;
    }

    std::vector<VSFilterDependency> deps = { { d->node, d->field > 1 ? rpGeneral : rpStrictSpatial } };
    if (d->sclip)
        deps.push_back({ d->sclip, rpStrictSpatial });
    if (d->mclip)
        deps.push_back({ d->mclip, d->field > 1 ? rpGeneral : rpStrictSpatial });

    vsapi->createVideoFilter(out, "EEDI3", &d->vi, eedi3GetFrame, eedi3Free, fmParallel, deps.data(), deps.size(), d.get(), core);
    d.release();
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.eedi3", "eedi3m", "Enhanced Edge Directed Interpolation 3", VS_MAKE_VERSION(7, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("EEDI3",
                             "clip:vnode;"
                             "field:int;"
                             "dh:int:opt;"
                             "planes:int[]:opt;"
                             "alpha:float:opt;"
                             "beta:float:opt;"
                             "gamma:float:opt;"
                             "nrad:int:opt;"
                             "mdis:int:opt;"
                             "hp:int:opt;"
                             "ucubic:int:opt;"
                             "cost3:int:opt;"
                             "vcheck:int:opt;"
                             "vthresh0:float:opt;"
                             "vthresh1:float:opt;"
                             "vthresh2:float:opt;"
                             "sclip:vnode:opt;"
                             "mclip:vnode:opt;"
                             "opt:int:opt;",
                             "clip:vnode;",
                             eedi3Create,
                             nullptr,
                             plugin);
}
