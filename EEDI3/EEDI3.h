#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#ifdef EEDI3_X86
#include "vectorclass/vectorclass.h"
#endif

static constexpr int MARGIN_H = 12; // Left and right margins for the virtual source frame
static constexpr int MARGIN_V = 4; // Top and bottom margins

using aligned_float = std::unique_ptr<float[], decltype(&vsh::vsh_aligned_free)>;
using aligned_int = std::unique_ptr<int[], decltype(&vsh::vsh_aligned_free)>;

struct EEDI3Data final {
    VSNode* node, * sclip, * mclip;
    VSVideoInfo vi;
    VSVideoFormat padFormat;
    int field, nrad, mdis, vcheck;
    bool dh, process[3], ucubic, cost3;
    float alpha, beta, gamma, vthresh2;
    float remainingWeight, rcpVthresh0, rcpVthresh1, rcpVthresh2;
    int vectorSize, alignment, tpitch, tpitchVector, mdisVector, peak;
    std::mutex mutex;
    std::unordered_map<std::thread::id, aligned_float> srcVector, ccosts, pcosts;
    std::unordered_map<std::thread::id, aligned_int> pbackt;
    std::unordered_map<std::thread::id, std::unique_ptr<bool[]>> bmask;
    std::unordered_map<std::thread::id, std::unique_ptr<int[]>> fpath, dmap;
    std::unordered_map<std::thread::id, std::unique_ptr<uint8_t[]>> mskVector;
    void (*filter)(const VSFrame* src, const VSFrame* scp, const VSFrame* mclip, VSFrame* mcp, VSFrame** pad, VSFrame* dst, void* srcVector, uint8_t* mskVector,
                   bool* VS_RESTRICT bmask, float* ccosts, float* pcosts, int* pbackt, int* VS_RESTRICT fpath, int* dmap, const int field_n,
                   const EEDI3Data* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
};

static void copyMask(const VSFrame* src, VSFrame* dst, const int plane, const bool dh, const int field_n, const VSAPI* vsapi) noexcept {
    const int off = dh ? 0 : field_n;
    const int mul = dh ? 1 : 2;

    vsh::bitblt(vsapi->getWritePtr(dst, plane),
                vsapi->getStride(dst, plane),
                vsapi->getReadPtr(src, plane) + vsapi->getStride(src, plane) * off,
                vsapi->getStride(src, plane) * mul,
                vsapi->getFrameWidth(src, plane),
                vsapi->getFrameHeight(dst, plane));
}

template<typename pixel_t>
static void copyPad(const VSFrame* src, VSFrame* dst, const int plane, const bool dh, const int off, const VSAPI* vsapi) noexcept {
    const int srcWidth = vsapi->getFrameWidth(src, plane);
    const int dstWidth = vsapi->getFrameWidth(dst, 0);
    const int srcHeight = vsapi->getFrameHeight(src, plane);
    const int dstHeight = vsapi->getFrameHeight(dst, 0);
    const ptrdiff_t srcStride = vsapi->getStride(src, plane) / sizeof(pixel_t);
    const ptrdiff_t dstStride = vsapi->getStride(dst, 0) / sizeof(pixel_t);
    const pixel_t* srcp = reinterpret_cast<const pixel_t*>(vsapi->getReadPtr(src, plane));
    pixel_t* VS_RESTRICT dstp = reinterpret_cast<pixel_t*>(vsapi->getWritePtr(dst, 0));

    if (!dh)
        vsh::bitblt(dstp + dstStride * (MARGIN_V + off) + MARGIN_H,
                    vsapi->getStride(dst, 0) * 2,
                    srcp + srcStride * off,
                    vsapi->getStride(src, plane) * 2,
                    srcWidth * sizeof(pixel_t),
                    srcHeight / 2);
    else
        vsh::bitblt(dstp + dstStride * (MARGIN_V + off) + MARGIN_H,
                    vsapi->getStride(dst, 0) * 2,
                    srcp,
                    vsapi->getStride(src, plane),
                    srcWidth * sizeof(pixel_t),
                    srcHeight);

    dstp += dstStride * (MARGIN_V + off);

    for (int y = MARGIN_V + off; y < dstHeight - MARGIN_V; y += 2) {
        for (int x = 0; x < MARGIN_H; x++)
            dstp[x] = dstp[MARGIN_H * 2 - x];

        for (int x = dstWidth - MARGIN_H, c = 2; x < dstWidth; x++, c += 2)
            dstp[x] = dstp[x - c];

        dstp += dstStride * 2;
    }

    dstp = reinterpret_cast<pixel_t*>(vsapi->getWritePtr(dst, 0));

    for (int y = off; y < MARGIN_V; y += 2)
        memcpy(dstp + dstStride * y, dstp + dstStride * (MARGIN_V * 2 - y), dstWidth * sizeof(pixel_t));

    for (int y = dstHeight - MARGIN_V + off, c = 2 + 2 * off; y < dstHeight; y += 2, c += 4)
        memcpy(dstp + dstStride * y, dstp + dstStride * (y - c), dstWidth * sizeof(pixel_t));
}

template<typename pixel_t>
static inline void interpolate(const pixel_t* src3p, const pixel_t* src1p, const pixel_t* src1n, const pixel_t* src3n, pixel_t* VS_RESTRICT dstp,
                               const bool* bmask, const int* fpath, int* VS_RESTRICT dmap, const int width, const bool ucubic, const int peak) noexcept {
    for (int x = 0; x < width; x++) {
        if (bmask && !bmask[x]) {
            dmap[x] = 0;

            if (ucubic)
                dstp[x] = std::clamp((9 * (src1p[x] + src1n[x]) - (src3p[x] + src3n[x]) + 8) / 16, 0, peak);
            else
                dstp[x] = (src1p[x] + src1n[x] + 1) / 2;
        } else {
            const int dir = fpath[x];
            const int dir3 = dir * 3;
            const int absDir3 = std::abs(dir3);

            dmap[x] = dir;

            if (ucubic && x >= absDir3 && x <= width - 1 - absDir3)
                dstp[x] = std::clamp((9 * (src1p[x + dir] + src1n[x - dir]) - (src3p[x + dir3] + src3n[x - dir3]) + 8) / 16, 0, peak);
            else
                dstp[x] = (src1p[x + dir] + src1n[x - dir] + 1) / 2;
        }
    }
}

template<>
inline void interpolate(const float* src3p, const float* src1p, const float* src1n, const float* src3n, float* VS_RESTRICT dstp, const bool* bmask,
                        const int* fpath, int* VS_RESTRICT dmap, const int width, const bool ucubic, [[maybe_unused]] const int peak) noexcept {
    for (int x = 0; x < width; x++) {
        if (bmask && !bmask[x]) {
            dmap[x] = 0;

            if (ucubic)
                dstp[x] = 0.5625f * (src1p[x] + src1n[x]) - 0.0625f * (src3p[x] + src3n[x]);
            else
                dstp[x] = (src1p[x] + src1n[x]) / 2.0f;
        } else {
            const int dir = fpath[x];
            const int dir3 = dir * 3;
            const int absDir3 = std::abs(dir3);

            dmap[x] = dir;

            if (ucubic && x >= absDir3 && x <= width - 1 - absDir3)
                dstp[x] = 0.5625f * (src1p[x + dir] + src1n[x - dir]) - 0.0625f * (src3p[x + dir3] + src3n[x - dir3]);
            else
                dstp[x] = (src1p[x + dir] + src1n[x - dir]) / 2.0f;
        }
    }
}

template<typename pixel_t, typename vector_t>
static inline void prepareLines(const pixel_t* srcp, vector_t* _dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                                const int srcY, const int vectorSize) noexcept {
    for (int y = srcY - 2; y < srcY + 2; y++) {
        vector_t* VS_RESTRICT dstp = _dstp;

        for (int y2 = y; y2 < y + vectorSize; y2++) {
            int realY = (y2 < 0) ? -1 - y2 : y2;
            if (realY >= height)
                realY = height * 2 - 1 - realY;
            realY = std::max(realY, 0);

            auto line = srcp + srcStride * realY;

            for (int x = 0; x < MARGIN_H; x++) {
                const int srcX = std::min(MARGIN_H - 1 - x, width - 1);
                dstp[x * vectorSize] = line[srcX];
            }

            for (int x = 0; x < width; x++)
                dstp[(MARGIN_H + x) * vectorSize] = line[x];

            for (int x = 0; x < MARGIN_H; x++) {
                const int srcX = std::max(width - 1 - x, 0);
                dstp[(MARGIN_H + width + x) * vectorSize] = line[srcX];
            }

            dstp++;
        }

        _dstp += dstStride * vectorSize;
    }
}

static inline void prepareMask(const uint8_t* srcp, uint8_t* VS_RESTRICT dstp, const int width, const int height, const ptrdiff_t stride, const int srcY,
                               const int vectorSize) noexcept {
    for (int y = srcY; y < srcY + vectorSize; y++) {
        int realY = y;
        if (realY >= height)
            realY = height * 2 - 1 - realY;
        realY = std::max(realY, 0);

        auto line = srcp + stride * realY;

        for (int x = 0; x < width; x++)
            dstp[x * vectorSize] = line[x];

        dstp++;
    }
}

template<typename pixel_t>
static void vCheck(const pixel_t* srcp, const pixel_t* scpp, pixel_t* VS_RESTRICT dstp, const int* dmap, void* _tline, const int field_n, const int width,
                   const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride, const EEDI3Data* VS_RESTRICT d) noexcept {
    for (int y = MARGIN_V + field_n; y < height - MARGIN_V; y += 2) {
        if (y >= 6 && y < height - 6) {
            auto dst3p = srcp - srcStride * 3 + MARGIN_H;
            auto dst2p = dstp - dstStride * 2;
            auto dst1p = dstp - dstStride;
            auto dst1n = dstp + dstStride;
            auto dst2n = dstp + dstStride * 2;
            auto dst3n = srcp + srcStride * 3 + MARGIN_H;
            pixel_t* VS_RESTRICT tline = reinterpret_cast<pixel_t*>(_tline);

            for (int x = 0; x < width; x++) {
                const int dirc = dmap[x];
                const pixel_t cint = scpp ? scpp[x] : std::clamp((9 * (dst1p[x] + dst1n[x]) - (dst3p[x] + dst3n[x]) + 8) / 16, 0, d->peak);

                if (dirc == 0) {
                    tline[x] = cint;
                    continue;
                }

                const int dirt = dmap[x - width];
                const int dirb = dmap[x + width];

                if (std::max(dirc * dirt, dirc * dirb) < 0 || (dirt == dirb && dirt == 0)) {
                    tline[x] = cint;
                    continue;
                }

                const int it = (dst2p[x + dirc] + dstp[x - dirc] + 1) / 2;
                const int ib = (dstp[x + dirc] + dst2n[x - dirc] + 1) / 2;
                const int vt = std::abs(dst2p[x + dirc] - dst1p[x + dirc]) + std::abs(dstp[x + dirc] - dst1p[x + dirc]);
                const int vb = std::abs(dst2n[x - dirc] - dst1n[x - dirc]) + std::abs(dstp[x - dirc] - dst1n[x - dirc]);
                const int vc = std::abs(dstp[x] - dst1p[x]) + std::abs(dstp[x] - dst1n[x]);

                const int d0 = std::abs(it - dst1p[x]);
                const int d1 = std::abs(ib - dst1n[x]);
                const int d2 = std::abs(vt - vc);
                const int d3 = std::abs(vb - vc);

                const int mdiff0 = (d->vcheck == 1) ? std::min(d0, d1) : (d->vcheck == 2 ? (d0 + d1 + 1) / 2 : std::max(d0, d1));
                const int mdiff1 = (d->vcheck == 1) ? std::min(d2, d3) : (d->vcheck == 2 ? (d2 + d3 + 1) / 2 : std::max(d2, d3));

                const float a0 = mdiff0 * d->rcpVthresh0;
                const float a1 = mdiff1 * d->rcpVthresh1;
                const float a2 = std::max((d->vthresh2 - std::abs(dirc)) * d->rcpVthresh2, 0.0f);
                const float a = std::min(std::max({ a0, a1, a2 }), 1.0f);

                tline[x] = static_cast<pixel_t>((1.0f - a) * dstp[x] + a * cint);
            }

            memcpy(dstp, tline, width * sizeof(pixel_t));
        }

        srcp += srcStride * 2;
        if (scpp)
            scpp += dstStride * 2;
        dstp += dstStride * 2;
        dmap += width;
    }
}

template<>
void vCheck(const float* srcp, const float* scpp, float* VS_RESTRICT dstp, const int* dmap, void* _tline, const int field_n, const int width, const int height,
            const ptrdiff_t srcStride, const ptrdiff_t dstStride, const EEDI3Data* VS_RESTRICT d) noexcept {
    for (int y = MARGIN_V + field_n; y < height - MARGIN_V; y += 2) {
        if (y >= 6 && y < height - 6) {
            auto dst3p = srcp - srcStride * 3 + MARGIN_H;
            auto dst2p = dstp - dstStride * 2;
            auto dst1p = dstp - dstStride;
            auto dst1n = dstp + dstStride;
            auto dst2n = dstp + dstStride * 2;
            auto dst3n = srcp + srcStride * 3 + MARGIN_H;
            float* VS_RESTRICT tline = reinterpret_cast<float*>(_tline);

            for (int x = 0; x < width; x++) {
                const int dirc = dmap[x];
                const float cint = scpp ? scpp[x] : 0.5625f * (dst1p[x] + dst1n[x]) - 0.0625f * (dst3p[x] + dst3n[x]);

                if (dirc == 0) {
                    tline[x] = cint;
                    continue;
                }

                const int dirt = dmap[x - width];
                const int dirb = dmap[x + width];

                if (std::max(dirc * dirt, dirc * dirb) < 0 || (dirt == dirb && dirt == 0)) {
                    tline[x] = cint;
                    continue;
                }

                const float it = (dst2p[x + dirc] + dstp[x - dirc]) / 2.0f;
                const float ib = (dstp[x + dirc] + dst2n[x - dirc]) / 2.0f;
                const float vt = std::abs(dst2p[x + dirc] - dst1p[x + dirc]) + std::abs(dstp[x + dirc] - dst1p[x + dirc]);
                const float vb = std::abs(dst2n[x - dirc] - dst1n[x - dirc]) + std::abs(dstp[x - dirc] - dst1n[x - dirc]);
                const float vc = std::abs(dstp[x] - dst1p[x]) + std::abs(dstp[x] - dst1n[x]);

                const float d0 = std::abs(it - dst1p[x]);
                const float d1 = std::abs(ib - dst1n[x]);
                const float d2 = std::abs(vt - vc);
                const float d3 = std::abs(vb - vc);

                const float mdiff0 = (d->vcheck == 1) ? std::min(d0, d1) : (d->vcheck == 2 ? (d0 + d1) / 2.0f : std::max(d0, d1));
                const float mdiff1 = (d->vcheck == 1) ? std::min(d2, d3) : (d->vcheck == 2 ? (d2 + d3) / 2.0f : std::max(d2, d3));

                const float a0 = mdiff0 * d->rcpVthresh0;
                const float a1 = mdiff1 * d->rcpVthresh1;
                const float a2 = std::max((d->vthresh2 - std::abs(dirc)) * d->rcpVthresh2, 0.0f);
                const float a = std::min(std::max({ a0, a1, a2 }), 1.0f);

                tline[x] = (1.0f - a) * dstp[x] + a * cint;
            }

            memcpy(dstp, tline, width * sizeof(float));
        }

        srcp += srcStride * 2;
        if (scpp)
            scpp += dstStride * 2;
        dstp += dstStride * 2;
        dmap += width;
    }
}
