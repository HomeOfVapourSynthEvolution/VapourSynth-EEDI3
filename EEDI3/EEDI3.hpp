#pragma once

#include "shared.hpp"

#ifdef VS_TARGET_CPU_X86
#define MAX_VECTOR_SIZE 512
#include "vectorclass/vectorclass.h"

static void copyMask(const VSFrameRef * src, VSFrameRef * dst, const int plane, const int field_n, const bool dh, const VSAPI * vsapi) noexcept {
    const int off = dh ? 0 : field_n;
    const int mul = dh ? 1 : 2;

    vs_bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane),
              vsapi->getReadPtr(src, plane) + vsapi->getStride(src, plane) * off, vsapi->getStride(src, plane) * mul,
              vsapi->getFrameWidth(src, plane), vsapi->getFrameHeight(dst, plane));
}

template<typename T1, typename T2>
static inline void prepareLines(const T1 * srcp, T2 * _dstp, const int width, const int height, const int srcStride, const int dstStride, const int srcY, const int vectorSize) noexcept {
    for (int y = srcY - 2; y < srcY + 2; y++) {
        T2 * VS_RESTRICT dstp = _dstp;

        for (int y2 = y; y2 < y + vectorSize; y2++) {
            int realY = (y2 < 0) ? -1 - y2 : y2;
            if (realY >= height)
                realY = height * 2 - 1 - realY;
            realY = std::max(realY, 0);

            const T1 * line = srcp + srcStride * realY;

            for (int x = 0; x < 12; x++) {
                const int srcX = std::min(12 - 1 - x, width - 1);
                dstp[x * vectorSize] = line[srcX];
            }

            for (int x = 0; x < width; x++)
                dstp[(12 + x) * vectorSize] = line[x];

            for (int x = 0; x < 12; x++) {
                const int srcX = std::max(width - 1 - x, 0);
                dstp[(width + 12 + x) * vectorSize] = line[srcX];
            }

            dstp++;
        }

        _dstp += dstStride * vectorSize;
    }
}

static inline void prepareMask(const uint8_t * srcp, uint8_t * VS_RESTRICT dstp, const int width, const int height, const int stride, const int srcY, const int vectorSize) noexcept {
    for (int y = srcY; y < srcY + vectorSize; y++) {
        int realY = y;
        if (realY >= height)
            realY = height * 2 - 1 - realY;
        realY = std::max(realY, 0);

        const uint8_t * line = srcp + stride * realY;

        for (int x = 0; x < width; x++)
            dstp[x * vectorSize] = line[x];

        dstp++;
    }
}
#endif

struct EEDI3Data {
    VSNodeRef * node, * sclip, * mclip;
    VSVideoInfo vi;
    int field, nrad, mdis, vcheck;
    bool dh, process[3], ucubic, cost3;
    float alpha, beta, gamma, vthresh2;
    int vectorSize, alignment, tpitch, tpitchVector, mdisVector, peak;
    float remainingWeight, rcpVthresh0, rcpVthresh1, rcpVthresh2;
    std::unordered_map<std::thread::id, int *> srcVector, pbackt, fpath, dmap, tline;
    std::unordered_map<std::thread::id, uint8_t *> mskVector;
    std::unordered_map<std::thread::id, bool *> bmask;
    std::unordered_map<std::thread::id, float *> ccosts, pcosts;
    void (*filter)(const VSFrameRef *, const VSFrameRef *, const VSFrameRef *, VSFrameRef *, VSFrameRef *, VSFrameRef **, const int, const EEDI3Data * const VS_RESTRICT, const VSAPI *);
};
