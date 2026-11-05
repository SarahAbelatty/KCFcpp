#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "kcf_types.hpp"

namespace ScalarImg
{

    inline unsigned char clamp8(float v)
    {
        int i = (int)std::floor(v + 0.5f);
        return (unsigned char)(i < 0 ? 0 : (i > 255 ? 255 : i));
    }

    // K1: bilinear resize, 8-bit interleaved, any channel count (11-bit fixed-point weights)
    inline void resizeLinear(const kcf::Image8 &s, int dw, int dh, kcf::Image8 &d)
    {
        const int SC = 2048, ch = s.ch;
        d.w = dw;
        d.h = dh;
        d.ch = ch;
        d.d.resize((size_t)dw * dh * ch);

        const double scx = (double)s.w / dw, scy = (double)s.h / dh;
        std::vector<int> xo(dw), xa0(dw), xa1(dw);
        for (int x = 0; x < dw; x++)
        {
            float fx = (float)((x + 0.5) * scx - 0.5);
            int sx = (int)std::floor(fx);
            fx -= sx;
            if (sx < 0)
            {
                fx = 0;
                sx = 0;
            }
            if (sx >= s.w - 1)
            {
                fx = 0;
                sx = s.w - 1;
            }
            xo[x] = sx;
            xa0[x] = (int)std::floor((1.f - fx) * SC + 0.5f);
            xa1[x] = SC - xa0[x];
        }
        std::vector<int> r0((size_t)dw * ch), r1((size_t)dw * ch);
        for (int y = 0; y < dh; y++)
        {
            float fy = (float)((y + 0.5) * scy - 0.5);
            int sy = (int)std::floor(fy);
            fy -= sy;
            if (sy < 0)
            {
                fy = 0;
                sy = 0;
            }
            if (sy >= s.h - 1)
            {
                fy = 0;
                sy = s.h - 1;
            }
            int sy1 = std::min(sy + 1, s.h - 1);
            int b0 = (int)std::floor((1.f - fy) * SC + 0.5f), b1 = SC - b0;

            for (int pass = 0; pass < 2; pass++)
            {
                const unsigned char *row = &s.d[(size_t)(pass ? sy1 : sy) * s.w * ch];
                int *out = pass ? r1.data() : r0.data();
                for (int x = 0; x < dw; x++)
                {
                    int sx = xo[x], sx1 = std::min(sx + 1, s.w - 1);
                    for (int c = 0; c < ch; c++)
                        out[x * ch + c] = row[sx * ch + c] * xa0[x] + row[sx1 * ch + c] * xa1[x];
                }
            }
            unsigned char *drow = &d.d[(size_t)y * dw * ch];
            for (int i = 0; i < dw * ch; i++)
                drow[i] = (unsigned char)((b0 * r0[i] + b1 * r1[i] + (1 << 21)) >> 22);
        }
    }

    // BGR -> Lab, 8-bit convention (L*255/100, a/b + 128)
    inline void bgr2lab8(const kcf::Image8 &s, kcf::Image8 &d)
    {
        static float g[256];
        static bool init = false;
        if (!init)
        {
            for (int i = 0; i < 256; i++)
            {
                float c = i / 255.f;
                g[i] = (c <= 0.04045f) ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
            }
            init = true;
        }
        auto f = [](float t)
        { return t > 0.008856f ? cbrtf(t) : 7.787f * t + 16.f / 116.f; };

        d.w = s.w;
        d.h = s.h;
        d.ch = 3;
        d.d.resize((size_t)s.w * s.h * 3);
        for (int i = 0; i < s.w * s.h; i++)
        {
            const unsigned char *p = &s.d[(size_t)i * 3];
            float B = g[p[0]], G = g[p[1]], R = g[p[2]];
            float X = (0.412453f * R + 0.357580f * G + 0.180423f * B) / 0.950456f;
            float Y = 0.212671f * R + 0.715160f * G + 0.072169f * B;
            float Z = (0.019334f * R + 0.119193f * G + 0.950227f * B) / 1.088754f;
            float fY = f(Y);
            float L = (Y > 0.008856f) ? 116.f * fY - 16.f : 903.3f * Y;
            unsigned char *o = &d.d[(size_t)i * 3];
            o[0] = clamp8(L * 255.f / 100.f);
            o[1] = clamp8(500.f * (f(X) - fY) + 128.f);
            o[2] = clamp8(200.f * (fY - f(Z)) + 128.f);
        }
    }

} // namespace ScalarImg