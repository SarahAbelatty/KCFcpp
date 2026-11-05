/*
Author: Christian Bailer
Contact address: Christian.Bailer@dfki.de
Department Augmented Vision DFKI

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#pragma once
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <utility>

typedef std::vector<float> FVec; // same typedef as in kcftracker.hpp

namespace FFTTools
{
    struct cpx
    {
        double re, im;
    };

    inline const std::vector<cpx> &twiddles(int N)
    {
        static std::map<int, std::vector<cpx>> cache;
        std::map<int, std::vector<cpx>>::iterator it = cache.find(N);
        if (it == cache.end())
        {
            std::vector<cpx> t(N);
            for (int j = 0; j < N; j++)
            {
                double a = -2.0 * 3.14159265358979323846 * j / N;
                t[j].re = std::cos(a);
                t[j].im = std::sin(a);
            }
            it = cache.insert(std::make_pair(N, t)).first;
        }
        return it->second;
    }

    // Recursive mixed-radix DFT of n points read from in[j*stride]; result in out[0..n-1].
    // tw = W_N table (n divides N). t = scratch of at least n elements.
    inline void fft_rec(const cpx *in, cpx *out, int n, int stride, bool inv,
                        const cpx *tw, int N, cpx *t)
    {
        if (n == 1)
        {
            out[0] = in[0];
            return;
        }
        int p = 2;
        while (n % p)
            p++;
        const int m = n / p;
        for (int r = 0; r < p; r++)
            fft_rec(in + (size_t)r * stride, out + r * m, m, stride * p, inv, tw, N, t);

        const int stepN = N / n, stepP = N / p;
        for (int k = 0; k < m; k++)
        {
            for (int r = 0; r < p; r++)
            {
                cpx v = out[r * m + k];
                cpx w = tw[(int)(((long long)r * k * stepN) % N)];
                if (inv)
                    w.im = -w.im;
                t[r].re = v.re * w.re - v.im * w.im;
                t[r].im = v.re * w.im + v.im * w.re;
            }
            for (int q = 0; q < p; q++)
            {
                double sr = 0, si = 0;
                for (int r = 0; r < p; r++)
                {
                    cpx w = tw[((r * q) % p) * stepP];
                    if (inv)
                        w.im = -w.im;
                    sr += t[r].re * w.re - t[r].im * w.im;
                    si += t[r].re * w.im + t[r].im * w.re;
                }
                out[k + q * m].re = sr;
                out[k + q * m].im = si;
            }
        }
    }

    inline void transform2d(std::vector<cpx> &a, int H, int W, bool inv)
    {
        const std::vector<cpx> &twW = twiddles(W);
        const std::vector<cpx> &twH = twiddles(H);
        const int M = std::max(H, W);
        std::vector<cpx> out(M), scr(M), col(H);
        for (int y = 0; y < H; y++)
        {
            fft_rec(&a[(size_t)y * W], out.data(), W, 1, inv, twW.data(), W, scr.data());
            std::copy(out.begin(), out.begin() + W, a.begin() + (size_t)y * W);
        }
        for (int x = 0; x < W; x++)
        {
            for (int y = 0; y < H; y++)
                col[y] = a[(size_t)y * W + x];
            fft_rec(col.data(), out.data(), H, 1, inv, twH.data(), H, scr.data());
            for (int y = 0; y < H; y++)
                a[(size_t)y * W + x] = out[y];
        }
        if (inv)
        {
            const double s = 1.0 / ((double)H * W);
            for (size_t i = 0; i < a.size(); i++)
            {
                a[i].re *= s;
                a[i].im *= s;
            }
        }
    }

    // real H*W (row-major) -> complex H*W, interleaved re/im
    inline FVec fft2(const float *re, int H, int W)
    {
        std::vector<cpx> a((size_t)H * W);
        for (size_t i = 0; i < a.size(); i++)
        {
            a[i].re = re[i];
            a[i].im = 0;
        }
        transform2d(a, H, W, false);
        FVec out(a.size() * 2);
        for (size_t i = 0; i < a.size(); i++)
        {
            out[2 * i] = (float)a[i].re;
            out[2 * i + 1] = (float)a[i].im;
        }
        return out;
    }

    // complex -> complex, scaled by 1/(H*W)
    inline FVec ifft2(const FVec &c, int H, int W)
    {
        std::vector<cpx> a((size_t)H * W);
        for (size_t i = 0; i < a.size(); i++)
        {
            a[i].re = c[2 * i];
            a[i].im = c[2 * i + 1];
        }
        transform2d(a, H, W, true);
        FVec out(a.size() * 2);
        for (size_t i = 0; i < a.size(); i++)
        {
            out[2 * i] = (float)a[i].re;
            out[2 * i + 1] = (float)a[i].im;
        }
        return out;
    }

    inline FVec real(const FVec &c)
    {
        FVec r(c.size() / 2);
        for (size_t i = 0; i < r.size(); i++)
            r[i] = c[2 * i];
        return r;
    }

    // out = a * conj(b)   (cv::mulSpectrums with conjB = true)
    inline FVec mulSpectrumsConj(const FVec &a, const FVec &b)
    {
        FVec o(a.size());
        for (size_t i = 0; i < a.size() / 2; i++)
        {
            float a0 = a[2 * i], a1 = a[2 * i + 1], b0 = b[2 * i], b1 = b[2 * i + 1];
            o[2 * i] = a0 * b0 + a1 * b1;
            o[2 * i + 1] = a1 * b0 - a0 * b1;
        }
        return o;
    }

    // Kept EXACTLY like the original repo, including its imaginary part (a1*b0 + a0*b1).
    // The mathematically correct quotient has a minus there; changing it changes the tracking results.
    inline FVec complexDivision(const FVec &a, const FVec &b)
    {
        FVec o(a.size());
        for (size_t i = 0; i < a.size() / 2; i++)
        {
            float a0 = a[2 * i], a1 = a[2 * i + 1], b0 = b[2 * i], b1 = b[2 * i + 1];
            float divisor = 1.f / (b0 * b0 + b1 * b1);
            o[2 * i] = (a0 * b0 + a1 * b1) * divisor;
            o[2 * i + 1] = (a1 * b0 + a0 * b1) * divisor;
        }
        return o;
    }

    // swap quadrants of a real H*W array (cx = W/2, cy = H/2), same as the original rearrange()
    inline void rearrange(FVec &img, int H, int W)
    {
        const int cx = W / 2, cy = H / 2;
        for (int y = 0; y < cy; y++)
            for (int x = 0; x < cx; x++)
            {
                std::swap(img[(size_t)y * W + x], img[(size_t)(y + cy) * W + (x + cx)]);
                std::swap(img[(size_t)y * W + (x + cx)], img[(size_t)(y + cy) * W + x]);
            }
    }

} // namespace FFTTools