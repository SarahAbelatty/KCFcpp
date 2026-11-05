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

#include <cassert>
#include <cmath>
#include <vector>
#include "kcf_types.hpp"

namespace RectTools
{

    template <typename t>
    inline t x2(const kcf::Rect_<t> &rect)
    {
        return rect.x + rect.width;
    }

    template <typename t>
    inline t y2(const kcf::Rect_<t> &rect)
    {
        return rect.y + rect.height;
    }

    template <typename t>
    inline void resize(kcf::Rect_<t> &rect, float scalex, float scaley = 0)
    {
        if (!scaley)
            scaley = scalex;
        rect.x -= rect.width * (scalex - 1.f) / 2.f;
        rect.width *= scalex;

        rect.y -= rect.height * (scaley - 1.f) / 2.f;
        rect.height *= scaley;
    }

    template <typename t>
    inline void limit(kcf::Rect_<t> &rect, kcf::Rect_<t> limit)
    {
        if (rect.x + rect.width > limit.x + limit.width)
            rect.width = (limit.x + limit.width - rect.x);
        if (rect.y + rect.height > limit.y + limit.height)
            rect.height = (limit.y + limit.height - rect.y);
        if (rect.x < limit.x)
        {
            rect.width -= (limit.x - rect.x);
            rect.x = limit.x;
        }
        if (rect.y < limit.y)
        {
            rect.height -= (limit.y - rect.y);
            rect.y = limit.y;
        }
        if (rect.width < 0)
            rect.width = 0;
        if (rect.height < 0)
            rect.height = 0;
    }

    template <typename t>
    inline void limit(kcf::Rect_<t> &rect, t width, t height, t x = 0, t y = 0)
    {
        kcf::Rect_<t> lim;
        lim.x = x;
        lim.y = y;
        lim.width = width;
        lim.height = height;
        limit(rect, lim);
    }

    // x = left border, y = top border, width = right border, height = bottom border
    template <typename t>
    inline kcf::Rect getBorder(const kcf::Rect_<t> &original, kcf::Rect_<t> &limited)
    {
        kcf::Rect res;
        res.x = limited.x - original.x;
        res.y = limited.y - original.y;
        res.width = x2(original) - x2(limited);
        res.height = y2(original) - y2(limited);
        assert(res.x >= 0 && res.y >= 0 && res.width >= 0 && res.height >= 0);
        return res;
    }

    // Replaces subwindow(..., BORDER_REPLICATE): crop `window` out of `in`, replicating the
    // edge pixels where the window leaves the image. Output is always window.width x window.height.
    inline void subwindow(const kcf::Image8 &in, const kcf::Rect &window, kcf::Image8 &out)
    {
        kcf::Rect cutWindow = window;
        limit(cutWindow, in.w, in.h);
        if (cutWindow.height <= 0 || cutWindow.width <= 0)
            assert(0);
        (void)getBorder(window, cutWindow); // keeps the original sanity assert

        out.w = window.width;
        out.h = window.height;
        out.ch = in.ch;
        out.d.resize((size_t)out.w * out.h * out.ch);

        for (int y = 0; y < out.h; y++)
        {
            int sy = window.y + y;
            sy = sy < 0 ? 0 : (sy > in.h - 1 ? in.h - 1 : sy);
            const unsigned char *srow = &in.d[(size_t)sy * in.w * in.ch];
            unsigned char *drow = &out.d[(size_t)y * out.w * out.ch];
            for (int x = 0; x < out.w; x++)
            {
                int sx = window.x + x;
                sx = sx < 0 ? 0 : (sx > in.w - 1 ? in.w - 1 : sx);
                for (int c = 0; c < in.ch; c++)
                    drow[x * in.ch + c] = srow[sx * in.ch + c];
            }
        }
    }

    // Replaces getGrayImage: BGR -> gray with OpenCV's 8-bit fixed-point weights, scaled to [0,1] floats.
    // (The caller subtracts 0.5 afterwards, as before.)
    inline void getGrayImage(const kcf::Image8 &img, std::vector<float> &gray)
    {
        const int n = img.w * img.h;
        gray.resize(n);
        const float s = 1 / 255.f;
        for (int i = 0; i < n; i++)
        {
            unsigned char g;
            if (img.ch == 3)
            {
                const unsigned char *p = &img.d[(size_t)i * 3];
                g = (unsigned char)((p[0] * 1868 + p[1] * 9617 + p[2] * 4899 + 8192) >> 14);
            }
            else
                g = img.d[i];
            gray[i] = (float)g * s;
        }
    }

}
