#include "resize_bilinear_scalar.hpp"

void resize_bilinear_scalar(const float* src, int src_w, int src_h, float* dst, int dst_w, int dst_h)
{
    const float scale_x = (float)src_w / (float)dst_w;
    const float scale_y = (float)src_h / (float)dst_h;
    for (int dy = 0; dy < dst_h; ++dy) {
        float fy  = (dy + 0.5f) * scale_y - 0.5f;
        int sy = (int)fy;
        if (fy < 0.0f && fy != (float)sy) sy -= 1;
        float fraction_y = fy - (float)sy;
        if (sy < 0)            { sy = 0;         fraction_y = 0.0f; }
        if (sy >= src_h - 1)   { sy = src_h - 2; fraction_y = 1.0f; }
        const float* top_row    = src + sy * src_w;
        const float* bottom_row = top_row + src_w;
        for (int dx = 0; dx < dst_w; ++dx) {
            float fx = (dx + 0.5f) * scale_x - 0.5f;
            int sx = (int)fx;
            if (fx < 0.0f && fx != (float)sx) sx -= 1;
            float u = fx - (float)sx;
            if (sx < 0)            { sx = 0;        u = 0.0f; }
            if (sx >= src_w - 1)   { sx = src_w - 2; u = 1.0f; }
            float tl = top_row[sx];
            float tr = top_row[sx + 1];
            float bl = bottom_row[sx];
            float br = bottom_row[sx + 1];
            float top    = tl * (1.0f - u) + tr * u;
            float bottom = bl * (1.0f - u) + br * u;
            dst[dy * dst_w + dx] = top * (1.0f - fraction_y) + bottom * fraction_y;
        }
    }
}
