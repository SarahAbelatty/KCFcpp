#pragma once
#include <vector>
#include <cmath>

namespace kcf
{

    struct Size
    {
        int width = 0, height = 0;
    };
    struct Point2f
    {
        float x = 0.f, y = 0.f;
    };

    template <typename T>
    struct Rect_
    {
        T x = 0, y = 0, width = 0, height = 0;
    };
    typedef Rect_<int> Rect;

    // 8-bit interleaved image (BGR or gray), replaces cv::Mat for frames
    struct Image8
    {
        int w = 0, h = 0, ch = 0;
        std::vector<unsigned char> d;
        bool empty() const { return d.empty(); }
        int stride() const { return w * ch; }
        const unsigned char *data() const { return d.data(); }
        unsigned char *data() { return d.data(); }
    };

    // cv::Rect_<float> -> cv::Rect conversion rounds half-to-even (cvRound), not truncation
    inline Rect toRect(const Rect_<float> &r)
    {
        Rect o;
        o.x = (int)lrintf(r.x);
        o.y = (int)lrintf(r.y);
        o.width = (int)lrintf(r.width);
        o.height = (int)lrintf(r.height);
        return o;
    }

} // namespace kcf