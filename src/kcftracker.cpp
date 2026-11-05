#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#include "scalar_img.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>

// ============================================================================
// K5: complex multiplication (interleaved re/im)
// ============================================================================
inline void complex_multiply_pure_scalar(const float *a, const float *b, float *out, int total)
{
    for (int i = 0; i < total; i++)
    {
        int idx = i * 2;
        out[idx] = (a[idx] * b[idx]) - (a[idx + 1] * b[idx + 1]);
        out[idx + 1] = (a[idx] * b[idx + 1]) + (a[idx + 1] * b[idx]);
    }
}

// ============================================================================
// K6: max search + sub-pixel refinement (interior pixels only, as in the original)
// ============================================================================
inline void find_peak_subpixel_pure_scalar(const float *response, int rows, int cols,
                                           float &max_val, float &sx, float &sy)
{
    int total = rows * cols, max_idx = 0;
    max_val = response[0];
    for (int i = 1; i < total; i++)
        if (response[i] > max_val)
        {
            max_val = response[i];
            max_idx = i;
        }
    const int my = max_idx / cols, mx = max_idx % cols;
    sx = (float)mx;
    sy = (float)my;

    if (mx > 0 && mx < cols - 1)
    {
        float l = response[my * cols + mx - 1], r = response[my * cols + mx + 1];
        float div = 2 * max_val - r - l;
        if (div != 0)
            sx += (float)(0.5 * (r - l) / div);
    }
    if (my > 0 && my < rows - 1)
    {
        float t = response[(my - 1) * cols + mx], b = response[(my + 1) * cols + mx];
        float div = 2 * max_val - b - t;
        if (div != 0)
            sy += (float)(0.5 * (b - t) / div);
    }
}

// ============================================================================
// KCFTracker
// ============================================================================
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{
    lambda = 0.0001;
    padding = 2.5;
    output_sigma_factor = 0.125;

    if (hog)
    {
        interp_factor = 0.012;
        sigma = 0.6;
        cell_size = 4;
        _hogfeatures = true;
        if (lab)
        {
            interp_factor = 0.005;
            sigma = 0.4;
            output_sigma_factor = 0.1;
            _labfeatures = true;
            cell_sizeQ = cell_size * cell_size;
        }
        else
        {
            _labfeatures = false;
        }
    }
    else
    {
        interp_factor = 0.075;
        sigma = 0.2;
        cell_size = 1;
        _hogfeatures = false;
        _labfeatures = false;
        if (lab)
            printf("Lab features are only used with HOG features.\n");
    }

    if (multiscale)
    {
        template_size = 96;
        scale_step = 1.05;
        scale_weight = 0.95;
        fixed_window = true;
    }
    else if (fixed_window)
    {
        template_size = 96;
        scale_step = 1;
    }
    else
    {
        template_size = 1;
        scale_step = 1;
    }
}

void KCFTracker::init(const kcf::Rect &roi, const kcf::Image8 &image)
{
    _roi.x = roi.x;
    _roi.y = roi.y;
    _roi.width = roi.width;
    _roi.height = roi.height;
    assert(roi.width >= 0 && roi.height >= 0);
    _tmpl = getFeatures(image, 1);
    _prob = createGaussianPeak(size_patch[0], size_patch[1]);
    _alphaf.assign((size_t)2 * size_patch[0] * size_patch[1], 0.f);
    train(_tmpl, 1.0);
}

kcf::Rect KCFTracker::update(const kcf::Image8 &image)
{
    if (_roi.x + _roi.width <= 0)
        _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0)
        _roi.y = -_roi.height + 1;
    if (_roi.x >= image.w - 1)
        _roi.x = image.w - 2;
    if (_roi.y >= image.h - 1)
        _roi.y = image.h - 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    float peak_value;
    kcf::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);

    if (scale_step != 1)
    {
        float new_peak_value;
        kcf::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);
        if (scale_weight * new_peak_value > peak_value)
        {
            res = new_res;
            peak_value = new_peak_value;
            _scale /= scale_step;
            _roi.width /= scale_step;
            _roi.height /= scale_step;
        }
        new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);
        if (scale_weight * new_peak_value > peak_value)
        {
            res = new_res;
            peak_value = new_peak_value;
            _scale *= scale_step;
            _roi.width *= scale_step;
            _roi.height *= scale_step;
        }
    }

    _roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale);
    _roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale);

    if (_roi.x >= image.w - 1)
        _roi.x = image.w - 1;
    if (_roi.y >= image.h - 1)
        _roi.y = image.h - 1;
    if (_roi.x + _roi.width <= 0)
        _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0)
        _roi.y = -_roi.height + 2;
    assert(_roi.width >= 0 && _roi.height >= 0);

    FVec x = getFeatures(image, 0);
    train(x, interp_factor);

    return kcf::toRect(_roi); // rounds like cv::Rect did
}

kcf::Point2f KCFTracker::detect(const FVec &z, const FVec &x, float &peak_value)
{
    using namespace FFTTools;
    const int H = size_patch[0], W = size_patch[1];

    FVec k = gaussianCorrelation(x, z);
    FVec kf = fft2(k.data(), H, W);

    FVec rc((size_t)2 * H * W);
    complex_multiply_pure_scalar(_alphaf.data(), kf.data(), rc.data(), H * W); // K5
    FVec res = real(ifft2(rc, H, W));

    float sx, sy;
    find_peak_subpixel_pure_scalar(res.data(), H, W, peak_value, sx, sy); // K6

    kcf::Point2f p;
    p.x = sx - (W / 2);
    p.y = sy - (H / 2);
    return p;
}

void KCFTracker::train(const FVec &x, float f)
{
    using namespace FFTTools;
    const int H = size_patch[0], W = size_patch[1];

    FVec k = gaussianCorrelation(x, x);
    FVec kf = fft2(k.data(), H, W);
    for (int i = 0; i < H * W; i++)
        kf[2 * i] += lambda; // lambda on the real part only
    FVec alphaf = complexDivision(_prob, kf);

    const float a = 1 - f;
    for (size_t i = 0; i < _tmpl.size(); i++)
        _tmpl[i] = a * _tmpl[i] + f * x[i];
    for (size_t i = 0; i < _alphaf.size(); i++)
        _alphaf[i] = a * _alphaf[i] + f * alphaf[i];
}

FVec KCFTracker::gaussianCorrelation(const FVec &x1, const FVec &x2)
{
    using namespace FFTTools;
    const int H = size_patch[0], W = size_patch[1], C = size_patch[2];
    const int N = H * W;

    FVec c(N, 0.f);
    for (int i = 0; i < C; i++)
    { // gray is just C == 1
        FVec a = fft2(&x1[(size_t)i * N], H, W);
        FVec b = fft2(&x2[(size_t)i * N], H, W);
        FVec prod = mulSpectrumsConj(a, b);
        FVec r = real(ifft2(prod, H, W));
        rearrange(r, H, W);
        for (int j = 0; j < N; j++)
            c[j] += r[j];
    }

    double s1 = 0, s2 = 0;
    for (size_t i = 0; i < x1.size(); i++)
    {
        float p = x1[i] * x1[i];
        s1 += (double)p;
    }
    for (size_t i = 0; i < x2.size(); i++)
    {
        float p = x2[i] * x2[i];
        s2 += (double)p;
    }

    const float tot = (float)(s1 + s2);
    const float inv = (float)(1.0 / (double)(H * W * C));
    const float sig = (float)(1.0 / (double)(sigma * sigma));

    FVec k(N);
    for (int j = 0; j < N; j++)
    {
        float d = (tot - 2.f * c[j]) * inv;
        if (d < 0.f)
            d = 0.f;
        k[j] = std::exp(-d * sig);
    }
    return k;
}

FVec KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    FVec res((size_t)sizey * sizex);
    int syh = sizey / 2, sxh = sizex / 2;
    float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh, jh = j - sxh;
            res[(size_t)i * sizex + j] = std::exp(mult * (float)(ih * ih + jh * jh));
        }
    return FFTTools::fft2(res.data(), sizey, sizex);
}

FVec KCFTracker::getFeatures(const kcf::Image8 &image, bool inithann, float scale_adjust)
{
    kcf::Rect extracted_roi;
    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;

    if (inithann)
    {
        int padded_w = _roi.width * padding;
        int padded_h = _roi.height * padding;

        if (template_size > 1)
        {
            if (padded_w >= padded_h)
                _scale = padded_w / (float)template_size;
            else
                _scale = padded_h / (float)template_size;
            _tmpl_sz.width = padded_w / _scale;
            _tmpl_sz.height = padded_h / _scale;
        }
        else
        {
            _tmpl_sz.width = padded_w;
            _tmpl_sz.height = padded_h;
            _scale = 1;
        }

        if (_hogfeatures)
        {
            _tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
            _tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
        }
        else
        {
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }

    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    // crop with replicated border, then K1 resize (8-bit, all channels)
    kcf::Image8 crop, resized;
    RectTools::subwindow(image, extracted_roi, crop);
    const kcf::Image8 *zp = &crop;
    if (crop.w != _tmpl_sz.width || crop.h != _tmpl_sz.height)
    {
        ScalarImg::resizeLinear(crop, _tmpl_sz.width, _tmpl_sz.height, resized);
        zp = &resized;
    }
    const kcf::Image8 &z = *zp;

    FVec feat; // channel-major: feat[c*N + p]
    int N;

    if (_hogfeatures)
    {
        // K2/K3 + normalization + PCA
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps(z.d.data(), z.w, z.h, z.stride(), z.ch, cell_size, &map);
        normalizeAndTruncate(map, 0.2f);
        PCAFeatureMaps(map);
        size_patch[0] = map->sizeY;
        size_patch[1] = map->sizeX;
        size_patch[2] = map->numFeatures;

        N = map->sizeX * map->sizeY;
        const int C = map->numFeatures;
        feat.assign((size_t)(C + (_labfeatures ? nClusters : 0)) * N, 0.f);
        for (int p = 0; p < N; p++)
            for (int c = 0; c < C; c++)
                feat[(size_t)c * N + p] = map->map[p * C + c];
        freeFeatureMapObject(&map);

        if (_labfeatures)
        {
            kcf::Image8 lab;
            ScalarImg::bgr2lab8(z, lab);
            const float *cent = &data[0][0];
            const int lstride = lab.stride();

            int cntCell = 0;
            for (int cY = cell_size; cY < z.h - cell_size; cY += cell_size)
            {
                for (int cX = cell_size; cX < z.w - cell_size; cX += cell_size)
                {
                    for (int y = cY; y < cY + cell_size; ++y)
                    {
                        for (int x = cX; x < cX + cell_size; ++x)
                        {
                            const unsigned char *px = &lab.d[(size_t)y * lstride + x * 3];
                            float l = (float)px[0], a = (float)px[1], b = (float)px[2];
                            float minDist = FLT_MAX;
                            int minIdx = 0;
                            for (int k = 0; k < nClusters; ++k)
                            {
                                float dl = l - cent[3 * k], da = a - cent[3 * k + 1], db = b - cent[3 * k + 2];
                                float dist = dl * dl + da * da + db * db;
                                if (dist < minDist)
                                {
                                    minDist = dist;
                                    minIdx = k;
                                }
                            }
                            feat[(size_t)(C + minIdx) * N + cntCell] += 1.0 / cell_sizeQ;
                        }
                    }
                    cntCell++;
                }
            }
            size_patch[2] += nClusters;
        }
    }
    else
    {
        std::vector<float> g;
        RectTools::getGrayImage(z, g);
        size_patch[0] = z.h;
        size_patch[1] = z.w;
        size_patch[2] = 1;
        N = z.w * z.h;
        feat.resize(N);
        for (int i = 0; i < N; i++)
            feat[i] = g[i] - 0.5f; // as in the paper
    }

    if (inithann)
        createHanningMats();

    // K4: Hanning multiply
    const size_t total = (size_t)size_patch[2] * N;
    for (size_t i = 0; i < total; i++)
        feat[i] *= hann[i];

    return feat;
}

void KCFTracker::createHanningMats()
{
    const int H = size_patch[0], W = size_patch[1];
    const double PI_D = 3.14159265358979323846;
    std::vector<float> hann1t(W, 0.f), hann2t(H, 0.f);
    for (int i = 0; i < W; i++)
        hann1t[i] = 0.5 * (1 - std::cos(2 * PI_D * i / (W - 1)));
    for (int i = 0; i < H; i++)
        hann2t[i] = 0.5 * (1 - std::cos(2 * PI_D * i / (H - 1)));

    std::vector<float> hann2d((size_t)H * W, 0.f);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            hann2d[(size_t)y * W + x] = hann2t[y] * hann1t[x];

    if (_hogfeatures)
    {
        const int N = H * W;
        hann.assign((size_t)size_patch[2] * N, 0.f);
        for (int i = 0; i < size_patch[2]; i++)
            for (int j = 0; j < N; j++)
                hann[(size_t)i * N + j] = hann2d[j];
    }
    else
    {
        hann = hann2d;
    }
}

float KCFTracker::subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;
    if (divisor == 0)
        return 0;
    return 0.5 * (right - left) / divisor;
}