/*
Unified Pure Scalar KCF Tracker (Student 1 + 2 + 3 Integration)
*/
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

// ============================================================================
// STUDENT 1: K1 (Pure C++ Bilinear Resize)
// ============================================================================
extern void resize_bilinear_scalar(const float* src, int src_w, int src_h, float* dst, int dst_w, int dst_h);

// ============================================================================
// STUDENT 3: K5 & K6 (Pure C++ Complex Mult & Peak Detection)
// ============================================================================
inline void complex_multiply_pure_scalar(const float* a, const float* b, float* out, int total) {
    for (int i = 0; i < total; i++) {
        int idx = i * 2; // Stride of 2 floats (8 bytes) for interleaved complex data
        out[idx]     = (a[idx] * b[idx]) - (a[idx+1] * b[idx+1]); // Real
        out[idx+1]   = (a[idx] * b[idx+1]) + (a[idx+1] * b[idx]); // Imaginary
    }
}

inline void find_peak_subpixel_pure_scalar(const float* response, int rows, int cols, 
                                           float& max_val, int& mx, int& my, float& sx, float& sy) {
    max_val = -FLT_MAX; int max_idx = -1;
    int total = rows * cols;
    for (int i = 0; i < total; i++) {
        if (response[i] > max_val) { max_val = response[i]; max_idx = i; }
    }
    my = max_idx / cols; mx = max_idx % cols;
    
    float c = response[my*cols+mx];
    float l = response[my*cols + (mx>0 ? mx-1 : mx)];
    float r = response[my*cols + (mx<cols-1 ? mx+1 : mx)];
    float t = response[(my>0 ? my-1 : my)*cols + mx];
    float b = response[(my<rows-1 ? my+1 : my)*cols + mx];
    
    float dx = l - 2*c + r;
    float dy = t - 2*c + b;
    
    sx = mx + (std::abs(dx) > 1e-6f ? 0.5f * (l - r) / dx : 0.0f);
    sy = my + (std::abs(dy) > 1e-6f ? 0.5f * (t - b) / dy : 0.0f);
}

// ============================================================================
// KCFTracker Implementation
// ============================================================================

KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab) {
    lambda = 0.0001; padding = 2.5; output_sigma_factor = 0.125;
    if (hog) {
        interp_factor = 0.012; sigma = 0.6; cell_size = 4; _hogfeatures = true;
        if (lab) {
            interp_factor = 0.005; sigma = 0.4; output_sigma_factor = 0.1;
            _labfeatures = true; _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data);
            cell_sizeQ = cell_size * cell_size;
        } else { _labfeatures = false; }
    } else {
        interp_factor = 0.075; sigma = 0.2; cell_size = 1; _hogfeatures = false; _labfeatures = false;
    }
    if (multiscale) {
        template_size = 96; scale_step = 1.05; scale_weight = 0.95; fixed_window = true;
    } else if (fixed_window) {
        template_size = 96; scale_step = 1;
    } else {
        template_size = 1; scale_step = 1;
    }
}

void KCFTracker::init(const cv::Rect &roi, cv::Mat image) {
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);
    _tmpl = getFeatures(image, 1);
    _prob = createGaussianPeak(size_patch[0], size_patch[1]);
    _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0);
}

cv::Rect KCFTracker::update(cv::Mat image) {
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;
    float peak_value;
    cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);

    if (scale_step != 1) {
        float new_peak_value;
        cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);
        if (scale_weight * new_peak_value > peak_value) {
            res = new_res; peak_value = new_peak_value; _scale /= scale_step;
            _roi.width /= scale_step; _roi.height /= scale_step;
        }
        new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);
        if (scale_weight * new_peak_value > peak_value) {
            res = new_res; peak_value = new_peak_value; _scale *= scale_step;
            _roi.width *= scale_step; _roi.height *= scale_step;
        }
    }

    _roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale);
    _roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;
    assert(_roi.width >= 0 && _roi.height >= 0);

    cv::Mat x = getFeatures(image, 0);
    train(x, interp_factor);
    return _roi;
}

// --- STUDENT 3 INTEGRATION: Pure Scalar K5 & K6 ---
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value) {
    using namespace FFTTools;
    cv::Mat k = gaussianCorrelation(x, z);
    cv::Mat k_fft = fftd(k);

    // K5: Pure C++ Complex Multiplication (Zero-Copy Pointer Bridge)
    cv::Mat res_complex(_alphaf.size(), CV_32FC2, cv::Scalar(0));
    int total_pixels = _alphaf.rows * _alphaf.cols;
    complex_multiply_pure_scalar((const float*)_alphaf.data, (const float*)k_fft.data, 
                                 (float*)res_complex.data, total_pixels);

    cv::Mat res = real(fftd(res_complex, true));

    // K6: Pure C++ Peak Detection (Zero-Copy Pointer Bridge)
    float max_val, sx, sy; int mx, my;
    cv::Mat res_float = (res.type() != CV_32F) ? res.clone() : res;
    find_peak_subpixel_pure_scalar((const float*)res_float.data, res_float.rows, res_float.cols, 
                                   max_val, mx, my, sx, sy);
    
    peak_value = max_val;
    cv::Point2f p(sx, sy);
    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;
    return p;
}

void KCFTracker::train(cv::Mat x, float train_interp_factor) {
    using namespace FFTTools;
    cv::Mat k = gaussianCorrelation(x, x);
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;
}

cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2) {
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
    if (_hogfeatures) {
        cv::Mat caux, x1aux, x2aux;
        for (int i = 0; i < size_patch[2]; i++) {
            x1aux = x1.row(i).reshape(1, size_patch[0]);
            x2aux = x2.row(i).reshape(1, size_patch[0]);
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            caux = fftd(caux, true); rearrange(caux); caux.convertTo(caux, CV_32F);
            c = c + real(caux);
        }
    } else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true); rearrange(c); c = real(c);
    }
    cv::Mat d;
    cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (size_patch[0]*size_patch[1]*size_patch[2]), 0, d);
    cv::Mat k; cv::exp((-d / (sigma * sigma)), k);
    return k;
}

cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex) {
    cv::Mat_<float> res(sizey, sizex);
    int syh = sizey / 2, sxh = sizex / 2;
    float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);
    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
            res(i, j) = std::exp(mult * (float)((i - syh)*(i - syh) + (j - sxh)*(j - sxh)));
    return FFTTools::fftd(res);
}

// --- STUDENT 1 & 2 INTEGRATION: Pure Scalar Feature Extraction ---
cv::Mat KCFTracker::getFeatures(const cv::Mat &image, bool inithann, float scale_adjust) {
    cv::Rect extracted_roi;
    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;
    
    if (inithann) {
        int padded_w = _roi.width * padding, padded_h = _roi.height * padding;
        if (template_size > 1) {
            _scale = (padded_w >= padded_h) ? padded_w / (float)template_size : padded_h / (float)template_size;
            _tmpl_sz.width = padded_w / _scale; _tmpl_sz.height = padded_h / _scale;
        } else {
            _tmpl_sz.width = padded_w; _tmpl_sz.height = padded_h; _scale = 1;
        }
        if (_hogfeatures) {
            _tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size*2;
            _tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size*2;
        } else {
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2; _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }
    
    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;
    
    cv::Mat FeaturesMap;  
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
    
    // STUDENT 1: K1 (Pure C++ Resize)
    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
        cv::Mat z_gray;
        if (z.channels() == 3) cv::cvtColor(z, z_gray, cv::COLOR_BGR2GRAY); 
        else z_gray = z.clone();
        z_gray.convertTo(z_gray, CV_32F);
        
        std::vector<float> src_buf(z_gray.rows * z_gray.cols);
        std::vector<float> dst_buf(_tmpl_sz.height * _tmpl_sz.width);
        std::memcpy(src_buf.data(), z_gray.data, sizeof(float) * src_buf.size()); // Zero-copy read
        
        resize_bilinear_scalar(src_buf.data(), z_gray.cols, z_gray.rows, 
                               dst_buf.data(), _tmpl_sz.width, _tmpl_sz.height);
        
        z = cv::Mat(_tmpl_sz.height, _tmpl_sz.width, CV_32F);
        std::memcpy(z.data, dst_buf.data(), sizeof(float) * dst_buf.size()); // Zero-copy write
    }

    // STUDENT 2: K2, K3 (Pure C++ HOG, NO IplImage!)
    if (_hogfeatures) {
        CvLSVMFeatureMapCaskade *map;
        // NEW SIGNATURE: Pass raw pointers directly!
        getFeatureMaps((const unsigned char*)z.data, z.cols, z.rows, (int)z.step1(), 1, cell_size, &map);
        
        normalizeAndTruncate(map, 0.2f);
        PCAFeatureMaps(map);
        
        size_patch[0] = map->sizeY; size_patch[1] = map->sizeX; size_patch[2] = map->numFeatures;
        FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX * map->sizeY), CV_32F, map->map);
        FeaturesMap = FeaturesMap.t();
        freeFeatureMapObject(&map);
    } else {
        FeaturesMap = RectTools::getGrayImage(z);
        FeaturesMap -= (float)0.5;
        size_patch[0] = z.rows; size_patch[1] = z.cols; size_patch[2] = 1;  
    }
    
    if (inithann) createHanningMats();
    
    // STUDENT 2: K4 (Pure C++ Hanning Multiply)
    if (!FeaturesMap.isContinuous()) FeaturesMap = FeaturesMap.clone();
    float *f = FeaturesMap.ptr<float>(0);
    int total = size_patch[2] * size_patch[0] * size_patch[1];
    for (int i = 0; i < total; i++) f[i] *= hann[i];
    
    return FeaturesMap;
}

void KCFTracker::createHanningMats() {
    const int H = size_patch[0], W = size_patch[1];
    const double PI_D = 3.14159265358979323846;
    std::vector<float> hann1t(W, 0.f), hann2t(H, 0.f);
    for (int i = 0; i < W; i++) hann1t[i] = 0.5 * (1 - std::cos(2 * PI_D * i / (W - 1)));
    for (int i = 0; i < H; i++) hann2t[i] = 0.5 * (1 - std::cos(2 * PI_D * i / (H - 1)));
    
    std::vector<float> hann2d((size_t)H * W, 0.f);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            hann2d[(size_t)y * W + x] = hann2t[y] * hann1t[x];
            
    if (_hogfeatures) {
        const int N = H * W;
        hann.assign((size_t)size_patch[2] * N, 0.f);
        for (int i = 0; i < size_patch[2]; i++)
            for (int j = 0; j < N; j++)
                hann[(size_t)i * N + j] = hann2d[j];
    } else {
        hann = hann2d;
    }
}

float KCFTracker::subPixelPeak(float left, float center, float right) {
    float divisor = 2 * center - right - left;
    if (divisor == 0) return 0;
    return 0.5 * (right - left) / divisor;
}