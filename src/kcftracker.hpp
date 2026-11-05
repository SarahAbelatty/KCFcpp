/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: horizontal area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


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
#include <cassert>
#include "tracker.h"

// Data layouts used by the scalar tracker:
//   real feature map : std::vector<float>, channel-major, f[c*N + p], N = size_patch[0]*size_patch[1],
//                      p = y*size_patch[1] + x   (same layout as the old FeaturesMap Mat, C rows x N cols)
//   complex 2D array : std::vector<float>, interleaved re/im, size 2*H*W   (same as the old CV_32FC2 Mat)
//   real 2D array    : std::vector<float>, row-major, size H*W             (same as the old CV_32F Mat)
typedef std::vector<float> FVec;

class KCFTracker : public Tracker
{
public:
    // Constructor
    KCFTracker(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true);

    // Initialize tracker
    virtual void init(const kcf::Rect &roi, const kcf::Image8 &image);

    // Update position based on the new frame
    virtual kcf::Rect update(const kcf::Image8 &image);

    float interp_factor;       // linear interpolation factor for adaptation
    float sigma;               // gaussian kernel bandwidth
    float lambda;              // regularization
    int cell_size;             // HOG cell size
    int cell_sizeQ;            // cell size^2, to avoid repeated operations
    float padding;             // extra area surrounding the target
    float output_sigma_factor; // bandwidth of gaussian target
    int template_size;         // template size
    float scale_step;          // scale step for multi-scale estimation
    float scale_weight;        // to downweight detection scores of other scales for added stability

protected:
    // Detect object in the current frame.
    kcf::Point2f detect(const FVec &z, const FVec &x, float &peak_value);

    // train tracker with a single image
    void train(const FVec &x, float train_interp_factor);

    // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input
    // images X and Y, which must both be MxN. They must also be periodic (ie., pre-processed
    // with a cosine window). Returns a real H*W array.
    FVec gaussianCorrelation(const FVec &x1, const FVec &x2);

    // Create Gaussian Peak (already FFT'd, complex H*W). Function called only in the first frame.
    FVec createGaussianPeak(int sizey, int sizex);

    // Obtain sub-window from image, with replication-padding and extract features
    FVec getFeatures(const kcf::Image8 &image, bool inithann, float scale_adjust = 1.0f);

    // Initialize Hanning window. Function called only in the first frame.
    void createHanningMats();

    // Calculate sub-pixel peak for one dimension
    float subPixelPeak(float left, float center, float right);

    FVec _alphaf; // complex, size_patch[0]*size_patch[1]
    FVec _prob;   // complex, size_patch[0]*size_patch[1]
    FVec _tmpl;   // real, size_patch[2] * size_patch[0]*size_patch[1]

private:
    int size_patch[3]; // [0]=rows(H), [1]=cols(W), [2]=channels(C)
    FVec hann;         // channel-major window, size_patch[2]*H*W (gray: H*W)
    kcf::Size _tmpl_sz;
    float _scale;
    int _gaussian_size;
    bool _hogfeatures;
    bool _labfeatures;
};
