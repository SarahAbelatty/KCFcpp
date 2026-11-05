#include "fhog.hpp"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- STUDENT 1: K2 (Pure C++ Gradients) ---
static inline float k2_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    float r = x;
    for(int i=0; i<4; ++i) r = 0.5f * (r + x / r);
    return r;
}

static inline void k2_central_diff_x(const float* src, int w, int h, float* dst) {
    for (int y = 1; y < h - 1; ++y) {
        const float* row = src + y * w;
        float* out = dst + y * w;
        for (int x = 1; x < w - 1; ++x) out[x] = row[x + 1] - row[x - 1];
    }
}

static inline void k2_central_diff_y(const float* src, int w, int h, float* dst) {
    for (int y = 1; y < h - 1; ++y) {
        const float* rowu = src + (y - 1) * w;
        const float* rowd = src + (y + 1) * w;
        float* out = dst + y * w;
        for (int x = 1; x < w - 1; ++x) out[x] = rowd[x] - rowu[x];
    }
}

static const float K2_boundary_x[10] = {1.0f, 0.939f, 0.766f, 0.5f, 0.173f, -0.173f, -0.5f, -0.766f, -0.939f, -1.0f};
static const float K2_boundary_y[10] = {0.0f, 0.342f, 0.642f, 0.866f, 0.984f, 0.984f, 0.866f, 0.642f, 0.342f, 0.0f};

// --- STUDENT 1 & 2: K2/K3 (Pure C++ HOG) ---
int getFeatureMaps(const unsigned char *img, int width, int height, int stride, int numChannels, const int k, CvLSVMFeatureMapCaskade **map) {
    int sizeX = width / k;
    int sizeY = height / k;
    int p = 3 * NUM_SECTOR;
    int stringSize = sizeX * p;
    allocFeatureMapObject(map, sizeX, sizeY, p);

    // Convert uchar to float buffer for gradients
    float* img_float = (float*)malloc(sizeof(float) * width * height);
    for(int i=0; i<width*height; ++i) img_float[i] = (float)img[i * numChannels]; 

    float *dx = (float*)calloc(width * height, sizeof(float));
    float *dy = (float*)calloc(width * height, sizeof(float));
    
    k2_central_diff_x(img_float, width, height, dx);
    k2_central_diff_y(img_float, width, height, dy);
    free(img_float);

    float *r = (float *)malloc(sizeof(float) * (width * height));
    int *alfa = (int *)malloc(sizeof(int) * (width * height * 2));

    for(int j = 1; j < height - 1; j++) {
        for(int i = 1; i < width - 1; i++) {
            float x = dx[j * width + i];
            float y = dy[j * width + i];
            r[j * width + i] = k2_sqrtf(x * x + y * y);
            
            float max = K2_boundary_x[0] * x + K2_boundary_y[0] * y;
            int maxi = 0;
            for (int kk = 0; kk < NUM_SECTOR; kk++) {
                float dotProd = K2_boundary_x[kk] * x + K2_boundary_y[kk] * y;
                if (dotProd > max) { max = dotProd; maxi = kk; }
                else if (-dotProd > max) { max = -dotProd; maxi = kk + NUM_SECTOR; }
            }
            alfa[j * width * 2 + i * 2] = maxi % NUM_SECTOR;
            alfa[j * width * 2 + i * 2 + 1] = maxi;
        }
    }

    int *nearest = (int *)malloc(sizeof(int) * k);
    float *w = (float*)malloc(sizeof(float) * (k * 2));
    for(int i = 0; i < k / 2; i++) nearest[i] = -1;
    for(int i = k / 2; i < k; i++) nearest[i] = 1;
    for(int j = 0; j < k; j++) {
        float a_x = (j < k/2) ? (k/2 - j - 0.5f) : (j - k/2 + 0.5f);
        float b_x = (j < k/2) ? (k/2 + j + 0.5f) : (-j + k/2 - 0.5f + k);
        w[j * 2] = 1.0f/a_x * ((a_x * b_x) / (a_x + b_x));
        w[j * 2 + 1] = 1.0f/b_x * ((a_x * b_x) / (a_x + b_x));
    }

    for(int i = 0; i < sizeY; i++) {
        for(int j = 0; j < sizeX; j++) {
            for(int ii = 0; ii < k; ii++) {
                for(int jj = 0; jj < k; jj++) {
                    if ((i * k + ii > 0) && (i * k + ii < height - 1) && (j * k + jj > 0) && (j * k + jj < width - 1)) {
                        int d = (k * i + ii) * width + (j * k + jj);
                        (*map)->map[i * stringSize + j * p + alfa[d * 2]] += r[d] * w[ii * 2] * w[jj * 2];
                        (*map)->map[i * stringSize + j * p + alfa[d * 2 + 1] + NUM_SECTOR] += r[d] * w[ii * 2] * w[jj * 2];
                        
                        if ((i + nearest[ii] >= 0) && (i + nearest[ii] <= sizeY - 1)) {
                            (*map)->map[(i + nearest[ii]) * stringSize + j * p + alfa[d * 2]] += r[d] * w[ii * 2 + 1] * w[jj * 2];
                            (*map)->map[(i + nearest[ii]) * stringSize + j * p + alfa[d * 2 + 1] + NUM_SECTOR] += r[d] * w[ii * 2 + 1] * w[jj * 2];
                        }
                        if ((j + nearest[jj] >= 0) && (j + nearest[jj] <= sizeX - 1)) {
                            (*map)->map[i * stringSize + (j + nearest[jj]) * p + alfa[d * 2]] += r[d] * w[ii * 2] * w[jj * 2 + 1];
                            (*map)->map[i * stringSize + (j + nearest[jj]) * p + alfa[d * 2 + 1] + NUM_SECTOR] += r[d] * w[ii * 2] * w[jj * 2 + 1];
                        }
                        if ((i + nearest[ii] >= 0) && (i + nearest[ii] <= sizeY - 1) && (j + nearest[jj] >= 0) && (j + nearest[jj] <= sizeX - 1)) {
                            (*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * p + alfa[d * 2]] += r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
                            (*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * p + alfa[d * 2 + 1] + NUM_SECTOR] += r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
                        }
                    }
                }
            }
        }
    }
    free(dx); free(dy); free(w); free(nearest); free(r); free(alfa);
    return LATENT_SVM_OK;
}

// --- STUDENT 2: K3 (Pure C++ Norm/Truncate & PCA) ---
int normalizeAndTruncate(CvLSVMFeatureMapCaskade *map, const float alfa) {
    int sizeX = map->sizeX, sizeY = map->sizeY;
    int p = NUM_SECTOR, xp = NUM_SECTOR * 3, pp = NUM_SECTOR * 12;
    float *partOfNorm = (float *)malloc(sizeof(float) * (sizeX * sizeY));
    
    for(int i = 0; i < sizeX * sizeY; i++) {
        float val = 0.0f;
        int pos = i * map->numFeatures;
        for(int j = 0; j < p; j++) val += map->map[pos + j] * map->map[pos + j];
        partOfNorm[i] = val;
    }
    
    sizeX -= 2; sizeY -= 2;
    float *newData = (float *)malloc(sizeof(float) * (sizeX * sizeY * pp));
    
    for(int i = 1; i <= sizeY; i++) {
        for(int j = 1; j <= sizeX; j++) {
            int pos1 = i * (sizeX + 2) * xp + j * xp;
            int pos2 = (i-1) * sizeX * pp + (j-1) * pp;
            
            float v1 = sqrtf(partOfNorm[i*(sizeX+2)+j] + partOfNorm[i*(sizeX+2)+(j+1)] + partOfNorm[(i+1)*(sizeX+2)+j] + partOfNorm[(i+1)*(sizeX+2)+(j+1)]) + FLT_EPSILON;
            for(int ii=0; ii<p; ii++) newData[pos2+ii] = map->map[pos1+ii]/v1;
            for(int ii=0; ii<2*p; ii++) newData[pos2+ii+p*4] = map->map[pos1+ii+p]/v1;
            
            float v2 = sqrtf(partOfNorm[i*(sizeX+2)+j] + partOfNorm[i*(sizeX+2)+(j+1)] + partOfNorm[(i-1)*(sizeX+2)+j] + partOfNorm[(i-1)*(sizeX+2)+(j+1)]) + FLT_EPSILON;
            for(int ii=0; ii<p; ii++) newData[pos2+ii+p] = map->map[pos1+ii]/v2;
            for(int ii=0; ii<2*p; ii++) newData[pos2+ii+p*6] = map->map[pos1+ii+p]/v2;
            
            float v3 = sqrtf(partOfNorm[i*(sizeX+2)+j] + partOfNorm[i*(sizeX+2)+(j-1)] + partOfNorm[(i+1)*(sizeX+2)+j] + partOfNorm[(i+1)*(sizeX+2)+(j-1)]) + FLT_EPSILON;
            for(int ii=0; ii<p; ii++) newData[pos2+ii+p*2] = map->map[pos1+ii]/v3;
            for(int ii=0; ii<2*p; ii++) newData[pos2+ii+p*8] = map->map[pos1+ii+p]/v3;
            
            float v4 = sqrtf(partOfNorm[i*(sizeX+2)+j] + partOfNorm[i*(sizeX+2)+(j-1)] + partOfNorm[(i-1)*(sizeX+2)+j] + partOfNorm[(i-1)*(sizeX+2)+(j-1)]) + FLT_EPSILON;
            for(int ii=0; ii<p; ii++) newData[pos2+ii+p*3] = map->map[pos1+ii]/v4;
            for(int ii=0; ii<2*p; ii++) newData[pos2+ii+p*10] = map->map[pos1+ii+p]/v4;
        }
    }
    for(int i = 0; i < sizeX * sizeY * pp; i++) if(newData[i] > alfa) newData[i] = alfa;
    
    map->numFeatures = pp; map->sizeX = sizeX; map->sizeY = sizeY;
    free(map->map); free(partOfNorm); map->map = newData;
    return LATENT_SVM_OK;
}

int PCAFeatureMaps(CvLSVMFeatureMapCaskade *map) {
    int sizeX = map->sizeX, sizeY = map->sizeY, p = map->numFeatures;
    int pp = NUM_SECTOR * 3 + 4, xp = NUM_SECTOR, yp = 4;
    float nx = 1.0f / sqrtf((float)(xp * 2)), ny = 1.0f / sqrtf((float)(yp));
    float *newData = (float *)malloc(sizeof(float) * (sizeX * sizeY * pp));
    
    for(int i = 0; i < sizeY; i++) {
        for(int j = 0; j < sizeX; j++) {
            int pos1 = (i * sizeX + j) * p, pos2 = (i * sizeX + j) * pp, k = 0;
            for(int jj = 0; jj < xp * 2; jj++) {
                float val = 0; for(int ii = 0; ii < yp; ii++) val += map->map[pos1 + yp * xp + ii * xp * 2 + jj];
                newData[pos2 + k++] = val * ny;
            }
            for(int jj = 0; jj < xp; jj++) {
                float val = 0; for(int ii = 0; ii < yp; ii++) val += map->map[pos1 + ii * xp + jj];
                newData[pos2 + k++] = val * ny;
            }
            for(int ii = 0; ii < yp; ii++) {
                float val = 0; for(int jj = 0; jj < 2 * xp; jj++) val += map->map[pos1 + yp * xp + ii * xp * 2 + jj];
                newData[pos2 + k++] = val * nx;
            }
        }
    }
    map->numFeatures = pp; free(map->map); map->map = newData;
    return LATENT_SVM_OK;
}

int allocFeatureMapObject(CvLSVMFeatureMapCaskade **obj, const int sizeX, const int sizeY, const int numFeatures) {
    (*obj) = (CvLSVMFeatureMapCaskade *)malloc(sizeof(CvLSVMFeatureMapCaskade));
    (*obj)->sizeX = sizeX; (*obj)->sizeY = sizeY; (*obj)->numFeatures = numFeatures;
    (*obj)->map = (float *)calloc(sizeX * sizeY * numFeatures, sizeof(float));
    return LATENT_SVM_OK;
}

int freeFeatureMapObject(CvLSVMFeatureMapCaskade **obj) {
    if(*obj == NULL) return LATENT_SVM_MEM_NULL;
    free((*obj)->map); free(*obj); (*obj) = NULL;
    return LATENT_SVM_OK;
}