#include "dense_interpolation.h" // 假设包含头文件声明（见下方说明）
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

// ==============================
// Helper: Wendland C2 CSRBF
// ==============================
inline float wendlandC2(float r, float supportRadius) {
    if (r >= supportRadius || supportRadius <= 0.0f) return 0.0f;
    float t = r / supportRadius;
    float om = 1.0f - t;
    return om * om * om * om * (1.0f + 4.0f * t); // (1-t)^4 * (1+4t)
}

// ==============================
// Helper: Interpolate CSRBF at a single point
// ==============================
bool interpolateCSRBFAtPoint(
    cv::flann::Index& flannIndex,  // ← 移除了 const
    const std::vector<cv::Point2f>& controlPts,
    const std::vector<float>& values,
    float gx, float gy,
    float supportRadius,
    float& outValue)
{
    constexpr int MIN_NEIGHBORS = 6;
    const int maxK = static_cast<int>(controlPts.size());
    if (maxK == 0) {
        outValue = std::numeric_limits<float>::quiet_NaN();
        return false;
    }

    cv::Mat query = (cv::Mat_<float>(1, 2) << gx, gy);
    std::vector<std::vector<int>> indices;
    std::vector<std::vector<float>> dists;

    flannIndex.radiusSearch(query, indices, dists,
        supportRadius, maxK,
        cv::flann::SearchParams(32));

    if (indices.empty() || indices[0].size() < MIN_NEIGHBORS) {
        outValue = std::numeric_limits<float>::quiet_NaN();
        return false;
    }

    const auto& idxList = indices[0];
    const int n = static_cast<int>(idxList.size());

    // Build local point list
    std::vector<cv::Point2f> localPts(n);
    for (int i = 0; i < n; ++i) {
        localPts[i] = controlPts[idxList[i]];
    }

    // Build Phi (n x n)
    cv::Mat Phi = cv::Mat::zeros(n, n, CV_64F);
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float dx = localPts[i].x - localPts[j].x;
            float dy = localPts[i].y - localPts[j].y;
            float r = std::sqrt(dx * dx + dy * dy);
            double phi_val = wendlandC2(r, supportRadius);
            Phi.at<double>(i, j) = phi_val;
            if (i != j) Phi.at<double>(j, i) = phi_val;
        }
    }

    // Build P = [1, x, y] (n x 3)
    cv::Mat P = cv::Mat::ones(n, 3, CV_64F);
    for (int i = 0; i < n; ++i) {
        P.at<double>(i, 1) = localPts[i].x;
        P.at<double>(i, 2) = localPts[i].y;
    }

    // RHS: [values; 0,0,0]
    cv::Mat rhs(n + 3, 1, CV_64F);
    for (int i = 0; i < n; ++i) {
        rhs.at<double>(i, 0) = static_cast<double>(values[idxList[i]]);
    }
    rhs.at<double>(n, 0) = 0.0;
    rhs.at<double>(n + 1, 0) = 0.0;
    rhs.at<double>(n + 2, 0) = 0.0;

    // System matrix A = [Phi  P; P^T  0]
    cv::Mat A = cv::Mat::zeros(n + 3, n + 3, CV_64F);
    Phi.copyTo(A(cv::Rect(0, 0, n, n)));
    P.copyTo(A(cv::Rect(n, 0, 3, n)));
    cv::Mat PT;
    cv::transpose(P, PT);
    PT.copyTo(A(cv::Rect(0, n, n, 3)));
    // bottom-right 3x3 remains zero

    // Solve
    cv::Mat coeffs;
    bool solved = cv::solve(A, rhs, coeffs, cv::DECOMP_LU);
    if (!solved) {
        outValue = std::numeric_limits<float>::quiet_NaN();
        return false;
    }

    // Evaluate
    double fx = 0.0;
    for (int i = 0; i < n; ++i) {
        float dx = gx - localPts[i].x;
        float dy = gy - localPts[i].y;
        float r = std::sqrt(dx * dx + dy * dy);
        double phi_val = wendlandC2(r, supportRadius);
        fx += coeffs.at<double>(i, 0) * phi_val;
    }
    fx += coeffs.at<double>(n, 0) +
        coeffs.at<double>(n + 1, 0) * gx +
        coeffs.at<double>(n + 2, 0) * gy;

    outValue = static_cast<float>(fx);
    return true;
}

// ==============================
// Main Function 1: Save to GeoTIFF
// ==============================
bool computeDenseFromPointsAndSaveGeoTIFF(
    const std::vector<cv::Point2f>& prevPts,
    const std::vector<cv::Point2f>& currPts,
    int imgWidth, int imgHeight,
    GeoTIFFWriter& writer,
    InterpMethod method,
    double kernelSigma,
    double kernelRadius,
    int blockWidth,
    int blockHeight)
{
    if (prevPts.size() != currPts.size()) return false;
    if (prevPts.empty()) return false;

    // Compute displacement vectors
    std::vector<cv::Point2f> validPrev;
    std::vector<float> pvx, pvy;
    for (size_t i = 0; i < prevPts.size(); ++i) {
        float dx = currPts[i].x - prevPts[i].x;
        float dy = currPts[i].y - prevPts[i].y;
        // Optional: filter outliers here
        validPrev.push_back(prevPts[i]);
        pvx.push_back(dx);
        pvy.push_back(dy);
    }

    if (validPrev.empty()) return false;

    // Global methods (e.g., TPS_GLOBAL) — not optimized here
    if (method == INTERP_TPS_GLOBAL) {
        // TODO: Keep original global TPS implementation
        // This version focuses on local methods
        // You can call your existing TPS code here
        return false; // placeholder
    }

    // Local methods: IDW_LOCAL or CSRBF
    if (method == INTERP_IDW_LOCAL || method == INTERP_CSRBF) {
        // Build FLANN index
        cv::Mat features(static_cast<int>(validPrev.size()), 2, CV_32F);
        for (size_t i = 0; i < validPrev.size(); ++i) {
            features.at<float>(static_cast<int>(i), 0) = validPrev[i].x;
            features.at<float>(static_cast<int>(i), 1) = validPrev[i].y;
        }
        cv::flann::KDTreeIndexParams indexParams(4);
        cv::flann::Index flannIndex(features, indexParams);

        float radius = static_cast<float>(kernelRadius > 0 ? kernelRadius : 50.0);
        const float eps = 1e-9f;
        const float twoSigma2 = 2.0f * static_cast<float>(kernelSigma) * static_cast<float>(kernelSigma);

        for (int by = 0; by < imgHeight; by += blockHeight) {
            for (int bx = 0; bx < imgWidth; bx += blockWidth) {
                int w = std::min(blockWidth, imgWidth - bx);
                int h = std::min(blockHeight, imgHeight - by);

                std::vector<float> bufX(w * h, std::numeric_limits<float>::quiet_NaN());
                std::vector<float> bufY(w * h, std::numeric_limits<float>::quiet_NaN());

                for (int yy = 0; yy < h; ++yy) {
                    for (int xx = 0; xx < w; ++xx) {
                        float gx = static_cast<float>(bx + xx);
                        float gy = static_cast<float>(by + yy);

                        if (method == INTERP_IDW_LOCAL) {
                            cv::Mat query = (cv::Mat_<float>(1, 2) << gx, gy);
                            std::vector<std::vector<int>> indices;
                            std::vector<std::vector<float>> dists;
                            flannIndex.radiusSearch(query, indices, dists, radius,
                                static_cast<int>(validPrev.size()),
                                cv::flann::SearchParams(32));

                            if (!indices.empty() && !indices[0].empty()) {
                                double numX = 0.0, numY = 0.0, den = 0.0;
                                const auto& idxList = indices[0];
                                const auto& distList = dists[0];
                                for (size_t j = 0; j < idxList.size(); ++j) {
                                    int id = idxList[j];
                                    float r2 = distList[j] + eps; // squared distance
                                    float wgt = std::exp(-r2 / twoSigma2);
                                    numX += pvx[id] * wgt;
                                    numY += pvy[id] * wgt;
                                    den += wgt;
                                }
                                if (den > eps) {
                                    bufX[yy * w + xx] = static_cast<float>(numX / den);
                                    bufY[yy * w + xx] = static_cast<float>(numY / den);
                                }
                            }
                        }
                        else if (method == INTERP_CSRBF) {
                            interpolateCSRBFAtPoint(flannIndex, validPrev, pvx, gx, gy, radius, bufX[yy * w + xx]);
                            interpolateCSRBFAtPoint(flannIndex, validPrev, pvy, gx, gy, radius, bufY[yy * w + xx]);
                        }
                    }
                }

                const float* ptrs[2] = { bufX.data(), bufY.data() };
                if (!writer.writeBlock(bx, by, w, h, ptrs)) {
                    return false;
                }
            }
        }
        return true;
    }

    return false; // unsupported method
}

// ==============================
// Main Function 2: Output to Blocks
// ==============================
bool computeDenseToBlocks(
    const std::vector<cv::Point2f>& prevPts,
    const std::vector<cv::Point2f>& currPts,
    int imgWidth, int imgHeight,
    std::vector<std::vector<std::vector<float>>>& denseBlocksX,
    std::vector<std::vector<std::vector<float>>>& denseBlocksY,
    InterpMethod method,
    double kernelSigma,
    double kernelRadius,
    int blockWidth,
    int blockHeight)
{
    if (prevPts.size() != currPts.size()) return false;
    size_t numBlocksY = (imgHeight + blockHeight - 1) / blockHeight;
    size_t numBlocksX = (imgWidth + blockWidth - 1) / blockWidth;
    denseBlocksX.assign(numBlocksY, std::vector<std::vector<float>>(numBlocksX));
    denseBlocksY.assign(numBlocksY, std::vector<std::vector<float>>(numBlocksX));

    if (prevPts.empty()) return true;

    std::vector<cv::Point2f> validPrev;
    std::vector<float> pvx, pvy;
    for (size_t i = 0; i < prevPts.size(); ++i) {
        float dx = currPts[i].x - prevPts[i].x;
        float dy = currPts[i].y - prevPts[i].y;
        validPrev.push_back(prevPts[i]);
        pvx.push_back(dx);
        pvy.push_back(dy);
    }

    if (method == INTERP_TPS_GLOBAL) {
        // Placeholder for global method
        return false;
    }

    if (method == INTERP_IDW_LOCAL || method == INTERP_CSRBF) {
        cv::Mat features(static_cast<int>(validPrev.size()), 2, CV_32F);
        for (size_t i = 0; i < validPrev.size(); ++i) {
            features.at<float>(static_cast<int>(i), 0) = validPrev[i].x;
            features.at<float>(static_cast<int>(i), 1) = validPrev[i].y;
        }
        cv::flann::KDTreeIndexParams indexParams(4);
        cv::flann::Index flannIndex(features, indexParams);

        float radius = static_cast<float>(kernelRadius > 0 ? kernelRadius : 50.0);
        const float eps = 1e-9f;
        const float twoSigma2 = 2.0f * static_cast<float>(kernelSigma) * static_cast<float>(kernelSigma);

        for (int by = 0; by < imgHeight; by += blockHeight) {
            int blockY = by / blockHeight;
            for (int bx = 0; bx < imgWidth; bx += blockWidth) {
                int blockX = bx / blockWidth;
                int w = std::min(blockWidth, imgWidth - bx);
                int h = std::min(blockHeight, imgHeight - by);

                denseBlocksX[blockY][blockX].assign(w * h, std::numeric_limits<float>::quiet_NaN());
                denseBlocksY[blockY][blockX].assign(w * h, std::numeric_limits<float>::quiet_NaN());

                for (int yy = 0; yy < h; ++yy) {
                    for (int xx = 0; xx < w; ++xx) {
                        float gx = static_cast<float>(bx + xx);
                        float gy = static_cast<float>(by + yy);

                        if (method == INTERP_IDW_LOCAL) {
                            cv::Mat query = (cv::Mat_<float>(1, 2) << gx, gy);
                            std::vector<std::vector<int>> indices;
                            std::vector<std::vector<float>> dists;
                            flannIndex.radiusSearch(query, indices, dists, radius,
                                static_cast<int>(validPrev.size()),
                                cv::flann::SearchParams(32));

                            if (!indices.empty() && !indices[0].empty()) {
                                double numX = 0.0, numY = 0.0, den = 0.0;
                                const auto& idxList = indices[0];
                                const auto& distList = dists[0];
                                for (size_t j = 0; j < idxList.size(); ++j) {
                                    int id = idxList[j];
                                    float r2 = distList[j] + eps;
                                    float wgt = std::exp(-r2 / twoSigma2);
                                    numX += pvx[id] * wgt;
                                    numY += pvy[id] * wgt;
                                    den += wgt;
                                }
                                if (den > eps) {
                                    denseBlocksX[blockY][blockX][yy * w + xx] = static_cast<float>(numX / den);
                                    denseBlocksY[blockY][blockX][yy * w + xx] = static_cast<float>(numY / den);
                                }
                            }
                        }
                        else if (method == INTERP_CSRBF) {
                            interpolateCSRBFAtPoint(flannIndex, validPrev, pvx, gx, gy, radius,
                                denseBlocksX[blockY][blockX][yy * w + xx]);
                            interpolateCSRBFAtPoint(flannIndex, validPrev, pvy, gx, gy, radius,
                                denseBlocksY[blockY][blockX][yy * w + xx]);
                        }
                    }
                }
            }
        }
        return true;
    }

    return false;
}