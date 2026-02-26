#include "optflowoffsets.h"
#include "../RSTools/ImageBlockWriter.h"
#include "../RSTools/ImageBlockReader.h"
#include "../RSTools/GDALImageReader.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <map>
#include <opencv2/flann.hpp>

using namespace std;

static double tpsU(double r) {
	if (r <= 1e-12) return 0.0;
	double r2 = r * r;
	return r2 * log(r2);
}

// ---------------------------------------------------------------------------
// Reconstruct pre-image using dense offsets stored in a GeoTIFF (offsetsPath).
// For each block in offsets.tif, read the corresponding offsets (dx,dy),
// read the necessary region from the pre-image (prePath) and perform
// inverse mapping (dst <- src at dst - offset) and write output block to outPath.
// This implementation assumes offsets image and pre-image share the same
// pixel grid / alignment.
// ---------------------------------------------------------------------------

static float bilinearSampleFloat(const cv::Mat& img, double x, double y) {
	if (x < 0 || y < 0 || x >= img.cols - 1 || y >= img.rows - 1) {
		int ix = static_cast<int>(std::floor(x + 0.5));
		int iy = static_cast<int>(std::floor(y + 0.5));
		if (ix >= 0 && iy >= 0 && ix < img.cols && iy < img.rows) return img.at<float>(iy, ix);
		return std::numeric_limits<float>::quiet_NaN();
	}
	int x0 = static_cast<int>(std::floor(x));
	int y0 = static_cast<int>(std::floor(y));
	int x1 = x0 + 1;
	int y1 = y0 + 1;
	float dx = static_cast<float>(x - x0);
	float dy = static_cast<float>(y - y0);
	float v00 = img.at<float>(y0, x0);
	float v10 = img.at<float>(y0, x1);
	float v01 = img.at<float>(y1, x0);
	float v11 = img.at<float>(y1, x1);
	float v0 = v00 * (1.0f - dx) + v10 * dx;
	float v1 = v01 * (1.0f - dx) + v11 * dx;
	return v0 * (1.0f - dy) + v1 * dy;
}

static bool copyBandToFloat(const ReadResult* rr, int bandIndex, std::vector<float>& out) {
	if (!rr || !rr->success) return false;
	if (bandIndex < 0 || bandIndex >= rr->bands) return false;
	int w = rr->width, h = rr->height; size_t n = static_cast<size_t>(w) * h;
	out.assign(n, std::numeric_limits<float>::quiet_NaN());
	switch (rr->dataType) {
	case ImageDataType::Float32: {
		const float* src = rr->getBandData<float>(bandIndex);
		if (!src) return false;
		for (size_t i = 0; i < n; ++i) out[i] = src[i];
		return true;
	}
	case ImageDataType::Float64: {
		const double* src = rr->getBandData<double>(bandIndex);
		if (!src) return false;
		for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(src[i]);
		return true;
	}
	case ImageDataType::Byte: {
		const uint8_t* src = rr->getBandData<uint8_t>(bandIndex);
		if (!src) return false;
		for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(src[i]);
		return true;
	}
	case ImageDataType::UInt16: {
		const uint16_t* src = rr->getBandData<uint16_t>(bandIndex);
		if (!src) return false;
		for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(src[i]);
		return true;
	}
	case ImageDataType::Int16: {
		const int16_t* src = rr->getBandData<int16_t>(bandIndex);
		if (!src) return false;
		for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(src[i]);
		return true;
	}
	case ImageDataType::Int32: {
		const int32_t* src = rr->getBandData<int32_t>(bandIndex);
		if (!src) return false;
		for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(src[i]);
		return true;
	}
	case ImageDataType::UInt32: {
		const uint32_t* src = rr->getBandData<uint32_t>(bandIndex);
		if (!src) return false;
		for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(src[i]);
		return true;
	}
	default:
		return false;
	}
}

extern "C" bool RSTools_ReconstructFromOffsets(const char* offsetsPath, const char* prePath, const char* outPath, int blockWidth, int blockHeight) {
	if (!offsetsPath || !prePath || !outPath) return false;

	ImageBlockReader offsReader(offsetsPath, std::max(1, blockWidth), std::max(1, blockHeight), 0);
	if (!offsReader.isOpen()) return false;

	ImageInfo* preInfo = RSTools_GetImageInfo(prePath);
	if (!preInfo) return false;

	int outW = offsReader.imageWidth();
	int outH = offsReader.imageHeight();

	ImageBlockWriter writer(outPath, outW, outH, preInfo->bands, preInfo->geoTransform, preInfo->projection);
	if (!writer.isOpen()) { RSTools_DestroyImageInfo(preInfo); return false; }

	ReadResult* offsRR = nullptr; BlockSpec spec;
	while (offsReader.next(&offsRR, &spec)) {
		if (!offsRR || !offsRR->success) { if (offsRR) RSTools_DestroyReadResult(offsRR); continue; }

		int bw = offsRR->width; int bh = offsRR->height;
		if (offsRR->bands < 2) { RSTools_DestroyReadResult(offsRR); continue; }

		std::vector<float> offX, offY;
		if (!copyBandToFloat(offsRR, 0, offX) || !copyBandToFloat(offsRR, 1, offY)) { RSTools_DestroyReadResult(offsRR); continue; }

		// compute sampling region in pre image by sampling offsets
		std::vector<double> minXs, minYs, maxXs, maxYs;
		int stepY = std::max(1, bh / 10); int stepX = std::max(1, bw / 10);
		for (int yy = 0; yy < bh; yy += stepY) {
			for (int xx = 0; xx < bw; xx += stepX) {
				size_t idx = yy * bw + xx;
				float dx = offX[idx]; float dy = offY[idx];
				if (std::isnan(dx) || std::isnan(dy)) continue;
				double outAbsX = spec.tileX + xx;
				double outAbsY = spec.tileY + yy;
				double srcX = outAbsX - dx;
				double srcY = outAbsY - dy;
				minXs.push_back(srcX); minYs.push_back(srcY); maxXs.push_back(srcX); maxYs.push_back(srcY);
			}
		}

		if (minXs.empty()) {
			// write nodata block
			std::vector<const float*> bandPtrs(preInfo->bands, nullptr);
			writer.writeBlock(spec.tileX, spec.tileY, spec.tileWidth, spec.tileHeight, bandPtrs.data());
			RSTools_DestroyReadResult(offsRR);
			continue;
		}

		double readMinX = *std::min_element(minXs.begin(), minXs.end()) - 10;
		double readMinY = *std::min_element(minYs.begin(), minYs.end()) - 10;
		double readMaxX = *std::max_element(maxXs.begin(), maxXs.end()) + 10;
		double readMaxY = *std::max_element(maxYs.begin(), maxYs.end()) + 10;

		int readX = std::max(0, static_cast<int>(std::floor(readMinX)));
		int readY = std::max(0, static_cast<int>(std::floor(readMinY)));
		int readW = std::min(preInfo->width - readX, static_cast<int>(std::ceil(readMaxX)) - readX);
		int readH = std::min(preInfo->height - readY, static_cast<int>(std::ceil(readMaxY)) - readY);

		if (readW <= 0 || readH <= 0) {
			std::vector<const float*> bandPtrs(preInfo->bands, nullptr);
			writer.writeBlock(spec.tileX, spec.tileY, spec.tileWidth, spec.tileHeight, bandPtrs.data());
			RSTools_DestroyReadResult(offsRR);
			continue;
		}

		ReadResult* preRR = RSTools_ReadImage(prePath, readX, readY, readW, readH);
		if (!preRR || !preRR->success) {
			if (preRR) RSTools_DestroyReadResult(preRR);
			std::vector<const float*> bandPtrs(preInfo->bands, nullptr);
			writer.writeBlock(spec.tileX, spec.tileY, spec.tileWidth, spec.tileHeight, bandPtrs.data());
			RSTools_DestroyReadResult(offsRR);
			continue;
		}

		// convert preRR bands to float mats
		std::vector<cv::Mat> preBandsFloat;
		for (int b = 0; b < preRR->bands && b < preInfo->bands; ++b) {
			std::vector<float> tmp; if (!copyBandToFloat(preRR, b, tmp)) { tmp.assign(static_cast<size_t>(readW) * readH, std::numeric_limits<float>::quiet_NaN()); }
			cv::Mat m(readH, readW, CV_32F);
			for (int ry = 0; ry < readH; ++ry) for (int rx = 0; rx < readW; ++rx) m.at<float>(ry, rx) = tmp[ry * readW + rx];
			preBandsFloat.push_back(std::move(m));
		}

		// prepare output bands
		int outBW = spec.tileWidth; int outBH = spec.tileHeight;
		std::vector<std::vector<float>> outBands(preInfo->bands, std::vector<float>(static_cast<size_t>(outBW) * outBH, std::numeric_limits<float>::quiet_NaN()));

		// inverse mapping
		for (int y = 0; y < outBH; ++y) {
			for (int x = 0; x < outBW; ++x) {
				int localIdx = y * bw + x; // careful: bw may equal outBW but spec.readWidth may differ; use offset coords
				int srcIdxX = (y * bw + x);
				size_t idxOff = static_cast<size_t>(y) * bw + x;
				if (idxOff >= offX.size() || idxOff >= offY.size()) continue;
				float dx = offX[idxOff]; float dy = offY[idxOff];
				if (std::isnan(dx) || std::isnan(dy)) continue;
				double outAbsX = spec.tileX + x;
				double outAbsY = spec.tileY + y;
				double srcAbsX = outAbsX - dx;
				double srcAbsY = outAbsY - dy;
				double localX = srcAbsX - readX;
				double localY = srcAbsY - readY;
				// sample each band
				for (int b = 0; b < preBandsFloat.size(); ++b) {
					float v = bilinearSampleFloat(preBandsFloat[b], localX, localY);
					if (!std::isnan(v)) outBands[b][y * outBW + x] = v;
				}
			}
		}

		// prepare pointers and write
		std::vector<const float*> ptrs(preInfo->bands, nullptr);
		for (int b = 0; b < preInfo->bands; ++b) ptrs[b] = outBands[b].data();
		writer.writeBlock(spec.tileX, spec.tileY, spec.tileWidth, spec.tileHeight, ptrs.data());

		RSTools_DestroyReadResult(preRR);
		RSTools_DestroyReadResult(offsRR);
	}

	RSTools_DestroyImageInfo(preInfo);
	return true;
}

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
bool OpticalFlowOffset::computeDenseFromPointsAndSaveGeoTIFF(
    const std::vector<cv::Point2f>& prevPts,
    const std::vector<cv::Point2f>& currPts,
    int imgWidth, int imgHeight,
    ImageBlockWriter& writer,
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
        cv::flann::Index flannIndex;//(features, indexParams);
		flannIndex.build(features, indexParams);

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
                            //std::vector<std::vector<int>> indices;
                            //std::vector<std::vector<float>> dists;
                            cv::Mat indices,dists;
                            flannIndex.knnSearch(query, indices, dists,static_cast<int>(10),              // k nearest
                                cv::flann::SearchParams(50));

                            if (!indices.empty() && !indices.row(0).empty()) {
                                double numX = 0.0, numY = 0.0, den = 0.0;
                                const auto& idxList = indices.row(0);
                                const auto& distList = dists.row(0);
                                for (size_t j = 0; j < indices.cols; ++j) {
                                    int id = idxList.at<int>(j);
                                    float r2 = distList.at<float>(j) + eps; // squared distance
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