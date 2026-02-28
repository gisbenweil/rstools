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
#include <future>
#include <queue>
#include <condition_variable>
#include <algorithm>
#include <map>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <opencv2/flann.hpp>

using namespace std;

static double tpsU(double r) {
	if (r <= 1e-12) return 0.0;
	double r2 = r * r;
	return r2 * log(r2);
}

// Variant 2: compute only corner offsets per block and bilinearly interpolate inside block
bool OpticalFlowOffset::computeDenseFromPointsAndSaveGeoTIFF2(
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
        validPrev.push_back(prevPts[i]);
        pvx.push_back(dx);
        pvy.push_back(dy);
    }
    if (validPrev.empty()) return false;

    // Build FLANN index
    cv::Mat features(static_cast<int>(validPrev.size()), 2, CV_32F);
    for (size_t i = 0; i < validPrev.size(); ++i) {
        features.at<float>(static_cast<int>(i), 0) = validPrev[i].x;
        features.at<float>(static_cast<int>(i), 1) = validPrev[i].y;
    }
    cv::flann::KDTreeIndexParams indexParams(4);
    cv::flann::Index flannIndex;
    flannIndex.build(features, indexParams);

    float radius = static_cast<float>(kernelRadius > 0 ? kernelRadius : 50.0);
    const float eps = 1e-9f;
    const float twoSigma2 = 2.0f * static_cast<float>(kernelSigma) * static_cast<float>(kernelSigma);
    const int knn = std::min(static_cast<int>(validPrev.size()), 30);

    int blocksX = static_cast<int>(std::ceil(imgWidth / static_cast<double>(blockWidth)));
    int blocksY = static_cast<int>(std::ceil(imgHeight / static_cast<double>(blockHeight)));
    int totalBlocks = std::max(1, blocksX) * std::max(1, blocksY);
    std::atomic<int> processedBlocks(0);
    auto startTime = std::chrono::steady_clock::now();

    for (int by = 0; by < imgHeight; by += blockHeight) {
        for (int bx = 0; bx < imgWidth; bx += blockWidth) {
            int w = std::min(blockWidth, imgWidth - bx);
            int h = std::min(blockHeight, imgHeight - by);

            std::vector<float> bufX(w * h, std::numeric_limits<float>::quiet_NaN());
            std::vector<float> bufY(w * h, std::numeric_limits<float>::quiet_NaN());

            // corner absolute coordinates
            float cx[4];
            float cy[4];
            cx[0] = static_cast<float>(bx);
            cy[0] = static_cast<float>(by);
            cx[1] = static_cast<float>(bx + w - 1);
            cy[1] = static_cast<float>(by);
            cx[2] = static_cast<float>(bx);
            cy[2] = static_cast<float>(by + h - 1);
            cx[3] = static_cast<float>(bx + w - 1);
            cy[3] = static_cast<float>(by + h - 1);

            float cornerX[4];
            float cornerY[4];
            bool cornerValid[4] = { false, false, false, false };

            // compute offsets only at 4 corners
            cv::Mat query(1, 2, CV_32F);
            cv::Mat indices, dists;
            for (int ci = 0; ci < 4; ++ci) {
                float gx = cx[ci];
                float gy = cy[ci];
                query.at<float>(0, 0) = gx;
                query.at<float>(0, 1) = gy;

                if (method == INTERP_IDW_LOCAL) {
                    flannIndex.knnSearch(query, indices, dists, knn, cv::flann::SearchParams());
                    if (!indices.empty() && indices.cols > 0) {
                        double numx = 0.0, numy = 0.0, den = 0.0;
                        int cols = indices.cols;
                        for (int j = 0; j < cols; ++j) {
                            int id = indices.at<int>(0, j);
                            float r2 = dists.at<float>(0, j) + eps;
                            float wgt = std::exp(-r2 / twoSigma2);
                            numx += pvx[id] * wgt;
                            numy += pvy[id] * wgt;
                            den += wgt;
                        }
                        if (den > eps) {
                            cornerX[ci] = static_cast<float>(numx / den);
                            cornerY[ci] = static_cast<float>(numy / den);
                            cornerValid[ci] = true;
                        }
                    }
                }
                else if (method == INTERP_CSRBF) {
                    float vx, vy;
                    bool okx = interpolateCSRBFAtPoint(flannIndex, validPrev, pvx, gx, gy, radius, vx);
                    bool oky = interpolateCSRBFAtPoint(flannIndex, validPrev, pvy, gx, gy, radius, vy);
                    if (okx && oky && !std::isnan(vx) && !std::isnan(vy)) {
                        cornerX[ci] = vx;
                        cornerY[ci] = vy;
                        cornerValid[ci] = true;
                    }
                }
            }

            // fill block by bilinear interpolation of corner values; handle missing corners by renormalizing weights
            for (int yy = 0; yy < h; ++yy) {
                for (int xx = 0; xx < w; ++xx) {
                    float tx = (w == 1) ? 0.0f : static_cast<float>(xx) / static_cast<float>(w - 1);
                    float ty = (h == 1) ? 0.0f : static_cast<float>(yy) / static_cast<float>(h - 1);
                    float w00 = (1.0f - tx) * (1.0f - ty);
                    float w10 = tx * (1.0f - ty);
                    float w01 = (1.0f - tx) * ty;
                    float w11 = tx * ty;
                    float ws[4] = { w00, w10, w01, w11 };
                    double numX = 0.0, numY = 0.0, den = 0.0;
                    for (int ci = 0; ci < 4; ++ci) {
                        if (cornerValid[ci]) {
                            numX += ws[ci] * cornerX[ci];
                            numY += ws[ci] * cornerY[ci];
                            den += ws[ci];
                        }
                    }
                    if (den > eps) {
                        bufX[yy * w + xx] = static_cast<float>(numX / den);
                        bufY[yy * w + xx] = static_cast<float>(numY / den);
                    }
                }
                // optional: per-row block progress
                if ((yy + 1) % std::max(1, h / 10) == 0 || yy + 1 == h) {
                    int rowDone = yy + 1;
                    double pct = 100.0 * rowDone / h;
                    std::cout << "\rBlock (" << bx << "," << by << ") progress: " << static_cast<int>(pct) << "% (" << rowDone << "/" << h << " rows)    " << std::flush;
                    if (rowDone == h) std::cout << std::endl;
                }
            }

            const float* ptrs[2] = { bufX.data(), bufY.data() };
            if (!writer.writeBlock(bx, by, w, h, ptrs)) {
                return false;
            }

            int proc = ++processedBlocks;
            if (proc % 10 == 0 || proc == totalBlocks) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - startTime).count();
                double perBlock = (proc > 0) ? (elapsed / proc) : 0.0;
                double remaining = std::max(0, totalBlocks - proc);
                double eta = perBlock * remaining;
                double percent = proc * 100.0 / totalBlocks;
                std::cout << "\rProgress: " << proc << "/" << totalBlocks
                    << " (" << std::fixed << std::setprecision(1) << percent << "%)"
                    << " Elapsed: " << std::fixed << std::setprecision(1) << elapsed << "s"
                    << " ETA: " << std::fixed << std::setprecision(1) << eta << "s    " << std::flush;
                if (proc == totalBlocks) std::cout << std::endl;
            }
        }
    }
    return true;
}

// Helper: convert float bands to target data type and write via writer
static bool convertAndWriteBlock(ImageBlockWriter& writer,
    const ImageInfo* preInfo,
    const std::vector<std::vector<float>>& outBands,
    int outBW, int outBH,
    const BlockSpec& tSpec)
{
    ImageDataType outType = preInfo->dataType;
    auto getNoData = [&](int bandIndex, double fallback)->double {
        if (bandIndex >= 0 && bandIndex < static_cast<int>(preInfo->noDataValues.size())) return preInfo->noDataValues[bandIndex];
        return fallback;
    };

    switch (outType) {
    case ImageDataType::Byte: {
        std::vector<std::vector<uint8_t>> buf(preInfo->bands);
        for (int b = 0; b < preInfo->bands; ++b) {
            buf[b].resize(static_cast<size_t>(outBW) * outBH);
            double nod = getNoData(b, 0.0);
            for (size_t i = 0; i < buf[b].size(); ++i) {
                float v = outBands[b][i];
                if (std::isnan(v)) buf[b][i] = static_cast<uint8_t>(std::max(0.0, std::min(255.0, nod)));
                else buf[b][i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, v)));
            }
        }
        std::vector<const uint8_t*> ptrs(preInfo->bands, nullptr);
        for (int b = 0; b < preInfo->bands; ++b) ptrs[b] = buf[b].data();
        return writer.writeBlock<uint8_t>(tSpec.tileX, tSpec.tileY, tSpec.tileWidth, tSpec.tileHeight, ptrs.data());
    }
    case ImageDataType::UInt16: {
        std::vector<std::vector<uint16_t>> buf(preInfo->bands);
        for (int b = 0; b < preInfo->bands; ++b) {
            buf[b].resize(static_cast<size_t>(outBW) * outBH);
            double nod = getNoData(b, 0.0);
            for (size_t i = 0; i < buf[b].size(); ++i) {
                float v = outBands[b][i];
                if (std::isnan(v)) buf[b][i] = static_cast<uint16_t>(std::max(0.0, std::min(65535.0, nod)));
                else buf[b][i] = static_cast<uint16_t>(std::max(0.0f, std::min(65535.0f, v)));
            }
        }
        std::vector<const uint16_t*> ptrs(preInfo->bands, nullptr);
        for (int b = 0; b < preInfo->bands; ++b) ptrs[b] = buf[b].data();
        return writer.writeBlock<uint16_t>(tSpec.tileX, tSpec.tileY, tSpec.tileWidth, tSpec.tileHeight, ptrs.data());
    }
    case ImageDataType::Int16: {
        std::vector<std::vector<int16_t>> buf(preInfo->bands);
        for (int b = 0; b < preInfo->bands; ++b) {
            buf[b].resize(static_cast<size_t>(outBW) * outBH);
            double nod = getNoData(b, 0.0);
            for (size_t i = 0; i < buf[b].size(); ++i) {
                float v = outBands[b][i];
                if (std::isnan(v)) buf[b][i] = static_cast<int16_t>(std::max((double)std::numeric_limits<int16_t>::min(), std::min((double)std::numeric_limits<int16_t>::max(), nod)));
                else buf[b][i] = static_cast<int16_t>(std::max((double)std::numeric_limits<int16_t>::min(), std::min((double)std::numeric_limits<int16_t>::max(), static_cast<double>(v))));
            }
        }
        std::vector<const int16_t*> ptrs(preInfo->bands, nullptr);
        for (int b = 0; b < preInfo->bands; ++b) ptrs[b] = buf[b].data();
        return writer.writeBlock<int16_t>(tSpec.tileX, tSpec.tileY, tSpec.tileWidth, tSpec.tileHeight, ptrs.data());
    }
    case ImageDataType::UInt32: {
        std::vector<std::vector<uint32_t>> buf(preInfo->bands);
        for (int b = 0; b < preInfo->bands; ++b) {
            buf[b].resize(static_cast<size_t>(outBW) * outBH);
            double nod = getNoData(b, 0.0);
            for (size_t i = 0; i < buf[b].size(); ++i) {
                float v = outBands[b][i];
                if (std::isnan(v)) buf[b][i] = static_cast<uint32_t>(std::max(0.0, std::min(static_cast<double>(std::numeric_limits<uint32_t>::max()), nod)));
                else buf[b][i] = static_cast<uint32_t>(std::max(0.0f, static_cast<float>(std::min(static_cast<double>(std::numeric_limits<uint32_t>::max()), static_cast<double>(v)))));
            }
        }
        std::vector<const uint32_t*> ptrs(preInfo->bands, nullptr);
        for (int b = 0; b < preInfo->bands; ++b) ptrs[b] = buf[b].data();
        return writer.writeBlock<uint32_t>(tSpec.tileX, tSpec.tileY, tSpec.tileWidth, tSpec.tileHeight, ptrs.data());
    }
    case ImageDataType::Int32: {
        std::vector<std::vector<int32_t>> buf(preInfo->bands);
        for (int b = 0; b < preInfo->bands; ++b) {
            buf[b].resize(static_cast<size_t>(outBW) * outBH);
            double nod = getNoData(b, 0.0);
            for (size_t i = 0; i < buf[b].size(); ++i) {
                float v = outBands[b][i];
                if (std::isnan(v)) buf[b][i] = static_cast<int32_t>(std::max(static_cast<double>(std::numeric_limits<int32_t>::min()), std::min(static_cast<double>(std::numeric_limits<int32_t>::max()), nod)));
                else buf[b][i] = static_cast<int32_t>(std::max(static_cast<double>(std::numeric_limits<int32_t>::min()), std::min(static_cast<double>(std::numeric_limits<int32_t>::max()), static_cast<double>(v))));
            }
        }
        std::vector<const int32_t*> ptrs(preInfo->bands, nullptr);
        for (int b = 0; b < preInfo->bands; ++b) ptrs[b] = buf[b].data();
        return writer.writeBlock<int32_t>(tSpec.tileX, tSpec.tileY, tSpec.tileWidth, tSpec.tileHeight, ptrs.data());
    }
    case ImageDataType::Float64: {
        std::vector<std::vector<double>> buf(preInfo->bands);
        for (int b = 0; b < preInfo->bands; ++b) {
            buf[b].resize(static_cast<size_t>(outBW) * outBH);
            double nod = getNoData(b, std::numeric_limits<double>::quiet_NaN());
            for (size_t i = 0; i < buf[b].size(); ++i) {
                float v = outBands[b][i];
                if (std::isnan(v)) buf[b][i] = nod;
                else buf[b][i] = static_cast<double>(v);
            }
        }
        std::vector<const double*> ptrs(preInfo->bands, nullptr);
        for (int b = 0; b < preInfo->bands; ++b) ptrs[b] = buf[b].data();
        return writer.writeBlock<double>(tSpec.tileX, tSpec.tileY, tSpec.tileWidth, tSpec.tileHeight, ptrs.data());
    }
    case ImageDataType::Float32:
    default: {
        std::vector<const float*> ptrs(preInfo->bands, nullptr);
        for (int b = 0; b < preInfo->bands; ++b) ptrs[b] = outBands[b].data();
        return writer.writeBlock<float>(tSpec.tileX, tSpec.tileY, tSpec.tileWidth, tSpec.tileHeight, ptrs.data());
    }
    }
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

    // writer will be created later with the same data type as the pre-image

    // Producer-consumer: read offset tiles one-by-one and push to a bounded queue
    struct OffsetTile {
        BlockSpec spec;
        int bw;
        int bh;
        std::vector<float> offX;
        std::vector<float> offY;
    };

    // Create writer with same data type as pre image
    ImageBlockWriter writer(outPath, outW, outH, preInfo->bands, preInfo->geoTransform, preInfo->dataType, preInfo->projection);
    if (!writer.isOpen()) { RSTools_DestroyImageInfo(preInfo); std::cerr << "[Reconstruct] Failed to open output writer: " << outPath << std::endl; return false; }
    // set nodata values if available
    for (size_t bi = 0; bi < preInfo->noDataValues.size(); ++bi) {
        writer.setBandNoDataValue(static_cast<int>(bi), preInfo->noDataValues[bi]);
    }

    const unsigned int maxThreads = std::max(1u, std::thread::hardware_concurrency());
    const size_t maxQueueSize = std::max<size_t>(4, maxThreads * 2);

    std::queue<OffsetTile> taskQueue;
    std::mutex queueMutex;
    std::condition_variable queueCv;
    bool doneProducing = false;

    std::mutex writerMutex;
    std::atomic<size_t> processedCount{ 0 };
    std::atomic<size_t> totalTasks{ 0 };
    auto startTime = std::chrono::steady_clock::now();

    auto formatDurationStr = [&](double secs)->std::string {
        int s = static_cast<int>(std::round(std::max(0.0, secs)));
        int h = s / 3600;
        int m = (s % 3600) / 60;
        int ss = s % 60;
        std::ostringstream oss;
        if (h > 0) oss << h << "h";
        if (m > 0 || h > 0) oss << m << "m";
        oss << ss << "s";
        return oss.str();
    };

    // worker function
    auto workerFunc = [&](int workerId) {
        while (true) {
            OffsetTile tile;
            {
                std::unique_lock<std::mutex> lk(queueMutex);
                queueCv.wait(lk, [&] { return !taskQueue.empty() || doneProducing; });
                if (taskQueue.empty() && doneProducing) break;
                tile = std::move(taskQueue.front());
                taskQueue.pop();
                lk.unlock();
                queueCv.notify_one();
            }

            const BlockSpec& tSpec = tile.spec;
            int bw = tile.bw; int bh = tile.bh;

            try {
                // compute sampling region in pre image by sampling offsets
                std::vector<double> minXs, minYs, maxXs, maxYs;
                int stepY = std::max(1, bh / 10); int stepX = std::max(1, bw / 10);
                for (int yy = 0; yy < bh; yy += stepY) {
                    for (int xx = 0; xx < bw; xx += stepX) {
                        size_t idx = yy * bw + xx;
                        float dx = tile.offX[idx]; float dy = tile.offY[idx];
                        if (std::isnan(dx) || std::isnan(dy)) continue;
                        double outAbsX = tSpec.tileX + xx;
                        double outAbsY = tSpec.tileY + yy;
                        double srcX = outAbsX - dx;
                        double srcY = outAbsY - dy;
                        minXs.push_back(srcX); minYs.push_back(srcY); maxXs.push_back(srcX); maxYs.push_back(srcY);
                    }
                }

                if (minXs.empty()) {
                    // write nodata block
                    std::vector<const float*> bandPtrs(preInfo->bands, nullptr);
                    {
                        std::lock_guard<std::mutex> lg(writerMutex);
                        bool wok = writer.writeBlock(tSpec.tileX, tSpec.tileY, tSpec.tileWidth, tSpec.tileHeight, bandPtrs.data());
                        if (!wok) std::cerr << "[Reconstruct][Error] Failed to write nodata block at (" << tSpec.tileX << "," << tSpec.tileY << ")" << std::endl;
                    }
                    processedCount.fetch_add(1);
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
                    {
                        std::lock_guard<std::mutex> lg(writerMutex);
                        bool wok = writer.writeBlock(tSpec.tileX, tSpec.tileY, tSpec.tileWidth, tSpec.tileHeight, bandPtrs.data());
                        if (!wok) std::cerr << "[Reconstruct][Error] Invalid read region for block at (" << tSpec.tileX << "," << tSpec.tileY << ")" << std::endl;
                    }
                    processedCount.fetch_add(1);
                    continue;
                }

                ReadResult* preRR = RSTools_ReadImage(prePath, readX, readY, readW, readH);
                if (!preRR || !preRR->success) {
                    if (preRR) RSTools_DestroyReadResult(preRR);
                    std::vector<const float*> bandPtrs(preInfo->bands, nullptr);
                    {
                        std::lock_guard<std::mutex> lg(writerMutex);
                        bool wok = writer.writeBlock(tSpec.tileX, tSpec.tileY, tSpec.tileWidth, tSpec.tileHeight, bandPtrs.data());
                        if (!wok) std::cerr << "[Reconstruct][Error] Failed to write nodata block after read failure at (" << tSpec.tileX << "," << tSpec.tileY << ")" << std::endl;
                    }
                    processedCount.fetch_add(1);
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
                int outBW = tSpec.tileWidth; int outBH = tSpec.tileHeight;
                std::vector<std::vector<float>> outBands(preInfo->bands, std::vector<float>(static_cast<size_t>(outBW) * outBH, std::numeric_limits<float>::quiet_NaN()));

                // inverse mapping
                for (int y = 0; y < outBH; ++y) {
                    for (int x = 0; x < outBW; ++x) {
                        size_t idxOff = static_cast<size_t>(y) * bw + x;
                        if (idxOff >= tile.offX.size() || idxOff >= tile.offY.size()) continue;
                        float dx = tile.offX[idxOff]; float dy = tile.offY[idxOff];
                        if (std::isnan(dx) || std::isnan(dy)) continue;
                        double outAbsX = tSpec.tileX + x;
                        double outAbsY = tSpec.tileY + y;
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

                // prepare pointers and write (serialize writer access)
                bool writeOk = false;
                {
                    std::lock_guard<std::mutex> lg(writerMutex);
                    writeOk = convertAndWriteBlock(writer, preInfo, outBands, outBW, outBH, tSpec);
                }
                if (!writeOk) std::cerr << "[Reconstruct][Error] Failed to write block at (" << tSpec.tileX << "," << tSpec.tileY << ")" << std::endl;

                RSTools_DestroyReadResult(preRR);
                size_t done = processedCount.fetch_add(1) + 1;
                // progress log
                size_t total = totalTasks.load();
                if (done % 10 == 0 || (total > 0 && done == total)) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - startTime).count();
                    double per = elapsed / std::max<size_t>(1, done);
                    double remaining = (total > 0) ? per * (total - done) : 0.0;
                    double percent = (total > 0) ? (double)done * 100.0 / (double)total : 0.0;
                    std::cout << "\r[Reconstruct] " << done << "/" << (total>0?total:0) << " (" << std::fixed << std::setprecision(1) << percent << "%)" << " elapsed: " << formatDurationStr(elapsed) << " ETA: " << formatDurationStr(remaining) << "    " << std::flush;
                    if (total > 0 && done == total) std::cout << std::endl;
                }
            }
            catch (const std::exception& ex) {
                std::cerr << "[Reconstruct][Exception] " << ex.what() << std::endl;
                processedCount.fetch_add(1);
            }
        }
    };

    // start worker threads
    std::vector<std::thread> workers;
    workers.reserve(maxThreads);
    for (unsigned int wi = 0; wi < maxThreads; ++wi) workers.emplace_back(workerFunc, static_cast<int>(wi));

    // producer: read offset tiles and push into queue
    ReadResult* offsRR2 = nullptr; BlockSpec spec2;
    while (offsReader.next(&offsRR2, &spec2)) {
        if (!offsRR2 || !offsRR2->success) { if (offsRR2) RSTools_DestroyReadResult(offsRR2); continue; }

        int bw = offsRR2->width; int bh = offsRR2->height;
        if (offsRR2->bands < 2) { RSTools_DestroyReadResult(offsRR2); continue; }

        std::vector<float> offX, offY;
        if (!copyBandToFloat(offsRR2, 0, offX) || !copyBandToFloat(offsRR2, 1, offY)) { RSTools_DestroyReadResult(offsRR2); continue; }

        OffsetTile tile{ spec2, bw, bh, std::move(offX), std::move(offY) };
        {
            std::unique_lock<std::mutex> lk(queueMutex);
            queueCv.wait(lk, [&] { return taskQueue.size() < maxQueueSize; });
            taskQueue.push(std::move(tile));
            totalTasks.fetch_add(1);
        }
        queueCv.notify_one();
        RSTools_DestroyReadResult(offsRR2);
    }

    // signal completion
    {
        std::lock_guard<std::mutex> lk(queueMutex);
        doneProducing = true;
    }
    queueCv.notify_all();

    // join workers
    for (auto &t : workers) if (t.joinable()) t.join();

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

        // progress tracking
        int blocksX = static_cast<int>(std::ceil(imgWidth / static_cast<double>(blockWidth)));
        int blocksY = static_cast<int>(std::ceil(imgHeight / static_cast<double>(blockHeight)));
        int totalBlocks = std::max(1, blocksX) * std::max(1, blocksY);
        std::atomic<int> processedBlocks(0);
        auto startTime = std::chrono::steady_clock::now();

        // Precompute block list to process sequentially; parallelize inside each block
        struct BlockItem { int bx; int by; int w; int h; };
        std::vector<BlockItem> blocks;
        blocks.reserve(totalBlocks);
        for (int by = 0; by < imgHeight; by += blockHeight) {
            for (int bx = 0; bx < imgWidth; bx += blockWidth) {
                int w = std::min(blockWidth, imgWidth - bx);
                int h = std::min(blockHeight, imgHeight - by);
                blocks.push_back({ bx, by, w, h });
            }
        }

        const unsigned int hwThreads = std::max(1u, std::thread::hardware_concurrency());
        std::mutex writerMutex;
        const int knn = std::min(static_cast<int>(validPrev.size()), 30);

        for (size_t bi = 0; bi < blocks.size(); ++bi) {
            const BlockItem& it = blocks[bi];
            std::vector<float> bufX(it.w * it.h, std::numeric_limits<float>::quiet_NaN());
            std::vector<float> bufY(it.w * it.h, std::numeric_limits<float>::quiet_NaN());

            // divide rows among threads
            int rows = it.h;
            unsigned int threadsForBlock = std::min<unsigned int>(hwThreads, static_cast<unsigned int>(rows));
            if (threadsForBlock == 0) threadsForBlock = 1;

            std::atomic<int> rowsDone(0);
            std::vector<std::thread> workers;
            workers.reserve(threadsForBlock);

            auto rowWorker = [&](int tid, int rowStart, int rowEnd) {
                cv::Mat query(1, 2, CV_32F);
                cv::Mat indices, dists;
                for (int yy = rowStart; yy < rowEnd; ++yy) {
                    for (int xx = 0; xx < it.w; ++xx) {
                        float gx = static_cast<float>(it.bx + xx);
                        float gy = static_cast<float>(it.by + yy);
                        query.at<float>(0, 0) = gx;
                        query.at<float>(0, 1) = gy;

                        if (method == INTERP_IDW_LOCAL) {
                            flannIndex.knnSearch(query, indices, dists, knn, cv::flann::SearchParams());
                            if (!indices.empty() && indices.cols > 0) {
                                double numX = 0.0, numY = 0.0, den = 0.0;
                                int cols = indices.cols;
                                for (int j = 0; j < cols; ++j) {
                                    int id = indices.at<int>(0, j);
                                    float r2 = dists.at<float>(0, j) + eps; // squared distance
                                    float wgt = std::exp(-r2 / twoSigma2);
                                    numX += pvx[id] * wgt;
                                    numY += pvy[id] * wgt;
                                    den += wgt;
                                }
                                if (den > eps) {
                                    bufX[yy * it.w + xx] = static_cast<float>(numX / den);
                                    bufY[yy * it.w + xx] = static_cast<float>(numY / den);
                                }
                            }
                        }
                        else if (method == INTERP_CSRBF) {
                            interpolateCSRBFAtPoint(flannIndex, validPrev, pvx, gx, gy, radius, bufX[yy * it.w + xx]);
                            interpolateCSRBFAtPoint(flannIndex, validPrev, pvy, gx, gy, radius, bufY[yy * it.w + xx]);
                        }
                    }
                    int done = ++rowsDone;
                    // block-internal progress every 10% or every 20 rows
                    if (done % std::max(1, rows / 10) == 0 || (rows <= 20 && done % 5 == 0) || done == rows) {
                        double pct = (100.0 * done) / rows;
                        std::cout << "\rBlock " << bi + 1 << "/" << blocks.size() << ": " << static_cast<int>(pct) << "% (" << done << "/" << rows << " rows)    " << std::flush;
                        if (done == rows) std::cout << std::endl;
                    }
                }
            };

            // spawn workers dividing rows
            int base = rows / static_cast<int>(threadsForBlock);
            int rem = rows % static_cast<int>(threadsForBlock);
            int r0 = 0;
            for (unsigned int t = 0; t < threadsForBlock; ++t) {
                int rcount = base + (t < static_cast<unsigned int>(rem) ? 1 : 0);
                int r1 = r0 + rcount;
                if (rcount > 0) workers.emplace_back(rowWorker, static_cast<int>(t), r0, r1);
                r0 = r1;
            }

            for (auto &th : workers) if (th.joinable()) th.join();

            // write block (serialize writer access)
            {
                std::lock_guard<std::mutex> lg(writerMutex);
                const float* ptrs[2] = { bufX.data(), bufY.data() };
                if (!writer.writeBlock(it.bx, it.by, it.w, it.h, ptrs)) {
                    // writing failure: best effort to continue
                }
            }

            int proc = ++processedBlocks;
            if (proc % 10 == 0 || proc == totalBlocks) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - startTime).count();
                double perBlock = (proc > 0) ? (elapsed / proc) : 0.0;
                double remaining = std::max(0, totalBlocks - proc);
                double eta = perBlock * remaining;
                double percent = proc * 100.0 / totalBlocks;
                std::cout << "\rProgress: " << proc << "/" << totalBlocks
                    << " (" << std::fixed << std::setprecision(1) << percent << "%)"
                    << " Elapsed: " << std::fixed << std::setprecision(1) << elapsed << "s"
                    << " ETA: " << std::fixed << std::setprecision(1) << eta << "s    " << std::flush;
                if (proc == totalBlocks) std::cout << std::endl;
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