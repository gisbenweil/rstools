#pragma once
#ifndef DENSE_INTERPOLATION_H
#define DENSE_INTERPOLATION_H

#include <vector>
#include <opencv2/core.hpp>

// Assume you have this defined elsewhere, e.g.:
enum InterpMethod {
    INTERP_TPS_GLOBAL,
    INTERP_IDW_LOCAL,
    INTERP_CSRBF
};

// Forward declaration (you must provide actual implementation)
class GeoTIFFWriter {
public:
    bool writeBlock(int x, int y, int w, int h, const float* ptrs[2]);
    // ... other members
};

bool computeDenseFromPointsAndSaveGeoTIFF(
    const std::vector<cv::Point2f>& prevPts,
    const std::vector<cv::Point2f>& currPts,
    int imgWidth, int imgHeight,
    GeoTIFFWriter& writer,
    InterpMethod method = INTERP_IDW_LOCAL,
    double kernelSigma = 10.0,
    double kernelRadius = 50.0,
    int blockWidth = 256,
    int blockHeight = 256);

bool computeDenseToBlocks(
    const std::vector<cv::Point2f>& prevPts,
    const std::vector<cv::Point2f>& currPts,
    int imgWidth, int imgHeight,
    std::vector<std::vector<std::vector<float>>>& denseBlocksX,
    std::vector<std::vector<std::vector<float>>>& denseBlocksY,
    InterpMethod method = INTERP_IDW_LOCAL,
    double kernelSigma = 10.0,
    double kernelRadius = 50.0,
    int blockWidth = 256,
    int blockHeight = 256);

#endif // DENSE_INTERPOLATION_H