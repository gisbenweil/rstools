#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include <memory>
#include <GDALImageBase.h>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <chrono>
#include <filesystem>
#include "../RSTools/ImageBlockWriter.h"
#include "GDALImageBase.h"
#include <string>

// 工具函数：获取数据类型大小
static size_t getDataTypeSize(ImageDataType type) {
    switch (type) {
    case Byte: return sizeof(unsigned char);
    case UInt16: return sizeof(unsigned short);
    case Int16: return sizeof(short);
    case UInt32: return sizeof(unsigned int);
    case Int32: return sizeof(int);
    case Float32: return sizeof(float);
    case Float64: return sizeof(double);
    default: return 0;
    }
};

// 光流结果结构
struct DenseOpticalFlowResult {
    cv::Mat flowX;      // X方向光流
    cv::Mat flowY;      // Y方向光流
    cv::Mat magnitude;  // 光流幅度
    cv::Mat angle;      // 光流方向
    bool success;
    std::string errorMessage;
};
struct SparseOpticalFlowResult {
    std::vector<cv::Point2f> prevPoints; // 前一帧点
    std::vector<cv::Point2f> currPoints; // 当前帧点
    std::vector<uchar> status;           // 跟踪状态
    std::vector<float> err;              // 跟踪误差
    bool success;
    std::string errorMessage;
};

DenseOpticalFlowResult calculateOpticalFlowStandard(const ReadResult& prevBlock,
    const ReadResult& currBlock,
    double pyramidScale = 0.5,
    int pyramidLevels = 3,
    int windowSize = 15,
    int iterations = 3,
    int polyN = 5,
    double polySigma = 1.1);

// 辅助函数：将ReadResult转换为OpenCV Mat
cv::Mat convertToCVMat(const ReadResult& result, int bandIndex = 0);

cv::Mat toFloatMat(const ReadResult* res, int bandIndex = 0);
std::vector<cv::Point2f> detectHybridFeatures(const cv::Mat& gray);

class OpticalFlowOffset {
private:
    cv::Mat prev_gray, curr_gray;
    std::vector<cv::Point2f> prev_points, curr_points;
    cv::Size winSize;
    int maxLevel;
    cv::TermCriteria criteria;
public:
    OpticalFlowOffset() : winSize(21, 21),
        maxLevel(3),
        criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01) {
    }
    SparseOpticalFlowResult calculateOpticalFlowOffset(const ReadResult& prevBlock,
        const ReadResult& currBlock);

    enum InterpMethod {
        TPS_GLOBAL = 0,
        IDW_LOCAL = 1,
        KNN_KRIGING_APPROX = 2
    };

    bool computeDenseFromPointsAndSaveGeoTIFF(
        const std::vector<cv::Point2f>& prevPts,
        const std::vector<cv::Point2f>& currPts,
        const std::vector<uchar>& status,
        int imgWidth,
        int imgHeight,
        int blockWidth,
        int blockHeight,
        const std::string& outPath,
        const GeoTransform& gt,
        const std::string& projectionWkt,
        InterpMethod method = TPS_GLOBAL,
        float kernelSigma = 10.0f,
        int kernelRadius = 15,
        double regularization = 1e-3,
        int maxGlobalPoints = 1500);

    // Compute dense offsets into per-block buffers (no file I/O).
    // blocksOut: list of blocks as tuples (bx,by,w,h) in row-major order
    // bufXOut/bufYOut: per-block float buffers (row-major w*h)
    bool computeDenseToBlocks(
        const std::vector<cv::Point2f>& prevPts,
        const std::vector<cv::Point2f>& currPts,
        const std::vector<uchar>& status,
        int imgWidth,
        int imgHeight,
        int blockWidth,
        int blockHeight,
        std::vector<std::tuple<int,int,int,int>>& blocksOut,
        std::vector<std::vector<float>>& bufXOut,
        std::vector<std::vector<float>>& bufYOut,
        InterpMethod method = TPS_GLOBAL,
        float kernelSigma = 10.0f,
        int kernelRadius = 15,
        double regularization = 1e-3,
        int maxGlobalPoints = 1500);

    // 新增：从稀疏匹配点构建密集偏移并按块保存
    // blockWidth/blockHeight: 每个保存块的像素大小
    // outDir: 输出目录（必须存在）
    // kernelSigma/kernelRadius: 插值核参数（sigma, 半径）
    // New: compute dense offsets and save as GeoTIFF using ImageBlockWriter
    // outPath: output GeoTIFF path
    // gt/proj: geotransform and projection for output
    bool computeDenseOffsetsAndSaveBlocks(const ReadResult& prevBlock,
        const ReadResult& currBlock,
        int blockWidth,
        int blockHeight,
        const std::string& outPath,
        const GeoTransform& gt,
        const std::string& projectionWkt,
        float kernelSigma = 10.0f,
        int kernelRadius = 15);
};