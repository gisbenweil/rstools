#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include <memory>
#include <cstring>
#include "RSToolsExport.h"
#include "GDALImageBase.h"


// 假设的ImageDataType枚举
enum ImageDataType {
    DT_Byte,
    DT_UInt16,
    DT_Int16,
    DT_UInt32,
    DT_Int32,
    DT_Float32,
    DT_Float64
};
// 工具函数：获取数据类型大小
size_t getDataTypeSize(ImageDataType type) {
    switch (type) {
    case DT_Byte: return sizeof(unsigned char);
    case DT_UInt16: return sizeof(unsigned short);
    case DT_Int16: return sizeof(short);
    case DT_UInt32: return sizeof(unsigned int);
    case DT_Int32: return sizeof(int);
    case DT_Float32: return sizeof(float);
    case DT_Float64: return sizeof(double);
    default: return 0;
    }
};

// 光流结果结构
struct OpticalFlowResult {
    cv::Mat flowX;      // X方向光流
    cv::Mat flowY;      // Y方向光流
    cv::Mat magnitude;  // 光流幅度
    cv::Mat angle;      // 光流方向
    bool success;
    std::string errorMessage;
};

// 辅助函数：将ReadResult转换为OpenCV Mat
cv::Mat convertToCVMat(const ReadResult& result, int bandIndex = 0) {
    if (!result.success || !result.data) {
        throw std::runtime_error("Invalid data in ReadResult");
    }

    if (bandIndex >= result.bands) {
        throw std::runtime_error("Band index out of range");
    }

    // 计算波段数据起始位置
    size_t bandOffset = 0;
    if (!result.bandOffsets.empty() && bandIndex < result.bandOffsets.size()) {
        bandOffset = result.bandOffsets[bandIndex];
    }
    else {
        // 如果没有指定偏移量，假设数据是交错存储的
        bandOffset = bandIndex * (result.width * result.height *
            getDataTypeSize(result.dataType));
    }

    char* bandData = static_cast<char*>(result.data) + bandOffset;

    // 根据数据类型创建Mat
    cv::Mat mat;
    int cvType = 0;
    double scale = 1.0;

    switch (result.dataType) {
    case DT_Byte:
        cvType = CV_8UC1;
        mat = cv::Mat(result.height, result.width, cvType, bandData);
        break;

    case DT_UInt16:
        cvType = CV_16UC1;
        scale = 255.0 / 65535.0; // 缩放到0-255范围
        mat = cv::Mat(result.height, result.width, cvType, bandData);
        mat.convertTo(mat, CV_8UC1, scale);
        break;

    case DT_Int16:
        cvType = CV_16SC1;
        mat = cv::Mat(result.height, result.width, cvType, bandData);
        // 将16位有符号转换为8位无符号
        mat.convertTo(mat, CV_8UC1, 0.5, 128);
        break;

    case DT_Float32:
        cvType = CV_32FC1;
        mat = cv::Mat(result.height, result.width, cvType, bandData);
        // 归一化到0-255
        double minVal, maxVal;
        cv::minMaxLoc(mat, &minVal, &maxVal);
        mat = (mat - minVal) * (255.0 / (maxVal - minVal));
        mat.convertTo(mat, CV_8UC1);
        break;

    case DT_Float64:
        cvType = CV_64FC1;
        mat = cv::Mat(result.height, result.width, cvType, bandData);
        // 归一化到0-255
        cv::minMaxLoc(mat, &minVal, &maxVal);
        mat = (mat - minVal) * (255.0 / (maxVal - minVal));
        mat.convertTo(mat, CV_8UC1);
        break;

    default:
        throw std::runtime_error("Unsupported data type");
    }

    // 处理NoData值
    if (!result.noDataValues.empty() && bandIndex < result.noDataValues.size()) {
        double noDataValue = result.noDataValues[bandIndex];
        cv::Mat mask = (mat == noDataValue);
        cv::inpaint(mat, mask, mat, 3, cv::INPAINT_TELEA);
    }

    return mat.clone(); // 返回深拷贝
}

// 计算光流的主函数 - 使用标准视频模块
OpticalFlowResult calculateOpticalFlowStandard(
    const ReadResult& prevBlock,
    const ReadResult& currBlock,
    double pyramidScale = 0.5,
    int pyramidLevels = 3,
    int windowSize = 15,
    int iterations = 3,
    int polyN = 5,
    double polySigma = 1.1) {

    OpticalFlowResult result;

    try {
        // 检查输入有效性
        if (!prevBlock.success || !currBlock.success) {
            result.success = false;
            result.errorMessage = "One or both input blocks are invalid";
            return result;
        }

        if (prevBlock.width != currBlock.width ||
            prevBlock.height != currBlock.height) {
            result.success = false;
            result.errorMessage = "Block dimensions do not match";
            return result;
        }

        // 将数据转换为灰度图像
        cv::Mat prevGray, currGray;

        if (prevBlock.bands > 1) {
            prevGray = convertToCVMat(prevBlock, 0);
            currGray = convertToCVMat(currBlock, 0);
        }
        else {
            prevGray = convertToCVMat(prevBlock, 0);
            currGray = convertToCVMat(currBlock, 0);
        }

        // 确保图像大小一致
        if (prevGray.size() != currGray.size()) {
            cv::resize(currGray, currGray, prevGray.size());
        }

        // 应用高斯模糊减少噪声
        cv::GaussianBlur(prevGray, prevGray, cv::Size(5, 5), 0);
        cv::GaussianBlur(currGray, currGray, cv::Size(5, 5), 0);

        // 计算光流 - 使用Farneback算法
        cv::Mat flow(prevGray.size(), CV_32FC2);

        cv::calcOpticalFlowFarneback(
            prevGray, currGray, flow,
            pyramidScale,    // 金字塔缩放因子
            pyramidLevels,   // 金字塔层数
            windowSize,      // 窗口大小
            iterations,      // 迭代次数
            polyN,           // 多项式邻域大小
            polySigma,       // 多项式标准差
            0               // 标志位
        );

        // 分离X和Y方向的光流
        std::vector<cv::Mat> flowChannels;
        cv::split(flow, flowChannels);
        result.flowX = flowChannels[0];
        result.flowY = flowChannels[1];

        // 计算光流幅度和方向
        cv::cartToPolar(result.flowX, result.flowY,
            result.magnitude, result.angle, true);

        result.success = true;

    }
    catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Error calculating optical flow: ") + e.what();
    }

    return result;
}