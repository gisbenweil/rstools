#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include <memory>
#include <GDALImageBase.h>

#include <opencv2/video/tracking.hpp>
#include <chrono>
#include <filesystem>
#include "../RSTools/ImageBlockWriter.h"
#include <string>

enum InterpMethod {
	INTERP_TPS_GLOBAL,
	INTERP_IDW_LOCAL,
	INTERP_CSRBF
};


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



// 辅助函数：将ReadResult转换为OpenCV Mat
cv::Mat convertToCVMat(const ReadResult& result, int bandIndex = 0);

cv::Mat toFloatMat(const ReadResult* res, int bandIndex = 0);
std::vector<cv::Point2f> detectHybridFeatures(const cv::Mat& gray);

void removeOutliersUsingStats(
	std::vector<cv::Point2f>& points1,
	std::vector<cv::Point2f>& points2,
	std::vector<uchar>& status,
	float maxStdDevMultiplier = 2.0f);


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


	bool computeDenseFromPointsAndSaveGeoTIFF(
		const std::vector<cv::Point2f>& prevPts,
		const std::vector<cv::Point2f>& currPts,
		int imgWidth, int imgHeight,
		ImageBlockWriter& writer,
		InterpMethod method,
		double kernelSigma,
		double kernelRadius,
		int blockWidth,
		int blockHeight);

	// Compute dense offsets into per-block buffers (no file I/O).
	// blocksOut: list of blocks as tuples (bx,by,w,h) in row-major order
	// bufXOut/bufYOut: per-block float buffers (row-major w*h)
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
		int blockHeight);


};