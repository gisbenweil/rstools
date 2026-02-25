#pragma once
#include <opencv2/opencv.hpp>
#include <GDALImageBase.h>
using namespace cv;
// 双线性采样（输入为 single-channel CV_32F 图像）
static float bilinearSample(const cv::Mat& img, double x, double y)
{
	if (x < 0 || y < 0 || x >= img.cols - 1 || y >= img.rows - 1) {
		// 边界外直接返回 0
		int ix = static_cast<int>(std::floor(x + 0.5));
		int iy = static_cast<int>(std::floor(y + 0.5));
		if (ix >= 0 && iy >= 0 && ix < img.cols && iy < img.rows) return img.at<float>(iy, ix);
		return 0.0f;
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

static Mat convertTo8Bit(const void* data, int width, int height,
	ImageDataType dataType,
	double minVal = NAN, double maxVal = NAN) {
	Mat result(height, width, CV_8U);

	// 计算最小最大值
	bool autoMinMax = std::isnan(minVal) || std::isnan(maxVal);
	if (autoMinMax) {
		switch (dataType) {
		case ImageDataType::Byte: {
			const uint8_t* ptr = static_cast<const uint8_t*>(data);
			minVal = std::numeric_limits<uint8_t>::max();
			maxVal = std::numeric_limits<uint8_t>::min();
			for (int i = 0; i < width * height; ++i) {
				uint8_t val = ptr[i];
				if (val < minVal) minVal = val;
				if (val > maxVal) maxVal = val;

			}
			break;
		}
		case ImageDataType::UInt16: {
			const uint16_t* ptr = static_cast<const uint16_t*>(data);
			minVal = std::numeric_limits<uint16_t>::max();
			maxVal = std::numeric_limits<uint16_t>::min();
			for (int i = 0; i < width * height; ++i) {
				uint16_t val = ptr[i];
				if (val < minVal) minVal = val;
				if (val > maxVal) maxVal = val;
			}
			break;
		}
								  // 其他数据类型类似处理...
		}
	}

	// 线性拉伸到0-255
	double scale = 255.0 / (maxVal - minVal);

	switch (dataType) {
	case ImageDataType::Byte: {
		const uint8_t* src = static_cast<const uint8_t*>(data);
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				double val = src[y * width + x];
				result.at<uint8_t>(y, x) = static_cast<uint8_t>((val - minVal) * scale);
			}
		}
		break;
	}
							// 其他数据类型类似处理...
	}

	return result;
}

static Mat createRGBImage(const ReadResult& result,
	int redBand, int greenBand, int blueBand) {
	if (result.bands < 3) {
		return Mat();
	}

	Mat rgb(result.height, result.width, CV_8UC3);

	// 获取各波段数据
	Mat r = convertTo8Bit(result.getBandData<uchar>(redBand - 1),
		result.width, result.height, result.dataType);
	Mat g = convertTo8Bit(result.getBandData<uchar>(greenBand - 1),
		result.width, result.height, result.dataType);
	Mat b = convertTo8Bit(result.getBandData<uchar>(blueBand - 1),
		result.width, result.height, result.dataType);

	// 合并通道
	std::vector<Mat> channels = { b, g, r };
	merge(channels, rgb);

	return rgb;
}
