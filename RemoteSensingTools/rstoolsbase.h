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

static std::string ImageDataTypeToString(ImageDataType t) {
	using R = ImageDataType;
	switch (t) {
	case R::Byte: return "Byte";
	case R::UInt16: return "UInt16";
	case R::Int16: return "Int16";
	case R::UInt32: return "UInt32";
	case R::Int32: return "Int32";
	case R::Float32: return "Float32";
	case R::Float64: return "Float64";
	default: return "Unknown";
	}
}

// 结果结构：合并边界信息与偏移
struct CombinedBoundsResult {
	// 地理范围（经/投影单位）
	double minX, maxX, minY, maxY;
	// 合并画布的 GeoTransform（取左影像的像素参数）
	GeoTransform combinedGT;
	// 合并画布像素尺寸
	int combinedWidth;
	int combinedHeight;
	// 左影像偏移（像素，浮点与整数）
	double leftOffsetXf, leftOffsetYf;
	int leftOffsetXi, leftOffsetYi;
	// 右影像偏移（像素，浮点与整数）
	double rightOffsetXf, rightOffsetYf;
	int rightOffsetXi, rightOffsetYi;
};

// 计算两幅影像的合并地理边界，并返回每幅影像相对于合并边界的像素偏移。
// 要求：两影像像素大小(pixelWidth, pixelHeight)相同（含符号、旋转也应一致或可接受）。
static CombinedBoundsResult computeCombinedBounds(const ImageInfo* left, const ImageInfo* right) {
	CombinedBoundsResult res{};
	// 计算四角地理坐标（每幅影像）
	auto cornersGeo = [](const ImageInfo* info) {
		std::array<std::pair<double, double>, 4> corners;
		// (0,0), (width,0), (0,height), (width,height)
		info->geoTransform.pixelToGeo(0.0, 0.0, corners[0].first, corners[0].second);
		info->geoTransform.pixelToGeo(info->width, 0.0, corners[1].first, corners[1].second);
		info->geoTransform.pixelToGeo(0.0, info->height, corners[2].first, corners[2].second);
		info->geoTransform.pixelToGeo(info->width, info->height, corners[3].first, corners[3].second);
		return corners;
		};

	auto lc = cornersGeo(left);
	auto rc = cornersGeo(right);

	double lminX = std::min({ lc[0].first, lc[1].first, lc[2].first, lc[3].first });
	double lmaxX = std::max({ lc[0].first, lc[1].first, lc[2].first, lc[3].first });
	double lminY = std::min({ lc[0].second, lc[1].second, lc[2].second, lc[3].second });
	double lmaxY = std::max({ lc[0].second, lc[1].second, lc[2].second, lc[3].second });

	double rminX = std::min({ rc[0].first, rc[1].first, rc[2].first, rc[3].first });
	double rmaxX = std::max({ rc[0].first, rc[1].first, rc[2].first, rc[3].first });
	double rminY = std::min({ rc[0].second, rc[1].second, rc[2].second, rc[3].second });
	double rmaxY = std::max({ rc[0].second, rc[1].second, rc[2].second, rc[3].second });

	// 合并范围（地理坐标）
	res.minX = std::min(lminX, rminX);
	res.maxX = std::max(lmaxX, rmaxX);
	res.minY = std::min(lminY, rminY);
	res.maxY = std::max(lmaxY, rmaxY);
	std::cout << "合并范围：" << res.minX << ", " << res.maxX << ", " << res.minY << ", " << res.maxY << std::endl;

	// 构建合并画布的 GeoTransform：左上角为 (minX, maxY)
	GeoTransform gt;// = left->geoTransform;
	gt.xOrigin = res.minX;
	gt.yOrigin = res.maxY;
	gt.pixelHeight = left->geoTransform.pixelHeight;
	gt.pixelWidth = left->geoTransform.pixelWidth;
	gt.rotationX = left->geoTransform.rotationX;
	gt.rotationY = left->geoTransform.rotationY;

	// 保持 pixelWidth, pixelHeight, rotationX/Y 与 left 一致（已在调用处要求一致）
	res.combinedGT = gt;

	// 计算合并像素尺寸：将右下角 (maxX, minY) 转为像素坐标
	double pxRight = 0.0, pyBottom = 0.0;
	bool ok = res.combinedGT.geoToPixel(res.maxX, res.minY, pxRight, pyBottom);
	if (!ok) {
		// 若转换失败，返回零尺寸
		res.combinedWidth = 0;
		res.combinedHeight = 0;
	}
	else {
		// 向上取整以包含边界
		res.combinedWidth = static_cast<int>(std::ceil(pxRight));
		res.combinedHeight = static_cast<int>(std::ceil(pyBottom));
		if (res.combinedWidth < 0) res.combinedWidth = 0;
		if (res.combinedHeight < 0) res.combinedHeight = 0;
	}

	// 计算左右影像左上角在合并画布中的像素位置
	double lx = 0.0, ly = 0.0;
	double rx = 0.0, ry = 0.0;
	res.combinedGT.geoToPixel(left->geoTransform.xOrigin, left->geoTransform.yOrigin, lx, ly);
	res.combinedGT.geoToPixel(right->geoTransform.xOrigin, right->geoTransform.yOrigin, rx, ry);

	res.leftOffsetXf = lx;
	res.leftOffsetYf = ly;
	res.leftOffsetXi = static_cast<int>(std::ceil(lx));
	res.leftOffsetYi = static_cast<int>(std::ceil(ly));

	res.rightOffsetXf = rx;
	res.rightOffsetYf = ry;
	res.rightOffsetXi = static_cast<int>(std::ceil(rx));
	res.rightOffsetYi = static_cast<int>(std::ceil(ry));

	return res;
}