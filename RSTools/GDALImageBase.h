#pragma once
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <gdal_priv.h>
#include <gdal.h>
#include <ogrsf_frmts.h>
#include <algorithm>
#include "RSToolsExport.h"



/**
 * @brief 遥感影像数据类型枚举
 */
enum  ImageDataType {
	Unknown = GDT_Unknown,
	Byte = GDT_Byte,           // 8位无符号整数
	UInt16 = GDT_UInt16,       // 16位无符号整数
	Int16 = GDT_Int16,         // 16位有符号整数
	UInt32 = GDT_UInt32,       // 32位无符号整数
	Int32 = GDT_Int32,         // 32位有符号整数
	Float32 = GDT_Float32,     // 32位浮点数
	Float64 = GDT_Float64      // 64位浮点数
};

/**
 * @brief 地理信息结构体
 */
struct GeoTransform {
	double xOrigin;     // 左上角X坐标
	double yOrigin;     // 左上角Y坐标
	double pixelWidth;  // 像素宽度
	double pixelHeight; // 像素高度
	double rotationX;   // X轴旋转
	double rotationY;   // Y轴旋转

	GeoTransform() : xOrigin(0), yOrigin(0), pixelWidth(1),
		pixelHeight(-1), rotationX(0), rotationY(0) {
	}

	/**
	 * @brief 从GDAL的6参数转换矩阵构造
	 */
	static GeoTransform fromGDAL(const double* transform) {
		GeoTransform gt;
		if (transform) {
			gt.xOrigin = transform[0];
			gt.pixelWidth = transform[1];
			gt.rotationY = transform[2];
			gt.yOrigin = transform[3];
			gt.rotationX = transform[4];
			gt.pixelHeight = transform[5];
		}
		return gt;
	}

	/**
	 * @brief 转换为GDAL的6参数数组
	 */
	void toGDAL(double* transform) const {
		if (transform) {
			transform[0] = xOrigin;
			transform[1] = pixelWidth;
			transform[2] = rotationY;
			transform[3] = yOrigin;
			transform[4] = rotationX;
			transform[5] = pixelHeight;
		}
	}

	/**
	 * @brief 地理坐标转像素坐标
	 */
	bool geoToPixel(double geoX, double geoY, double& pixelX, double& pixelY) const {
		double denominator = pixelWidth * pixelHeight - rotationX * rotationY;
		if (std::abs(denominator) < 1e-16) {
			return false;
		}

		pixelX = ((geoX - xOrigin) * pixelHeight - (geoY - yOrigin) * rotationY) / denominator;
		pixelY = ((geoY - yOrigin) * pixelWidth - (geoX - xOrigin) * rotationX) / denominator;
		return true;
	}

	/**
	 * @brief 像素坐标转地理坐标
	 */
	void pixelToGeo(double pixelX, double pixelY, double& geoX, double& geoY) const {
		geoX = xOrigin + pixelX * pixelWidth + pixelY * rotationY;
		geoY = yOrigin + pixelX * rotationX + pixelY * pixelHeight;
	}


    
};

/**
 * @brief 影像读取配置
 */
struct ReadConfig {
	// 读取波段，空表示读取所有波段
	std::vector<int> bands;

	// 重采样方法
	std::string resampleMethod = "nearest"; // nearest, bilinear, cubic

	// 是否缓存瓦片
	bool cacheTiles = false;

	// 缓存大小（MB）
	int cacheSizeMB = 256;

	// 读取进度回调
	std::function<void(double progress, const std::string& message)> progressCallback;

	// 超时时间（毫秒）
	int timeoutMS = 30000;

	// 读取失败时是否抛出异常
	bool throwOnError = false;
};

/**
 * @brief 读取区域定义
 */
struct ReadArea {
	int x;          // 起始X坐标
	int y;          // 起始Y坐标
	int width;      // 读取宽度
	int height;     // 读取高度
	int level;      // 金字塔层级（0为原始层级）

	ReadArea() : x(0), y(0), width(0), height(0), level(0) {}
	ReadArea(int x, int y, int w, int h, int l = 0)
		: x(x), y(y), width(w), height(h), level(l) {
	}

	bool isValid() const {
		return width > 0 && height > 0;
	}

	/**
	 * @brief 从地理范围构造读取区域
	 */
	static ReadArea fromGeoExtent(const GeoTransform& gt,
		double minX, double minY,
		double maxX, double maxY,
		int level = 0) {
		ReadArea area;

		// 将地理坐标转换为像素坐标
		double px1, py1, px2, py2;
		if (gt.geoToPixel(minX, maxY, px1, py1) &&
			gt.geoToPixel(maxX, minY, px2, py2)) {

			// 考虑金字塔缩放
			int scale = 1 << level;

			area.x = static_cast<int>(std::floor(px1 / scale));
			area.y = static_cast<int>(std::floor(py1 / scale));
			area.width = static_cast<int>(std::ceil(px2 - px1 / scale)) ;
			area.height = static_cast<int>(std::ceil(py2 - py1/ scale));
			area.level = level;

			// 确保区域在有效范围内
			if (area.width < 0) {
				area.x += area.width;
				area.width = -area.width;
			}
			if (area.height < 0) {
				area.y += area.height;
				area.height = -area.height;
			}
		}

		return area;
	}
};

/**
 * @brief 影像信息结构体
 */
struct ImageInfo {
	std::string filePath;          // 文件路径
	std::string format;            // 格式名称
	int width;                     // 宽度（像素）
	int height;                    // 高度（像素）
	int bands;                     // 波段数
	ImageDataType dataType;        // 数据类型
	GeoTransform geoTransform;     // 地理变换参数
	std::string projection;        // 投影信息
	std::string metadata;          // 元数据（JSON格式）
	bool hasGeoInfo;               // 是否有地理信息
	bool hasPyramid;               // 是否有金字塔
	std::vector<int> pyramidLevels; // 金字塔层级列表
	std::vector<double> noDataValues; // 各波段的NoData值

	// 获取金字塔层级的尺寸
	std::pair<int, int> getLevelSize(int level) const {
		if (level == 0) return { width, height };
		int scale = 1 << level;
		return { std::max(1, width / scale), std::max(1, height / scale) };
	}
};

/**
 * @brief 读取结果结构体
 */
struct ReadResult {
	bool success;                  // 是否成功
	std::string errorMessage;      // 错误信息
	void* data;                    // 数据指针
	int width;                     // 实际读取宽度
	int height;                    // 实际读取高度
	int bands;                     // 实际读取波段数
	size_t dataSize;               // 数据大小（字节）
	ImageDataType dataType;        // 数据类型
	std::vector<size_t> bandOffsets; // 各波段数据偏移量
	std::vector<double> noDataValues; // 各波段的NoData值

	ReadResult() : success(false), data(nullptr),
		width(0), height(0), bands(0),
		dataSize(0), dataType(ImageDataType::Unknown) {
	}

	~ReadResult() {
		if (data) {
			free(data);
			data = nullptr;
		}
	}

	// 禁用拷贝构造和赋值，使用移动语义
	ReadResult(const ReadResult&) = delete;
	ReadResult& operator=(const ReadResult&) = delete;

	// 移动构造
	ReadResult(ReadResult&& other) noexcept
		: success(other.success),
		errorMessage(std::move(other.errorMessage)),
		data(other.data),
		width(other.width),
		height(other.height),
		bands(other.bands),
		dataSize(other.dataSize),
		dataType(other.dataType),
		bandOffsets(std::move(other.bandOffsets)),
		noDataValues(std::move(other.noDataValues)) {
		other.data = nullptr;
		other.success = false;
	}

	// 移动赋值
	ReadResult& operator=(ReadResult&& other) noexcept {
		if (this != &other) {
			if (data) free(data);

			success = other.success;
			errorMessage = std::move(other.errorMessage);
			data = other.data;
			width = other.width;
			height = other.height;
			bands = other.bands;
			dataSize = other.dataSize;
			dataType = other.dataType;
			bandOffsets = std::move(other.bandOffsets);
			noDataValues = std::move(other.noDataValues);

			other.data = nullptr;
			other.success = false;
		}
		return *this;
	}

	/**
	 * @brief 获取指定波段的数据指针
	 */
	template<typename T>
	T* getBandData(int band) const {
		if (!success || !data || band < 0 || band >= bands) {
			return nullptr;
		}

		if (bandOffsets.empty()) {
			// 如果没有偏移量，假设数据是交错的（band-interleaved）
			return reinterpret_cast<T*>(data) + band * width * height;
		}
		else {
			// 数据是分离的（band-separated）
			return reinterpret_cast<T*>(reinterpret_cast<char*>(data) + bandOffsets[band]);
		}
	}

	/**
	 * @brief 获取指定位置的像素值
	 */
	std::vector<double> getPixel(int x, int y) const {
		std::vector<double> pixel(bands, 0.0);
		if (!success || !data || x < 0 || x >= width || y < 0 || y >= height) {
			return pixel;
		}

		int pixelIndex = y * width + x;

		switch (dataType) {
		case ImageDataType::Byte: {
			auto* bandData = getBandData<uint8_t>(0);
			for (int b = 0; b < bands; b++) {
				pixel[b] = static_cast<double>(bandData[b * width * height + pixelIndex]);
			}
			break;
		}
		case ImageDataType::UInt16: {
			auto* bandData = getBandData<uint16_t>(0);
			for (int b = 0; b < bands; b++) {
				pixel[b] = static_cast<double>(bandData[b * width * height + pixelIndex]);
			}
			break;
		}
		case ImageDataType::Int16: {
			auto* bandData = getBandData<int16_t>(0);
			for (int b = 0; b < bands; b++) {
				pixel[b] = static_cast<double>(bandData[b * width * height + pixelIndex]);
			}
			break;
		}
		case ImageDataType::UInt32: {
			auto* bandData = getBandData<uint32_t>(0);
			for (int b = 0; b < bands; b++) {
				pixel[b] = static_cast<double>(bandData[b * width * height + pixelIndex]);
			}
			break;
		}
		case ImageDataType::Int32: {
			auto* bandData = getBandData<int32_t>(0);
			for (int b = 0; b < bands; b++) {
				pixel[b] = static_cast<double>(bandData[b * width * height + pixelIndex]);
			}
			break;
		}
		case ImageDataType::Float32: {
			auto* bandData = getBandData<float>(0);
			for (int b = 0; b < bands; b++) {
				pixel[b] = static_cast<double>(bandData[b * width * height + pixelIndex]);
			}
			break;
		}
		case ImageDataType::Float64: {
			auto* bandData = getBandData<double>(0);
			for (int b = 0; b < bands; b++) {
				pixel[b] = bandData[b * width * height + pixelIndex];
			}
			break;
		}
		default:
			break;
		}

		return pixel;
	}
};

