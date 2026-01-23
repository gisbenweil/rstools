#include "ImageBlockReader.h"
#include <gdal_priv.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>

// Helper: map GDALDataType to ImageDataType (局部复制，与 GDALImageReader 中的映射一致)
static ImageDataType MapGDALType(GDALDataType t) {
	switch (t) {
	case GDT_Byte: return ImageDataType::Byte;
	case GDT_UInt16: return ImageDataType::UInt16;
	case GDT_Int16: return ImageDataType::Int16;
	case GDT_UInt32: return ImageDataType::UInt32;
	case GDT_Int32: return ImageDataType::Int32;
	case GDT_Float32: return ImageDataType::Float32;
	case GDT_Float64: return ImageDataType::Float64;
	default: return ImageDataType::Unknown;
	}
}

// 将 double noData 值写入目标缓冲（按元素），用于初始化超出范围部分
template<typename T>
static void FillNoData(T* dst, size_t count, double nd) {
	if (!dst) return;
	T v = static_cast<T>(nd);
	for (size_t i = 0; i < count; ++i) dst[i] = v;
}

ImageBlockReader::ImageBlockReader(const std::string& path, int blockWidth, int blockHeight, int padding, const ReadArea* range)
	: path_(path), blockW_(std::max(1, blockWidth)), blockH_(std::max(1, blockHeight)),
	  padding_(std::max(0, padding)), imgW_(0), imgH_(0),
	  tilesX_(0), tilesY_(0), curXIdx_(0), curYIdx_(0), open_(false), processed_(0), ds_(nullptr) {

	// 打开数据集一次以获取尺寸并在后续重用
	ds_ = static_cast<GDALDataset*>(GDALOpen(path_.c_str(), GA_ReadOnly));
	if (!ds_) {
		open_ = false;
		return;
	}
	imgW_ = ds_->GetRasterXSize();
	imgH_ = ds_->GetRasterYSize();

	// 设置范围
	if (range && range->isValid()) {
		range_.x = range->x;
		range_.y = range->y;
		// 确保不超出影像
		range_.width = range->width;
		range_.height = range->height;
		if (range_.width <= 0 || range_.height <= 0) {
			// 无效，回退为整幅影像
			range_.x = 0; range_.y = 0; range_.width = imgW_; range_.height = imgH_;
		}
	} else {
		range_.x = 0; range_.y = 0; range_.width = imgW_; range_.height = imgH_;
	}

	// 计算 tiles
	tilesX_ = static_cast<int>(std::ceil(static_cast<double>(range_.width) / blockW_));
	tilesY_ = static_cast<int>(std::ceil(static_cast<double>(range_.height) / blockH_));

	// 防护
	if (tilesX_ <= 0) tilesX_ = 1;
	if (tilesY_ <= 0) tilesY_ = 1;

	open_ = true;
	curXIdx_ = 0;
	curYIdx_ = 0;
	processed_ = 0;
}

ImageBlockReader::~ImageBlockReader() {
	if (ds_) {
		GDALClose(ds_);
		ds_ = nullptr;
	}
}

bool ImageBlockReader::isOpen() const {
	return open_ && ds_;
}

void ImageBlockReader::reset() {
	curXIdx_ = 0;
	curYIdx_ = 0;
	processed_ = 0;
}

int ImageBlockReader::getTotalBlocks() const {
	return tilesX_ * tilesY_;
}

int ImageBlockReader::getProcessedBlocks() const {
	return processed_;
}

int ImageBlockReader::imageWidth() const { return imgW_; }
int ImageBlockReader::imageHeight() const { return imgH_; }

static bool ReadTileInternal(GDALDataset* ds, int tileX, int tileY, int tileW, int tileH, int padding,
	ReadArea const& range, int imgW, int imgH, int blockW, int blockH, ReadResult** out, BlockSpec* spec) {
	// tileX,tileY are absolute image coords for the tile's top-left within range coordinate.
	if (!out) return false;
	*out = nullptr;

	// 期望窗口（可能超出图像）
	int desiredReadX = tileX - padding;
	int desiredReadY = tileY - padding;
	int desiredRight = tileX + tileW + padding;
	int desiredBottom = tileY + tileH + padding;
	int desiredReadW = desiredRight - desiredReadX;
	int desiredReadH = desiredBottom - desiredReadY;

	if (desiredReadW <= 0 || desiredReadH <= 0) return false;

	// 计算与影像的重叠区域（用于从文件读取）
	int overlapX0 = std::max(0, desiredReadX);
	int overlapY0 = std::max(0, desiredReadY);
	int overlapX1 = std::min(imgW, desiredRight);
	int overlapY1 = std::min(imgH, desiredBottom);
	int overlapW = overlapX1 - overlapX0;
	int overlapH = overlapY1 - overlapY0;

	// 创建 ReadResult 并分配缓冲（按期望尺寸，即可能包含超出部分）
	auto rr = new ReadResult();
	rr->success = false;
	rr->width = desiredReadW;
	rr->height = desiredReadH;

	// 波段与类型
	int bands = ds->GetRasterCount();
	rr->bands = bands;
	GDALRasterBand* b0 = ds->GetRasterBand(1);
	GDALDataType gdalType = b0 ? b0->GetRasterDataType() : GDT_Byte;
	rr->dataType = MapGDALType(gdalType);

	int bytesPerPixel = GDALGetDataTypeSizeBytes(gdalType);
	if (bytesPerPixel <= 0) bytesPerPixel = 1;

	// 先收集每波段的 NoData 值（用于填充超出区域）
	std::vector<double> bandNoData(bands, 0.0);
	for (int b = 0; b < bands; ++b) {
		GDALRasterBand* rb = ds->GetRasterBand(b + 1);
		if (!rb) continue;
		int hasNoData = 0;
		double nd = rb->GetNoDataValue(&hasNoData);
		bandNoData[b] = hasNoData ? nd : 0.0;
	}

	size_t bandElemCount = static_cast<size_t>(desiredReadW) * static_cast<size_t>(desiredReadH);
	size_t totalBytes = bandElemCount * bytesPerPixel * static_cast<size_t>(bands);

	rr->data = malloc(totalBytes);
	if (!rr->data) {
		rr->success = false;
		rr->errorMessage = "Out of memory";
		delete rr;
		return false;
	}
	// 初始化为各波段的 NoData（逐波段写入）
	char* basePtr = static_cast<char*>(rr->data);
	for (int b = 0; b < bands; ++b) {
		char* bandPtr = basePtr + static_cast<size_t>(b) * bandElemCount * bytesPerPixel;
		size_t elems = bandElemCount;
		switch (gdalType) {
		case GDT_Byte:
			FillNoData<uint8_t>(reinterpret_cast<uint8_t*>(bandPtr), elems, bandNoData[b]);
			break;
		case GDT_UInt16:
			FillNoData<uint16_t>(reinterpret_cast<uint16_t*>(bandPtr), elems, bandNoData[b]);
			break;
		case GDT_Int16:
			FillNoData<int16_t>(reinterpret_cast<int16_t*>(bandPtr), elems, bandNoData[b]);
			break;
		case GDT_UInt32:
			FillNoData<uint32_t>(reinterpret_cast<uint32_t*>(bandPtr), elems, bandNoData[b]);
			break;
		case GDT_Int32:
			FillNoData<int32_t>(reinterpret_cast<int32_t*>(bandPtr), elems, bandNoData[b]);
			break;
		case GDT_Float32:
			FillNoData<float>(reinterpret_cast<float*>(bandPtr), elems, bandNoData[b]);
			break;
		case GDT_Float64:
			FillNoData<double>(reinterpret_cast<double*>(bandPtr), elems, bandNoData[b]);
			break;
		default:
			std::memset(bandPtr, 0, elems * bytesPerPixel);
			break;
		}
	}

	// 若与影像有重叠区域，从文件读取该重叠区域并拷贝到目标缓冲的正确偏移位置
	bool anyFail = false;
	if (overlapW > 0 && overlapH > 0) {
		// 为每个波段分别读取 overlap 区域到临时缓冲，再拷贝到最终缓冲
		size_t overlapElems = static_cast<size_t>(overlapW) * static_cast<size_t>(overlapH);
		size_t overlapBytes = overlapElems * static_cast<size_t>(bytesPerPixel);
		void* tempBuf = malloc(overlapBytes);
		if (!tempBuf) {
			rr->errorMessage = "Out of memory (temp overlap)";
			free(rr->data);
			rr->data = nullptr;
			delete rr;
			return false;
		}

		for (int b = 0; b < bands; ++b) {
			GDALRasterBand* rb = ds->GetRasterBand(b + 1);
			if (!rb) continue;

			// 读取重叠区域
			CPLErr err = rb->RasterIO(GF_Read, overlapX0, overlapY0, overlapW, overlapH,
									 tempBuf, overlapW, overlapH, gdalType,
									 0, 0);
			if (err != CE_None) {
				rr->success = false;
				rr->errorMessage = std::string("GDAL RasterIO failed on band ") + std::to_string(b + 1);
				anyFail = true;
				break;
			}

			// 拷贝到目标缓冲的正确位置
			char* bandPtr = basePtr + static_cast<size_t>(b) * bandElemCount * bytesPerPixel;
			// 目标起始偏移（像素）
			int dstX0 = overlapX0 - desiredReadX; // >=0
			int dstY0 = overlapY0 - desiredReadY; // >=0

			// 按行拷贝
			for (int row = 0; row < overlapH; ++row) {
				char* srcRow = static_cast<char*>(tempBuf) + static_cast<size_t>(row) * overlapW * bytesPerPixel;
				char* dstRow = bandPtr + (static_cast<size_t>(dstY0 + row) * static_cast<size_t>(desiredReadW) + static_cast<size_t>(dstX0)) * bytesPerPixel;
				std::memcpy(dstRow, srcRow, static_cast<size_t>(overlapW) * bytesPerPixel);
			}

			// 保存 NoData 值（如有）
			int hasNoData = 0;
			double nd = rb->GetNoDataValue(&hasNoData);
			if (hasNoData) {
				if (rr->noDataValues.size() < static_cast<size_t>(bands)) rr->noDataValues.resize(bands, 0.0);
				rr->noDataValues[b] = nd;
			}
		}

		free(tempBuf);
	}

	if (anyFail) {
		if (rr->data) free(rr->data);
		rr->data = nullptr;
		delete rr;
		*out = nullptr;
		return false;
	}

	rr->success = true;
	rr->dataSize = totalBytes;
	rr->bandOffsets.clear();
	rr->errorMessage.clear();

	// 填充 spec（如果需要）——使用期望窗口（可能超出图像）
	if (spec) {
		spec->tileX = tileX;
		spec->tileY = tileY;
		spec->tileWidth = tileW;
		spec->tileHeight = tileH;
		spec->readX = desiredReadX;
		spec->readY = desiredReadY;
		spec->readWidth = desiredReadW;
		spec->readHeight = desiredReadH;
		spec->padding = padding;
	}

	*out = rr;
	return true;
}

bool ImageBlockReader::next(ReadResult** out, BlockSpec* spec) {
	if (!out) return false;
	*out = nullptr;

	if (!isOpen()) return false;
	if (curYIdx_ >= tilesY_) {
		return false;
	}

	// 计算当前 tile 的逻辑坐标（在 range_ 内）
	int tileX = range_.x + curXIdx_ * blockW_;
	int tileY = range_.y + curYIdx_ * blockH_;
	int tileW = std::min(blockW_, range_.x + range_.width - tileX);
	int tileH = std::min(blockH_, range_.y + range_.height - tileY);

	bool ok = ReadTileInternal(ds_, tileX, tileY, tileW, tileH, padding_, range_, imgW_, imgH_, blockW_, blockH_, out, spec);
	if (!ok) {
		// 跳过并继续到下一个（保持与原行为一致）
		++curXIdx_;
		if (curXIdx_ >= tilesX_) { curXIdx_ = 0; ++curYIdx_; }
		return next(out, spec);
	}

	// 前进索引并标记
	++processed_;
	++curXIdx_;
	if (curXIdx_ >= tilesX_) {
		curXIdx_ = 0;
		++curYIdx_;
	}

	return true;
}

// 新增：根据分块全局索引读取，不改变内部迭代状态
bool ImageBlockReader::readBlockByIndex(int index, ReadResult** out, BlockSpec* spec) {
	if (!out) return false;
	*out = nullptr;
	if (!isOpen()) return false;
	if (index < 0 || index >= getTotalBlocks()) return false;

	int tx = index % tilesX_;
	int ty = index / tilesX_;

	int tileX = range_.x + tx * blockW_;
	int tileY = range_.y + ty * blockH_;
	int tileW = std::min(blockW_, range_.x + range_.width - tileX);
	int tileH = std::min(blockH_, range_.y + range_.height - tileY);

	return ReadTileInternal(ds_, tileX, tileY, tileW, tileH, padding_, range_, imgW_, imgH_, blockW_, blockH_, out, spec);
}

// ---------------- C API 实现 ----------------
extern "C" {

RSTOOLS_API void* RSTools_ImageBlockReader_Create(const char* path, int blockWidth, int blockHeight, int padding, const ReadArea* range) {
	if (!path) return nullptr;
	ImageBlockReader* p = new (std::nothrow) ImageBlockReader(path, blockWidth, blockHeight, padding, range);
	if (!p) return nullptr;
	if (!p->isOpen()) { delete p; return nullptr; }
	return static_cast<void*>(p);
}

RSTOOLS_API void RSTools_ImageBlockReader_Destroy(void* handle) {
	if (!handle) return;
	ImageBlockReader* p = static_cast<ImageBlockReader*>(handle);
	delete p;
}

RSTOOLS_API bool RSTools_ImageBlockReader_IsOpen(void* handle) {
	if (!handle) return false;
	ImageBlockReader* p = static_cast<ImageBlockReader*>(handle);
	return p->isOpen();
}

RSTOOLS_API void RSTools_ImageBlockReader_Reset(void* handle) {
	if (!handle) return;
	ImageBlockReader* p = static_cast<ImageBlockReader*>(handle);
	p->reset();
}

RSTOOLS_API int RSTools_ImageBlockReader_GetTotalBlocks(void* handle) {
	if (!handle) return 0;
	ImageBlockReader* p = static_cast<ImageBlockReader*>(handle);
	return p->getTotalBlocks();
}

RSTOOLS_API int RSTools_ImageBlockReader_GetProcessedBlocks(void* handle) {
	if (!handle) return 0;
	ImageBlockReader* p = static_cast<ImageBlockReader*>(handle);
	return p->getProcessedBlocks();
}

RSTOOLS_API int RSTools_ImageBlockReader_ImageWidth(void* handle) {
	if (!handle) return 0;
	ImageBlockReader* p = static_cast<ImageBlockReader*>(handle);
	return p->imageWidth();
}

RSTOOLS_API int RSTools_ImageBlockReader_ImageHeight(void* handle) {
	if (!handle) return 0;
	ImageBlockReader* p = static_cast<ImageBlockReader*>(handle);
	return p->imageHeight();
}

RSTOOLS_API bool RSTools_ImageBlockReader_Next(void* handle, ReadResult** out, BlockSpec* spec) {
	if (!handle || !out) return false;
	ImageBlockReader* p = static_cast<ImageBlockReader*>(handle);
	return p->next(out, spec);
}

// 新增 C API：按索引读取指定分块
RSTOOLS_API bool RSTools_ImageBlockReader_ReadByIndex(void* handle, int index, ReadResult** out, BlockSpec* spec) {
	if (!handle || !out) return false;
	ImageBlockReader* p = static_cast<ImageBlockReader*>(handle);
	return p->readBlockByIndex(index, out, spec);
}

} // extern "C"