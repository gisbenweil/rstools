#include "GDALImageReader.h"
#include <gdal_priv.h>
#include <cpl_conv.h> // CPLMalloc, CSLCount
#include <memory>
#include <cstring>
#include <cctype>

extern "C" {

RSTOOLS_API void RSTools_Initialize() {
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
    GDALAllRegister();
}

// Helper: map GDALDataType to ImageDataType
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

RSTOOLS_API ReadResult* RSTools_ReadImage(const char* path, int x, int y, int width, int height) {
    if (!path) return nullptr;

    GDALDataset* ds = static_cast<GDALDataset*>(GDALOpen(path, GA_ReadOnly));
    if (!ds) return nullptr;

    auto result = new ReadResult();

    int imgW = ds->GetRasterXSize();
    int imgH = ds->GetRasterYSize();
    int bands = ds->GetRasterCount();
    if (bands <= 0) { GDALClose(ds); delete result; return nullptr; }

    if (width <= 0) width = imgW;
    if (height <= 0) height = imgH;
    if (x < 0) x = 0;
    if (y < 0) y = 0;

    // clip to image
    if (x + width > imgW) width = imgW - x;
    if (y + height > imgH) height = imgH - y;
    if (width <= 0 || height <= 0) { GDALClose(ds); delete result; return nullptr; }

    GDALRasterBand* band0 = ds->GetRasterBand(1);
    GDALDataType gdalType = band0->GetRasterDataType();
    result->dataType = MapGDALType(gdalType);
    result->width = width;
    result->height = height;
    result->bands = bands;

    // bytes per element
    int bytesPerPixel = GDALGetDataTypeSizeBytes(gdalType);
    if (bytesPerPixel <= 0) bytesPerPixel = 1;

    // allocate contiguous memory: band0 | band1 | ...
    size_t bandElemCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    size_t totalBytes = bandElemCount * bytesPerPixel * static_cast<size_t>(bands);
    result->data = malloc(totalBytes);
    if (!result->data) {
        result->success = false;
        result->errorMessage = "Out of memory";
        GDALClose(ds);
        return result;
    }

    // 按波段逐个读取，存放为 band0...bandN 连续块（与 GDALImageBase::ReadResult::getPixel 假定一致）
    char* basePtr = static_cast<char*>(result->data);
    for (int b = 0; b < bands; ++b) {
        GDALRasterBand* rb = ds->GetRasterBand(b + 1);
        char* bandPtr = basePtr + static_cast<size_t>(b) * bandElemCount * bytesPerPixel;
        CPLErr err = rb->RasterIO(GF_Read, x, y, width, height,
                                  bandPtr, width, height, gdalType,
                                  0, 0);
        if (err != CE_None) {
            result->success = false;
            result->errorMessage = std::string("GDAL RasterIO failed on band ") + std::to_string(b + 1);
            free(result->data);
            result->data = nullptr;
            GDALClose(ds);
            return result;
        }

        // 获取并保存 NoData 值（可选）
        int hasNoData = 0;
        double nd = rb->GetNoDataValue(&hasNoData);
        if (hasNoData) {
            if (result->noDataValues.size() < static_cast<size_t>(bands)) result->noDataValues.resize(bands, 0.0);
            result->noDataValues[b] = nd;
        }
    }

    // 填充其他信息
    result->success = true;
    result->dataSize = totalBytes;
    // 不填充 bandOffsets（保持为空），GDALImageBase::ReadResult::getPixel 使用这种布局
    result->bandOffsets.clear();

    // 填充 ImageInfo 可按需扩展（这里只演示基本信息）
    result->errorMessage.clear();

    GDALClose(ds);
    return result;
}

RSTOOLS_API void RSTools_DestroyReadResult(ReadResult* result) {
    if (!result) return;
    delete result;
}

// ------------- 影像信息等函数保持不变 -------------
RSTOOLS_API ImageInfo* RSTools_GetImageInfo(const char* path) {
    if (!path) return nullptr;

    GDALDataset* ds = static_cast<GDALDataset*>(GDALOpen(path, GA_ReadOnly));
    if (!ds) return nullptr;

    auto info = new ImageInfo();
    info->filePath = path;

    GDALDriver* drv = ds->GetDriver();
    if (drv) info->format = drv->GetDescription() ? drv->GetDescription() : "";

    info->width = ds->GetRasterXSize();
    info->height = ds->GetRasterYSize();
    info->bands = ds->GetRasterCount();

    if (info->bands > 0) {
        GDALRasterBand* b0 = ds->GetRasterBand(1);
        if (b0) {
            info->dataType = MapGDALType(b0->GetRasterDataType());
        }
    } else {
        info->dataType = ImageDataType::Unknown;
    }

    // GeoTransform
    double gt[6];
    if (ds->GetGeoTransform(gt) == CE_None) {
        info->geoTransform = GeoTransform::fromGDAL(gt);
        info->hasGeoInfo = true;
    } else {
        info->hasGeoInfo = false;
    }

    // Projection
    const char* proj = ds->GetProjectionRef();
    info->projection = proj ? proj : "";

    // Metadata: 简单拼接 key=value 行（可按需改为 JSON）
    char** md = ds->GetMetadata();
    if (md && *md) {
        std::string meta;
        int i = 0;
        while (md[i]) {
            if (!meta.empty()) meta += "\n";
            meta += md[i];
            ++i;
        }
        info->metadata = meta;
    } else {
        info->metadata.clear();
    }

    // Pyramid (overviews)：使用第一个波段的 overviews 信息，记录每个 overview 的 downsample factor（整数）
    info->pyramidLevels.clear();
    if (info->bands > 0) {
        GDALRasterBand* rb = ds->GetRasterBand(1);
        if (rb) {
            int ovCount = rb->GetOverviewCount();
            for (int i = 0; i < ovCount; ++i) {
                GDALRasterBand* ov = rb->GetOverview(i);
                if (ov) {
                    int ovW = ov->GetXSize();
                    if (ovW > 0) {
                        int scale = info->width / ovW;
                        if (scale < 1) scale = 1;
                        info->pyramidLevels.push_back(scale);
                    }
                }
            }
        }
    }
    info->hasPyramid = !info->pyramidLevels.empty();

    // NoData per band
    info->noDataValues.clear();
    for (int b = 0; b < info->bands; ++b) {
        GDALRasterBand* rb = ds->GetRasterBand(b + 1);
        if (!rb) {
            info->noDataValues.push_back(0.0);
            continue;
        }
        int hasNoData = 0;
        double nd = rb->GetNoDataValue(&hasNoData);
        if (hasNoData) info->noDataValues.push_back(nd);
        else info->noDataValues.push_back(0.0);
    }

    GDALClose(ds);
    return info;
}

RSTOOLS_API void RSTools_DestroyImageInfo(ImageInfo* info) {
    if (!info) return;
    delete info;
}

} // extern "C"


// ---------------- 新增：C++ 重载实现（同名，但为 C++ 链接） ----------------
// 使用 GDALRasterIOExtraArg 指定重采样算法，由 GDAL 在 RasterIO 中完成重采样。
// 支持 resampleMethod: "nearest", "bilinear", "cubic"（不区分大小写）。
RSTOOLS_API ReadResult* RSTools_ReadImage(const char* path, int x, int y, int width, int height, int outWidth, int outHeight, const char* resampleMethod) {
    if (!path) return nullptr;

    GDALDataset* ds = static_cast<GDALDataset*>(GDALOpen(path, GA_ReadOnly));
    if (!ds) return nullptr;

    auto result = new ReadResult();

    int imgW = ds->GetRasterXSize();
    int imgH = ds->GetRasterYSize();
    int bands = ds->GetRasterCount();
    if (bands <= 0) { GDALClose(ds); delete result; return nullptr; }

    if (width <= 0) width = imgW;
    if (height <= 0) height = imgH;
    if (x < 0) x = 0;
    if (y < 0) y = 0;

    // clip to image
    if (x + width > imgW) width = imgW - x;
    if (y + height > imgH) height = imgH - y;
    if (width <= 0 || height <= 0) { GDALClose(ds); delete result; return nullptr; }

    // 输出尺寸处理：如果未指定输出尺寸，返回原窗口大小
    if (outWidth <= 0) outWidth = width;
    if (outHeight <= 0) outHeight = height;

    GDALRasterBand* band0 = ds->GetRasterBand(1);
    GDALDataType gdalType = band0->GetRasterDataType();
    result->dataType = MapGDALType(gdalType);
    result->width = outWidth;   // 返回的宽高为输出尺寸
    result->height = outHeight;
    result->bands = bands;

    // bytes per element for target type
    int bytesPerPixel = GDALGetDataTypeSizeBytes(gdalType);
    if (bytesPerPixel <= 0) bytesPerPixel = 1;

    size_t outBandElemCount = static_cast<size_t>(outWidth) * static_cast<size_t>(outHeight);
    size_t totalOutBytes = outBandElemCount * bytesPerPixel * static_cast<size_t>(bands);

    result->data = malloc(totalOutBytes);
    if (!result->data) {
        result->success = false;
        result->errorMessage = "Out of memory";
        GDALClose(ds);
        return result;
    }

    // prepare GDALRasterIOExtraArg
    GDALRasterIOExtraArg extraArg;
    INIT_RASTERIO_EXTRA_ARG(extraArg);

    std::string method = resampleMethod ? resampleMethod : "nearest";
    for (auto &c : method) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (method == "bilinear") {
        extraArg.eResampleAlg = GRIORA_Bilinear;
    } else if (method == "cubic") {
        extraArg.eResampleAlg = GRIORA_Cubic;
    } else {
        extraArg.eResampleAlg = GRIORA_NearestNeighbour;
    }

    // 对每个波段：调用 RasterIO，指定输出缓冲尺寸与 extraArg，由 GDAL 负责重采样
    char* basePtr = static_cast<char*>(result->data);
    for (int b = 0; b < bands; ++b) {
        GDALRasterBand* rb = ds->GetRasterBand(b + 1);
        char* bandPtr = basePtr + static_cast<size_t>(b) * outBandElemCount * bytesPerPixel;

        CPLErr err = rb->RasterIO(GF_Read, x, y, width, height,
                                  bandPtr, outWidth, outHeight, gdalType,
                                  0, 0, &extraArg);
        if (err != CE_None) {
            result->success = false;
            result->errorMessage = std::string("GDAL RasterIO (scaled) failed on band ") + std::to_string(b + 1);
            free(result->data);
            result->data = nullptr;
            GDALClose(ds);
            return result;
        }

        // 获取并保存 NoData 值（可选）
        int hasNoData = 0;
        double nd = rb->GetNoDataValue(&hasNoData);
        if (hasNoData) {
            if (result->noDataValues.size() < static_cast<size_t>(bands)) result->noDataValues.resize(bands, 0.0);
            result->noDataValues[b] = nd;
        }
    }

    // 填充其他信息
    result->success = true;
    result->dataSize = totalOutBytes;
    result->bandOffsets.clear();
    result->errorMessage.clear();

    GDALClose(ds);
    return result;
}