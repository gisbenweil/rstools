#include "ImageBlockWriter.h"
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <vector>
#include <cmath>
#include <cstring> // For memset when initializing non-float types

// Helper function to convert ImageDataType to GDALDataType
static GDALDataType ImageDataTypeToGDALType(ImageDataType dt) {
    switch (dt) {
    case ImageDataType::Byte:    return GDT_Byte;
    case ImageDataType::UInt16:  return GDT_UInt16;
    case ImageDataType::Int16:   return GDT_Int16;
    case ImageDataType::UInt32:  return GDT_UInt32;
    case ImageDataType::Int32:   return GDT_Int32;
    case ImageDataType::Float32: return GDT_Float32;
    case ImageDataType::Float64: return GDT_Float64;
    default:                     return GDT_Float32; // Default fallback
    }
}

ImageBlockWriter::ImageBlockWriter(const std::string& path, int width, int height, int bands,
    const GeoTransform& gt, ImageDataType dataType,
    const std::string& projectionWkt)
    : ds_(nullptr), width_(width), height_(height), bands_(bands), path_(path), dataType_(dataType)
{
    gdalDataType_ = ImageDataTypeToGDALType(dataType_);

    GDALAllRegister();
    GDALDriver* drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!drv) return;

    // 创建选项：压缩与瓦片，使大文件高效
    char** papszOptions = nullptr;
    papszOptions = CSLSetNameValue(papszOptions, "TILED", "YES");
    papszOptions = CSLSetNameValue(papszOptions, "BLOCKXSIZE", "512");
    papszOptions = CSLSetNameValue(papszOptions, "BLOCKYSIZE", "512");
    papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");

    ds_ = drv->Create(path.c_str(), width_, height_, bands_, gdalDataType_, papszOptions);
    CSLDestroy(papszOptions);
    if (!ds_) return;

    // 设置 GeoTransform
    double gdalTrans[6];
    gt.toGDAL(gdalTrans);
    ds_->SetGeoTransform(gdalTrans);

    // 设置投影
    if (!projectionWkt.empty()) {
        ds_->SetProjection(projectionWkt.c_str());
    }

    // 初始化每个波段为 NaN 或 0 (取决于数据类型)
    for (int b = 1; b <= bands_; ++b) {
        GDALRasterBand* rb = ds_->GetRasterBand(b);
        if (!rb) continue;
        // 尝试设置 NoData 为 NaN（对于浮点数）或 0（对于整数）
        if (dataType_ == ImageDataType::Float32 || dataType_ == ImageDataType::Float64) {
            rb->SetNoDataValue(std::numeric_limits<double>::quiet_NaN());
        }
        else {
            rb->SetNoDataValue(0); // 默认整数类型的 NoData 值
        }

        // 根据数据类型初始化数据
        size_t elementSize = 0;
        switch (dataType_) {
        case ImageDataType::Byte: elementSize = sizeof(uint8_t); break;
        case ImageDataType::UInt16: elementSize = sizeof(uint16_t); break;
        case ImageDataType::Int16: elementSize = sizeof(int16_t); break;
        case ImageDataType::UInt32: elementSize = sizeof(uint32_t); break;
        case ImageDataType::Int32: elementSize = sizeof(int32_t); break;
        case ImageDataType::Float32: elementSize = sizeof(float); break;
        case ImageDataType::Float64: elementSize = sizeof(double); break;
        }

        if (elementSize > 0) {
            std::vector<char> rowData(width_ * elementSize, 0);
            if (dataType_ == ImageDataType::Float32) {
                std::vector<float> tempRow(width_, std::numeric_limits<float>::quiet_NaN());
                memcpy(rowData.data(), tempRow.data(), width_ * sizeof(float));
            }
            // For other types, we initialize with 0, which is already done by vector constructor.

            for (int y = 0; y < height_; ++y) {
                CPLErr err = rb->RasterIO(GF_Write, 0, y, width_, 1, rowData.data(),
                    width_, 1, gdalDataType_, 0, 0);
                if (err != CE_None) {
                    // Log or handle error if needed
                    break; // Stop initialization on error
                }
            }
        }
    }
}

// 便捷构造函数：创建 Float32 类型的 GeoTIFF (保持向后兼容)
ImageBlockWriter::ImageBlockWriter(const std::string& path, int width, int height, int bands,
    const GeoTransform& gt, const std::string& projectionWkt)
    : ImageBlockWriter(path, width, height, bands, gt, ImageDataType::Float32, projectionWkt) {
    // Delegates to the main constructor with Float32 type
}

ImageBlockWriter::~ImageBlockWriter() {
    if (ds_) {
        GDALClose(ds_);
        ds_ = nullptr;
    }
}

bool ImageBlockWriter::isOpen() const {
    return ds_ != nullptr;
}
// Note: template implementation moved to header (ImageBlockWriter.h)

void ImageBlockWriter::setBandNoDataValue(int bandIndex, double nodata) {
    if (!ds_) return;
    if (bandIndex < 0 || bandIndex >= bands_) return;
    GDALRasterBand* rb = ds_->GetRasterBand(bandIndex + 1);
    if (!rb) return;
    rb->SetNoDataValue(nodata);
}