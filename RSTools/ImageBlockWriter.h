#pragma once
#include <string>
#include <gdal_priv.h>
#include "GDALImageBase.h" // 确保包含 ImageDataType 定义

class RSTOOLS_API ImageBlockWriter {
public:
    // 主构造函数：创建多波段 GeoTIFF，支持多种数据类型
    // path: 输出文件路径
    // width/height: 栅格像素尺寸（合并画布）
    // bands: 波段数（本场景为2）
    // gt: 输出 GeoTransform（像素/地理映射）
    // dataType: 图像数据的类型（见 ImageDataType 枚举）
    // projectionWkt: 投影（WKT），可为空
    ImageBlockWriter(const std::string& path, int width, int height, int bands,
        const GeoTransform& gt, ImageDataType dataType,
        const std::string& projectionWkt = std::string());

    // 便捷构造函数：创建 Float32 类型的 GeoTIFF (保持向后兼容)
    // path: 输出文件路径
    // width/height: 栅格像素尺寸（合并画布）
    // bands: 波段数（本场景为2）
    // gt: 输出 GeoTransform（像素/地理映射）
    // projectionWkt: 投影（WKT），可为空
    ImageBlockWriter(const std::string& path, int width, int height, int bands,
        const GeoTransform& gt, const std::string& projectionWkt = std::string());

    ~ImageBlockWriter();

    bool isOpen() const;

    // 将波段数据写入到输出图像的指定窗口（坐标为输出画布像素坐标）
    // bandDataPtrs: 指向每个波段缓冲区的指针（每个缓冲区按行主序，尺寸为 w*h）
    //              缓冲区的数据类型必须与构造函数中指定的 dataType 相匹配。
    //              例如，如果 dataType 是 Float32，则 bandDataPtrs[i] 必须指向 float 数组。
    // 波段数量必须等于构造时指定的 bands
    // Template implementation must be visible to all translation units that use it.
    template<typename T>
    bool writeBlock(int outX, int outY, int w, int h, const T* const bandDataPtrs[]) {
        if (!ds_) return false;
        if (outX < 0 || outY < 0 || w <= 0 || h <= 0) return false;
        if (outX + w > width_) w = width_ - outX;
        if (outY + h > height_) h = height_ - outY;
        if (w <= 0 || h <= 0) return false;

        for (int b = 0; b < bands_; ++b) {
            GDALRasterBand* rb = ds_->GetRasterBand(b + 1);
            if (!rb) continue;
            const void* src = bandDataPtrs[b];
            // 如果 src 为 nullptr，跳过写入（保持已有值）
            if (!src) continue;

            // GDAL RasterIO expects a void* buffer for writing; cast away const safely
            void* writeBuf = const_cast<void*>(src);
            // Use pixel and line spacing as 0 (packed tightly)
            CPLErr err = rb->RasterIO(GF_Write, outX, outY, w, h,
                writeBuf, w, h, gdalDataType_, 0, 0);
            if (err != CE_None) return false;
        }
        return true;
    }

    // 设置每个波段的 NoData 值（注意：值类型应与内部数据类型兼容）
    void setBandNoDataValue(int bandIndex, double nodata);

private:
    GDALDataset* ds_;
    int width_;
    int height_;
    int bands_;
    std::string path_;
    ImageDataType dataType_; // 记录内部使用的数据类型
    GDALDataType gdalDataType_; // 对应的 GDAL 数据类型
};