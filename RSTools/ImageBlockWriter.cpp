#include "ImageBlockWriter.h"
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <vector>
#include <cmath>

ImageBlockWriter::ImageBlockWriter(const std::string& path, int width, int height, int bands,
    const GeoTransform& gt, const std::string& projectionWkt)
    : ds_(nullptr), width_(width), height_(height), bands_(bands), path_(path)
{
    GDALAllRegister();
    GDALDriver* drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!drv) return;

    // 创建选项：压缩与瓦片，使大文件高效
    char** papszOptions = nullptr;
    papszOptions = CSLSetNameValue(papszOptions, "TILED", "YES");
    papszOptions = CSLSetNameValue(papszOptions, "BLOCKXSIZE", "512");
    papszOptions = CSLSetNameValue(papszOptions, "BLOCKYSIZE", "512");
    papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");

    ds_ = drv->Create(path.c_str(), width_, height_, bands_, GDT_Float32, papszOptions);
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

    // 初始化每个波段为 NaN（NoData）
    for (int b = 1; b <= bands_; ++b) {
        GDALRasterBand* rb = ds_->GetRasterBand(b);
        if (!rb) continue;
        // 尝试设置 NoData 为 NaN（GDAL 支持 double NaN）
        rb->SetNoDataValue(std::numeric_limits<double>::quiet_NaN());
        // 初始化为 NaN
        std::vector<float> row(width_, std::numeric_limits<float>::quiet_NaN());
        for (int y = 0; y < height_; ++y) {
            rb->RasterIO(GF_Write, 0, y, width_, 1, row.data(), width_, 1, GDT_Float32, 0, 0);
        }
    }
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

bool ImageBlockWriter::writeBlock(int outX, int outY, int w, int h, const float* const bandDataPtrs[]) {
    if (!ds_) return false;
    if (outX < 0 || outY < 0 || w <= 0 || h <= 0) return false;
    if (outX + w > width_) w = width_ - outX;
    if (outY + h > height_) h = height_ - outY;
    if (w <= 0 || h <= 0) return false;

    for (int b = 0; b < bands_; ++b) {
        GDALRasterBand* rb = ds_->GetRasterBand(b + 1);
        if (!rb) continue;
        const float* src = bandDataPtrs[b];
        // 如果 src 为 nullptr，跳过写入（保持已有值）
        if (!src) continue;

        CPLErr err = rb->RasterIO(GF_Write, outX, outY, w, h,
            const_cast<float*>(src), w, h, GDT_Float32, 0, 0);
        if (err != CE_None) return false;
    }
    return true;
}

void ImageBlockWriter::setBandNoDataValue(int bandIndex, double nodata) {
    if (!ds_) return;
    if (bandIndex < 0 || bandIndex >= bands_) return;
    GDALRasterBand* rb = ds_->GetRasterBand(bandIndex + 1);
    if (!rb) return;
    rb->SetNoDataValue(nodata);
}