#pragma once
#include <string>
#include <gdal_priv.h>
#include "GDALImageBase.h"

class RSTOOLS_API ImageBlockWriter {
public:
    // 创建一个多波段 Float32 GeoTIFF（bands 个波段）
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
    // bandDataPtrs: 指向每个波段缓冲区的指针（每个缓冲区为 float，按行主序，尺寸为 w*h）
    // 波段数量必须等于构造时指定的 bands
    bool writeBlock(int outX, int outY, int w, int h, const float* const bandDataPtrs[]);

    // 设置每个波段的 NoData 值（默认使用 NaN）
    void setBandNoDataValue(int bandIndex, double nodata);

private:
    GDALDataset* ds_;
    int width_;
    int height_;
    int bands_;
    std::string path_;
};