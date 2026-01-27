// RemoteSensingTools.cpp : 测试程序：初始化 GDAL 并读取影像，打印基本信息和左上角像素值。
//

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include "GDALImageBase.h"
#include "GDALImageReader.h"
#include "ImageBlockReader.h"
#include "ImageBlockWriter.h"
#include "opencv2/opencv.hpp"

#include "optflowoffsets.h"

using namespace cv;

Mat convertTo8Bit(const void* data, int width, int height,
    ImageDataType dataType,
    double minVal=NAN, double maxVal=NAN) {
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

Mat createRGBImage(const ReadResult& result,
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
        std::array<std::pair<double,double>,4> corners;
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
    } else {
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

int main(int argc, char** argv)
{
    

    // 测试1：显示OpenCV版本
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;



    //std::string lpath = "F:/变形测试/磐安县/磐安县.img";
    //std::string rpath = "F:/变形测试/磐安县.img";
    std::string lpath = "F:/变形测试/磐安县_clip_before1.tif";
    std::string rpath = "F:/变形测试/磐安县_clip_after1.tif";
   
    // 初始化 GDAL
    RSTools_Initialize();
    ImageInfo* linfo = RSTools_GetImageInfo(lpath.c_str());
    ImageInfo* rinfo = RSTools_GetImageInfo(rpath.c_str());
    if (!linfo || !rinfo) {
        std::cerr << "RSTools_GetImageInfo 返回空指针，可能无法打开文件或路径错误。\n";
        return 1;
    }

    std::cout << "----Left ImageInfo ----\n";
    std::cout << "文件: " << linfo->filePath << "\n";
    std::cout << "格式: " << linfo->format << "\n";
    std::cout << "尺寸: " << linfo->width << " x " << linfo->height << "\n";
    std::cout << "波段数: " << linfo->bands << "\n";
    std::cout << "数据类型: " << ImageDataTypeToString(linfo->dataType) << "\n";
    std::cout << "hasGeoInfo: " << (linfo->hasGeoInfo ? "true" : "false") << "\n";
    if (linfo->hasGeoInfo) {
        const auto& gt = linfo->geoTransform;


        std::cout << std::fixed << std::setprecision(8);
        std::cout << "GeoTransform: xOrigin=" << gt.xOrigin
            << ", yOrigin=" << gt.yOrigin
            << ", pixelWidth=" << gt.pixelWidth
            << ", pixelHeight=" << gt.pixelHeight
            << ", rotationX=" << gt.rotationX
            << ", rotationY=" << gt.rotationY << "\n";
    }
    if (!linfo->projection.empty()) {
        std::cout << "投影: " << linfo->projection << "\n";
    }
    if (!linfo->metadata.empty()) {
        std::cout << "元数据:\n" << linfo->metadata << "\n";
    }
    std::cout << "----Right ImageInfo ----\n";
    std::cout << "文件: " << rinfo->filePath << "\n";
    std::cout << "格式: " << rinfo->format << "\n";
    std::cout << "尺寸: " << rinfo->width << " x " << rinfo->height << "\n";
    std::cout << "波段数: " << rinfo->bands << "\n";
    std::cout << "数据类型: " << ImageDataTypeToString(rinfo->dataType) << "\n";
    std::cout << "hasGeoInfo: " << (rinfo->hasGeoInfo ? "true" : "false") << "\n";
    if (rinfo->hasGeoInfo) {
        const auto& gt = rinfo->geoTransform;
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "GeoTransform: xOrigin=" << gt.xOrigin
            << ", yOrigin=" << gt.yOrigin
            << ", pixelWidth=" << gt.pixelWidth
            << ", pixelHeight=" << gt.pixelHeight
            << ", rotationX=" << gt.rotationX
            << ", rotationY=" << gt.rotationY << "\n";
    }
    if (!rinfo->projection.empty()) {
        std::cout << "投影: " << rinfo->projection << "\n";
    }
    if (!rinfo->metadata.empty()) {
        std::cout << "元数据:\n" << rinfo->metadata << "\n";
    }
    if (linfo->hasGeoInfo && rinfo->hasGeoInfo) {
        const auto& lgt = linfo->geoTransform;
        const auto& rgt = rinfo->geoTransform;
        if (std::abs(lgt.pixelWidth - rgt.pixelWidth) > 1e-8 ||
            std::abs(lgt.pixelHeight - rgt.pixelHeight) > 1e-8) {
            std::cout << "像素大小不一致，无法比较。\n";
            return 0;
        }


        // 新增：计算两个影像的最大边界（地理坐标）及在该边界下左右影像的像素偏移
        CombinedBoundsResult cb = computeCombinedBounds(linfo, rinfo);
        const ReadArea leftArea = ReadArea::fromGeoExtent(
            linfo->geoTransform,
            cb.minX, cb.minY,
            cb.maxX, cb.maxY,
            0);
        const ReadArea rightArea = ReadArea::fromGeoExtent(
            rinfo->geoTransform,
            cb.minX, cb.minY,
            cb.maxX, cb.maxY,0);
        std::cout << "左影像在合并画布中的读取区域: x=" << leftArea.x << ", y=" << leftArea.y
                << ", width=" << leftArea.width << ", height=" << leftArea.height << "\n";
        std::cout << "右影像在合并画布中的读取区域: x=" << rightArea.x << ", y=" << rightArea.y
                << ", width=" << rightArea.width << ", height=" << rightArea.height << "\n";
        //cb.
        std::cout << "---- 合并边界（地理坐标） ----\n";
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "minX=" << cb.minX << ", maxX=" << cb.maxX << ", minY=" << cb.minY << ", maxY=" << cb.maxY << "\n";
        std::cout << "合并画布像素尺寸: " << cb.combinedWidth << " x " << cb.combinedHeight << "\n";
        std::cout << "合并画布 GeoTransform: xOrigin=" << cb.combinedGT.xOrigin << "\n"
            << ", yOrigin=" << cb.combinedGT.yOrigin << "\n"
            << ", pixelWidth=" << cb.combinedGT.pixelWidth << "\n"
            << ", pixelHeight=" << cb.combinedGT.pixelHeight << "\n"
            << ", rotationX=" << cb.combinedGT.rotationX << "\n"
            << ", rotationY=" << cb.combinedGT.rotationY << "\n";

        std::cout << "---- 左影像在合并画布中的偏移 ----\n";
        std::cout << "浮点偏移: (" << cb.leftOffsetXf << ", " << cb.leftOffsetYf << ")\n";
        std::cout << "整数偏移: (" << cb.leftOffsetXi << ", " << cb.leftOffsetYi << ")\n";

        std::cout << "---- 右影像在合并画布中的偏移 ----\n";
        std::cout << "浮点偏移: (" << cb.rightOffsetXf << ", " << cb.rightOffsetYf << ")\n";
        std::cout << "整数偏移: (" << cb.rightOffsetXi << ", " << cb.rightOffsetYi << ")\n";
        // 输出文件路径
    std::string outOffsetPath = "F:/变形测试/offsets.tif";

    ImageBlockReader rreader(rpath, 512, 512, 16, &rightArea);
    ImageBlockReader lreader(lpath, 512, 512, 16, &leftArea);

    // 匹配参数（可调整）
    const int templHalf = 3; // 模板半尺寸 => 模板 7x7
    const int templW = templHalf * 2 + 1;
    const int searchRadius = 16; // 在右影像中心周围搜索 +/-searchRadius

    // prepare per-output-block containers to accumulate sparse matches during tile traversal
    const int outBlockW = 512;
    const int outBlockH = 512;
    int blocksX = static_cast<int>(std::ceil(cb.combinedWidth / static_cast<double>(outBlockW)));
    int blocksY = static_cast<int>(std::ceil(cb.combinedHeight / static_cast<double>(outBlockH)));
    int totalBlocks = std::max(1, blocksX) * std::max(1, blocksY);

    std::vector<std::vector<cv::Point2f>> prevPtsPerBlock(totalBlocks);
    std::vector<std::vector<cv::Point2f>> currPtsPerBlock(totalBlocks);
    std::vector<std::vector<uchar>> statusPerBlock(totalBlocks);

    while (true) {
        ReadResult* rres = nullptr;
        ReadResult* lres = nullptr;
        BlockSpec rspec;
        BlockSpec lspec;
        if (!rreader.next(&rres, &rspec) || !lreader.next(&lres, &lspec))
        {
            if (rres) { RSTools_DestroyReadResult(rres); rres = nullptr; }
            if (lres) { RSTools_DestroyReadResult(lres); lres = nullptr; }
            break;
        }

        if (!lres || !rres) {
            if (rres) RSTools_DestroyReadResult(rres);
            if (lres) RSTools_DestroyReadResult(lres);
            continue;
        }

        // 仅使用第一波段进行匹配（若无波段或数据异常则跳过）
        if (lres->bands < 1 || rres->bands < 1) {
            RSTools_DestroyReadResult(rres);
            RSTools_DestroyReadResult(lres);
            continue;
        }

        cv::Mat lband = convertToCVMat(*lres, 0);
        cv::Mat rband = convertToCVMat(*rres, 0);

        OpticalFlowOffset ofCalculator = OpticalFlowOffset();
        SparseOpticalFlowResult sparse = ofCalculator.calculateOpticalFlowOffset(*lres, *rres);
        // transform sparse points to combined canvas coordinates and assign to output blocks
        if (sparse.success && !sparse.prevPoints.empty() && sparse.prevPoints.size() == sparse.currPoints.size()) {
            for (size_t i = 0; i < sparse.prevPoints.size(); ++i) {
                if (i < sparse.status.size() && sparse.status[i] == 0) continue;
                double prevCombinedX = cb.leftOffsetXf + lspec.readX + sparse.prevPoints[i].x;
                double prevCombinedY = cb.leftOffsetYf + lspec.readY + sparse.prevPoints[i].y;
                double currCombinedX = cb.rightOffsetXf + rspec.readX + sparse.currPoints[i].x;
                double currCombinedY = cb.rightOffsetYf + rspec.readY + sparse.currPoints[i].y;

                // clamp
                if (prevCombinedX < 0 || prevCombinedY < 0 || prevCombinedX >= cb.combinedWidth || prevCombinedY >= cb.combinedHeight) continue;

                int bxIdx = static_cast<int>(prevCombinedX) / outBlockW;
                int byIdx = static_cast<int>(prevCombinedY) / outBlockH;
                if (bxIdx < 0) bxIdx = 0; if (byIdx < 0) byIdx = 0;
                if (bxIdx >= blocksX) bxIdx = blocksX - 1; if (byIdx >= blocksY) byIdx = blocksY - 1;

                int blockIndex = byIdx * blocksX + bxIdx;
                prevPtsPerBlock[blockIndex].emplace_back(static_cast<float>(prevCombinedX), static_cast<float>(prevCombinedY));
                currPtsPerBlock[blockIndex].emplace_back(static_cast<float>(currCombinedX), static_cast<float>(currCombinedY));
                statusPerBlock[blockIndex].push_back(sparse.status[i]);
            }
        }

        // 释放
        RSTools_DestroyReadResult(rres);
        RSTools_DestroyReadResult(lres);
    }

    // After collecting sparse matches per output block, flatten to global lists for densification
    std::vector<cv::Point2f> allPrevPts;
    std::vector<cv::Point2f> allCurrPts;
    std::vector<uchar> allStatus;
    for (size_t bi = 0; bi < prevPtsPerBlock.size(); ++bi) {
        for (size_t j = 0; j < prevPtsPerBlock[bi].size(); ++j) {
            allPrevPts.push_back(prevPtsPerBlock[bi][j]);
            allCurrPts.push_back(currPtsPerBlock[bi][j]);
            if (j < statusPerBlock[bi].size()) allStatus.push_back(statusPerBlock[bi][j]);
            else allStatus.push_back(1);
        }
    }

    OpticalFlowOffset densifier;
    std::vector<std::tuple<int,int,int,int>> blocksOut;
    std::vector<std::vector<float>> bufXOut, bufYOut;
    // parameters
    OpticalFlowOffset::InterpMethod method = OpticalFlowOffset::TPS_GLOBAL;
    float kernelSigma = 8.0f;
    int kernelRadius = 16;
    double regularization = 1e-3;
    int maxGlobalPoints = 2000;

    bool ok = densifier.computeDenseToBlocks(allPrevPts, allCurrPts, allStatus,
        cb.combinedWidth, cb.combinedHeight,
        512, 512,
        blocksOut, bufXOut, bufYOut,
        method, kernelSigma, kernelRadius, regularization, maxGlobalPoints);

    if (!ok) {
        std::cerr << "computeDenseToBlocks failed" << std::endl;
    } else {
        // write buffers to GeoTIFF using ImageBlockWriter in main thread sequentially
        ImageBlockWriter writer(outOffsetPath, cb.combinedWidth, cb.combinedHeight, 2, cb.combinedGT, linfo->projection);
        if (!writer.isOpen()) {
            std::cerr << "无法创建输出文件: " << outOffsetPath << std::endl;
        } else {
            for (size_t i = 0; i < blocksOut.size(); ++i) {
                int bx, by, w, h; std::tie(bx, by, w, h) = blocksOut[i];
                const float* ptrs[2] = { bufXOut[i].data(), bufYOut[i].data() };
                if (!writer.writeBlock(bx, by, w, h, ptrs)) {
                    std::cerr << "写入块失败: " << bx << "," << by << std::endl;
                }
            }
            std::cout << "生成密集偏移 GeoTIFF 成功: " << outOffsetPath << std::endl;
        }
    }
    }

    // 释放 info（由 DLL 分配，必须调用对应销毁接口）
    RSTools_DestroyImageInfo(linfo);

    // 释放 info（由 DLL 分配，必须调用对应销毁接口）
    RSTools_DestroyImageInfo(rinfo);

    std::cout << "---- 完成 ----\n";
    return 0;
}
