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

#include "optflow.h"

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
    /*if (argc >= 2) {
        lpath = argv[1];
    }
    else {
        std::cout << "请输入原始影像文件路径（或回车取消）: ";
        std::getline(std::cin, lpath);
    }
    if (argc >= 3) {
        rpath = argv[2];
    }
    else
    {
        std::cout << "请输入参考影像文件路径（或回车取消）: ";
        std::getline(std::cin, rpath);
    }
    if (lpath.empty() || rpath.empty()) {
        std::cout << "路径不对，退出。\n";
        return 0;
    }*/
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
    // 创建两波段输出（band0 = dx, band1 = dy），大小为合并画布
    ImageBlockWriter writer(outOffsetPath, cb.combinedWidth, cb.combinedHeight, 2, cb.combinedGT, linfo->projection);
    if (!writer.isOpen()) {
        std::cerr << "无法创建输出文件: " << outOffsetPath << std::endl;
        // 继续但不会写入
    }

    ImageBlockReader rreader(rpath, 512, 512, 16, &rightArea);
    ImageBlockReader lreader(lpath, 512, 512, 16, &leftArea);

    // 匹配参数（可调整）
    const int templHalf = 3; // 模板半尺寸 => 模板 7x7
    const int templW = templHalf * 2 + 1;
    const int searchRadius = 16; // 在右影像中心周围搜索 +/-searchRadius

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

        // 将左/右缓冲的第一波段转换为 CV_32F 矩阵（以读出的缓冲区为坐标）
        auto toFloatMat = [](const ReadResult* res, int bandIndex)->cv::Mat {
            if (!res) return cv::Mat();
            int w = res->width;
            int h = res->height;
            cv::Mat m(h, w, CV_32F);
            switch (res->dataType) {
            case ImageDataType::Byte: {
                auto* p = res->getBandData<uint8_t>(bandIndex);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        m.at<float>(y, x) = static_cast<float>(p[y * w + x]);
                break;
            }
            case ImageDataType::UInt16: {
                auto* p = res->getBandData<uint16_t>(bandIndex);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        m.at<float>(y, x) = static_cast<float>(p[y * w + x]);
                break;
            }
            case ImageDataType::Int16: {
                auto* p = res->getBandData<int16_t>(bandIndex);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        m.at<float>(y, x) = static_cast<float>(p[y * w + x]);
                break;
            }
            case ImageDataType::UInt32: {
                auto* p = res->getBandData<uint32_t>(bandIndex);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        m.at<float>(y, x) = static_cast<float>(p[y * w + x]);
                break;
            }
            case ImageDataType::Int32: {
                auto* p = res->getBandData<int32_t>(bandIndex);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        m.at<float>(y, x) = static_cast<float>(p[y * w + x]);
                break;
            }
            case ImageDataType::Float32: {
                auto* p = res->getBandData<float>(bandIndex);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        m.at<float>(y, x) = p[y * w + x];
                break;
            }
            case ImageDataType::Float64: {
                auto* p = res->getBandData<double>(bandIndex);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        m.at<float>(y, x) = static_cast<float>(p[y * w + x]);
                break;
            }
            default:
                m.setTo(std::numeric_limits<float>::quiet_NaN());
                break;
            }
            return m;
        };

        cv::Mat lband = toFloatMat(lres, 0);
        cv::Mat rband = toFloatMat(rres, 0);

        // 生成 NoData 检测值（若 ReadResult 提供了 noDataValues，则使用第一个值）
        bool leftHasNoData = !lres->noDataValues.empty();
        double leftNoDataVal = leftHasNoData ? lres->noDataValues[0] : std::numeric_limits<double>::quiet_NaN();
        bool rightHasNoData = !rres->noDataValues.empty();
        double rightNoDataVal = rightHasNoData ? rres->noDataValues[0] : std::numeric_limits<double>::quiet_NaN();

        // 构造输出偏移缓冲（与左块尺寸相同），并初始化为 NaN
        cv::Mat offsetX(lres->height, lres->width, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
        cv::Mat offsetY(lres->height, lres->width, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

        // 遍历左块像素（跳过边缘，保证模板完整）
        for (int ly = templHalf; ly < lres->height - templHalf; ++ly) {
            for (int lx = templHalf; lx < lres->width - templHalf; ++lx) {
                // 全局左影像像素坐标
                int leftGlobalX = lspec.readX + lx;
                int leftGlobalY = lspec.readY + ly;

                // 判断左像素是否为 NoData（简单以中心像素判断）
                bool leftIsNoData = false;
                if (leftHasNoData) {
                    float v = lband.at<float>(ly, lx);
                    if (std::isfinite(static_cast<double>(v)) && v == static_cast<float>(leftNoDataVal)) leftIsNoData = true;
                    // NaN 也视为无效
                    if (!std::isfinite(v)) leftIsNoData = true;
                } else {
                    float v = lband.at<float>(ly, lx);
                    if (!std::isfinite(v)) leftIsNoData = true; // 若数据本身为 NaN
                }

                // 将左像素位置映射到地理坐标，再映射到右影像像素坐标（浮点）
                double geoX = 0.0, geoY = 0.0;
                linfo->geoTransform.pixelToGeo(static_cast<double>(leftGlobalX), static_cast<double>(leftGlobalY), geoX, geoY);
                double rightPxF = 0.0, rightPyF = 0.0;
                bool okMap = rinfo->geoTransform.geoToPixel(geoX, geoY, rightPxF, rightPyF);
                if (!okMap) continue;

                // 计算右影像在右块缓冲中的中心坐标（浮点）
                double rightBufCx = rightPxF - rspec.readX;
                double rightBufCy = rightPyF - rspec.readY;

                // 搜索窗口范围（在右块缓冲坐标）
                int searchX0 = static_cast<int>(std::floor(rightBufCx)) - searchRadius;
                int searchY0 = static_cast<int>(std::floor(rightBufCy)) - searchRadius;
                int searchX1 = static_cast<int>(std::floor(rightBufCx)) + searchRadius;
                int searchY1 = static_cast<int>(std::floor(rightBufCy)) + searchRadius;

                // 保证搜索窗口和模板在右块缓冲内
                int rW = rres->width;
                int rH = rres->height;
                if (searchX0 < 0) searchX0 = 0;
                if (searchY0 < 0) searchY0 = 0;
                if (searchX1 >= rW) searchX1 = rW - 1;
                if (searchY1 >= rH) searchY1 = rH - 1;

                int sw = searchX1 - searchX0 + 1;
                int sh = searchY1 - searchY0 + 1;
                if (sw <= templW || sh <= templW) continue; // 搜索窗口必须大于模板

                // 提取模板（左）与搜索区域（右）作为 float 矩阵
                cv::Rect templRect(lx - templHalf, ly - templHalf, templW, templW);
                cv::Mat templ = lband(templRect);
                // 检查模板是否主要为 NoData (若中心 NoData 且模板绝大多数 NoData，则跳过)
                if (leftHasNoData) {
                    // 若模板中心就是 NoData，跳过
                    float centerVal = templ.at<float>(templHalf, templHalf);
                    if (!std::isfinite(centerVal) || centerVal == static_cast<float>(leftNoDataVal)) {
                        continue;
                    }
                } else {
                    float centerVal = templ.at<float>(templHalf, templHalf);
                    if (!std::isfinite(centerVal)) continue;
                }

                cv::Rect searchRect(searchX0, searchY0, sw, sh);
                cv::Mat searchRegion = rband(searchRect);

                // 若搜索区域或模板包含大量无效值，跳过以避免误匹配
                // 简单策略：require center pixel of candidate to be valid when considering result later.

                // 使用 matchTemplate 查找最佳匹配（SSD 归一化）
                cv::Mat result;
                // matchTemplate 要求模板尺寸 <= 搜索区域
                cv::matchTemplate(searchRegion, templ, result, cv::TM_SQDIFF_NORMED);

                // 找最小值位置
                double minVal, maxVal;
                cv::Point minLoc, maxLoc;
                cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

                // minLoc 为搜索区域内的左上角位置，使模板最佳匹配
                // 计算对应的右影像像素中心位置（绝对右影像像素坐标）
                int matchTopLeftX = searchX0 + minLoc.x;
                int matchTopLeftY = searchY0 + minLoc.y;
                double matchCenterRightX = static_cast<double>(rspec.readX + matchTopLeftX + templHalf);
                double matchCenterRightY = static_cast<double>(rspec.readY + matchTopLeftY + templHalf);

                // 判断右侧对应中心像素是否为 NoData
                bool rightCenterNoData = false;
                if (rightHasNoData) {
                    int rx = static_cast<int>(std::round(matchCenterRightX)) - rspec.readX;
                    int ry = static_cast<int>(std::round(matchCenterRightY)) - rspec.readY;
                    if (rx < 0 || rx >= rres->width || ry < 0 || ry >= rres->height) rightCenterNoData = true;
                    else {
                        float rv = rband.at<float>(ry, rx);
                        if (!std::isfinite(rv) || rv == static_cast<float>(rightNoDataVal)) rightCenterNoData = true;
                    }
                } else {
                    int rx = static_cast<int>(std::round(matchCenterRightX)) - rspec.readX;
                    int ry = static_cast<int>(std::round(matchCenterRightY)) - rspec.readY;
                    if (rx < 0 || rx >= rres->width || ry < 0 || ry >= rres->height) rightCenterNoData = true;
                    else {
                        float rv = rband.at<float>(ry, rx);
                        if (!std::isfinite(rv)) rightCenterNoData = true;
                    }
                }

                // 如果两侧均为无效，则跳过
                if (leftIsNoData && rightCenterNoData) continue;

                // 计算偏移：右影像像素坐标（浮点） - 左影像全局像素坐标（浮点）
                float dx = static_cast<float>(matchCenterRightX - static_cast<double>(leftGlobalX));
                float dy = static_cast<float>(matchCenterRightY - static_cast<double>(leftGlobalY));

                offsetX.at<float>(ly, lx) = dx;
                offsetY.at<float>(ly, lx) = dy;
            }
        }

        // 将 offsetX/offsetY 写入到合并画布（按合并画布像素坐标）
        // 计算左块左上角（全局）像素在合并画布的像素坐标
        double geoX0 = 0.0, geoY0 = 0.0;
        // leftGlobal origin pixel coords = lspec.readX, lspec.readY
        linfo->geoTransform.pixelToGeo(static_cast<double>(lspec.readX), static_cast<double>(lspec.readY), geoX0, geoY0);
        double outPxF = 0.0, outPyF = 0.0;
        bool okOut = cb.combinedGT.geoToPixel(geoX0, geoY0, outPxF, outPyF);
        if (okOut && writer.isOpen()) {
            int outX = static_cast<int>(std::floor(outPxF + 0.5));
            int outY = static_cast<int>(std::floor(outPyF + 0.5));

            // 准备连续缓冲并写入（注意 writer 接受 float* 指向行主序连续数据）
            // 我们需要把 offset mats 截取为写入区域（可能超出合并画布边界）
            int writeW = lres->width;
            int writeH = lres->height;
            // 若超出合并画布边界则裁剪
            if (outX < 0) {
                int dx = -outX;
                if (dx >= writeW) { /*全部超出*/ writeW = 0; }
                else {
                    // shift data pointer horizontally by dx
                    // We'll copy to temp buffer below
                    outX = 0;
                }
            }
            if (outY < 0) {
                int dy = -outY;
                if (dy >= writeH) { writeH = 0; }
                else {
                    outY = 0;
                }
            }
            if (outX + writeW > cb.combinedWidth) writeW = cb.combinedWidth - outX;
            if (outY + writeH > cb.combinedHeight) writeH = cb.combinedHeight - outY;

            if (writeW > 0 && writeH > 0) {
                // 创建连续缓冲并复制数据行（写区域以左块内部相同起点）
                std::vector<float> bufDx(writeW * writeH, std::numeric_limits<float>::quiet_NaN());
                std::vector<float> bufDy(writeW * writeH, std::numeric_limits<float>::quiet_NaN());
                for (int y = 0; y < writeH; ++y) {
                    int srcY = y;
                    int dstY = y;
                    for (int x = 0; x < writeW; ++x) {
                        int srcX = x;
                        bufDx[dstY * writeW + x] = offsetX.at<float>(srcY, srcX);
                        bufDy[dstY * writeW + x] = offsetY.at<float>(srcY, srcX);
                    }
                }
                const float* bandsPtr[2] = { bufDx.data(), bufDy.data() };
                writer.writeBlock(outX, outY, writeW, writeH, bandsPtr);
            }
        }

        // 释放
        RSTools_DestroyReadResult(rres);
        RSTools_DestroyReadResult(lres);
    }
    }

    // 释放 info（由 DLL 分配，必须调用对应销毁接口）
    RSTools_DestroyImageInfo(linfo);

    // 释放 info（由 DLL 分配，必须调用对应销毁接口）
    RSTools_DestroyImageInfo(rinfo);

    std::cout << "---- 完成 ----\n";
    return 0;
}
