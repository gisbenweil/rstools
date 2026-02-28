#include "optflowoffsets.h"

#include <fstream>
#include <filesystem>
#include <cmath>
#include <numeric>
#include "../RSTools/ImageBlockWriter.h"


#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

/**
 * 抽稀特征点 - 改进版混合特征检测器
 * 使用距离阈值和网格划分来减少特征点密度
 */
std::vector<cv::Point2f> detectHybridFeatures(const cv::Mat& gray) {
    std::vector<cv::Point2f> allPoints;

    // 设置最小距离阈值，用于控制特征点密度
    float minDistance = 15.0f;

    // 1. 角点特征
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, 200, 0.01, minDistance);
    allPoints.insert(allPoints.end(), corners.begin(), corners.end());

    // 2. FAST特征
    std::vector<cv::KeyPoint> fastKeypoints;
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
    fast->detect(gray, fastKeypoints);

    std::vector<cv::Point2f> fastPoints;
    cv::KeyPoint::convert(fastKeypoints, fastPoints);

    // 去除与角点太近的FAST点
    for (const auto& pt : fastPoints) {
        bool tooClose = false;
        for (const auto& corner : corners) {
            if (cv::norm(pt - corner) < minDistance) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) {
            allPoints.push_back(pt);
        }
    }

    // 3. 网格采样补充（仅在低密度区域）
    int gridSize = 40; // 增大网格尺寸以降低密度
    std::vector<std::vector<bool>> gridOccupied(gray.rows / gridSize + 1,
        std::vector<bool>(gray.cols / gridSize + 1, false));

    // 标记已有特征点所在的网格
    for (const auto& pt : allPoints) {
        int gridX = static_cast<int>(pt.x) / gridSize;
        int gridY = static_cast<int>(pt.y) / gridSize;
        if (gridX < gridOccupied[0].size() && gridY < gridOccupied.size()) {
            gridOccupied[gridY][gridX] = true;
        }
    }

    // 在空网格中补充特征点
    for (int y = gridSize; y < gray.rows; y += gridSize) {
        for (int x = gridSize; x < gray.cols; x += gridSize) {
            int gridX = x / gridSize;
            int gridY = y / gridSize;

            if (gridX < gridOccupied[0].size() && gridY < gridOccupied.size() && !gridOccupied[gridY][gridX]) {
                allPoints.emplace_back(x, y);
            }
        }
    }

    // 4. 最终抽稀 - 使用非极大值抑制策略
    std::vector<cv::Point2f> refinedPoints;
    std::vector<bool> kept(allPoints.size(), true);

    for (size_t i = 0; i < allPoints.size(); ++i) {
        if (!kept[i]) continue;

        for (size_t j = i + 1; j < allPoints.size(); ++j) {
            if (cv::norm(allPoints[i] - allPoints[j]) < minDistance) {
                kept[j] = false; // 标记为移除
            }
        }
        refinedPoints.push_back(allPoints[i]);
    }

    return refinedPoints;
}

/**
 * 额外的抽稀函数 - 使用K-means聚类进一步减少特征点数量
 */
std::vector<cv::Point2f> sparsifyWithClustering(const std::vector<cv::Point2f>& points, int maxPoints = 100) {
    if (points.size() <= maxPoints) {
        return points; // 不需要抽稀
    }

    // 使用K-means聚类减少点数
    cv::Mat data(points.size(), 2, CV_32F);
    for (size_t i = 0; i < points.size(); ++i) {
        data.at<float>(i, 0) = points[i].x;
        data.at<float>(i, 1) = points[i].y;
    }

    std::vector<int> labels;
    cv::Mat centers;
    cv::kmeans(data, maxPoints, labels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
        3, cv::KMEANS_PP_CENTERS, centers);

    // 从每个聚类中心获取代表性点
    std::vector<cv::Point2f> result;
    for (int i = 0; i < maxPoints; ++i) {
        result.emplace_back(centers.at<float>(i, 0), centers.at<float>(i, 1));
    }

    return result;
}

// 组合使用的示例函数
std::vector<cv::Point2f> detectAndSparsifyFeatures(const cv::Mat& gray, int maxPoints = 100) {
    auto initialPoints = detectHybridFeatures(gray);
    return sparsifyWithClustering(initialPoints, maxPoints);
}


void removeOutliersUsingStats(
    std::vector<cv::Point2f>& points1,
    std::vector<cv::Point2f>& points2,
    std::vector<uchar>& status,
    float maxStdDevMultiplier) {

    // 1. 计算运动向量
    std::vector<cv::Point2f> motions;
    std::vector<float> magnitudes;

    for (size_t i = 0; i < status.size(); i++) {
        if (status[i] == 1) {
            cv::Point2f motion = points2[i] - points1[i];
            motions.push_back(motion);
            magnitudes.push_back(cv::norm(motion));
        }
    }

    if (motions.empty()) return;

    // 2. 计算运动向量的均值和标准差
    cv::Scalar mean_motion, stddev_motion;
    cv::meanStdDev(motions, mean_motion, stddev_motion);

    // 3. 计算运动幅度的均值和标准差
    cv::Scalar mean_mag, stddev_mag;
    cv::meanStdDev(magnitudes, mean_mag, stddev_mag);

    // 4. 基于3σ原则过滤异常点
    float max_dx = mean_motion[0] + maxStdDevMultiplier * stddev_motion[0];
    float min_dx = mean_motion[0] - maxStdDevMultiplier * stddev_motion[0];
    float max_dy = mean_motion[1] + maxStdDevMultiplier * stddev_motion[1];
    float min_dy = mean_motion[1] - maxStdDevMultiplier * stddev_motion[1];
    float max_mag = mean_mag[0] + maxStdDevMultiplier * stddev_mag[0];

    // 5. 过滤异常点
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i] == 1) {
            cv::Point2f motion = points2[i] - points1[i];
            float mag = cv::norm(motion);

            // 检查是否超出统计范围
            if (motion.x < min_dx || motion.x > max_dx ||
                motion.y < min_dy || motion.y > max_dy ||
                mag > max_mag) {
                status[i] = 0;  // 标记为异常点
            }
        }
    }
}

SparseOpticalFlowResult OpticalFlowOffset::calculateOpticalFlowOffset(
    const ReadResult& prevBlock,
    const ReadResult& currBlock) {
    SparseOpticalFlowResult result;
    try {
        // 检查输入有效性
        if (!prevBlock.success || !currBlock.success) {
            result.success = false;
            result.errorMessage = "One or both input blocks are invalid";
            return result;
        }

        // 处理不同尺寸的影像，将较小的影像填充到较大影像的尺寸
        cv::Mat prev_gray_original, curr_gray_original;
        if (prevBlock.bands > 1) {
            prev_gray_original = convertToCVMat(prevBlock, 0);
            curr_gray_original = convertToCVMat(currBlock, 0);
        }
        else {
            prev_gray_original = convertToCVMat(prevBlock, 0);
            curr_gray_original = convertToCVMat(currBlock, 0);
        }

        if (prev_gray_original.empty() || curr_gray_original.empty()) {
            result.success = false;
            result.errorMessage = "One or both input frames are empty";
            return result;
        }

        // 检查尺寸是否不同，如果不同则填充较小的影像
        cv::Mat prev_gray, curr_gray;
        if (prev_gray_original.size() != curr_gray_original.size()) {
            // 获取最大尺寸
            int max_rows = std::max(prev_gray_original.rows, curr_gray_original.rows);
            int max_cols = std::max(prev_gray_original.cols, curr_gray_original.cols);

            // 填充前影像
            if (prev_gray_original.rows < max_rows || prev_gray_original.cols < max_cols) {
                int pad_top = 0, pad_bottom = max_rows - prev_gray_original.rows;
                int pad_left = 0, pad_right = max_cols - prev_gray_original.cols;
                cv::copyMakeBorder(prev_gray_original, prev_gray, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));
            }
            else {
                prev_gray = prev_gray_original;
            }

            // 填充后影像
            if (curr_gray_original.rows < max_rows || curr_gray_original.cols < max_cols) {
                int pad_top = 0, pad_bottom = max_rows - curr_gray_original.rows;
                int pad_left = 0, pad_right = max_cols - curr_gray_original.cols;
                cv::copyMakeBorder(curr_gray_original, curr_gray, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));
            }
            else {
                curr_gray = curr_gray_original;
            }

            // 确保填充后的图像尺寸一致
            if (prev_gray.size() != curr_gray.size()) {
                // 如果仍然不一致，调整为相同尺寸
                cv::resize(curr_gray, curr_gray, prev_gray.size());
            }
        }
        else {
            // 如果尺寸相同，则直接使用
            prev_gray = prev_gray_original;
            curr_gray = curr_gray_original;
        }

        // 使用混合特征检测获得稀疏点
        std::vector<cv::Point2f> points1 = detectAndSparsifyFeatures(prev_gray, 500);
        std::vector<cv::Point2f> points2 = detectAndSparsifyFeatures(curr_gray, 500);

        // 使用LK光流精炼匹配（注意：这里调用者传入的points2并非直接对应，但后续筛选基于status/err）
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, points1, points2, status, err, winSize, maxLevel, criteria);

        // 可视化（可选）
        cv::Mat frame1_color, frame2_color;
        cv::cvtColor(prev_gray, frame1_color, cv::COLOR_GRAY2BGR);
        cv::cvtColor(prev_gray, frame2_color, cv::COLOR_GRAY2BGR);

        std::vector<cv::Point2f> good_points1;
        std::vector<cv::Point2f> good_points2;

        removeOutliersUsingStats(points1, points2, status, 0.5);

        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] == 1 && err[i] < 30.0) {
                good_points1.push_back(points1[i]);
                good_points2.push_back(points2[i]);
            }
        }

        for (size_t i = 0; i < good_points1.size(); i++) {
            cv::circle(frame1_color, good_points1[i], 3, cv::Scalar(0, 255, 0), -1);
            cv::circle(frame1_color, good_points2[i], 3, cv::Scalar(0, 0, 255), -1);
        }

        //cv::Mat combined;
        //cv::hconcat(frame1_color, frame2_color, combined);
        for (size_t i = 0; i < good_points1.size(); i++) {
            cv::Point pt2_in_combined = cv::Point(
                good_points2[i].x,
                good_points2[i].y
            );
            cv::line(frame1_color, good_points1[i], pt2_in_combined,
                cv::Scalar(255, 255, 0), 1);
        }

        //cv::imshow("Optical Flow", frame1_color);
        //cv::waitKey(0);

        result.prevPoints = good_points1;
        result.currPoints = good_points2;
        result.status = status;
        result.success = true;
    }
    catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Error calculating optical flow offset: ") + e.what();
    }
    return result;
}


// -------------------- Dense Farneback function (unchanged) --------------------
DenseOpticalFlowResult calculateOpticalFlowStandard(
    const ReadResult& prevBlock,
    const ReadResult& currBlock,
    double pyramidScale,
    int pyramidLevels,
    int windowSize,
    int iterations,
    int polyN,
    double polySigma) {

    DenseOpticalFlowResult result;

    try {
        // 检查输入有效性
        if (!prevBlock.success || !currBlock.success) {
            result.success = false;
            result.errorMessage = "One or both input blocks are invalid";
            return result;
        }

        if (prevBlock.width != currBlock.width ||
            prevBlock.height != currBlock.height) {
            result.success = false;
            result.errorMessage = "Block dimensions do not match";
            return result;
        }

        // 将数据转换为灰度图像
        cv::Mat prevGray, currGray;

        if (prevBlock.bands > 1) {
            prevGray = convertToCVMat(prevBlock, 0);
            currGray = convertToCVMat(currBlock, 0);
        }
        else {
            prevGray = convertToCVMat(prevBlock, 0);
            currGray = convertToCVMat(currBlock, 0);
        }

        // 确保图像大小一致
        if (prevGray.size() != currGray.size()) {
            cv::resize(currGray, currGray, prevGray.size());
        }

        // 应用高斯模糊减少噪声
        cv::GaussianBlur(prevGray, prevGray, cv::Size(5, 5), 0);
        cv::GaussianBlur(currGray, currGray, cv::Size(5, 5), 0);

        // 计算光流 - 使用Farneback算法
        cv::Mat flow(prevGray.size(), CV_32FC2);

        cv::calcOpticalFlowFarneback(
            prevGray, currGray, flow,
            pyramidScale,    // 金字塔缩放因子
            pyramidLevels,   // 金字塔层数
            windowSize,      // 窗口大小
            iterations,      // 迭代次数
            polyN,           // 多项式邻域大小
            polySigma,       // 多项式标准差
            0               // 标志位
        );

        // 分离X和Y方向的光流
        std::vector<cv::Mat> flowChannels;
        cv::split(flow, flowChannels);
        result.flowX = flowChannels[0];
        result.flowY = flowChannels[1];

        // 计算光流幅度和方向
        cv::cartToPolar(result.flowX, result.flowY,
            result.magnitude, result.angle, true);

        result.success = true;

    }
    catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Error calculating optical flow: ") + e.what();
    }

    return result;
}

// 将左/右缓冲的第一波段转换为 CV_32F 矩阵（以读出的缓冲区为坐标）
cv::Mat toFloatMat(const ReadResult* res, int bandIndex) {
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


cv::Mat convertToCVMat(const ReadResult& result, int bandIndex) {
    if (!result.success || !result.data) {
        throw std::runtime_error("Invalid data in ReadResult");
    }

    if (bandIndex >= result.bands) {
        throw std::runtime_error("Band index out of range");
    }

    // 计算波段数据起始位置
    size_t bandOffset = 0;
    if (!result.bandOffsets.empty() && bandIndex < result.bandOffsets.size()) {
        bandOffset = result.bandOffsets[bandIndex];
    }
    else {
        // 如果没有指定偏移量，假设数据是交错存储的
        bandOffset = bandIndex * (result.width * result.height *
            getDataTypeSize(result.dataType));
    }

    char* bandData = static_cast<char*>(result.data) + bandOffset;

    // 根据数据类型创建Mat
    cv::Mat mat;
    int cvType = 0;
    double scale = 1.0;

    switch (result.dataType) {
    case Byte:
        cvType = CV_8UC1;
        mat = cv::Mat(result.height, result.width, cvType, bandData);
        break;

    case UInt16:
        cvType = CV_16UC1;
        scale = 255.0 / 65535.0; // 缩放到0-255范围
        mat = cv::Mat(result.height, result.width, cvType, bandData);
        mat.convertTo(mat, CV_8UC1, scale);
        break;

    case Int16:
        cvType = CV_16SC1;
        mat = cv::Mat(result.height, result.width, cvType, bandData);
        // 将16位有符号转换为8位无符号
        mat.convertTo(mat, CV_8UC1, 0.5, 128);
        break;

    case Float32:
        cvType = CV_32FC1;
        mat = cv::Mat(result.height, result.width, cvType, bandData);
        // 归一化到0-255
        {
            double minVal, maxVal;
            cv::minMaxLoc(mat, &minVal, &maxVal);
            if (maxVal - minVal > 1e-9) {
                mat = (mat - minVal) * (255.0 / (maxVal - minVal));
            }
            else {
                mat = cv::Mat::zeros(mat.size(), mat.type());
            }
            mat.convertTo(mat, CV_8UC1);
        }
        break;

    case Float64:
        cvType = CV_64FC1;
        mat = cv::Mat(result.height, result.width, cvType, bandData);
        // 归一化到0-255
        {
            double minVal, maxVal;
            cv::minMaxLoc(mat, &minVal, &maxVal);
            if (maxVal - minVal > 1e-9) {
                mat = (mat - minVal) * (255.0 / (maxVal - minVal));
            }
            else {
                mat = cv::Mat::zeros(mat.size(), mat.type());
            }
            mat.convertTo(mat, CV_8UC1);
        }
        break;

    default:
        throw std::runtime_error("Unsupported data type");
    }

    // 处理NoData值
    if (!result.noDataValues.empty() && bandIndex < result.noDataValues.size()) {
        double noDataValue = result.noDataValues[bandIndex];
        cv::Mat mask = (mat == noDataValue);
        cv::inpaint(mat, mask, mat, 3, cv::INPAINT_TELEA);
    }

    return mat.clone(); // 返回深拷贝
}
