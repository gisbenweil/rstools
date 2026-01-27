#include "optflowoffsets.h"

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

        if (prevBlock.width != currBlock.width ||
            prevBlock.height != currBlock.height) {
            result.success = false;
            result.errorMessage = "Block dimensions do not match";
            return result;
        }

        // 将数据转换为灰度图像
        //cv::Mat prevGray, currGray;

        if (prevBlock.bands > 1) {
            prev_gray = convertToCVMat(prevBlock, 0);
            curr_gray = convertToCVMat(currBlock, 0);
        }
        else {
            prev_gray = convertToCVMat(prevBlock, 0);
            curr_gray = convertToCVMat(currBlock, 0);
        }

        // 确保图像大小一致
        //if (prevGray.size() != currGray.size()) {
        //    cv::resize(currGray, currGray, prevGray.size());
        //}
		//cv::imshow("Prev Gray", prev_gray);
		//cv::imshow("Curr Gray", curr_gray);
		//cv::waitKey(0);



        if (prev_gray.empty() || curr_gray.empty()) {
            result.success = false;
            result.errorMessage = "One or both input frames are empty";
            return result;
        }
        // 方法1：结合ORB特征和LK光流
        //cv::Ptr<cv::ORB> orb = cv::ORB::create(100);
        //std::vector<cv::KeyPoint> kps1, kps2;
        //cv::Mat desc1, desc2;

        //orb->detectAndCompute(prev_gray, cv::noArray(), kps1, desc1);
        //orb->detectAndCompute(prev_gray, cv::noArray(), kps2, desc2);



        //// 转换为Point2f用于LK
        std::vector<cv::Point2f> points1, points2;
        //cv::KeyPoint::convert(kps1, points1);
        //cv::KeyPoint::convert(kps2, points2);


		points1 = detectHybridFeatures(prev_gray);
		points2 = detectHybridFeatures(curr_gray);

        // 使用LK光流精炼匹配
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray, prev_gray, points1, points2,
            status, err);
        // 4. 可视化结果
        cv::Mat frame1_color, frame2_color;
        cv::cvtColor(prev_gray, frame1_color, cv::COLOR_GRAY2BGR);
        cv::cvtColor(prev_gray, frame2_color, cv::COLOR_GRAY2BGR);

        // 3. 筛选出跟踪成功的点
        std::vector<cv::Point2f> good_points1;
        std::vector<cv::Point2f> good_points2;

        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] == 1 && err[i] < 30.0) {  // 状态为1表示成功跟踪
                good_points1.push_back(points1[i]);
                good_points2.push_back(points2[i]);
            }
        }

        // 绘制跟踪点
        for (size_t i = 0; i < good_points1.size(); i++) {
            // 第一帧中的点（绿色）
            cv::circle(frame1_color, good_points1[i], 3, cv::Scalar(0, 255, 0), -1);
            // 第二帧中的点（红色）
            cv::circle(frame2_color, good_points2[i], 3, cv::Scalar(0, 0, 255), -1);
        }

        // 绘制运动轨迹
        cv::Mat combined;
        cv::hconcat(frame1_color, frame2_color, combined);

        for (size_t i = 0; i < good_points1.size(); i++) {
            cv::Point pt2_in_combined = cv::Point(
                good_points2[i].x + prev_gray.cols,
                good_points2[i].y
            );
            cv::line(combined, good_points1[i], pt2_in_combined,
                cv::Scalar(255, 255, 0), 1);
        }

        // 5. 保存和显示结果
        //cv::imwrite("optical_flow_result.jpg", combined);

        cv::imshow("Optical Flow", combined);
        cv::waitKey(0);
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


// 计算光流的主函数 - 使用标准视频模块
DenseOpticalFlowResult calculateOpticalFlowStandard(
    const ReadResult& prevBlock,
    const ReadResult& currBlock,
    double pyramidScale ,
    int pyramidLevels ,
    int windowSize ,
    int iterations ,
    int polyN ,
    double polySigma ) {

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
cv::Mat toFloatMat (const ReadResult* res, int bandIndex) {
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


cv::Mat convertToCVMat(const ReadResult& result, int bandIndex ) {
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
        double minVal, maxVal;
        cv::minMaxLoc(mat, &minVal, &maxVal);
        mat = (mat - minVal) * (255.0 / (maxVal - minVal));
        mat.convertTo(mat, CV_8UC1);
        break;

    case Float64:
        cvType = CV_64FC1;
        mat = cv::Mat(result.height, result.width, cvType, bandData);
        // 归一化到0-255
        cv::minMaxLoc(mat, &minVal, &maxVal);
        mat = (mat - minVal) * (255.0 / (maxVal - minVal));
        mat.convertTo(mat, CV_8UC1);
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


// 多种特征检测器组合
std::vector<cv::Point2f> detectHybridFeatures(const cv::Mat& gray) {
    std::vector<cv::Point2f> allPoints;

    // 1. 角点特征
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, 200, 0.01, 10);
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
            if (cv::norm(pt - corner) < 10) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) {
            allPoints.push_back(pt);
        }
    }

    // 3. 网格采样补充
    int gridSize = 30;
    for (int y = gridSize; y < gray.rows; y += gridSize) {
        for (int x = gridSize; x < gray.cols; x += gridSize) {
            // 检查该区域是否已有特征点
            bool hasPoint = false;
            for (const auto& pt : allPoints) {
                if (std::abs(pt.x - x) < gridSize / 2 &&
                    std::abs(pt.y - y) < gridSize / 2) {
                    hasPoint = true;
                    break;
                }
            }
            if (!hasPoint) {
                allPoints.emplace_back(x, y);
            }
        }
    }

    return allPoints;
}