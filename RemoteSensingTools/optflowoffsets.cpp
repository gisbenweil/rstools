#include "optflowoffset.h"

// 计算光流的主函数 - 使用标准视频模块
OpticalFlowResult calculateOpticalFlowStandard(
    const ReadResult& prevBlock,
    const ReadResult& currBlock,
    double pyramidScale ,
    int pyramidLevels ,
    int windowSize ,
    int iterations ,
    int polyN ,
    double polySigma ) {

    OpticalFlowResult result;

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


