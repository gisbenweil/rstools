#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include <memory>
#include <GDALImageBase.h>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <chrono>




class OpticalFlowOffset { 
};

class OpticalFlowTracker {
private:
    std::vector<cv::Point2f> l_points, r_points;
    cv::Mat l_gray, r_gray;
    cv::Size winSize;
    int maxLevel;
    cv::TermCriteria criteria;

public:
    OpticalFlowTracker()
        : winSize(21, 21),
        maxLevel(3),
        criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01) {
    }

    void processFrame(cv::Mat& frame) {
        cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);

        if (!prev_gray.empty()) {
            // 计算光流
            std::vector<uchar> status;
            std::vector<float> err;

            cv::calcOpticalFlowPyrLK(prev_gray, curr_gray,
                prev_points, curr_points,
                status, err,
                winSize, maxLevel,
                criteria);

            // 筛选好的跟踪点
            std::vector<cv::Point2f> good_prev, good_curr;
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i] == 1 && err[i] < 20.0) {
                    good_prev.push_back(prev_points[i]);
                    good_curr.push_back(curr_points[i]);
                }
            }

            // 绘制光流
            drawOpticalFlow(frame, good_prev, good_curr);

            // 更新点为下一帧准备
            prev_points = good_curr;

            // 如果点太少，重新检测
            if (prev_points.size() < 20) {
                detectFeatures(curr_gray);
            }
        }
        else {
            // 第一帧，检测特征点
            detectFeatures(curr_gray);
        }

        // 更新前一帧
        curr_gray.copyTo(prev_gray);
    }

private:
    void detectFeatures(const cv::Mat& gray) {

        cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
        std::vector<cv::KeyPoint> kps1, kps2;
        cv::Mat desc1, desc2;

        orb->detectAndCompute(l_gray, cv::noArray(), kps1, desc1);
        orb->detectAndCompute(r_gray, cv::noArray(), kps2, desc2);

        // 转换为Point2f用于LK
        std::vector<cv::Point2f> points1, points2;
        cv::KeyPoint::convert(kps1, points1);
        cv::KeyPoint::convert(kps2, points2);

        // 使用LK光流精炼匹配
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(frame1, frame2, points1, points2,
            status, err);
        prev_points.clear();
        cv::goodFeaturesToTrack(gray, prev_points,
            100, 0.3, 7, cv::Mat(), 7, false, 0.04);
    }

    void drawOpticalFlow(cv::Mat& frame,
        const std::vector<cv::Point2f>& prev,
        const std::vector<cv::Point2f>& curr) {
        for (size_t i = 0; i < prev.size(); i++) {
            // 绘制轨迹线
            cv::arrowedLine(frame, prev[i], curr[i],
                cv::Scalar(0, 255, 0), 2, 8, 0, 0.3);

            // 绘制当前点
            cv::circle(frame, curr[i], 3, cv::Scalar(0, 0, 255), -1);

            // 绘制前一帧点（半透明）
            cv::circle(frame, prev[i], 3, cv::Scalar(255, 255, 0, 128), -1);
        }
    }
};


// 工具函数：获取数据类型大小
static size_t getDataTypeSize(ImageDataType type) {
    switch (type) {
    case Byte: return sizeof(unsigned char);
    case UInt16: return sizeof(unsigned short);
    case Int16: return sizeof(short);
    case UInt32: return sizeof(unsigned int);
    case Int32: return sizeof(int);
    case Float32: return sizeof(float);
    case Float64: return sizeof(double);
    default: return 0;
    }
};

// 光流结果结构
struct OpticalFlowResult {
    cv::Mat flowX;      // X方向光流
    cv::Mat flowY;      // Y方向光流
    cv::Mat magnitude;  // 光流幅度
    cv::Mat angle;      // 光流方向
    bool success;
    std::string errorMessage;
};
OpticalFlowResult calculateOpticalFlowStandard(const ReadResult& prevBlock,
    const ReadResult& currBlock,
    double pyramidScale = 0.5,
    int pyramidLevels = 3,
    int windowSize = 15,
    int iterations = 3,
    int polyN = 5,
    double polySigma = 1.1);



// 辅助函数：将ReadResult转换为OpenCV Mat
cv::Mat convertToCVMat(const ReadResult& result, int bandIndex = 0);

cv::Mat toFloatMat(const ReadResult* res, int bandIndex = 0);