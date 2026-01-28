#include "optflowoffsets.h"
#include "../RSTools/ImageBlockWriter.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <map>
#include <opencv2/flann.hpp>

using namespace std;

static double tpsU(double r) {
    if (r <= 1e-12) return 0.0;
    double r2 = r * r;
    return r2 * log(r2);
}

bool OpticalFlowOffset::computeDenseToBlocks(
    const std::vector<cv::Point2f>& prevPts,
    const std::vector<cv::Point2f>& currPts,
    const std::vector<uchar>& status,
    int imgWidth,
    int imgHeight,
    int blockWidth,
    int blockHeight,
    std::vector<std::tuple<int,int,int,int>>& blocksOut,
    std::vector<std::vector<float>>& bufXOut,
    std::vector<std::vector<float>>& bufYOut,
    InterpMethod method,
    float kernelSigma,
    int kernelRadius,
    double regularization,
    int maxGlobalPoints) {

    removeOutliersUsingStats(
        const_cast<std::vector<cv::Point2f>&>(prevPts),
        const_cast<std::vector<cv::Point2f>&>(currPts),
        const_cast<std::vector<uchar>&>(status),
		5.0f);

    if (imgWidth <= 0 || imgHeight <= 0) return false;

    // collect valid matched points
    std::vector<cv::Point2f> p0, p1;
    for (size_t i = 0; i < prevPts.size() && i < currPts.size(); ++i) {
        if (i < status.size() && status[i] == 0) continue;
        p0.push_back(prevPts[i]);
        p1.push_back(currPts[i]);
    }
    if (p0.empty()) return false;

    std::vector<float> valX(p0.size()), valY(p0.size());
    std::vector<double> mags(p0.size());
    for (size_t i = 0; i < p0.size(); ++i) {
        double dx = p1[i].x - p0[i].x;
        double dy = p1[i].y - p0[i].y;
        valX[i] = static_cast<float>(dx);
        valY[i] = static_cast<float>(dy);
        mags[i] = std::sqrt(dx*dx + dy*dy);
    }

    // filter outliers using median + MAD
    auto mags_sorted = mags;
    std::sort(mags_sorted.begin(), mags_sorted.end());
    double median = mags_sorted[mags_sorted.size()/2];
    std::vector<double> absdev(mags.size());
    for (size_t i=0;i<mags.size();++i) absdev[i] = std::abs(mags[i]-median);
    std::sort(absdev.begin(), absdev.end());
    double mad = absdev[absdev.size()/2];
    double thresh = median + std::max(3.0*mad, 3.0);

    std::vector<cv::Point2f> pf0; std::vector<cv::Point2f> pf1; std::vector<float> pvx; std::vector<float> pvy;
    for (size_t i=0;i<p0.size();++i) {
        if (mags[i] <= thresh) {
            pf0.push_back(p0[i]);
            pf1.push_back(p1[i]);
            pvx.push_back(valX[i]);
            pvy.push_back(valY[i]);
        }
    }
    if (pf0.empty()) return false;

    // prepare blocks list
    blocksOut.clear(); bufXOut.clear(); bufYOut.clear();
    for (int by=0; by<imgHeight; by+=blockHeight) {
        for (int bx=0; bx<imgWidth; bx+=blockWidth) {
            int w = std::min(blockWidth, imgWidth - bx);
            int h = std::min(blockHeight, imgHeight - by);
            blocksOut.emplace_back(bx, by, w, h);
            bufXOut.emplace_back(w*h, std::numeric_limits<float>::quiet_NaN());
            bufYOut.emplace_back(w*h, std::numeric_limits<float>::quiet_NaN());
        }
    }

    size_t B = blocksOut.size();
    std::atomic<size_t> blockIndex(0);
    unsigned int numThreads = std::max(1u, std::min((unsigned int)std::thread::hardware_concurrency(), 8u));

    if (method == TPS_GLOBAL) {
        // subsample control points
        std::vector<cv::Point2f> spts; std::vector<double> svalsX; std::vector<double> svalsY;
        if ((int)pf0.size() > maxGlobalPoints) {
            int gx = static_cast<int>(std::sqrt(maxGlobalPoints));
            int gy = gx;
            std::vector<int> count(gx*gy,0);
            std::vector<double> sumx(gx*gy,0), sumy(gx*gy,0), sumvx(gx*gy,0), sumvy(gx*gy,0);
            for (size_t i=0;i<pf0.size();++i) {
                int cx = std::min(gx-1, std::max(0, static_cast<int>(pf0[i].x * gx / std::max(1, imgWidth))));
                int cy = std::min(gy-1, std::max(0, static_cast<int>(pf0[i].y * gy / std::max(1, imgHeight))));
                int idx = cy*gx + cx;
                count[idx]++; sumx[idx]+=pf0[i].x; sumy[idx]+=pf0[i].y; sumvx[idx]+=pvx[i]; sumvy[idx]+=pvy[i];
            }
            for (int j=0;j<gx*gy;++j) if (count[j]>0) {
                spts.emplace_back(sumx[j]/count[j], sumy[j]/count[j]);
                svalsX.emplace_back(sumvx[j]/count[j]);
                svalsY.emplace_back(sumvy[j]/count[j]);
            }
        } else {
            for (size_t i=0;i<pf0.size();++i) { spts.push_back(pf0[i]); svalsX.push_back(pvx[i]); svalsY.push_back(pvy[i]); }
        }

        int N = static_cast<int>(spts.size()); if (N==0) return false;
        cv::Mat A = cv::Mat::zeros(N+3,N+3,CV_64F);
        for (int i=0;i<N;++i) for (int j=0;j<N;++j) { double dx=spts[i].x-spts[j].x; double dy=spts[i].y-spts[j].y; double r=sqrt(dx*dx+dy*dy); A.at<double>(i,j)=tpsU(r); }
        for (int i=0;i<N;++i) A.at<double>(i,i)+=regularization;
        for (int i=0;i<N;++i) { A.at<double>(i,N+0)=1;A.at<double>(i,N+1)=spts[i].x;A.at<double>(i,N+2)=spts[i].y; A.at<double>(N+0,i)=1;A.at<double>(N+1,i)=spts[i].x;A.at<double>(N+2,i)=spts[i].y; }
        cv::Mat rhsX = cv::Mat::zeros(N+3,1,CV_64F), rhsY = cv::Mat::zeros(N+3,1,CV_64F);
        for (int i=0;i<N;++i) { rhsX.at<double>(i,0)=svalsX[i]; rhsY.at<double>(i,0)=svalsY[i]; }
        cv::Mat coeffX, coeffY; if (!cv::solve(A,rhsX,coeffX,cv::DECOMP_SVD)) return false; if (!cv::solve(A,rhsY,coeffY,cv::DECOMP_SVD)) return false;

        // threaded evaluation of blocks
        auto worker = [&](unsigned int threadId){
            size_t idx;
            while ((idx = blockIndex.fetch_add(1)) < B) {
                int bx,by,w,h; std::tie(bx,by,w,h)=blocksOut[idx];
                auto &bufX = bufXOut[idx]; auto &bufY = bufYOut[idx];
                for (int yy=0; yy<h; ++yy) {
                    for (int xx=0; xx<w; ++xx) {
                        double gx = bx + xx; double gy = by + yy; double sumX=0,sumY=0;
                        for (int k=0;k<N;++k) { double dx=gx-spts[k].x; double dy=gy-spts[k].y; double r=sqrt(dx*dx+dy*dy); double Uk=tpsU(r); sumX+=coeffX.at<double>(k,0)*Uk; sumY+=coeffY.at<double>(k,0)*Uk; }
                        sumX += coeffX.at<double>(N+0,0) + coeffX.at<double>(N+1,0)*gx + coeffX.at<double>(N+2,0)*gy;
                        sumY += coeffY.at<double>(N+0,0) + coeffY.at<double>(N+1,0)*gx + coeffY.at<double>(N+2,0)*gy;
                        bufX[yy*w+xx]=static_cast<float>(sumX); bufY[yy*w+xx]=static_cast<float>(sumY);
                    }
                }
            }
        };

        vector<thread> ths; ths.reserve(numThreads);
        for (unsigned int t=0;t<numThreads;++t) ths.emplace_back(worker, t);
        for (auto &t: ths) if (t.joinable()) t.join();
        return true;
    } else {
        // local methods (IDW_LOCAL or approx Kriging)
        const float twoSigma2 = 2.0f * kernelSigma * kernelSigma; const float eps=1e-9f;
        auto workerLocal = [&](unsigned int tid){
            size_t idx;
            while ((idx = blockIndex.fetch_add(1)) < B) {
                int bx,by,w,h; std::tie(bx,by,w,h)=blocksOut[idx];
                auto &bufX = bufXOut[idx]; auto &bufY = bufYOut[idx];
                int ext = kernelRadius*2; int sx = std::max(0,bx-ext); int sy = std::max(0,by-ext); int exb = std::min(imgWidth-1,bx+w+ext); int ey = std::min(imgHeight-1,by+h+ext);
                std::vector<int> idxs; idxs.reserve(pf0.size());
                for (size_t i=0;i<pf0.size();++i) if (pf0[i].x>=sx && pf0[i].x<=exb && pf0[i].y>=sy && pf0[i].y<=ey) idxs.push_back((int)i);
                if (!idxs.empty()) {
                    for (int yy=0; yy<h; ++yy) {
                        for (int xx=0; xx<w; ++xx) {
                            int gx = bx+xx; int gy = by+yy; double numX=0,numY=0,den=0;
                            for (int id: idxs) {
                                double dx = gx - pf0[id].x; double dy = gy - pf0[id].y; double r2 = dx*dx + dy*dy + eps; double wgt = exp(-r2 / twoSigma2);
                                numX += pvx[id] * wgt; numY += pvy[id] * wgt; den += wgt;
                            }
                            if (den > eps) { bufX[yy*w+xx] = static_cast<float>(numX/den); bufY[yy*w+xx] = static_cast<float>(numY/den); }
                        }
                    }
                }
            }
        };

        vector<thread> ths; ths.reserve(numThreads);
        for (unsigned int t=0;t<numThreads;++t) ths.emplace_back(workerLocal, t);
        for (auto &t: ths) if (t.joinable()) t.join();
        return true;
    }
}

bool OpticalFlowOffset::computeDenseFromPointsAndSaveGeoTIFF(
    const std::vector<cv::Point2f>& prevPts,
    const std::vector<cv::Point2f>& currPts,
    const std::vector<uchar>& status,
    int imgWidth,
    int imgHeight,
    int blockWidth,
    int blockHeight,
    const std::string& outPath,
    const GeoTransform& gt,
    const std::string& projectionWkt,
    InterpMethod method,
    float kernelSigma,
    int kernelRadius,
    double regularization,
    int maxGlobalPoints) {

    std::vector<std::tuple<int,int,int,int>> blocksOut;
    std::vector<std::vector<float>> bufXOut, bufYOut;
    if (!computeDenseToBlocks(prevPts, currPts, status, imgWidth, imgHeight,
        blockWidth, blockHeight, blocksOut, bufXOut, bufYOut,
        method, kernelSigma, kernelRadius, regularization, maxGlobalPoints)) {
        return false;
    }

    ImageBlockWriter writer(outPath, imgWidth, imgHeight, 2, gt, projectionWkt);
    if (!writer.isOpen()) return false;

    for (size_t i = 0; i < blocksOut.size(); ++i) {
        int bx, by, w, h; std::tie(bx, by, w, h) = blocksOut[i];
        const float* ptrs[2] = { bufXOut[i].data(), bufYOut[i].data() };
        if (!writer.writeBlock(bx, by, w, h, ptrs)) return false;
    }

    return true;
}
