// RemoteSensingTools.cpp : 测试程序：初始化 GDAL 并读取影像，打印基本信息和左上角像素值。
//

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <sstream>
#include "rstoolsbase.h"
#include "GDALImageBase.h"
#include "GDALImageReader.h"
#include "ImageBlockReader.h"
#include "ImageBlockWriter.h"
#include "opencv2/opencv.hpp"

#include "optflowoffsets.h"

// forward declaration for reconstruction implemented in optflowoffsets_impl.cpp
extern "C" bool RSTools_ReconstructFromOffsets(const char* offsetsPath, const char* prePath, const char* outPath, int blockWidth, int blockHeight);

using namespace cv;
using namespace std;

bool calculateOffsetsAndSave(string lpath,string rpath,string outOffsetPath)
{
	ImageInfo* linfo = RSTools_GetImageInfo(lpath.c_str());
	ImageInfo* rinfo = RSTools_GetImageInfo(rpath.c_str());
	if (!linfo || !rinfo) {
		std::cerr << "RSTools_GetImageInfo 返回空指针，可能无法打开文件或路径错误, 请检查输入文件。\n";
        return false;
	}

    // cleanup helper to ensure ImageInfo are released on all paths
    auto cleanup = [&](bool ret)->bool {
        if (linfo) { RSTools_DestroyImageInfo(linfo); linfo = nullptr; }
        if (rinfo) { RSTools_DestroyImageInfo(rinfo); rinfo = nullptr; }
        return ret;
    };

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
            return cleanup(false);
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
			cb.maxX, cb.maxY, 0);
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

		// --- 进度信息准备 ---
		const int readTileW = 512; // 与 ImageBlockReader 构造一致
		const int readTileH = 512;
		int tilesX = static_cast<int>(std::ceil(leftArea.width / static_cast<double>(readTileW)));
		int tilesY = static_cast<int>(std::ceil(leftArea.height / static_cast<double>(readTileH)));
		int totalTiles = std::max(1, tilesX) * std::max(1, tilesY);
		int processedTiles = 0;
		auto startTime = std::chrono::steady_clock::now();
		auto formatDuration = [](double secs)->std::string {
			int s = static_cast<int>(std::round(std::max(0.0, secs)));
			int h = s / 3600;
			int m = (s % 3600) / 60;
			int ss = s % 60;
			std::ostringstream oss;
			if (h > 0) oss << h << "h";
			if (m > 0 || h > 0) oss << m << "m";
			oss << ss << "s";
			return oss.str();
			};

		// 存储所有稀疏匹配点对
		std::vector<cv::Point2f> allPrevPts;
		std::vector<cv::Point2f> allCurrPts;
		std::vector<uchar> allStatus;

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

			// 计数并输出进度（按每10块或结束时刷新）
			++processedTiles;
			if (processedTiles % 10 == 0 || processedTiles == totalTiles) {
				auto now = std::chrono::steady_clock::now();
				double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - startTime).count();
				double perTile = (processedTiles > 0) ? (elapsed / processedTiles) : 0.0;
				double remaining = std::max(0, totalTiles - processedTiles);
				double eta = perTile * remaining;
				double percent = processedTiles * 100.0 / totalTiles;
				std::cout << "\r处理进度: " << processedTiles << "/" << totalTiles
					<< " (" << std::fixed << std::setprecision(1) << percent << "%)"
					<< " 已用: " << formatDuration(elapsed)
					<< " ETA: " << formatDuration(eta) << "    " << std::flush;
				if (processedTiles == totalTiles) std::cout << std::endl;
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

					// 保存到全局列表
					allPrevPts.push_back(cv::Point2f(prevCombinedX, prevCombinedY));
					allCurrPts.push_back(cv::Point2f(currCombinedX, currCombinedY));
					allStatus.push_back(sparse.status[i]);
				}
			}

			// 释放
			RSTools_DestroyReadResult(rres);
			RSTools_DestroyReadResult(lres);
		}



		// After collecting sparse matches per output block, flatten to global lists for densification
		for (size_t bi = 0; bi < prevPtsPerBlock.size(); ++bi) {
			for (size_t j = 0; j < prevPtsPerBlock[bi].size(); ++j) {
				allPrevPts.push_back(prevPtsPerBlock[bi][j]);
				allCurrPts.push_back(currPtsPerBlock[bi][j]);
				if (j < statusPerBlock[bi].size()) allStatus.push_back(statusPerBlock[bi][j]);
				else allStatus.push_back(1);
			}
		}

		OpticalFlowOffset densifier;

		// parameters
		InterpMethod method = INTERP_IDW_LOCAL;
		float kernelSigma = 50.0f;
		int kernelRadius = 60;
		double regularization = 1e-3;
		int maxGlobalPoints = 2000;

		// 显示插值进度
		std::cout << "开始密集插值处理..." << std::endl;
		auto densifyStart = std::chrono::steady_clock::now();



		ImageBlockWriter offset_writer(outOffsetPath, cb.combinedWidth, cb.combinedHeight, 2, cb.combinedGT, linfo->projection);
		InterpMethod inter_method = InterpMethod::INTERP_IDW_LOCAL;
        bool ok = densifier.computeDenseFromPointsAndSaveGeoTIFF2(allPrevPts, allCurrPts,
            cb.combinedWidth, cb.combinedHeight,
            offset_writer, inter_method,
            kernelSigma, kernelRadius, 1024, 1024);
        return cleanup(ok);
	}
    // 如果未进入上面的 hasGeoInfo 分支，仍需释放 info
    if (linfo) RSTools_DestroyImageInfo(linfo);
    if (rinfo) RSTools_DestroyImageInfo(rinfo);
    return false;
}


int main(int argc, char** argv)
{


	// 测试1：显示OpenCV版本
	std::cout << "OpenCV Version: " << CV_VERSION << std::endl;



	std::string lpath = "F:/变形测试/磐安县/磐安县.img";
	std::string rpath = "F:/变形测试/磐安县.img";
	//std::string lpath = "F:/变形测试/磐安县_clip_before1.tif";
	//std::string rpath = "F:/变形测试/磐安县_clip_after1.tif";

	// 偏移文件路径
	std::string outOffsetPath = "F:/变形测试/offsets_pa.tif";
	//左影像重建结果路径
	std::string reconPath = "F:/变形测试/reconstructed_pa.tif";

	// 初始化 GDAL
	RSTools_Initialize();
	bool ok = calculateOffsetsAndSave(lpath, rpath, outOffsetPath);
	//bool ok = true;
	
	if (!ok) {
		std::cerr << "computeDenseToBlocks failed" << std::endl;
	}
    else {
        // 调用已实现的分块重建函数（在 optflowoffsets_impl.cpp 中实现）
        bool recOk = RSTools_ReconstructFromOffsets(outOffsetPath.c_str(), lpath.c_str(), reconPath.c_str(), 512, 512);
        if (!recOk) {
            std::cerr << "Reconstruction from offsets failed" << std::endl;
        }
        else {
            std::cout << "重建影像生成成功: " << reconPath << std::endl;
        }
    }
		
	//	// ----- 根据前影像与偏移反算生成后影像（优化的分块实现）-----

	//	ImageDataType reconDataType = linfo->dataType; // 使用与左影像相同的数据类型
	//	// 创建输出writer
	//	// *** 修改点: 使用新的主构造函数，使用 reconDataType ***
	//	ImageBlockWriter reconWriter(reconPath, cb.combinedWidth, cb.combinedHeight, linfo->bands, cb.combinedGT, reconDataType, linfo->projection);
	//	if (!reconWriter.isOpen()) {
	//		std::cerr << "无法创建重建输出文件: " << reconPath << std::endl;
	//	}
	//	else {
	//		// 使用与偏移量计算相同的块大小
	//		const int reconBlockW = 512;
	//		const int reconBlockH = 512;
	//		int reconBlocksX = static_cast<int>(std::ceil(cb.combinedWidth / static_cast<double>(reconBlockW)));
	//		int reconBlocksY = static_cast<int>(std::ceil(cb.combinedHeight / static_cast<double>(reconBlockH)));

	//		std::cout << "开始分块重建影像，总块数: " << reconBlocksX * reconBlocksY << std::endl;

	//		// 为每个输出块进行处理
	//		for (int by = 0; by < reconBlocksY; ++by) {
	//			for (int bx = 0; bx < reconBlocksX; ++bx) {
	//				int outX = bx * reconBlockW;
	//				int outY = by * reconBlockH;
	//				int blockW = std::min(reconBlockW, cb.combinedWidth - outX);
	//				int blockH = std::min(reconBlockH, cb.combinedHeight - outY);

	//				// 查找对应的偏移量块（从之前计算的结果中）
	//				cv::Mat flowX = cv::Mat::zeros(blockH, blockW, CV_32F);
	//				cv::Mat flowY = cv::Mat::zeros(blockH, blockW, CV_32F);

	//				// 初始化为NaN
	//				flowX.setTo(cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
	//				flowY.setTo(cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

	//				// 从之前计算的密集偏移量中查找对应的数据
	//				// 由于之前的代码将偏移量存储在bufXOut和bufYOut中，我们可以直接使用

	//				// 找到对应的块索引
	//				size_t blockIdx = by * reconBlocksX + bx;
	//				if (blockIdx < bufXOut.size() && blockIdx < bufYOut.size()) {
	//					const std::vector<float>& xData = bufXOut[blockIdx];
	//					const std::vector<float>& yData = bufYOut[blockIdx];

	//					// 验证数据大小
	//					if (xData.size() >= static_cast<size_t>(blockW * blockH) &&
	//						yData.size() >= static_cast<size_t>(blockW * blockH)) {

	//						// 填充flowX和flowY
	//						for (int y = 0; y < blockH; ++y) {
	//							for (int x = 0; x < blockW; ++x) {
	//								size_t idx = y * blockW + x;
	//								flowX.at<float>(y, x) = xData[idx];
	//								flowY.at<float>(y, x) = yData[idx];
	//							}
	//						}
	//					}
	//				}

	//				// 现在我们有了当前块的偏移量，需要读取对应的左影像区域
	//				// 计算当前输出块在左影像中的大致范围
	//				std::vector<double> minXVals, minYVals, maxXVals, maxYVals;

	//				// 采样偏移量来估计需要读取的左影像区域
	//				for (int y = 0; y < blockH; y += std::max(1, blockH / 10)) {
	//					for (int x = 0; x < blockW; x += std::max(1, blockW / 10)) {
	//						float dx = flowX.at<float>(y, x);
	//						float dy = flowY.at<float>(y, x);

	//						if (!std::isnan(dx) && !std::isnan(dy)) {
	//							// 输出位置
	//							double outAbsX = outX + x;
	//							double outAbsY = outY + y;

	//							// 源位置（在左影像的合并坐标系中）
	//							double srcXInCombined = outAbsX - dx;
	//							double srcYInCombined = outAbsY - dy;

	//							// 转换到左影像本地坐标
	//							double srcXInLeft = srcXInCombined - cb.leftOffsetXf;
	//							double srcYInLeft = srcYInCombined - cb.leftOffsetYf;

	//							minXVals.push_back(srcXInLeft);
	//							minYVals.push_back(srcYInLeft);
	//							maxXVals.push_back(srcXInLeft);
	//							maxYVals.push_back(srcYInLeft);
	//						}
	//					}
	//				}

	//				if (minXVals.empty()) {
	//					// 没有有效的偏移量，跳过此块或用NoData填充
	//					// *** 修改点: 创建一个指针数组来传递给 writeBlock ***
	//					std::vector<const void*> bandPtrs(linfo->bands, nullptr);
	//					reconWriter.writeBlock(outX, outY, blockW, blockH, bandPtrs.data());
	//					continue;
	//				}

	//				// 计算读取区域（添加一些padding）
	//				double readMinX = *std::min_element(minXVals.begin(), minXVals.end()) - 10;
	//				double readMinY = *std::min_element(minYVals.begin(), minYVals.end()) - 10;
	//				double readMaxX = *std::max_element(maxXVals.begin(), maxXVals.end()) + 10;
	//				double readMaxY = *std::max_element(maxYVals.begin(), maxYVals.end()) + 10;

	//				// 转换为整数像素坐标并裁剪到影像范围
	//				int readX = std::max(0, static_cast<int>(std::floor(readMinX)));
	//				int readY = std::max(0, static_cast<int>(std::floor(readMinY)));
	//				int readW = std::min(linfo->width - readX,
	//					static_cast<int>(std::ceil(readMaxX)) - readX);
	//				int readH = std::min(linfo->height - readY,
	//					static_cast<int>(std::ceil(readMaxY)) - readY);

	//				if (readW <= 0 || readH <= 0) {
	//					// 无效读取区域
	//					// *** 修改点: 创建一个指针数组来传递给 writeBlock ***
	//					std::vector<const void*> bandPtrs(linfo->bands, nullptr);
	//					reconWriter.writeBlock(outX, outY, blockW, blockH, bandPtrs.data());
	//					continue;
	//				}

	//				// 读取左影像块
	//				ReadResult* leftBlock = RSTools_ReadImage(
	//					lpath.c_str(), readX, readY, readW, readH
	//				);

	//				if (!leftBlock || !leftBlock->success) {
	//					if (leftBlock) RSTools_DestroyReadResult(leftBlock);
	//					// *** 修改点: 创建一个指针数组来传递给 writeBlock ***
	//					std::vector<const void*> bandPtrs(linfo->bands, nullptr);
	//					reconWriter.writeBlock(outX, outY, blockW, blockH, bandPtrs.data());
	//					continue;
	//				}

	//				// 转换左影像块到float格式
	//				std::vector<cv::Mat> leftBandFloats;
	//				for (int b = 0; b < leftBlock->bands && b < linfo->bands; ++b) {
	//					cv::Mat bandMat = convertToCVMat(*leftBlock, b);
	//					cv::Mat bandFloat;
	//					bandMat.convertTo(bandFloat, CV_32F);
	//					leftBandFloats.push_back(bandFloat);
	//				}

	//				// 为输出分配内存 (根据 reconDataType 调整)
	//				// 为了灵活性，我们先用 float 计算，最后再转为所需类型
	//				std::vector<std::vector<float>> outputBandsFloat(linfo->bands);
	//				for (auto& band : outputBandsFloat) {
	//					band.resize(blockW * blockH, std::numeric_limits<float>::quiet_NaN());
	//				}

	//				// 执行反向映射
	//				for (int y = 0; y < blockH; ++y) {
	//					for (int x = 0; x < blockW; ++x) {
	//						float dx = flowX.at<float>(y, x);
	//						float dy = flowY.at<float>(y, x);

	//						if (std::isnan(dx) || std::isnan(dy)) {
	//							continue;
	//						}

	//						// 计算源位置
	//						double outAbsX = outX + x;
	//						double outAbsY = outY + y;
	//						double srcXInCombined = outAbsX - dx;
	//						double srcYInCombined = outAbsY - dy;
	//						double srcXInLeft = srcXInCombined - cb.leftOffsetXf;
	//						double srcYInLeft = srcYInCombined - cb.leftOffsetYf;

	//						// 转换到左影像块的局部坐标
	//						double localX = srcXInLeft - readX;
	//						double localY = srcYInLeft - readY;

	//						// 双线性采样
	//						for (int b = 0; b < leftBandFloats.size(); ++b) {
	//							float sampledValue = bilinearSample(leftBandFloats[b], localX, localY);
	//							if (!std::isnan(sampledValue)) {
	//								outputBandsFloat[b][y * blockW + x] = sampledValue;
	//							}
	//						}
	//					}
	//				}

	//				// 将 float 结果转换为最终输出数据类型
	//				std::vector<std::vector<uint8_t>> finalOutputBands(linfo->bands); // 使用 uint8_t 作为通用容器
	//				for (int b = 0; b < linfo->bands; ++b) {
	//					size_t element_size = 0;
	//					switch (reconDataType) {
	//					case ImageDataType::Byte: element_size = sizeof(uint8_t); break;
	//					case ImageDataType::UInt16: element_size = sizeof(uint16_t); break;
	//					case ImageDataType::Int16: element_size = sizeof(int16_t); break;
	//					case ImageDataType::UInt32: element_size = sizeof(uint32_t); break;
	//					case ImageDataType::Int32: element_size = sizeof(int32_t); break;
	//					case ImageDataType::Float32: element_size = sizeof(float); break;
	//					case ImageDataType::Float64: element_size = sizeof(double); break;
	//					}
	//					finalOutputBands[b].resize(blockW * blockH * element_size);

	//					// 转换逻辑示例 (仅展示 Float32 和 Byte)
	//					if (reconDataType == ImageDataType::Float32) {
	//						float* dest_ptr = reinterpret_cast<float*>(finalOutputBands[b].data());
	//						for (size_t i = 0; i < outputBandsFloat[b].size(); ++i) {
	//							dest_ptr[i] = outputBandsFloat[b][i]; // 直接复制
	//						}
	//					}
	//					else if (reconDataType == ImageDataType::Byte) {
	//						uint8_t* dest_ptr = reinterpret_cast<uint8_t*>(finalOutputBands[b].data());
	//						for (size_t i = 0; i < outputBandsFloat[b].size(); ++i) {
	//							dest_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, outputBandsFloat[b][i]))); // 转换并截断
	//						}
	//					}
	//					else {
	//						// 其他类型转换逻辑应在此处实现
	//						// 为简化，这里仍以 float 处理，实际应用中需补充
	//						float* dest_ptr = reinterpret_cast<float*>(finalOutputBands[b].data());
	//						for (size_t i = 0; i < outputBandsFloat[b].size(); ++i) {
	//							dest_ptr[i] = outputBandsFloat[b][i];
	//						}
	//					}
	//				}

	//				// 创建指向最终数据的指针数组
	//				// *** 修改点: 创建一个指针数组来传递给 writeBlock ***
	//				std::vector<const void*> bandPtrs;
	//				for (int b = 0; b < linfo->bands; ++b) {
	//					bandPtrs.push_back(finalOutputBands[b].data());
	//				}

	//				// 写入输出
	//				reconWriter.writeBlock(outX, outY, blockW, blockH, bandPtrs.data());

	//				// 清理
	//				RSTools_DestroyReadResult(leftBlock);
	//			}
	//		}

	//		std::cout << "重建影像生成成功: " << reconPath << std::endl;
	//	}
	//}



	std::cout << "---- 完成 ----\n";
	return 0;
}
