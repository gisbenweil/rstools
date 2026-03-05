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
#include <filesystem>
#include "rstoolsbase.h"
#include "GDALImageBase.h"
#include "GDALImageReader.h"
#include "ImageBlockReader.h"
#include "ImageBlockWriter.h"
#include "opencv2/opencv.hpp"

#include "optflowoffsets.h"
using namespace cv;
using namespace std;


// forward declaration for reconstruction implemented in optflowoffsets_impl.cpp
extern "C" bool RSTools_ReconstructFromOffsets(const char* offsetsPath, const char* prePath, const char* outPath, int blockWidth, int blockHeight);



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


    // 默认路径（可通过命令行替换）
    //std::string lpath = "F:/变形测试/磐安县_clip_before1.tif";
    //std::string rpath = "F:/变形测试/磐安县_clip_after1.tif";
    //std::string outOffsetPath = "F:/变形测试/offsets_.tif";
    //std::string reconPath = "F:/变形测试/reconstructed_.tif";
	std::string lpath = "F:/变形测试/磐安县.img";
	std::string rpath = "F:/变形测试/磐安县/磐安县.img";
	std::string outOffsetPath = "F:/变形测试/offsets_pa_.tif";
	std::string reconPath = "F:/变形测试/reconstructed_pa_.tif";

    // 默认行为：计算偏移并重建
    bool doComputeOffsets = true;
    bool doReconstruct = true;

    // 简单命令行解析（支持开关）
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--skip-offsets" || a == "--no-offsets") {
            doComputeOffsets = false;
        } else if (a == "--skip-recon" || a == "--no-recon") {
            doReconstruct = false;
        } else if (a == "--left" && i + 1 < argc) {
            lpath = argv[++i];
        } else if (a == "--right" && i + 1 < argc) {
            rpath = argv[++i];
        } else if (a == "--offsets" && i + 1 < argc) {
            outOffsetPath = argv[++i];
        } else if (a == "--recon" && i + 1 < argc) {
            reconPath = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0] << " [--left <left.tif>] [--right <right.tif>] ";
            std::cout << "[--offsets <offsets.tif>] [--recon <recon.tif>] [--skip-offsets] [--skip-recon]" << std::endl;
            return 0;
        }
    }

    // 初始化 GDAL
    RSTools_Initialize();

    // 参数有效性检查
    if (!doComputeOffsets && !doReconstruct) {
        std::cerr << "Both compute and reconstruct are disabled. Nothing to do." << std::endl;
        return 1;
    }

    // 当需要计算偏移时，左/右影像必须存在
    if (doComputeOffsets) {
        if (!std::filesystem::exists(lpath)) {
            std::cerr << "Left image not found: " << lpath << std::endl;
            return 1;
        }
        if (!std::filesystem::exists(rpath)) {
            std::cerr << "Right image not found: " << rpath << std::endl;
            return 1;
        }
    }

    // 当只重建时，偏移文件必须存在
    if (doReconstruct && !doComputeOffsets) {
        if (!std::filesystem::exists(outOffsetPath)) {
            std::cerr << "Offsets file not found (required for reconstruction): " << outOffsetPath << std::endl;
            return 1;
        }
    }

    bool computeOk = true;
    if (doComputeOffsets) {
        computeOk = calculateOffsetsAndSave(lpath, rpath, outOffsetPath);
        if (!computeOk) {
            std::cerr << "computeOffsetsAndSave failed" << std::endl;
        }
    }

    if (doReconstruct) {
        // ensure offsets exist
        if (!std::filesystem::exists(outOffsetPath)) {
            std::cerr << "Offsets file not found for reconstruction: " << outOffsetPath << std::endl;
            return 1;
        }
        // only attempt reconstruction if compute succeeded or compute was skipped
        if (computeOk) {
            bool recOk = RSTools_ReconstructFromOffsets(outOffsetPath.c_str(), lpath.c_str(), reconPath.c_str(), 2048, 2048);
            if (!recOk) {
                std::cerr << "Reconstruction from offsets failed" << std::endl;
            } else {
                std::cout << "重建影像生成成功: " << reconPath << std::endl;
            }
        }
    }

    std::cout << "---- 完成 ----\n";
    return 0;
}
