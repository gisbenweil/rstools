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
#include "gdal_priv.h"
#include "cpl_conv.h"
using namespace cv;
using namespace std;


// forward declaration for reconstruction implemented in optflowoffsets_impl.cpp
extern "C" bool RSTools_ReconstructFromOffsets(const char* offsetsPath, const char* prePath, const char* outPath, int blockWidth, int blockHeight);





int main(int argc, char** argv)
{


    // 默认路径（可通过命令行替换）
    std::string lpath = "F:/变形测试/磐安县_clip_before1.tif";
    std::string rpath = "F:/变形测试/磐安县_clip_after1.tif";
    std::string outOffsetPath = "F:/变形测试/offsets_.tif";
    std::string reconPath = "F:/变形测试/reconstructed_.tif";



	//std::string lpath = "F:/变形测试/磐安县.img";
	//std::string rpath = "F:/变形测试/磐安县/磐安县.img";
	//std::string outOffsetPath = "F:/变形测试/offsets_pa_.tif";
	//std::string reconPath = "F:/变形测试/reconstructed_pa_.tif";

	//buildOverviews(outOffsetPath);
	//buildOverviews(reconPath);

    // 默认行为：计算偏移并重建
    bool doComputeOffsets = true;
    bool doReconstruct = true;
    // 默认分块大小（用于读取/输出块）
    int blockSize = 512;
    // 重建时使用的分块大小（默认保持原来的较大值）
    int reconBlockSize = 2048;

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
        } else if (a == "--block-size" && i + 1 < argc) {
            blockSize = std::max(16, atoi(argv[++i]));
        } else if (a == "--recon-block-size" && i + 1 < argc) {
            reconBlockSize = std::max(16, atoi(argv[++i]));
        } else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0] << " [--left <left.tif>] [--right <right.tif>] ";
            std::cout << "[--offsets <offsets.tif>] [--recon <recon.tif>] [--block-size <n>] [--recon-block-size <n>] [--skip-offsets] [--skip-recon]" << std::endl;
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
        computeOk = calculateOffsetsAndSave(lpath, rpath, outOffsetPath, blockSize);
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
            bool recOk = RSTools_ReconstructFromOffsets(outOffsetPath.c_str(), lpath.c_str(), reconPath.c_str(), reconBlockSize, reconBlockSize);
            if (!recOk) {
                std::cerr << "Reconstruction from offsets failed" << std::endl;
            } else {
                std::cout << "重建影像生成成功: " << reconPath << std::endl;
                // build overviews for the reconstructed file
                buildOverviews(reconPath);
            }
        }
    }

    std::cout << "---- 完成 ----\n";
    return 0;
}
