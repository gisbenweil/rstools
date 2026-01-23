#pragma once
#include "RSToolsExport.h"
#include "GDALImageBase.h"
#include <string>

extern "C" {

// 初始化 GDAL（可选，但推荐在使用前调用一次）
RSTOOLS_API void RSTools_Initialize();

// 读取整个影像或指定窗口（如果 width/height 为 0 则读取整影像）
// 返回指针由 RSTools_DestroyReadResult 释放
// path: UTF-8 或本地编码（视 GDAL 编译与系统设置）
// x,y,width,height: 像素窗口
RSTOOLS_API ReadResult* RSTools_ReadImage(const char* path, int x, int y, int width, int height);

// 释放由 RSTools_ReadImage 返回的 ReadResult
RSTOOLS_API void RSTools_DestroyReadResult(ReadResult* result);

// 读取影像基本信息（不读取像素数据），返回由 DLL 分配的 ImageInfo*，请使用 RSTools_DestroyImageInfo 释放
RSTOOLS_API ImageInfo* RSTools_GetImageInfo(const char* path);

// 释放由 RSTools_GetImageInfo 返回的 ImageInfo
RSTOOLS_API void RSTools_DestroyImageInfo(ImageInfo* info);

} // extern "C"

// C++ 重载：读取并缩放到指定输出尺寸（outWidth/outHeight），并可指定重采样算法
// 支持 resampleMethod: "nearest", "bilinear", "cubic"（不区分大小写）
// 注意：这是 C++ 链接（mangled 名称），用于在 C++ 代码中调用重载版本。原 extern "C" 的接口保持不变以兼容已有二进制调用。
RSTOOLS_API ReadResult* RSTools_ReadImage(const char* path, int x, int y, int width, int height, int outWidth, int outHeight, const char* resampleMethod);