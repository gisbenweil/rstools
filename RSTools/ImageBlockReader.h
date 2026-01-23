#pragma once
#include "RSToolsExport.h"
#include "GDALImageBase.h"
#include <string>

/**
 * @brief 分块读取信息（返回给调用者的元信息）
 */
struct RSTOOLS_API BlockSpec {
	int tileX;      // 瓦片左上角（在指定范围/影像坐标系中）
	int tileY;
	int tileWidth;  // 瓦片的逻辑宽高（不含 padding，可能最后一列/行小于 blockSize）
	int tileHeight;

	// 实际读取窗口（包含 padding，并已裁剪到影像范围）
	int readX;
	int readY;
	int readWidth;
	int readHeight;

	int padding;    // 请求的 padding（可能被裁剪）
};

/**
 * @brief ImageBlockReader：按块读取影像，支持 padding 与自定义范围
 *
 * 使用示例：
 * ImageBlockReader reader(path, 512, 512, 16);
 * while (true) {
 *   ReadResult* res = nullptr;
 *   BlockSpec spec;
 *   if (!reader.next(&res, &spec)) break;
 *   // 处理 res（注意 res->width/res->height == spec.readWidth/readHeight）
 *   RSTools_DestroyReadResult(res);
 * }
 */
class RSTOOLS_API ImageBlockReader {
public:
	// path: 影像路径
	// blockWidth/blockHeight: 瓦片尺寸
	// padding: 向外扩展像素（四周），可为0
	// range: 可选读取范围（若 width/height 为0则视为整幅影像）
	ImageBlockReader(const std::string& path, int blockWidth, int blockHeight, int padding = 0, const ReadArea* range = nullptr);
	~ImageBlockReader();

	// 是否成功打开并可使用
	bool isOpen() const;

	// 重置迭代到起始块
	void reset();

	// 获取总块数（基于指定范围与 blockSize 计算）
	int getTotalBlocks() const;

	// 获取已处理块数
	int getProcessedBlocks() const;

	// 获取影像尺寸
	int imageWidth() const;
	int imageHeight() const;

	// 获取下一个块；返回 true 表示已返回一个块，out must be freed by caller via RSTools_DestroyReadResult.
	// 如果返回 false 表示无更多块或发生错误（out 设置为 nullptr）
	// 可选返回参数 spec 会被填充（tile 与实际读取窗口信息）
	bool next(ReadResult** out, BlockSpec* spec = nullptr);

	// 新增：按照全局分块索引直接读取指定分块（不改变迭代器状态）
	// index: [0, getTotalBlocks()-1]
	// 返回 true 并在 *out 填充 ReadResult*（由调用者通过 RSTools_DestroyReadResult 释放）
	bool readBlockByIndex(int index, ReadResult** out, BlockSpec* spec = nullptr);

private:
	std::string path_;
	int blockW_;
	int blockH_;
	int padding_;

	// 图片 / 范围
	int imgW_;
	int imgH_;
	ReadArea range_;

	// tiles grid
	int tilesX_;
	int tilesY_;

	// current indices
	int curXIdx_;
	int curYIdx_;

	// 状态
	bool open_;
	int processed_;

	// 单一打开的 GDALDataset
	GDALDataset* ds_;
};

// ---------------- C API 封装导出（DLL 友好、不透明句柄） ----------------
extern "C" {

// 创建并返回不透明句柄（请用 RSTools_ImageBlockReader_Destroy 释放）
RSTOOLS_API void* RSTools_ImageBlockReader_Create(const char* path, int blockWidth, int blockHeight, int padding, const ReadArea* range);

// 销毁句柄
RSTOOLS_API void RSTools_ImageBlockReader_Destroy(void* handle);

// 是否打开成功
RSTOOLS_API bool RSTools_ImageBlockReader_IsOpen(void* handle);

// 重置迭代
RSTOOLS_API void RSTools_ImageBlockReader_Reset(void* handle);

// 查询总块数 / 已处理块数 / 图片尺寸
RSTOOLS_API int RSTools_ImageBlockReader_GetTotalBlocks(void* handle);
RSTOOLS_API int RSTools_ImageBlockReader_GetProcessedBlocks(void* handle);
RSTOOLS_API int RSTools_ImageBlockReader_ImageWidth(void* handle);
RSTOOLS_API int RSTools_ImageBlockReader_ImageHeight(void* handle);

// 读取下一个块：返回 true 并输出 ReadResult*（由调用者通过 RSTools_DestroyReadResult 释放）
// 若无更多块或出错返回 false 并将 *out 置为 nullptr
RSTOOLS_API bool RSTools_ImageBlockReader_Next(void* handle, ReadResult** out, BlockSpec* spec);

// 新增 C API：按分块索引读取指定块
RSTOOLS_API bool RSTools_ImageBlockReader_ReadByIndex(void* handle, int index, ReadResult** out, BlockSpec* spec);

} // extern "C"