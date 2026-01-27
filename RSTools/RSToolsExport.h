#pragma once

// 导出宏：编译 DLL 时定义 RSTOOLS_EXPORTS（在项目属性 Preprocessor Definitions 中添加），使用方不定义
#ifdef _WIN32
  #ifdef RSTOOLS_EXPORTS
    #define RSTOOLS_API __declspec(dllexport)
  #else
    #define RSTOOLS_API __declspec(dllimport)
  #endif
#else
  #define RSTOOLS_API
#endif