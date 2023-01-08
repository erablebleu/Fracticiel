#ifndef FRACTICIEL_CUDA_INCLUDE_H_
#define FRACTICIEL_CUDA_INCLUDE_H_

#ifdef CUDA_DLL_EXPORTS
#define EXTERN extern "C" __declspec(dllexport)
#else
#define EXTERN extern "C" __declspec(dllimport)
#endif

#include <cstdint>

EXTERN const char* getLastErrorName();
EXTERN const char* getLastErrorString();
EXTERN const char* getLastErrorFileName() ;
EXTERN int getLastErrorFileLine() ;

EXTERN int32_t mandelbrot(uint32_t* result, uint32_t dW, uint32_t dH, double x, double y, double res, uint32_t maxLoopCnt, double maxMagnitude);
EXTERN int32_t buddhabrot(uint32_t* result, uint32_t dW, uint32_t dH, double x, double y, double res, uint32_t maxLoopCnt, double maxMagnitude, uint32_t maxValue);
EXTERN int32_t julia(uint32_t* result, uint32_t dW, uint32_t dH, double x, double y, double res, uint32_t maxLoopCnt, double maxMagnitude, double cx, double cy);
EXTERN int32_t colorizeBW(uint8_t* result, uint32_t* data, uint32_t sz, uint32_t min, uint32_t max);
EXTERN int32_t colorizeRGB(uint32_t* result, uint32_t* data, uint32_t sz, uint32_t min, uint32_t max);
EXTERN int32_t multisampling(uint32_t* result, uint32_t* data, uint32_t dW, uint32_t dH, uint32_t multisampling);

#endif // FRACTICIEL_CUDA_INCLUDE_H_