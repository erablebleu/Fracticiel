#ifndef FRACTICIEL_CUDA_COMMON_H_
#define FRACTICIEL_CUDA_COMMON_H_

#include "cuda_runtime.h"

#define CUDA_ASSERT_SUCCESS(func) { cudaError_t retVal = func; if(cudaSuccess != retVal) { lastErr = retVal; lastErrLine = __LINE__; lastErrFileName = __FILE__; return retVal; } }

extern cudaError_t lastErr;
extern int lastErrLine;
extern const char* lastErrFileName;

#endif // FRACTICIEL_CUDA_COMMON_H_