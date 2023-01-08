
#include "cuda_runtime.h"
#include "common.h"
#include "include.h"

cudaError_t lastErr = cudaSuccess;
int lastErrLine = 0;
const char* lastErrFileName = 0;

const char* getLastErrorName() {
   return cudaGetErrorName(lastErr);
}
const char* getLastErrorString() {
   return cudaGetErrorString(lastErr);
}
const char* getLastErrorFileName() {
   return lastErrFileName;
}
int getLastErrorFileLine() {
   return lastErrLine;
}