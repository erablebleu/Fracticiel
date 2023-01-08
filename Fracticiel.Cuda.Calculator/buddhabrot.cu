
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include.h"
#include "common.h"

__global__ void kernel_buddhabrot_divergence(bool* result, uint32_t dW, uint32_t dH, double x0, double y0, double res, uint32_t maxLoopCnt, double maxMagnitude) {
   int i = (blockIdx.x * blockDim.x + threadIdx.x);
   int j = (blockIdx.y * blockDim.y + threadIdx.y);

   if (i >= dW || j >= dH)
      return;

   double cx = x0 + (double)i * res;
   double cy = y0 + (double)j * res;
   bool* p = &result[j * dW + i];
   double x = 0.0f;
   double y = 0.0f;
   double tmp;
   uint32_t cnt = 0;

   maxMagnitude *= maxMagnitude;

   while ((x * x + y * y) <= maxMagnitude && cnt < maxLoopCnt)
   {
      tmp = x * x - y * y + cx;
      y = 2.0f * x * y + cy;
      x = tmp;
      cnt++;
   }

   *p = cnt >= maxLoopCnt;
}

__global__ void kernel_buddhabrot(uint32_t* result, bool* diverg, uint32_t dW, uint32_t dH, double x0, double y0, double res, uint32_t maxLoopCnt, double maxMagnitude, uint32_t maxValue) {
   int i = (blockIdx.x * blockDim.x + threadIdx.x);
   int j = (blockIdx.y * blockDim.y + threadIdx.y);

   if (i >= dW || j >= dH || diverg[j * dW + i])
      return;

   double cx = x0 + (double)i * res;
   double cy = y0 + (double)j * res;
   uint32_t cnt = 0;
   double x = 0.0f;
   double y = 0.0f;
   double tmp;

   maxMagnitude *= maxMagnitude;

   while ((x * x + y * y) <= maxMagnitude && cnt < maxLoopCnt)
   {
      tmp = x * x - y * y + cx;
      y = 2.0f * x * y + cy;
      x = tmp;

      i = (x - x0) / res;
      j = (y - y0) / res;

      if (i >= 0
          && i < dW
          && j >= 0
          && j < dH
          && result[j * dW + i] < maxValue) {
         result[j * dW + i]++;
      }
      cnt++;
   }
}

int32_t buddhabrot(uint32_t* result, uint32_t dW, uint32_t dH, double x, double y, double res, uint32_t maxLoopCnt, double maxMagnitude, uint32_t maxValue) {
   uint32_t* cuda_result = 0;
   bool* cuda_diverg = 0;

   CUDA_ASSERT_SUCCESS(cudaSetDevice(0));
   CUDA_ASSERT_SUCCESS(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_result, dW * dH * sizeof(uint32_t)));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_diverg, dW * dH * sizeof(bool)));
   dim3 threadsPerBlock(16, 16);
   dim3 numBlocks(dW / threadsPerBlock.x + 1, dH / threadsPerBlock.y + 1);
   kernel_buddhabrot_divergence << <numBlocks, threadsPerBlock >> > (cuda_diverg, dW, dH, x, y, res, maxLoopCnt, maxMagnitude);
   CUDA_ASSERT_SUCCESS(cudaGetLastError());
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   kernel_buddhabrot << <numBlocks, threadsPerBlock >> > (cuda_result, cuda_diverg, dW, dH, x, y, res, maxLoopCnt, maxMagnitude, maxValue);
   CUDA_ASSERT_SUCCESS(cudaGetLastError());
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   CUDA_ASSERT_SUCCESS(cudaMemcpy(result, cuda_result, dW * dH * sizeof(uint32_t), cudaMemcpyDeviceToHost));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_result));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_diverg));
   CUDA_ASSERT_SUCCESS(cudaDeviceReset());

   return cudaSuccess;
}