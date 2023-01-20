
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include.h"
#include "common.h"

__global__ void kernel_buddhabrot_divergence(bool* result, int32_t dW, int32_t dH, double x0, double y0, double res, int32_t maxLoopCnt, double maxMagnitude) {
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

__global__ void kernel_buddhabrot(uint32_t* result, bool* diverg, int32_t dW, int32_t dH, double x0, double y0, double res, int32_t maxLoopCnt, double maxMagnitude, uint32_t maxValue) {
   int i = (blockIdx.x * blockDim.x + threadIdx.x);
   int j = (blockIdx.y * blockDim.y + threadIdx.y);

   if (i >= dW || j >= dH || diverg[j * dW + i])
      return;

   // TODO: divergent loop here

   // TODO: random cx, cy
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

int32_t buddhabrot(uint32_t* result, const DataBlock* block, const Settings_Buddhabrot* settings) {
   uint32_t* cuda_result = 0;
   bool* cuda_diverg = 0;

   CUDA_ASSERT_SUCCESS(cudaSetDevice(0));
   CUDA_ASSERT_SUCCESS(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_result, block->Width * block->Height * sizeof(uint32_t)));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_diverg, block->Width * block->Height * sizeof(bool)));
   dim3 threadsPerBlock(16, 16);
   dim3 numBlocks(block->Width / threadsPerBlock.x + 1, block->Height / threadsPerBlock.y + 1);
   kernel_buddhabrot_divergence << <numBlocks, threadsPerBlock >> > (cuda_diverg, block->Width, block->Height, block->X, block->Y, block->Resolution, settings->LoopCount, settings->Magnitude);
   CUDA_ASSERT_SUCCESS(cudaGetLastError());
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   kernel_buddhabrot << <numBlocks, threadsPerBlock >> > (cuda_result, cuda_diverg, block->Width, block->Height, block->X, block->Y, block->Resolution, settings->LoopCount, settings->Magnitude, settings->MaxValue);
   CUDA_ASSERT_SUCCESS(cudaGetLastError());
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   CUDA_ASSERT_SUCCESS(cudaMemcpy(result, cuda_result, block->Width * block->Height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_result));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_diverg));
   CUDA_ASSERT_SUCCESS(cudaDeviceReset());

   return cudaSuccess;
}