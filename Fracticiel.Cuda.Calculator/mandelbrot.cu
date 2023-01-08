
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include.h"
#include "common.h"

__global__ void kernel_mandelbrot(uint32_t* result, uint32_t dW, uint32_t dH, double x0, double y0, double res, uint32_t maxLoopCnt, double maxMagnitude) {
   int i = (blockIdx.x * blockDim.x + threadIdx.x);
   int j = (blockIdx.y * blockDim.y + threadIdx.y);

   if (i >= dW || j >= dH)
      return;

   double cx = x0 + (double)i * res;
   double cy = y0 + (double)j * res;
   uint32_t* p = &result[j * dW + i];
   double x = 0.0f;
   double y = 0.0f;
   double tmp;

   maxMagnitude *= maxMagnitude;

   while ((x * x + y * y) <= maxMagnitude && *p < maxLoopCnt)
   {
      tmp = x * x - y * y + cx;
      y = 2.0f * x * y + cy;
      x = tmp;
      (*p)++;
   }
}

int32_t mandelbrot(uint32_t* result, uint32_t dW, uint32_t dH, double x, double y, double res, uint32_t maxLoopCnt, double maxMagnitude) {
   uint32_t* cuda_result = 0;

   CUDA_ASSERT_SUCCESS(cudaSetDevice(0));
   CUDA_ASSERT_SUCCESS(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_result, dW * dH * sizeof(uint32_t)));
   dim3 threadsPerBlock(16, 16);
   dim3 numBlocks(dW / threadsPerBlock.x + 1, dH / threadsPerBlock.y + 1);
   kernel_mandelbrot << <numBlocks, threadsPerBlock >> > (cuda_result, dW, dH, x, y, res, maxLoopCnt, maxMagnitude);
   CUDA_ASSERT_SUCCESS(cudaGetLastError());
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   CUDA_ASSERT_SUCCESS(cudaMemcpy(result, cuda_result, dW * dH * sizeof(uint32_t), cudaMemcpyDeviceToHost));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_result));
   CUDA_ASSERT_SUCCESS(cudaDeviceReset());

   return cudaSuccess;
}