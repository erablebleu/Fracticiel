
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include.h"
#include "common.h"

__global__ void kernel_mandelbrot(int32_t* result, int32_t dW, int32_t dH, double x0, double y0, double res, int32_t multiSampling, int32_t maxLoopCnt, double maxMagnitude) {
   int i = (blockIdx.x * blockDim.x + threadIdx.x);
   int j = (blockIdx.y * blockDim.y + threadIdx.y);

   if (i >= dW || j >= dH)
      return;

   long count = 0;
   double tmp;
   double dm = (double)multiSampling;

   maxMagnitude *= maxMagnitude;

   for (int k = 0; k < multiSampling; k++)
   for (int l = 0; l < multiSampling; l++)
   {
      double cx = x0 + ((double)i + (double)k / dm) * res;
      double cy = y0 + ((double)j + (double)l / dm) * res;
      double x = 0.0f;
      double y = 0.0f;
      double magnitude = x * x + y * y;
      for (int m = 0; m < maxLoopCnt && magnitude <= maxMagnitude; m++)
      {
         tmp = x * x - y * y + cx;
         y = 2.0f * x * y + cy;
         x = tmp;
         magnitude = x * x + y * y;
         count++;
      }
   }

   result[j * dW + i] = count / multiSampling / multiSampling;
}

int32_t mandelbrot(int32_t* result, const DataBlock* block, const Settings_Mandelbrot* settings) {
   int32_t* cuda_result = 0;

   CUDA_ASSERT_SUCCESS(cudaSetDevice(0));
   CUDA_ASSERT_SUCCESS(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_result, block->Width * block->Height * sizeof(int32_t)));
   dim3 threadsPerBlock(16, 16);
   dim3 numBlocks(block->Width / threadsPerBlock.x + 1, block->Height / threadsPerBlock.y + 1);
   kernel_mandelbrot << <numBlocks, threadsPerBlock >> > (cuda_result, block->Width, block->Height, block->X, block->Y, block->Resolution, block->MultiSampling, settings->LoopCount, settings->Magnitude);
   CUDA_ASSERT_SUCCESS(cudaGetLastError());
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   CUDA_ASSERT_SUCCESS(cudaMemcpy(result, cuda_result, block->Width * block->Height * sizeof(int32_t), cudaMemcpyDeviceToHost));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_result));
   CUDA_ASSERT_SUCCESS(cudaDeviceReset());

   return cudaSuccess;
}