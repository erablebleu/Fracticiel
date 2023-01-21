
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include.h"
#include "common.h"

__global__ void kernel_buddhabrot(uint32_t* result, int32_t dW, int32_t dH, double x0, double y0, double res, int32_t maxLoopCnt, double maxMagnitude, uint32_t maxValue, double* randoms) {

   maxMagnitude *= maxMagnitude; 
   
   int i = (blockIdx.x * blockDim.x + threadIdx.x);
   int j = (blockIdx.y * blockDim.y + threadIdx.y);

   if (i >= dW || j >= dH)
      return;

   double cx = randoms[2 * j * dW + i];
   double cy = randoms[2 * j * dW + i + 1];

   double x = 0.0f;
   double y = 0.0f;
   double tmp;
   double magnitude;

   for (int k = 0; k < maxLoopCnt; k++)
   {
      tmp = x * x - y * y + cx;
      y = 2.0f * x * y + cy;
      x = tmp;
      magnitude = x * x + y * y;

      if(magnitude > maxMagnitude)
         break;
   }

   if(magnitude <= maxMagnitude) // not divergent
      return;

   x = 0.0f;
   y = 0.0f;

   for (int k = 0; k < maxLoopCnt; k++)
   {
      tmp = x * x - y * y + cx;
      y = 2.0f * x * y + cy;
      x = tmp;
      magnitude = x * x + y * y;

      i = (x - x0) / res;
      j = (y - y0) / res;

      if (i >= 0 && i < dW
          && j >= 0 && j < dH) {
         result[j * dW + i]++;
      }

      if (magnitude > maxMagnitude)
         break;
   }
}

int32_t buddhabrot(uint32_t* result, const DataBlock* block, const Settings_Buddhabrot* settings, const double* randoms) {
   uint32_t* cuda_result = 0;
   double* cuda_randoms = 0;

   CUDA_ASSERT_SUCCESS(cudaSetDevice(0));
   CUDA_ASSERT_SUCCESS(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_result, block->Width * block->Height * sizeof(uint32_t)));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_randoms, 2 * block->Width * block->Height * sizeof(double)));
   CUDA_ASSERT_SUCCESS(cudaMemcpy(cuda_randoms, randoms, 2 * block->Width * block->Height * sizeof(double), cudaMemcpyHostToDevice));
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   dim3 threadsPerBlock(16, 16);
   dim3 numBlocks(block->Width / threadsPerBlock.x + 1, block->Height / threadsPerBlock.y + 1);
   kernel_buddhabrot << <numBlocks, threadsPerBlock >> > (cuda_result, block->Width, block->Height, block->X, block->Y, block->Resolution, settings->LoopCount, settings->Magnitude, settings->MaxValue, cuda_randoms);
   CUDA_ASSERT_SUCCESS(cudaGetLastError());
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   CUDA_ASSERT_SUCCESS(cudaMemcpy(result, cuda_result, block->Width * block->Height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_result));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_randoms));
   CUDA_ASSERT_SUCCESS(cudaDeviceReset());

   return cudaSuccess;
}