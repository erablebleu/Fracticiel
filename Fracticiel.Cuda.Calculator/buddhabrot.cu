
#include <stdio.h>

#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include "include.h"
#include "common.h"

__global__ void kernel_setup_curand(curandState* state) {
   int idx = (blockIdx.x * blockDim.x + threadIdx.x);
   curand_init(idx, idx, 0, &state[idx]);
}

QUALIFIERS bool kernel_buddhabrot_isDivergent(double cx, double cy, int32_t divergenceCount, double divergenceMagnitude) {
   double x = 0.0;
   double y = 0.0;
   double tmp;
   double magnitude;

   for (int k = 0; k < divergenceCount; k++)
   {
      tmp = x * x - y * y + cx;
      y = 2.0f * x * y + cy;
      x = tmp;
      magnitude = x * x + y * y;

      if (magnitude > divergenceMagnitude)
         return true;
   }
   return false;
}

QUALIFIERS void kernel_buddhabrot(int32_t* result,
                                  int32_t dW, int32_t dH, 
                                  double x0, double y0, 
                                  double cx, double cy,
                                  double res, 
                                  int32_t divergenceCount,
                                  double divergenceMagnitude) {
   int i, j;
   double x = 0.0;
   double y = 0.0;
   double tmp;
   double magnitude;

   if(!kernel_buddhabrot_isDivergent(cx, cy, divergenceCount, divergenceMagnitude))
      return;

   x = 0.0;
   y = 0.0;

   for (int k = 0; k < divergenceCount; k++)
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

      if (magnitude > divergenceMagnitude)
         break;
   }
}

__global__ void kernel_buddhabrot(curandState* state,
                                  int32_t* result,
                                  int32_t dW, int32_t dH,
                                  double x0, double y0,
                                  double resolution,
                                  int32_t divergenceCount,
                                  double divergenceMagnitude,
                                  int32_t startPointCount) {
   int idx = (blockIdx.x * blockDim.x + threadIdx.x);

   for(int l = 0; l < startPointCount; l++) {
      kernel_buddhabrot(result, dW, dH, x0, y0, 
                        (double)(-2.0) + (double)4.0 * curand_uniform_double(&state[idx]),
                        (double)(-2.0) + (double)4.0 * curand_uniform_double(&state[idx]),
                        resolution,
                        divergenceCount,
                        divergenceMagnitude);
   }
}

int32_t buddhabrot(int32_t* result, const DataBlock* block, const Settings_Buddhabrot* settings) {
   int32_t* cuda_result = 0;
   curandState* d_state = 0;
   int32_t blockCount = 256;
   int32_t threadCount = 256;

   CUDA_ASSERT_SUCCESS(cudaSetDevice(0));
   CUDA_ASSERT_SUCCESS(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
   CUDA_ASSERT_SUCCESS(cudaMalloc(&d_state, blockCount * threadCount * sizeof(curandState)));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_result, block->Width * block->Height * sizeof(int32_t)));
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   kernel_setup_curand << <blockCount, threadCount >> > (d_state);
   kernel_buddhabrot << <blockCount, threadCount >> > (d_state,
                                        cuda_result, 
                                        block->Width, block->Height, 
                                        block->X, block->Y, 
                                        block->Resolution,
                                        settings->LoopCount, 
                                        settings->Magnitude,
      settings->StartPointCount);
   CUDA_ASSERT_SUCCESS(cudaGetLastError());
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   CUDA_ASSERT_SUCCESS(cudaMemcpy(result, cuda_result, block->Width * block->Height * sizeof(int32_t), cudaMemcpyDeviceToHost));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_result));
   CUDA_ASSERT_SUCCESS(cudaFree(d_state));
   CUDA_ASSERT_SUCCESS(cudaDeviceReset());

   return cudaSuccess;
}