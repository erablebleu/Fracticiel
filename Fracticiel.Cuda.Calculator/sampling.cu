#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include.h"
#include "common.h"

__global__ void kernel_multisampling(uint32_t* result, uint32_t* data, int32_t dW, int32_t dH, int32_t sampling) {
   int x = (blockIdx.x * blockDim.x + threadIdx.x);
   int y = (blockIdx.y * blockDim.y + threadIdx.y);
   uint64_t val = 0;

   if (x >= dW || y >= dH)
      return;

   for (int i = 0; i < sampling; i++)
      for (int j = 0; j < sampling; j++) 
         val += data[x * sampling + i + (y * sampling + j) * dW * sampling];

   result[x + y * dW] = val / sampling;
}

int32_t multisampling(uint32_t* result, uint32_t* data, int32_t dW, int32_t dH, int32_t multisampling) {
   uint32_t* cuda_result = 0;
   uint32_t* cuda_data = 0;

   CUDA_ASSERT_SUCCESS(cudaSetDevice(0));
   CUDA_ASSERT_SUCCESS(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_result, dW * dH * sizeof(uint32_t)));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_data, dW * dH * multisampling * multisampling * sizeof(uint32_t)));
   CUDA_ASSERT_SUCCESS(cudaMemcpy(cuda_data, data, dW * multisampling * dH * multisampling * sizeof(uint32_t), cudaMemcpyHostToDevice));
   dim3 threadsPerBlock(16, 16);
   dim3 numBlocks(dW / threadsPerBlock.x + 1, dH / threadsPerBlock.y + 1);
   kernel_multisampling << <numBlocks, threadsPerBlock >> > (cuda_result, cuda_data, dW, dH, multisampling);
   CUDA_ASSERT_SUCCESS(cudaGetLastError());
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   CUDA_ASSERT_SUCCESS(cudaMemcpy(result, cuda_result, dW * dH * sizeof(uint32_t), cudaMemcpyDeviceToHost));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_result));
   CUDA_ASSERT_SUCCESS(cudaDeviceReset());

   return cudaSuccess;
}