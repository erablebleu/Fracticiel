
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include.h"
#include "common.h"

__global__ void kernel_colorizeBW(uint8_t* result, uint32_t* data, uint32_t sz, uint32_t min, uint32_t max) {
   int i = (blockIdx.x * blockDim.x + threadIdx.x);
   uint32_t val = data[i];
   if (i >= sz)
      return;
   if (val <= min)
      result[i] = 0;
   if (val >= max)
      result[i] = 255;
   else {
      result[i] = (val - min) * 255 / (max - min);
   }
}

int32_t colorizeBW(uint8_t* result, uint32_t* data, uint32_t sz, uint32_t min, uint32_t max) {
   uint8_t* cuda_result = 0;
   uint32_t* cuda_data = 0;

   CUDA_ASSERT_SUCCESS(cudaSetDevice(0));
   CUDA_ASSERT_SUCCESS(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_result, sz * sizeof(uint8_t)));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_data, sz * sizeof(uint32_t)));
   CUDA_ASSERT_SUCCESS(cudaMemcpy(cuda_data, data, sz * sizeof(uint32_t), cudaMemcpyHostToDevice));
   dim3 threadsPerBlock(256);
   dim3 numBlocks(sz / threadsPerBlock.x + 1);
   kernel_colorizeBW << <numBlocks, threadsPerBlock >> > (cuda_result, cuda_data, sz, min, max);
   CUDA_ASSERT_SUCCESS(cudaGetLastError());
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   CUDA_ASSERT_SUCCESS(cudaMemcpy(result, cuda_result, sz * sizeof(uint8_t), cudaMemcpyDeviceToHost));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_result));
   CUDA_ASSERT_SUCCESS(cudaDeviceReset());

   return cudaSuccess;
}

__global__ void kernel_colorizeRGB(uint32_t* result, uint32_t* data, uint32_t sz, uint32_t min, uint32_t max) {
   int i = (blockIdx.x * blockDim.x + threadIdx.x);
   uint32_t val = data[i];
   if (i >= sz)
      return;
   if (val <= min
       || val >= max)
      result[i] = 0xFF000000;
   else {
      result[i] = 0xFF000000 | ((val - min) * 0x00FFFFFF / (max - min));
   }
}

int32_t colorizeRGB(uint32_t* result, uint32_t* data, uint32_t sz, uint32_t min, uint32_t max) {
   uint32_t* cuda_result = 0;
   uint32_t* cuda_data = 0;

   CUDA_ASSERT_SUCCESS(cudaSetDevice(0));
   CUDA_ASSERT_SUCCESS(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_result, sz * sizeof(uint32_t)));
   CUDA_ASSERT_SUCCESS(cudaMalloc((void**)&cuda_data, sz * sizeof(uint32_t)));
   CUDA_ASSERT_SUCCESS(cudaMemcpy(cuda_data, data, sz * sizeof(uint32_t), cudaMemcpyHostToDevice));
   dim3 threadsPerBlock(256);
   dim3 numBlocks(sz / threadsPerBlock.x + 1);
   kernel_colorizeRGB << <numBlocks, threadsPerBlock >> > (cuda_result, cuda_data, sz, min, max);
   CUDA_ASSERT_SUCCESS(cudaGetLastError());
   CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
   CUDA_ASSERT_SUCCESS(cudaMemcpy(result, cuda_result, sz * sizeof(uint32_t), cudaMemcpyDeviceToHost));
   CUDA_ASSERT_SUCCESS(cudaFree(cuda_result));
   CUDA_ASSERT_SUCCESS(cudaDeviceReset());

   return cudaSuccess;
}