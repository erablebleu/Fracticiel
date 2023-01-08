#pragma once

#include "../Fracticiel.Cuda.Calculator/include.h"
#include <cstdint>
#include <corecrt_malloc.h>

#define ASSERT_SUCCESS(func) { int32_t cuda_result = func; if(0 != cuda_result) ThrowError(__func__, cuda_result); }

using namespace System;

namespace Fracticiel::Cuda {

   public ref class Calculator {
   private:
      static void ThrowError(const char* func, int32_t err) {
         throw gcnew System::Exception(System::String::Format("Method {0} - At: {1}.{2} : {3} {4}",
                                                              gcnew System::String(func),
                                                              gcnew System::String(getLastErrorFileName()),
                                                              getLastErrorFileLine(),
                                                              gcnew System::String(getLastErrorName()), 
                                                              gcnew System::String(getLastErrorString())));
      };
   public:
      static void Mandelbrot(array<unsigned int>^% result, unsigned int dW, unsigned int dH, double x, double y, double res, unsigned int maxLoopCnt, double maxMagnitude) {
         pin_ptr<uint32_t> p = &result[0];
         int32_t cuda_result = mandelbrot(p, dW, dH, x, y, res, maxLoopCnt, maxMagnitude);
         if (0 != cuda_result) ThrowError(__func__, cuda_result);
      };
      static void Buddhabrot(array<unsigned int>^% result, unsigned int dW, unsigned int dH, double x, double y, double res, unsigned int maxLoopCnt, double maxMagnitude, unsigned int maxValue) {
         pin_ptr<uint32_t> p = &result[0];
         ASSERT_SUCCESS(buddhabrot(p, dW, dH, x, y, res, maxLoopCnt, maxMagnitude, maxValue));
      };
      static void Julia(array<unsigned int>^% result, unsigned int dW, unsigned int dH, double x, double y, double res, unsigned int maxLoopCnt, double maxMagnitude, double cx, double cy) {
         pin_ptr<uint32_t> p = &result[0];
         ASSERT_SUCCESS(julia(p, dW, dH, x, y, res, maxLoopCnt, maxMagnitude, cx, cy));
      };
      static void ColorizeBW(array<unsigned char>^% result, array<unsigned int>^ data, unsigned int sz, unsigned int min, unsigned int max) {
         pin_ptr<uint8_t> pRes = &result[0];
         pin_ptr<uint32_t> pData = &data[0];
         ASSERT_SUCCESS(colorizeBW(pRes, pData, sz, min, max));
      };
      static void ColorizeRGB(array<unsigned int>^% result, array<unsigned int>^ data, unsigned int sz, unsigned int min, unsigned int max) {
         pin_ptr<uint32_t> pRes = &result[0];
         pin_ptr<uint32_t> pData = &data[0];
         ASSERT_SUCCESS(colorizeRGB(pRes, pData, sz, min, max));
      };
      static void Multisampling(array<unsigned int>^% result, array<unsigned int>^ data, unsigned int dW, unsigned int dH, unsigned int sampling) {
         pin_ptr<uint32_t> pRes = &result[0];
         pin_ptr<uint32_t> pData = &data[0];
         ASSERT_SUCCESS(multisampling(pRes, pData, dW, dH, sampling));
      };
   };
}