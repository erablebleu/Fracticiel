#ifndef FRACTICIEL_CUDA_INCLUDE_H_
#define FRACTICIEL_CUDA_INCLUDE_H_

#ifdef CUDA_DLL_EXPORTS
#define EXTERN extern "C" __declspec(dllexport)
#else
#define EXTERN extern "C" __declspec(dllimport)
#endif

#include <cstdint>

struct DataBlock {
   int32_t Width;
   int32_t Height;
   double X;
   double Y;
   double Resolution;
};

struct Settings_Mandelbrot {

   int32_t LoopCount;
   double Magnitude;
};
struct Settings_Buddhabrot {

   int32_t LoopCount;
   double Magnitude;
   int32_t MaxValue;
};
struct Settings_Julia {

   int32_t LoopCount;
   double Magnitude;
   double CX;
   double CY;
};

EXTERN const char* getLastErrorName();
EXTERN const char* getLastErrorString();
EXTERN const char* getLastErrorFileName();
EXTERN int getLastErrorFileLine();

EXTERN int32_t mandelbrot(uint32_t* result, const DataBlock* block, const Settings_Mandelbrot* settings);
EXTERN int32_t buddhabrot(uint32_t* result, const DataBlock* block, const Settings_Buddhabrot* settings);
EXTERN int32_t julia(uint32_t* result, const DataBlock* block, const Settings_Julia* settings);

EXTERN int32_t colorizeBW(uint8_t* result, uint32_t* data, int32_t sz, uint32_t min, uint32_t max);
EXTERN int32_t colorizeRGB(uint32_t* result, uint32_t* data, int32_t sz, uint32_t min, uint32_t max);
EXTERN int32_t multisampling(uint32_t* result, uint32_t* data, int32_t dW, int32_t dH, int32_t multisampling);

#endif // FRACTICIEL_CUDA_INCLUDE_H_