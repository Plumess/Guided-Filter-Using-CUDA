
#define BLOCK_W     32
#define BLOCK_H     32

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

void guided_filter_cuda(float* ptrT_device, unsigned char* guidedRGBImg_device, float* ptrGuidedT_device, int rows, int cols,
                int inStride, int outStride, int m_nGBlockSize, float fEps, cudaStream_t stream,
                float* fGuidedImg_device, float* pfInitN_device, float* pfInitMeanIp_device, float* pfInitMeanII_device,
                float* pfMeanP_device, float* pfN_device, float* pfMeanI_device, float* pfMeanIp_device,
                float* pfMeanII_device, float* pfvarI_device, float* pfCovIp_device, float* pfA_device, float* pfB_device,
                float* pfOutA_device, float* pfOutB_device, float* pfArrayCum_device);