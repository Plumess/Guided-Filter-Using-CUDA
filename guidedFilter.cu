#include "guidedFilter.cuh"

__global__ void to_float_point(float* ptrT,
								float *fGuidedImg, 
								unsigned char *guidedRGBImg, 
								float *pfInitN,
								float *pfInitMeanIp,
								float *pfInitMeanII,
								int rows,
								int cols, 
								int inStride,
								int outStride){

	int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if((j >= cols) || (i >= rows)){
        return;
    }
    //RGB
    //fGuidedImg[i * outStride + j] = float((0.299 * guidedRGBImg[i * inStride * 3 + j * 3 + 0] + 0.587 * guidedRGBImg[i * inStride * 3 + j * 3 + 1] + 0.114 * guidedRGBImg[i * inStride * 3 + j * 3 + 2]) / 255.);
	//BGR
    fGuidedImg[i * outStride + j] = float((0.114 * guidedRGBImg[i * inStride * 3 + j * 3 + 0] + 0.587 * guidedRGBImg[i * inStride * 3 + j * 3 + 1] + 0.299 * guidedRGBImg[i * inStride * 3 + j * 3 + 2]) / 255.);
    //N
	pfInitN[i * outStride + j] = 1.0f;
	pfInitMeanIp[i * outStride + j] = fGuidedImg[i * outStride + j] * ptrT[i * outStride + j];
	pfInitMeanII[i * outStride + j] = fGuidedImg[i * outStride + j] * fGuidedImg[i * outStride + j];
}


__global__ void init_pfArrayCum_Y(float* pfInArray, float *pfArrayCum, int rows, int cols){
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if(j >= cols){
		return;
	}
	pfArrayCum[j] = pfInArray[j];
}

__global__ void pfArrayCum_Y(float* pfInArray, float *pfArrayCum, int rows, int cols, int stride){
	int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if((j >= cols) || (i > 0)){
        return;
    }
	
	for(int k = 1; k < rows; k++){
	 	pfArrayCum[k * stride + j] = pfArrayCum[(k - 1) * stride + j] + pfInArray[k * stride + j];
	}
}

__global__ void diff_Y_axis( float* fOutArray, float *pfArrayCum, int nR, int rows, int cols, int stride){
	int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if((j >= cols) || (i >= rows)){
        return;
    }
	
	if(i <  (nR + 1)){
		fOutArray[i * stride + j] = pfArrayCum[(i + nR + 1) * stride + j];
	}
	if(i >=(nR + 1) && i < (rows - nR)){
		fOutArray[i * stride + j] = pfArrayCum[(i + nR) * stride + j] - pfArrayCum[(i - nR - 1) * stride + j];
	}
	if(i>=(rows - nR) && i < rows){
		fOutArray[i * stride + j] = pfArrayCum[(rows - 1) * stride + j] - pfArrayCum[(i - nR - 1) * stride + j];
	}
}


__global__ void init_pfArrayCum_X(float* fOutArray, float *pfArrayCum, int rows, int cols, int stride){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i >= rows){
        return;
    }
	pfArrayCum[i * stride] = fOutArray[i * stride];

}

__global__ void pfArrayCum_X(float* fOutArray, float *pfArrayCum, int rows, int cols, int stride){
	int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if((i >= rows) || (j > 0)){
        return;
    }

	for(int k = 1; k < cols; k++){
	 	pfArrayCum[i * stride + k] = pfArrayCum[i * stride + k - 1] + fOutArray[i * stride + k];
	}
}

__global__ void diff_X_axis( float* fOutArray, float *pfArrayCum, int nR, int rows, int cols, int stride){
	int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if((j >= cols) || (i >= rows)){
        return;
    }
	if(j < (nR + 1)){
		fOutArray[i * stride + j] = pfArrayCum[i * stride + j + nR];
	}
	if(j >= (nR + 1) && j < (cols - nR)){
		fOutArray[i * stride + j] = pfArrayCum[i * stride + j + nR] - pfArrayCum[i * stride + j - nR - 1];
	}
	if(j >= (cols - nR) && j < cols){
		fOutArray[i * stride + j] = pfArrayCum[i * stride + cols - 1] - pfArrayCum[i * stride + j - nR - 1];
	}
}


void BoxFilter_gpu(float* pfArrayCum, float* pfInArray, float* fOutArray, int nR, int rows, int cols, int stride, cudaStream_t stream){
	dim3 gridSize((cols + BLOCK_W - 1) / BLOCK_W, (rows + BLOCK_H - 1) / BLOCK_H);
    dim3 blockSize(BLOCK_W, BLOCK_H);

	init_pfArrayCum_Y<<<gridSize, blockSize, 0, stream>>>(pfInArray, pfArrayCum, rows, cols);
	pfArrayCum_Y<<<gridSize, blockSize, 0, stream>>>(pfInArray, pfArrayCum, rows, cols, stride);
	diff_Y_axis<<<gridSize, blockSize, 0, stream>>>(fOutArray, pfArrayCum, nR, rows, cols, stride);

	init_pfArrayCum_X<<<gridSize, blockSize, 0, stream>>>(fOutArray, pfArrayCum, rows, cols, stride);
	pfArrayCum_X<<<gridSize, blockSize, 0, stream>>>(fOutArray, pfArrayCum, rows, cols, stride);
	diff_X_axis<<<gridSize, blockSize, 0, stream>>>(fOutArray, pfArrayCum, nR, rows, cols, stride);
}

__global__ void set_value(float* pfMeanI,
						  float* pfMeanP,
						  float* pfN,
						  float* pfMeanIp,
						  float* pfCovIp,
						  float* pfMeanII,
						  float* pfvarI,
						  float* pfA,
						  float* pfB, 
						  float fEps,
						  int rows, 
						  int cols, 
						  int outStride){
	int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if((j >= cols) || (i >= rows)){
        return;
    }
	pfMeanI[i * outStride + j] = pfMeanI[i * outStride + j] / pfN[i * outStride + j];
	pfMeanP[i * outStride + j] = pfMeanP[i * outStride + j] / pfN[i * outStride + j];
	pfMeanIp[i * outStride + j] = pfMeanIp[i * outStride + j] / pfN[i * outStride + j];
	pfCovIp[i * outStride + j] = pfMeanIp[i * outStride + j] - pfMeanI[i * outStride + j] * pfMeanP[i * outStride + j];
	pfMeanII[i * outStride + j] = pfMeanII[i * outStride + j] / pfN[i * outStride + j];
	pfvarI[i * outStride + j] = pfMeanII[i * outStride + j] - pfMeanI[i * outStride + j] * pfMeanI[i * outStride + j];
	//a and b
	pfA[i * outStride + j] = pfCovIp[i * outStride + j] / (pfvarI[i * outStride + j] + fEps);
	pfB[i * outStride + j] = pfMeanP[i * outStride + j] - pfA[i * outStride + j] * pfMeanI[i * outStride + j];
}


__global__ void get_guide_output(float* ptrGuidedT,
							float* pfOutA,
							float* fGuidedImg,
							float* pfOutB,
							float* pfN,
							int rows, 
							int cols, 
							int outStride){
	int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if((j >= cols) || (i >= rows)){
        return;
    }
	ptrGuidedT[i * outStride + j] = 1.0;
	ptrGuidedT[i * outStride + j] = (pfOutA[i * outStride + j] * fGuidedImg[i * outStride + j] + pfOutB[i * outStride + j]) / pfN[i * outStride + j];
}

void guided_filter_cuda(float* ptrT_device, unsigned char* guidedRGBImg_device, float* ptrGuidedT_device, int rows, int cols,
			   int inStride, int outStride, int m_nGBlockSize, float fEps, cudaStream_t stream,
               float* fGuidedImg_device, float* pfInitN_device, float* pfInitMeanIp_device, float* pfInitMeanII_device,
               float* pfMeanP_device, float* pfN_device, float* pfMeanI_device, float* pfMeanIp_device,
               float* pfMeanII_device, float* pfvarI_device, float* pfCovIp_device, float* pfA_device, float* pfB_device,
               float* pfOutA_device, float* pfOutB_device, float* pfArrayCum_device){
	dim3 gridSize((cols + BLOCK_W - 1) / BLOCK_W, (rows + BLOCK_H - 1) / BLOCK_H);
    dim3 blockSize(BLOCK_W, BLOCK_H);
	to_float_point<<<gridSize, blockSize, 0, stream>>>(ptrT_device,
                                                       fGuidedImg_device,
											           guidedRGBImg_device,
											           pfInitN_device,
											           pfInitMeanIp_device,
											           pfInitMeanII_device,
											           rows,
											           cols,
											           inStride,
											           outStride);

	BoxFilter_gpu(pfArrayCum_device, pfInitN_device, pfN_device, m_nGBlockSize, rows, cols, outStride, stream);
	 //Mean_I
    BoxFilter_gpu(pfArrayCum_device, fGuidedImg_device, pfMeanI_device, m_nGBlockSize, rows, cols, outStride, stream);
	 //Mean_P
    BoxFilter_gpu(pfArrayCum_device, ptrT_device, pfMeanP_device, m_nGBlockSize, rows, cols, outStride, stream);
	//mean_IP
    BoxFilter_gpu(pfArrayCum_device, pfInitMeanIp_device, pfMeanIp_device, m_nGBlockSize, rows, cols, outStride, stream);
	//mean_II
    BoxFilter_gpu(pfArrayCum_device, pfInitMeanII_device, pfMeanII_device, m_nGBlockSize, rows, cols, outStride, stream);

	set_value<<<gridSize, blockSize, 0, stream>>>(pfMeanI_device,
										          pfMeanP_device,
										          pfN_device,
										          pfMeanIp_device,
										          pfCovIp_device,
										          pfMeanII_device,
										          pfvarI_device,
										          pfA_device,
										          pfB_device,
										          fEps,
										          rows,
										          cols,
										          outStride);
	BoxFilter_gpu(pfArrayCum_device, pfA_device, pfOutA_device, m_nGBlockSize, rows, cols, outStride, stream);
    BoxFilter_gpu(pfArrayCum_device, pfB_device, pfOutB_device, m_nGBlockSize, rows, cols, outStride, stream);
	get_guide_output<<<gridSize, blockSize, 0, stream>>>(ptrGuidedT_device,
												         pfOutA_device,
												         fGuidedImg_device,
												         pfOutB_device,
												         pfN_device,
												         rows,
												         cols,
												         outStride);
}