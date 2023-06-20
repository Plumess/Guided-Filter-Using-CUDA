#include "guidedFilter.cuh"
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;

void tmpMalloc(float*& fGuidedImg_device, float*& pfInitN_device, float*& pfInitMeanIp_device, float*& pfInitMeanII_device,
               float*& pfMeanP_device, float*& pfN_device, float*& pfMeanI_device, float*& pfMeanIp_device,
               float*& pfMeanII_device, float*& pfvarI_device, float*& pfCovIp_device, float*& pfA_device, float*& pfB_device,
               float*& pfOutA_device, float*& pfOutB_device, float*& pfArrayCum_device, int fSize, cudaStream_t streamIdx){

    cudaMallocAsync((void **)&fGuidedImg_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfInitN_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfInitMeanIp_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfInitMeanII_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfMeanP_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfN_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfMeanI_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfMeanIp_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfMeanII_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfvarI_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfCovIp_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfA_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfB_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfOutA_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfOutB_device, fSize, streamIdx);
    cudaMallocAsync((void **)&pfArrayCum_device, fSize, streamIdx);
}

void tmpFree(float* fGuidedImg_device, float* pfInitN_device, float* pfInitMeanIp_device, float* pfInitMeanII_device,
               float* pfMeanP_device, float* pfN_device, float* pfMeanI_device, float* pfMeanIp_device,
               float* pfMeanII_device, float* pfvarI_device, float* pfCovIp_device, float* pfA_device, float* pfB_device,
               float* pfOutA_device, float* pfOutB_device, float* pfArrayCum_device, cudaStream_t streamIdx){

    cudaFreeAsync(fGuidedImg_device, streamIdx);
    cudaFreeAsync(pfInitN_device, streamIdx);
    cudaFreeAsync(pfInitMeanIp_device, streamIdx);
    cudaFreeAsync(pfInitMeanII_device, streamIdx);
    cudaFreeAsync(pfMeanP_device, streamIdx);
    cudaFreeAsync(pfN_device, streamIdx);
    cudaFreeAsync(pfMeanI_device, streamIdx);
    cudaFreeAsync(pfMeanIp_device, streamIdx);
    cudaFreeAsync(pfMeanII_device, streamIdx);
    cudaFreeAsync(pfvarI_device, streamIdx);
    cudaFreeAsync(pfCovIp_device, streamIdx);
    cudaFreeAsync(pfA_device, streamIdx);
    cudaFreeAsync(pfB_device, streamIdx);
    cudaFreeAsync(pfOutA_device, streamIdx);
    cudaFreeAsync(pfOutB_device, streamIdx);
    cudaFreeAsync(pfArrayCum_device, streamIdx);
}


void Guided_Filter(float* input, unsigned char* guideRGB, float* guidedOut, int rows, int cols,
                   int r, float eps, cudaStream_t streamIdx){

    int inStride = cols;
    int outStride = cols;
    int rgbSize = rows * cols * 3 * sizeof(unsigned char);
    int fSize = rows * cols * sizeof(float);

    // Malloc Device Memory
    float* d_input, *d_guidedOut;
    unsigned char* d_guideRGB;
    cudaMallocAsync((void **)&d_input, fSize, streamIdx);
    cudaMallocAsync((void **)&d_guidedOut, fSize, streamIdx);
    cudaMallocAsync((void **)&d_guideRGB, rgbSize, streamIdx);

    // Copy from Host Memory
    cudaMemcpyAsync(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice, streamIdx);
    cudaMemcpyAsync(d_guideRGB, guideRGB, rows * cols * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice, streamIdx);

    // Malloc Temporary Memory
    float *fGuidedImg_device = nullptr, *pfInitN_device = nullptr, *pfInitMeanIp_device = nullptr, *pfInitMeanII_device = nullptr,
          *pfMeanP_device = nullptr, *pfN_device = nullptr, *pfMeanI_device = nullptr, *pfMeanIp_device = nullptr,
          *pfMeanII_device = nullptr, *pfvarI_device = nullptr, *pfCovIp_device = nullptr, *pfA_device = nullptr, *pfB_device = nullptr,
          *pfOutA_device = nullptr, *pfOutB_device = nullptr, *pfArrayCum_device = nullptr;

    tmpMalloc(fGuidedImg_device, pfInitN_device, pfInitMeanIp_device, pfInitMeanII_device,
              pfMeanP_device, pfN_device, pfMeanI_device, pfMeanIp_device,
              pfMeanII_device, pfvarI_device, pfCovIp_device, pfA_device, pfB_device,
              pfOutA_device, pfOutB_device, pfArrayCum_device, fSize, streamIdx);

    // Guided Filter using CUDA
    guided_filter_cuda(d_input, d_guideRGB, d_guidedOut, rows, cols,
                       inStride, outStride, r, eps, streamIdx,
                       fGuidedImg_device, pfInitN_device, pfInitMeanIp_device, pfInitMeanII_device,
                       pfMeanP_device, pfN_device, pfMeanI_device, pfMeanIp_device,
                       pfMeanII_device, pfvarI_device, pfCovIp_device, pfA_device, pfB_device,
                       pfOutA_device, pfOutB_device, pfArrayCum_device);

    // Out
    cudaMemcpyAsync(guidedOut, d_guidedOut, rows * cols * sizeof(float), cudaMemcpyDeviceToHost, streamIdx);

    // Free
    cudaFreeAsync(d_input, streamIdx);
    cudaFreeAsync(d_guideRGB, streamIdx);
    cudaFreeAsync(d_guidedOut, streamIdx);
    tmpFree(fGuidedImg_device, pfInitN_device, pfInitMeanIp_device, pfInitMeanII_device,
            pfMeanP_device, pfN_device, pfMeanI_device, pfMeanIp_device,
            pfMeanII_device, pfvarI_device, pfCovIp_device, pfA_device, pfB_device,
            pfOutA_device, pfOutB_device, pfArrayCum_device, streamIdx);
}

int main(int argc, char *argv[]) {
    string workspace = argv[1];
    string input_folder = workspace + "input/";
    string guide_folder = workspace + "guide/";
    string output_folder = workspace + "output/";

    int frameIdx = 0;
    int streamIdx = 0;

    int r = 60;
    float eps = 1e-2;
    int rows = 0;
    int cols = 0;
    float* input = nullptr;
    unsigned char* guideRGB = nullptr;
    float* guidedOut = nullptr;

    auto total_time = std::chrono::duration<double, std::micro>::zero();

    for (const auto & entry : std::filesystem::directory_iterator(input_folder)) {
        std::string input_path = entry.path().string();
        std::string filename = entry.path().filename().string(); // get filename include extensions
        std::string basename = entry.path().stem().string(); // get filename without extensions
        // remove "_transmission" postfix
        std::size_t pos = basename.find("_transmission");
        if (pos != std::string::npos) {
            basename = basename.substr(0, pos);
        }

        std::cout<< input_path << std::endl;

        // load gray png
        cv::Mat input_img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
        // load rgb guide
        std::string guide_path = guide_folder + basename + ".jpg";
        cv::Mat guide_img = cv::imread(guide_path, cv::IMREAD_COLOR);

        // init size
        if(frameIdx==0){
            rows = input_img.rows;
            cols = input_img.cols;
            input = new float[rows * cols];
            guideRGB = new unsigned char[rows * cols * 3];
            guidedOut = new float[rows * cols];
        }

        // convert input to float
        cv::Mat input_float;
        input_img.convertTo(input_float, CV_32F, 1/255.0);

        memcpy(input, input_float.data, rows * cols * sizeof(float));
        memcpy(guideRGB, guide_img.data, rows * cols * 3 * sizeof(unsigned char));

        auto t1 = std::chrono::system_clock::now();

        // guided filter cuda
        Guided_Filter(input, guideRGB, guidedOut, rows, cols, r, eps, reinterpret_cast<cudaStream_t>(streamIdx));

        auto t2 = std::chrono::system_clock::now();
        auto full_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        std::cout << "This Time: " << full_time.count() << " us" << std::endl;
        // ignore warmup
        if(frameIdx!=0)
            total_time += full_time;

        // save result
        cv::Mat guided_transmission(rows, cols, CV_32F, guidedOut);
        cv::Mat guided_transmission_8UC1;
        guided_transmission.convertTo(guided_transmission_8UC1, CV_8UC1, 255.0);
        std::string save_path = output_folder + basename + "_guided.png";
        cv::imwrite(save_path, guided_transmission_8UC1);

        frameIdx++;
    }

    // ignore warmup
    if(frameIdx!=0)
        std::cout << "\nAverage Time: " << total_time.count() / (frameIdx-1) << " us" << std::endl;

    delete[] input;
    delete[] guideRGB;
    delete[] guidedOut;

    return 0;
}

