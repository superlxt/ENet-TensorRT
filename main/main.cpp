#include <QCoreApplication>
#include "segNet.h"
#include "loadImage.h"
#include "commandLine.h"
#include "cudaMappedMemory.h"
#include <sys/time.h>
#include "opencv2/opencv.hpp"



uint64_t current_timestamp() {
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}


// main entry point
int main()
{
    std::cout << "Hello, World!" << std::endl;
    cv::VideoCapture cap(0);

    if(!cap.isOpened())
    {
        std::cout<<"There is no video in this location"<<std::endl;
        return -1;
    }
    cv::Mat imgFilename;

    const char* prototxt = "/media/lxt/CVPR/Github/ENet-TensorRT/network/no_bn.prototxt";
    const char* modelName = "/media/lxt/CVPR/Github/ENet-TensorRT/network/no_bn.caffemodel";
    const char* input="data";
    const char* output="deconv6_0_0";
    uint32_t maxBatchSize = 1;
    // create the segNet from pretrained or custom model by parsing the command line
    segNet* net = segNet::Create(prototxt, modelName, input, output, maxBatchSize);


    // enable layer timings for the console application
    net->EnableProfiler();

    // load image from file on disk

    int    imgWidth  = 256;
    int    imgHeight = 256;
    const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 3;


    float* imgCPU    = NULL;
    float* imgCUDA   = NULL;
    // allocate output image
    float* outCPU  = NULL;
    uint8_t* outCUDA = NULL;


    while(1)
    {
        cap>>imgFilename;
        if(imgFilename.empty())//如果某帧为空则退出循环
           break;
        double t1 = cv::getTickCount();

        //cv::Mat imgFilename=cv::imread("/home/nvidia/lxt/jetson-inference/data/images/test/1.png");
        resize(imgFilename, imgFilename, cv::Size(256,256));
        imgFilename.convertTo(imgFilename, CV_32F, 1, 0);


        loadImageBGR(imgFilename, (float3**)&imgCPU, (float3**)&imgCUDA, &imgWidth, &imgHeight);
        memcpy(imgCPU,imgFilename.data,imgSize);
        cudaAllocMapped((void**)&outCPU, (void**)&outCUDA, imgWidth * imgHeight * sizeof(float) * 3);


        // process image overlay
        cv::Mat out = net->Overlay(imgCUDA, outCUDA, imgWidth, imgHeight);
        std::cout<<"time:                xxxxxx"<<(cv::getTickCount()-t1)/cv::getTickFrequency()<<std::endl;
        cv::imshow("frame",out);
        cv::waitKey(1);



    }

    CUDA(cudaFreeHost(imgCPU));
    CUDA(cudaFreeHost(outCPU));
    delete net;
    return 0;
}
