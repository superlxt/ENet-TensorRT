/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "segNet.h"

#include "cudaMappedMemory.h"
#include "cudaOverlay.h"
#include "cudaResize.h"

#include "commandLine.h"
#include <iostream>

#include "opencv2/opencv.hpp"
// constructor
segNet::segNet() : tensorNet()
{
	mClassColors[0] = NULL;	// cpu ptr
	mClassColors[1] = NULL;  // gpu ptr

	mClassMap[0] = NULL;
	mClassMap[1] = NULL;
}


// destructor
segNet::~segNet()
{
	
}

// Create
segNet* segNet::Create( const char* prototxt, const char* model, const char* input_blob, const char* output_blob, uint32_t maxBatchSize )
{
	// create segmentation model
    segNet* net = new segNet();
	
	if( !net )
		return NULL;

	printf("\n");
	printf("segNet -- loading segmentation network model from:\n");
	printf("       -- prototxt:   %s\n", prototxt);
	printf("       -- model:      %s\n", model);
	printf("       -- input_blob  '%s'\n", input_blob);
	printf("       -- output_blob '%s'\n", output_blob);
	printf("       -- batch_size  %u\n\n", maxBatchSize);
	
	// load network
	std::vector<std::string> output_blobs;
	output_blobs.push_back(output_blob);
	
	if( !net->LoadNetwork(prototxt, model, NULL, input_blob, output_blobs, maxBatchSize) )
	{
		printf("segNet -- failed to initialize.\n");
		return NULL;
	}
	
	// initialize array of class colors
	const uint32_t numClasses = net->GetNumClasses();
	
	if( !cudaAllocMapped((void**)&net->mClassColors[0], (void**)&net->mClassColors[1], numClasses * sizeof(float4)) )
		return NULL;
    printf("       -- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  %u\n\n");
	for( uint32_t n=0; n < numClasses; n++ )
	{
		net->mClassColors[0][n*4+0] = 255.0f;	// r
		net->mClassColors[0][n*4+1] = 0.0f;	// g
		net->mClassColors[0][n*4+2] = 0.0f;	// b
		net->mClassColors[0][n*4+3] = 255.0f;	// a
	}
	
	// initialize array of classified argmax
	const int s_w = DIMS_W(net->mOutputs[0].dims);
	const int s_h = DIMS_H(net->mOutputs[0].dims);
	const int s_c = DIMS_C(net->mOutputs[0].dims);


	if( !cudaAllocMapped((void**)&net->mClassMap[0], (void**)&net->mClassMap[1], s_w * s_h * sizeof(uint8_t)) )
		return NULL;

	
	return net;
}



// declaration from imageNet.cu
cudaError_t cudaPreImageNet( float3* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight );


// Overlay
cv::Mat segNet::Overlay( float* rgba, uint8_t* output, uint32_t width, uint32_t height, const char* ignore_class )
{

    // downsample and convert to band-sequential BGR
    CUDA_FAILED(cudaPreImageNet((float3*)rgba, width, height, mInputCUDA, mWidth, mHeight));
    // process with GIE
    void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA };

    mContext->execute(1, inferenceBuffers);

    //PROFILER_REPORT();	// report total time, when profiling enabled
	
	// retrieve scores
    float* scores = mOutputs[0].CPU;
	const int s_w = DIMS_W(mOutputs[0].dims);
	const int s_h = DIMS_H(mOutputs[0].dims);
    const int s_c = DIMS_C(mOutputs[0].dims);



    cv::Mat imggg= cv::Mat::zeros(256,256,CV_8UC1);
    cv::Mat s0(256, 256, CV_32FC1, const_cast<float*> (scores));
    cv::Mat s1(256, 256, CV_32FC1, const_cast<float*> (scores+s_w*s_h));
//    for (int j=0;j<255;j++)
//            for(int i=0;i<255;i++)
//            {
//                if (s0.at<float>(i,j) > s1.at<float>(i,j))
//                        imggg.at<uchar>(i,j)=0;
//                else
//                        imggg.at<uchar>(i,j)=255;
//            }

    for (int j=0;j<255;j++)
    {
        float *X = s0.ptr<float>(j);
        float *Y = s1.ptr<float>(j);
        uchar *Z = imggg.ptr<uchar>(j);
        for(int i=0;i<255;i++)
        {
            if (X[i]>Y[i])
                Z[i] = 0;
            else
                Z[i] = 255;
        }
    }


    return imggg;
}


	
	
