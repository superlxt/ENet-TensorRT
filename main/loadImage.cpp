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
#include <iostream>
#include "loadImage.h"
#include "cudaMappedMemory.h"

//#include <QImage>
#include <opencv2/core/mat.hpp>




bool    loadImageBGR( cv::Mat frame, float3** cpu, float3** gpu, int* width, int* height)
{
    const uint32_t imgWidth  = 256;
    const uint32_t imgHeight = 256;
    const uint32_t imgPixels = imgWidth * imgHeight;
    const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 3;

    // allocate buffer for the image
    if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
    {
        printf(LOG_CUDA "failed to allocated bytes for image");
        return false;
    }

    return true;
}
