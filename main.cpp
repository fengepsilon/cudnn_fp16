#include <iostream>
#include "cuda_runtime.h"
#include "cublas.h"
#include "cublas_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include <cudnn.h>
#include <opencv2/opencv.hpp>
using namespace std;

#define PRINT_LOG fprintf(stdout,"file=%s:line=%d\n",__FILE__,__LINE__); fflush(stdout);

#define CUDNN_CHECK(err) { if(err != CUDNN_STATUS_SUCCESS){ printf("cudnn_err = %d:\n"); return -1; } }

template <typename Dtype>
inline void createActivationDescriptor(cudnnActivationDescriptor_t* activ_desc,
    cudnnActivationMode_t mode) {
  CUDNN_CHECK(cudnnCreateActivationDescriptor(activ_desc));
  CUDNN_CHECK(cudnnSetActivationDescriptor(*activ_desc, mode,
                                           CUDNN_PROPAGATE_NAN, Dtype(0)));
}

int main(int argc, char* argv[])
{
    {
    cudnnHandle_t               handle_;
    cudnnTensorDescriptor_t     bottom_desc_;
    cudnnTensorDescriptor_t     top_desc_;
    cudnnActivationDescriptor_t activ_desc_;



    cudnnStatus_t cudnn_err = cudnnCreate(&handle_); 
    cout << cudnn_err << endl;
    PRINT_LOG;

    cudnn_err = cudnnCreateTensorDescriptor(&bottom_desc_);
    cout << cudnn_err << endl;
    PRINT_LOG;

    cudnn_err = cudnnCreateTensorDescriptor(&top_desc_);
    cout << cudnn_err << endl;
    PRINT_LOG;
    
    cudnn_err = cudnnCreateActivationDescriptor(&activ_desc_);
    cout << cudnn_err << endl;
    PRINT_LOG;


    cudnn_err =  cudnnSetActivationDescriptor(activ_desc_, CUDNN_ACTIVATION_RELU,
                                            CUDNN_NOT_PROPAGATE_NAN, double(0.0));
    cout << cudnn_err << endl;
    PRINT_LOG;


    int n=1;
    int c=1024;
    int w=1;
    int h=1;
    const int stride_w = 1;
    const int stride_h = w * stride_w;
    const int stride_c = h * stride_h;
    const int stride_n = c * stride_c;

    cudnn_err =  cudnnSetTensor4dDescriptorEx(bottom_desc_, CUDNN_DATA_HALF,
            n, c, h, w, stride_n, stride_c, stride_h, stride_w);
    cout << cudnn_err << endl;
    PRINT_LOG;

    cudnn_err =  cudnnSetTensor4dDescriptorEx(top_desc_, CUDNN_DATA_HALF,
            n, c, h, w, stride_n, stride_c, stride_h, stride_w);
    cout << cudnn_err << endl;
    PRINT_LOG;




    float half_one = (1.0);
    float half_zero = (0.0);


    half* bottom_data;
    half* top_data ;


    cudaMalloc((void**)&bottom_data,sizeof(half)*n*c*w*h);
    cudaMalloc((void**)&top_data,sizeof(half)*n*c*w*h);

    half *bottom_data_host = new half[n*c*w*h];
    for(size_t i = 0; i < n*c*w*h; i=i+2)
    {
        bottom_data_host[i] =  (-1.0);
        bottom_data_host[i+1] =  (1.0);
    }   

    
    
	cudaMemcpy(bottom_data,bottom_data_host,sizeof(half)*n*c*w*h,cudaMemcpyHostToDevice);
	cudaMemcpy(top_data,bottom_data_host,sizeof(half)*n*c*w*h,cudaMemcpyHostToDevice);
    memset(bottom_data_host,0,sizeof(half)*n*w*c*h);

	cudaMemcpy(bottom_data_host,bottom_data,sizeof(half)*n*c*w*h,cudaMemcpyDeviceToHost);
    cout <<"before" << endl;
    for(size_t i = 0; i < 10; i++)
    {
        cout << bottom_data_host[i]<<endl;
    }
    double t0 = (double)cvGetTickCount();
    for(int k = 0; k < 10000; k++)
    cudnn_err =  cudnnActivationForward( handle_,
                            activ_desc_,
                            &half_one,
                            bottom_desc_, bottom_data,
                            &half_zero,
                            top_desc_, top_data);

    double t1 = (double)cvGetTickCount();
    printf("elasped time is %fms\n",(t1-t0)/cvGetTickFrequency()/1000.0);

    cout << cudnn_err << endl;
    PRINT_LOG;



	cudaMemcpy(bottom_data_host,top_data,sizeof(half)*n*c*w*h,cudaMemcpyDeviceToHost);




    cout <<"after" << endl;
    for(size_t i = 0; i < 10; i++)
    {
        cout << bottom_data_host[i]<<endl;
    }
    




    }



    PRINT_LOG;
    printf("hello world\n");
    return 0;
}