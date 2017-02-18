#ifndef SUPPLEMENT_GPU_MATH_FUNCTIONS_HPP
#define SUPPLEMENT_GPU_MATH_FUNCTIONS_HPP
#include "caffe/common.hpp"
//#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"
#include "cublas_v2.h"
//C=a*A*AT+b*C; all col major
void gpu_rankUpdate(float* C, const float* A, const int C_length, const int A_col, const float alpha=1.0,
                           const float beta=0.0)
{
    CUBLAS_CHECK(cublasSsyrk(Caffe::cublas_handle(),CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,
                             C_length,A_col,&alpha,A,C_length,&beta,C,C_length));
}

#endif // SUPPLEMENT_GPU_MATH_FUNCTIONS_HPP
