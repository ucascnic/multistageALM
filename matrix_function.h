#ifndef MATRIX_FUNCTION_H
#define MATRIX_FUNCTION_H

#include<cuda_runtime_api.h>
#include<cublas_v2.h>
#include<stdio.h>
#include<stdlib.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}




static __const__ double alpha_1   =  -1.0 ;
static __const__ double alpha1  =  1.0 ;
static __const__ double beta   =  0.0 ;

static const double minvalue = 1e-4;
static const double maxvalue = 500.0;

void matrix_ATtimesA(cublasHandle_t handle,double *f,int mf,int nf,double *res);

void matrix_times(cublasHandle_t handle,double *f,double *g,int mf,int nf, int kg,double *res);

void matrix_timesV(cublasHandle_t handle,double *f,double *g,int mf,int nf,double *res);

void matrix_timesT(cublasHandle_t handle,double *f,double *g,int mf,int nf, int kg,double *res);

void matrixT(cublasHandle_t handle, double *P, double *Pt, int m, int n);

void matrix_Ttimes(cublasHandle_t handle,double *f,double *g,int mf,int nf, int kg,double *res);

void matrix_AtimesAT(cublasHandle_t handle,double *f,int mf,int nf,double *res);

void matrixT_timesV(cublasHandle_t handle,double *f,double *g,int mf,int nf,double *res);

void show_res(double * ,int);

template <typename T>
void show_res_T(T *s,int n){
    T *res = (T *)malloc(n*sizeof(T));
    CHECK(cudaMemcpy(res,s,n*sizeof(T),cudaMemcpyDeviceToHost));
    for (int i = 0 ; i< n;++i){
        printf("%.6f\t",res[i]);
    }

    printf("\n");
    free(res);
}

template <typename T>
void show_res_T(T *s,int start, int end){
    int n = end - start;
    T *res = (T *)malloc(n*sizeof(T));
    CHECK(cudaMemcpy(res,s + start,n*sizeof(T),cudaMemcpyDeviceToHost));
    for (int i = 0 ; i< end - start ;++i){
        printf("%.6f\t",res[i]);
    }

    printf("\n");
    free(res);
}

template <typename T>
void show_res_int(T *s,int start, int end){
    int n = end - start;
    T *res = (T *)malloc(n*sizeof(T));
    CHECK(cudaMemcpy(res,s + start,n*sizeof(T),cudaMemcpyDeviceToHost));
    for (int i = 0 ; i< end - start ;++i){
        printf("%d\t",res[i]);
    }

    printf("\n");
    free(res);
}
#endif // MATRIX_FUNCTION_H
