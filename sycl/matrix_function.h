#ifndef MATRIX_FUNCTION_H
#define MATRIX_FUNCTION_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>
#include <stdio.h>
#include<stdlib.h>

/*
DPCT1009:3: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
#define CHECK(call)                                                            \
    { const int error = call; }

static __const__ double alpha_1   =  -1.0 ;
static __const__ double alpha1  =  1.0 ;
static __const__ double beta   =  0.0 ;

static const double minvalue = 1e-4;
static const double maxvalue = 500.0;

void matrix_ATtimesA(sycl::queue *handle, double *f, int mf, int nf,
                     double *res);

void matrix_times(sycl::queue *handle, double *f, double *g, int mf, int nf,
                  int kg, double *res);

void matrix_timesV(sycl::queue *handle, double *f, double *g, int mf, int nf,
                   double *res);

void matrix_timesT(sycl::queue *handle, double *f, double *g, int mf, int nf,
                   int kg, double *res);

void matrixT(sycl::queue *handle, double *P, double *Pt, int m, int n);

void matrix_Ttimes(sycl::queue *handle, double *f, double *g, int mf, int nf,
                   int kg, double *res);

void matrix_AtimesAT(sycl::queue *handle, double *f, int mf, int nf,
                     double *res);

void matrixT_timesV(sycl::queue *handle, double *f, double *g, int mf, int nf,
                    double *res);

void show_res(double * ,int);

template <typename T>
void show_res_T(T *s,int n){
    T *res = (T *)malloc(n*sizeof(T));
    /*
    DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    CHECK((dpct::get_default_queue().memcpy(res, s, n * sizeof(T)).wait(), 0));
    for (int i = 0 ; i< n;++i){
        printf("%.4f\t",res[i]);
    }

    printf("\n");
    free(res);
}

template <typename T>
void show_res_T(T *s,int start, int end){
    int n = end - start;
    T *res = (T *)malloc(n*sizeof(T));
    /*
    DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    CHECK(
        (dpct::get_default_queue().memcpy(res, s + start, n * sizeof(T)).wait(),
         0));
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
    /*
    DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    CHECK(
        (dpct::get_default_queue().memcpy(res, s + start, n * sizeof(T)).wait(),
         0));
    for (int i = 0 ; i< end - start ;++i){
        printf("%d\t",res[i]);
    }

    printf("\n");
    free(res);
}
#endif // MATRIX_FUNCTION_H
