#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "matrix_function.h"
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

void matrix_times(sycl::queue *handle, double *f, double *g, int mf, int nf,
                  int kg, double *res) {

    oneapi::mkl::blas::gemm(*handle, oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::nontrans, mf, kg, nf,
                            dpct::get_value(&alpha1, *handle), f, mf, g, nf,
                            dpct::get_value(&beta, *handle), res, mf);
}

void matrix_timesV(sycl::queue *handle, double *f, double *g, int mf, int nf,
                   double *res) {

    oneapi::mkl::blas::gemv(*handle, oneapi::mkl::transpose::nontrans, mf, nf,
                            dpct::get_value(&alpha1, *handle), f, mf, g, 1,
                            dpct::get_value(&beta, *handle), res, 1);
}
void matrixT_timesV(sycl::queue *handle, double *f, double *g, int mf, int nf,
                    double *res) {

    oneapi::mkl::blas::gemv(*handle, oneapi::mkl::transpose::trans, mf, nf,
                            dpct::get_value(&alpha1, *handle), f, mf, g, 1,
                            dpct::get_value(&beta, *handle), res, 1);
}
void matrix_timesT(sycl::queue *handle, double *f, double *g, int mf, int nf,
                   int kg, double *res) {

    oneapi::mkl::blas::gemm(*handle, oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::trans, mf, kg, nf,
                            dpct::get_value(&alpha1, *handle), f, mf, g, kg,
                            dpct::get_value(&beta, *handle), res, mf);
}

void matrix_AtimesAT(sycl::queue *handle, double *f, int mf, int nf,
                     double *res) {

    oneapi::mkl::blas::gemm(*handle, oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::trans, mf, mf, nf,
                            dpct::get_value(&alpha1, *handle), f, mf, f, mf,
                            dpct::get_value(&beta, *handle), res, mf);
}
void matrix_ATtimesA(sycl::queue *handle, double *f, int mf, int nf,
                     double *res) {

    oneapi::mkl::blas::gemm(*handle, oneapi::mkl::transpose::trans,
                            oneapi::mkl::transpose::nontrans, nf, nf, mf,
                            dpct::get_value(&alpha1, *handle), f, mf, f, mf,
                            dpct::get_value(&beta, *handle), res, nf);
}
void matrixT(sycl::queue *handle, double *P, double *Pt, int m, int n) {
    /*
    DPCT1007:12: Migration of this CUDA API is not supported by the Intel(R)
    DPC++ Compatibility Tool.
    */
    cublasDgeam(handle, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, n, m, &alpha1, P, m, &beta,
                Pt, n, Pt, n);
}


void show_res(double *s,int n){
    double *res = (double *)malloc(n*sizeof(double));
    /*
    DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CHECK((dpct::get_default_queue().memcpy(res, s, n * sizeof(double)).wait(),
           0));
    for (int i = 0 ; i< n;++i){
        printf("%.6f\t",res[i]);
    }

    printf("\n");
    free(res);
}





