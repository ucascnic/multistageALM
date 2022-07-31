#ifndef __CG_DD__
#define __CG_DD__
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>
#include "cnic_sparsematrix.h"
void Conjugate_gradient(sycl::queue *handle, double *A, double *b, double *x0,
                        double tol, int itMax, double *recources, int n);

void Conjugate_gradient_sp(sycl::queue *handle, CSRMatrix *A, double *b,
                           double *x0, double tol, int itMax, double *recources,
                           int n);

void Conjugate_gradient_debug(sycl::queue *handle, double *A, double *b,
                              double *x0, double tol, int itMax,
                              double *recources, int n);
void Conjugate_gradient_sp_bsr(sycl::queue *handle, BSRMatrix *A, double *b,
                               double *x0, double tol, int itMax,
                               double *recources, int n);

/*void Qr_sp_csr(cusolverSpHandle_t *cusolverH, csrqrInfo_t *info,
               oneapi::mkl::index_base *descrA, int *cusolver_status,
               CSRMatrix *A, double *b, double *x0, double tol, int itMax,
               double *buffer_qr);
*/

#include"resources.h"

/*void Qr_sp_buff(cusolverSpHandle_t *cusolverH, csrqrInfo_t *info,
                oneapi::mkl::index_base *descrA, int *cusolver_status,
                CSRMatrix *A, size_t *);
*/
#endif
