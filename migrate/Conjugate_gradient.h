#ifndef __CG_DD__
#define __CG_DD__
#include<cublas_v2.h>
#include"cnic_sparsematrix.h"
void Conjugate_gradient(cublasHandle_t handle,double *A,double *b, double *x0,
                        double tol,int itMax,double *recources,int n);

void Conjugate_gradient_sp(cublasHandle_t handle,CSRMatrix *A,double *b, double *x0,
                        double tol,int itMax,double *recources,int n);

void Conjugate_gradient_debug(cublasHandle_t handle,double *A,double *b, double *x0,
                        double tol,int itMax,double *recources,int n);
void Conjugate_gradient_sp_bsr(cublasHandle_t handle,BSRMatrix *A,double *b, double *x0,
                        double tol,int itMax,double *recources,int n);


#include<cusolverSp.h>
void Qr_sp_csr( cusolverSpHandle_t *cusolverH,
                csrqrInfo_t *info ,
                cusparseMatDescr_t *descrA,
                  cusolverStatus_t *cusolver_status,
                CSRMatrix *A,double *b, double *x0,
                               double tol,int itMax,double *buffer_qr);

#include"resources.h"
void Qr_sp_buff(cusolverSpHandle_t *cusolverH,
                csrqrInfo_t *info ,
                cusparseMatDescr_t *descrA ,
                cusolverStatus_t *cusolver_status,
                CSRMatrix *A, size_t *);
#endif
