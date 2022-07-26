#ifndef CNIC_MATHLIB_H
#define CNIC_MATHLIB_H
#include<stdlib.h>
#include<stdio.h>
#include<mkl.h>
#include<string.h>
#include<iostream>


// res = A*x
inline void matrix_times_vector(double *A, double *x,double *res,int m,int n){
    cblas_dgemv(CblasColMajor,CblasNoTrans,m,n,1.0,A,m,x,1,0.0,res,1);
}


inline void matrix_times_vector_plus(double *A, double *x,double *res,int m,int n){
    cblas_dgemv(CblasColMajor,CblasNoTrans,m,n,1.0,A,m,x,1,1.0,res,1);
}


inline void matrixT_times_vector(double *A, double *x,double *res,int m,int n){
    cblas_dgemv(CblasColMajor,CblasTrans,m,n,1.0,A,m,x,1,0.0,res,1);
}


inline void matrix_times_vector_float(float *A, float *x,float *res,int m,int n){
    cblas_sgemv(CblasColMajor,CblasNoTrans,m,n,1.0,A,m,x,1,0.0,res,1);
}

template <typename T>
void show_res(T *res, int n){
    for (int i = 0;i<n;++i)
        std::cout << res[i] << " ";
    std::cout <<  std::endl;
    std::cout <<  std::endl;

}

template <typename T>
bool check_res(T *res, T *res2, int n){
    for (int i = 0;i<n;++i)
        if ( std::abs(res[i] - res2[i]) > 1e-6){
            std::cout << "error at " << i << " " << " res1 = " <<  res[i] << " res2= " << res2[i] << std::endl;
            return false;
        }
    return true;
}

template <typename T>
void matrix_times_vector_cpu(T *A,T *x, T *res,int m,int n){

    for (int i = 0 ; i < m ; ++i){
        T temp  = 0.0;
        for (int j = 0;j < n; ++j){
            temp += A[i + j*m] * x[j];
        }
        res[i] = temp;
    }

}


// y = alpha*x + y
inline void vector_times_alpha_plus(double *x, double alpha, double *y,int n){
    cblas_daxpy(n,alpha,x,1,y,1);
}


// y = x
inline void vector_copy(double *x, double*y,  int n){
    memcpy(y,x,sizeof(double)*n);
}

// y = x + y
inline void vector_add(double *x, double*y,  int n){
    cblas_daxpy(n,1.0,x,1,y,1);
}

// y = y - x
inline void vector_minues(double *x, double*y,  int n){
    cblas_daxpy(n,-1.0,x,1,y,1);
}

inline void vector_norm_2(double *x, int n);





#endif // CNIC_MATHLIB_H
