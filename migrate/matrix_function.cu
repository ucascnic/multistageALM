#include"matrix_function.h"
#include<cuda_runtime_api.h>
#include <cublas_v2.h>

void matrix_times(cublasHandle_t handle,double *f,double *g,int mf,int nf, int kg,double *res){

    cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,mf,kg,nf,
                 &alpha1 ,f ,mf ,g ,nf ,&beta,res,mf);
}

void matrix_timesV(cublasHandle_t handle,double *f,double *g,int mf,int nf,double *res){

    cublasDgemv(handle, CUBLAS_OP_N, mf, nf, &alpha1,
                f, mf, g, 1, &beta, res,1);
}
void matrixT_timesV(cublasHandle_t handle,double *f,double *g,int mf,int nf,double *res){

    cublasDgemv(handle, CUBLAS_OP_T, mf, nf, &alpha1,
                f, mf, g, 1, &beta, res,1);
}
void matrix_timesT(cublasHandle_t handle,double *f,double *g,int mf,int nf, int kg,double *res){

    cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,mf,kg,nf,
                 &alpha1 ,f,mf,g,kg,&beta,res,mf);
}

void matrix_AtimesAT(cublasHandle_t handle,double *f,int mf,int nf,double *res){

    cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,mf,mf,nf,
                 &alpha1 ,f,mf,f,mf,&beta,res,mf);
}
void matrix_ATtimesA(cublasHandle_t handle,double *f,int mf,int nf,double *res){

    cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,nf,nf,mf,
                 &alpha1 ,f,mf,f,mf,&beta,res,nf);
}
void matrixT(cublasHandle_t handle, double *P, double *Pt, int m, int n){
    cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha1, P, m, &beta,
                Pt,n, Pt, n);

}


void show_res(double *s,int n){
    double *res = (double *)malloc(n*sizeof(double));
    CHECK(cudaMemcpy(res,s,n*sizeof(double),cudaMemcpyDeviceToHost));
    for (int i = 0 ; i< n;++i){
        printf("%.6f\t",res[i]);
    }

    printf("\n");
    free(res);
}





