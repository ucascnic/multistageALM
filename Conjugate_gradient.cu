#include"Conjugate_gradient.h"
#include<cublas_v2.h>
#include<cuda_runtime_api.h>
#include"matrix_function.h"
#include<stdio.h>
#include<stdlib.h>
#include <cusolverSp.h>


__global__ void updateXnew(double* xOld, double*dOld,double*xNew, double alpha, int nn){
    // xNew = xOld  + alpha.*dOld;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nn)
        return;
    xNew[x] = xOld[x] + alpha * dOld[x];
    xOld[x] = xNew[x];


}

__global__ void update_rew(double* rnew,double *b, int nn){
    // xNew = xOld  + alpha.*dOld;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nn)
        return;
    rnew[x] = b[x] - rnew[x];


}

__global__ void updatedOld(double *rNew,double *dOld,double alpha, int nn){
    //dOld = rNew + alpha* dOld;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nn)
        return;
    dOld[x] = rNew[x] + alpha * dOld[x];

}
__global__ void updatedOld(double * rold, double *rNew,double *dOld,double alpha, int nn){
    //dOld = rNew + alpha* dOld;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nn)
        return;
    dOld[x] = rNew[x] + alpha * dOld[x];
    rold[x] = rNew[x];

}
#include"toolbox_dongdong.h"
__constant__ double  const_number[3] = {1.0, 0.0, -1.0};

__constant__ double yi = 1.0;
__constant__ double fuyi = -1.0;
void Conjugate_gradient(cublasHandle_t handle,double *A,double *b, double *x0,
                        double tol,int itMax,double *recources,int n){
    // recources is the memeroy we allocated before for the programm
    // to use whatever it want to use
    int size_x = sizeof(double) * n;
    dim3 block( (n)/128+ 1);
    double *xOld = &recources[0];
    double *rOld = &recources[n];
    double *dOld = &recources[2*n];
    double *xNew = &recources[3*n];
    double *local_temp = &recources[4*n];
    double *rNew =  &recources[5*n];

    // xOld = x0;
    cudaMemcpy(xOld,x0,size_x,cudaMemcpyDeviceToDevice);

    // rOld = b-A*xOld;

    matrix_timesV(handle,A,xOld,n,n,rOld);

    update_rew<<<block,128>>>(rOld,b,n);


    // dOld  ;
    cudaMemcpy(dOld,rOld,size_x,cudaMemcpyDeviceToDevice);



    double bNorm = 0.0;
    cublasDnrm2(handle,n,b,1,&bNorm); //bNorm;

    double alpha_ = 0.0;
    double beta_ = 0.0;
    double alpha_temp = 0.0;

    double resNorm = 0.0;

    for (int k =0; k <= 100; ++k){

        //alpha = (dOld'*rOld)/(dOld'*A*dOld);
        cublasDdot(handle,n,dOld,1,rOld,1,&alpha_temp);




        matrix_timesV(handle,A,dOld,n,n,local_temp);
        cublasDdot(handle,n,local_temp,1,dOld,1,&alpha_);


        alpha_ = alpha_temp/alpha_;

//         printf("alpha = %.6f\n",alpha_);exit(0);


        // xNew = xOld  + alpha.*dOld;
        updateXnew<<<block,128>>>(xOld,dOld,xNew,alpha_,n);


        //rNew = b-A*xNew;
        matrix_timesV(handle,A,xNew,n,n,rNew);
        update_rew<<<block,128>>>(rNew,b,n);
        cublasDnrm2(handle,n,rNew,1,&resNorm);
        if (resNorm < tol * bNorm){

            cudaMemcpy(x0,xNew,size_x,cudaMemcpyDeviceToDevice);

            return ;

        }



        cublasDdot(handle,n,rNew,1,rNew,1,&alpha_temp);
        cublasDdot(handle,n,rOld,1,rOld,1,&alpha_);
        //beta = (rNew'*rNew)/(rOld'*rOld);
        beta_ = alpha_temp/alpha_;


        //dOld = rNew + beta* dOld;
        updatedOld<<<block,128>>>(rNew,dOld,beta_,n);

        //rOld = rNew;
        cudaMemcpy(rOld,rNew,size_x,cudaMemcpyDeviceToDevice);

        //xOld = xNew;
        cudaMemcpy(xOld,xNew,size_x,cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(x0,xNew,size_x,cudaMemcpyDeviceToDevice);

}
void Conjugate_gradient_debug(cublasHandle_t handle,double *A,double *b, double *x0,
                        double tol,int itMax,double *recources,int n){
    // recources is the memeroy we allocated before for the programm
    // to use whatever it want to use
    int size_x = sizeof(double) * n;
    dim3 block( (n)/128+ 1);
    double *xOld = &recources[0];
    double *rOld = &recources[n];
    double *dOld = &recources[2*n];
    double *xNew = &recources[3*n];
    double *local_temp = &recources[4*n];
    double *rNew =  &recources[5*n];

    // xOld = x0;
    cudaMemcpy(xOld,x0,size_x,cudaMemcpyDeviceToDevice);

    // rOld = b-A*xOld;

    matrix_timesV(handle,A,xOld,n,n,rOld);

    update_rew<<<block,128>>>(rOld,b,n);


    // dOld  ;
    cudaMemcpy(dOld,rOld,size_x,cudaMemcpyDeviceToDevice);



    double bNorm = 0.0;
    cublasDnrm2(handle,n,b,1,&bNorm); //bNorm;

    double alpha_ = 0.0;
    double beta_ = 0.0;
    double alpha_temp = 0.0;
    double resNorm = 0.0;


    for (int k =0; k <= itMax; ++k){

        //alpha = (dOld'*rOld)/(dOld'*A*dOld);
        cublasDdot(handle,n,dOld,1,rOld,1,&alpha_temp);




        matrix_timesV(handle,A,dOld,n,n,local_temp);
        cublasDdot(handle,n,local_temp,1,dOld,1,&alpha_);


        alpha_ = alpha_temp/alpha_;

//         printf("alpha = %.6f\n",alpha_);exit(0);


        // xNew = xOld  + alpha.*dOld;
        updateXnew<<<block,128>>>(xOld,dOld,xNew,alpha_,n);


        //rNew = b-A*xNew;
        //cudaMemcpy(rNew,b,size_x,cudaMemcpyDeviceToDevice);
        //cublasDgemv(handle, CUBLAS_OP_N, n, n, &yi,
        //            A, n, xNew, 1, &alpha_1, rNew,1);
        //show_res(&const_number[0],2);
        matrix_timesV(handle,A,xNew,n,n,rNew);


        update_rew<<<block,128>>>(rNew,b,n);

        cublasDnrm2(handle,n,rNew,1,&resNorm);
        printf("resnorm = %.2f\n",resNorm);exit(0);
        if (resNorm < tol * bNorm){

            cudaMemcpy(x0,xNew,size_x,cudaMemcpyDeviceToDevice);

            return ;

        }

        cublasDdot(handle,n,rNew,1,rNew,1,&alpha_temp);
        cublasDdot(handle,n,rOld,1,rOld,1,&alpha_);
        //beta = (rNew'*rNew)/(rOld'*rOld);
        beta_ = alpha_temp/alpha_;


        //dOld = rNew + beta* dOld;
        updatedOld<<<block,128>>>(rNew,dOld,beta_,n);

        //rOld = rNew;
        cudaMemcpy(rOld,rNew,size_x,cudaMemcpyDeviceToDevice);

        //xOld = xNew;
        cudaMemcpy(xOld,xNew,size_x,cudaMemcpyDeviceToDevice);
    }

}

#include"cnic_sparsematrix.h"
void Conjugate_gradient_sp(cublasHandle_t handle,CSRMatrix *A,double *b, double *x0,
                        double tol,int itMax,double *recources,int n){
    // recources is the memeroy we allocated before for the programm
    // to use whatever it want to use
    int size_x = sizeof(double) * n;
    dim3 block( (n)/128+ 1);
    double *xOld = &recources[0];
    double *rOld = &recources[n];
    double *dOld = &recources[2*n];
    double *xNew = &recources[3*n];
    double *local_temp = &recources[4*n];
    double *rNew =  &recources[5*n];

    // xOld = x0;
    cudaMemcpy(xOld,x0,size_x,cudaMemcpyDeviceToDevice);



    //matrix_timesV(handle,A,xOld,n,n,rOld);
    sp_matrix_times_V_ptrl(A,xOld,rOld);

    update_rew<<<block,128>>>(rOld,b,n);


    // dOld  ;
    cudaMemcpy(dOld,rOld,size_x,cudaMemcpyDeviceToDevice);

    double bNorm = 0.0;
    cublasDnrm2(handle,n,b,1,&bNorm); //bNorm;

    double alpha_ = 0.0;
    double beta_ = 0.0;
    double alpha_temp = 0.0;

    double resNorm = 0.0;

    for (int k =0; k <= itMax; ++k){

        //alpha = (dOld'*rOld)/(dOld'*A*dOld);
        cublasDdot(handle,n,dOld,1,rOld,1,&alpha_temp);




        //matrix_timesV(handle,A,dOld,n,n,local_temp);
        sp_matrix_times_V_ptrl(A,dOld,local_temp);
        cublasDdot(handle,n,local_temp,1,dOld,1,&alpha_);


        alpha_ = alpha_temp/alpha_;

//         printf("alpha = %.6f\n",alpha_);exit(0);


        // xNew = xOld  + alpha.*dOld;
        updateXnew<<<block,128>>>(xOld,dOld,xNew,alpha_,n);


        //rNew = b-A*xNew;
        //matrix_timesV(handle,A,xNew,n,n,rNew);
        sp_matrix_times_V_ptrl(A,xNew,rNew);

        update_rew<<<block,128>>>(rNew,b,n);
        cublasDnrm2(handle,n,rNew,1,&resNorm);
        if (resNorm < tol * bNorm){

            cudaMemcpy(x0,xNew,size_x,cudaMemcpyDeviceToDevice);

            return ;

        }



        cublasDdot(handle,n,rNew,1,rNew,1,&alpha_temp);
        cublasDdot(handle,n,rOld,1,rOld,1,&alpha_);
        //beta = (rNew'*rNew)/(rOld'*rOld);
        beta_ = alpha_temp/alpha_;


        //dOld = rNew + beta* dOld;
        updatedOld<<<block,128>>>(rNew,dOld,beta_,n);

        //rOld = rNew;
        cudaMemcpy(rOld,rNew,size_x,cudaMemcpyDeviceToDevice);

        //xOld = xNew;
        cudaMemcpy(xOld,xNew,size_x,cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(x0,xNew,size_x,cudaMemcpyDeviceToDevice);

}
void Conjugate_gradient_sp_bsr(cublasHandle_t handle,BSRMatrix *A,double *b, double *x0,
                        double tol,int itMax,double *recources,int n){
    // recources is the memeroy we allocated before for the programm
    // to use whatever it want to use
    int size_x = sizeof(double) * n;
    dim3 block( (n)/128+ 1);
    double *xOld = &recources[0];
    double *rOld = &recources[n];
    double *dOld = &recources[2*n];

    double *local_temp = &recources[4*n];
    double *rNew =  &recources[5*n];

    // xOld = x0;
    cudaMemcpy(xOld,x0,size_x,cudaMemcpyDeviceToDevice);


    //matrix_timesV(handle,A,xOld,n,n,rOld);
    sp_bsr_matrix_times_V_ptrl(A,xOld,rOld);

    update_rew<<<block,128>>>(rOld,b,n);


    // dOld  ;
    cudaMemcpy(dOld,rOld,size_x,cudaMemcpyDeviceToDevice);

    double bNorm = 0.0;
    cublasDnrm2(handle,n,b,1,&bNorm); //bNorm;

    double alpha_ = 0.0;
    double beta_ = 0.0;
    double alpha_temp = 0.0;

    double resNorm = 0.0;
    double *alpha_cu =  &recources[6*n];
    for (int k =0; k <= itMax; ++k){

        cublasDdot(handle,n,dOld,1,rOld,1,&alpha_temp);
        sp_bsr_matrix_times_V_ptrl(A,dOld,local_temp);
        cublasDdot(handle,n,local_temp,1,dOld,1,&alpha_);
        alpha_ = alpha_temp/alpha_;
        updateXnew<<<block,128>>>(xOld,dOld,x0,alpha_,n);
        sp_bsr_matrix_times_V_ptrl(A,x0,rNew);
        update_rew<<<block,128>>>(rNew,b,n);
        cublasDdot(handle,n,rNew,1,rNew,1,&resNorm);
        if (sqrt(resNorm) < tol * bNorm){
            //cudaMemcpy(x0,xNew,size_x,cudaMemcpyDeviceToDevice);
            return ;

        }
        cublasDdot(handle,n,rOld,1,rOld,1,&alpha_);
        beta_ = resNorm/alpha_;

        updatedOld<<<block,128>>>(rOld,rNew,dOld,beta_,n);






    }
    //cudaMemcpy(x0,xNew,size_x,cudaMemcpyDeviceToDevice);

}

#include<assert.h>
#include"resources.h"
void  Qr_sp_buff(cusolverSpHandle_t *cusolverH,
                csrqrInfo_t *info ,
                cusparseMatDescr_t *descrA ,
                 cusolverStatus_t *cusolver_status,
                CSRMatrix *A,size_t *size_qr){


    int m = A->n_rows;
    int nnzA = A->n_element;
    int *d_csrRowPtrA = thrust::raw_pointer_cast(A->csr_row_ptrl.data());
    int *d_csrColIndA = thrust::raw_pointer_cast(A->cucol.data());
    double *d_csrValA = thrust::raw_pointer_cast(A->cudata.data());
    *cusolver_status = cusolverSpXcsrqrAnalysisBatched(
        *cusolverH, m, m, nnzA,
        *descrA, d_csrRowPtrA, d_csrColIndA,
        *info);
    size_t size_internal;

    int batchSize = 1;
    assert(*cusolver_status == CUSOLVER_STATUS_SUCCESS);
    *cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
         *cusolverH, m, m, nnzA,
         *descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
         batchSize,
         *info,
         &size_internal,
         size_qr);
    assert(*cusolver_status == CUSOLVER_STATUS_SUCCESS);
    printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal);
    printf("numerical factorization needs working space %lld bytes\n", (long long)*size_qr);



}
void Qr_sp_csr( cusolverSpHandle_t *cusolverH,
                csrqrInfo_t *info ,
                cusparseMatDescr_t *descrA,
                  cusolverStatus_t *cusolver_status,
                CSRMatrix *A,double *b, double *x0,
                               double tol,int itMax,double *buffer_qr){


    int batchSize = 1;
    int m = A->n_rows;
    int nnzA = A->n_element;
    int *d_csrRowPtrA = thrust::raw_pointer_cast(A->csr_row_ptrl.data());
    int *d_csrColIndA = thrust::raw_pointer_cast(A->cucol.data());
    double *d_csrValA = thrust::raw_pointer_cast(A->cudata.data());

    double *d_b = b;
    double *d_x = x0;



    *cusolver_status = cusolverSpDcsrqrsvBatched(
        *cusolverH, m, m, nnzA,
        *descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_b, d_x,
        batchSize,
        *info,
        buffer_qr);

    std::cout << *cusolver_status << std::endl;

    assert(*cusolver_status == CUSOLVER_STATUS_SUCCESS);

}
