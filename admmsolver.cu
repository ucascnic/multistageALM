#include"admmsolver.h"
#include<stdio.h>
#include"resources.h"
#include<vector>
#include"cnic_sparsematrix.h"
//#define ARMA_ALLOW_FAKE_GCC
//#include<armadillo>
//using namespace arma;

#include"matrix_function.h"
#include<cuda_runtime_api.h>
#include"Conjugate_gradient.h"
#define THREAD_ 128

#define copy_double_device_data(a,b,n)  \
    cudaMemcpy(a,b,(n)*sizeof(double),cudaMemcpyDeviceToDevice);
#define copy_double_device_data_start_length(a,b,start,length) \
    cudaMemcpy(a+start,b,(length)*sizeof(double),cudaMemcpyDeviceToDevice);


#define matrix_cslice(a,matrix,nrows,col) cudaMemcpy(a,matrix+(nrows)*(col),(nrows)*sizeof(double),cudaMemcpyDeviceToDevice);

//r =  yk + ahatk - beq - lambda/rho;
__global__ void update_r(double *r,
                         double *yk,
                         double *ahatk,
                         double *beq,
                         double *lambda,
                         double rho,int nlambda){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nlambda)
        return;
    r[x] =  yk[x] + ahatk[x] - beq[x] - lambda[x]/rho;

}
__global__ void update_xk(double *xk,
                          double *old_x,
                          double *b,
                          double *S,
                          double *r,
                          double *a,
                          double *LA,
                          double *UA,
                          double *DD,
                          int n,
                          int d,
                          int r_end,
                          double tau,
                          double rho){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=n+d)
        return;

    if(x<n-1)
    {
        xk[x] = (tau*old_x[x] - b[x] - rho *  S[x]   * r[x]) /(2.0*a[x] + rho *   S[x] *S[x] + tau);
        xk[x] = min(max(xk[x],LA[x]),UA[x]);

    }
    else
    {
        if (x>n-1){

            xk[x] = (tau*old_x[x] - b[x] - rho * DD[x-n] * r[x])
                    /(2.0*a[x] + rho  * DD[x-n] * DD[x-n]  + tau);
            xk[x] =  max(xk[x],0.0);

//            if(x == (n+d - 1)){
//                printf("old_x[%d] = %.6f\n",x+1,old_x[x]);
//                printf("tau[%d] = %.6f\n",x+1,tau);
//                printf("b[%d] = %.6f\n",x+1,b[x]);
//                printf("rho[%d] = %.6f\n",x+1,rho);
//                printf("DD[%d] = %.6f\n",x+1,DD[x-n]);
//                printf("r[%d] = %.6f\n",x+1,r[x]);
//                printf("a[%d] = %.6f\n",x+1,a[x]);


//            }

        }
        else{

            xk[n-1] =  (tau * old_x[n-1] - b[n-1] - rho * S[n-1] *
                    (r[n-1] - r[r_end]))/(2.0*rho * S[n-1] * S[n-1] + 2.0*a[n-1] + tau);
            xk[n-1] = min(max(xk[n-1],LA[n-1]),UA[n-1]);

        }

    }



}


// haty = haty - N1 - lambday/tau2;
// haty = max(haty,0);
__global__
void update_hat_y(double *haty,double *N1,double *lambday,double tau2,int N1_size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=N1_size)
        return;
     haty[x] = haty[x] - N1[x] - lambday[x]/tau2;
     haty[x] = max(haty[x],0.0);

}

//b2 =  -haty - N1 - lambday/tau2;
__global__ void update_b2(double *b2,double *haty,double *N1,double *lambday,double tau2, int nlambday){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nlambday)
        return;
    b2[x] =  -haty[x] - N1[x] - lambday[x]/tau2;

}

__global__ void update_b1(double *b1,double*b1_temp,double*beq,double*lambda,double rho,int n){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=n)
        return;
    b1[x] += b1_temp[x] - beq[x] - lambda[x]/rho;
}

__global__
void update_bb(double *bb,
               double *bb_temp,
               double rho,
               double tau2,
               int nn){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nn)
        return;

    bb[x] = -(rho * bb_temp[x] +  tau2 *bb[x] );

}
inline
double* allocate_data(Resources<double>*resources,vec &b){
    double *b_cu = resources->allocate_resource(b.n_elem);
    cudaMemcpy(b_cu,b.memptr(),sizeof(double)*b.n_elem,cudaMemcpyHostToDevice);

    return b_cu;
}


__global__ void
update_residual_kernel(double*r,double*old_x,double*old_y,double*beq,double*lambda,
                       double rho,int nlambda){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=nlambda)
        return;
     r[x] = old_x[x] + old_y[x]  - beq[x] - lambda[x]/rho;

}
inline void
update_residual(double*r,double*old_x,double*old_y,double*beq,double*lambda,double rho,int nlambda){
    dim3 block(nlambda/THREAD_+1);
    update_residual_kernel<<<block,THREAD_>>>(r,old_x,old_y,beq,lambda,rho,nlambda);

}

__global__ void
update_ahatk_kernel(double *ahatk,double *residual,int n,int d){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=(d+1))
        return;
     ahatk[x] =   -residual[n+x];
     ahatk[x] = max(ahatk[x],0.0);

}

inline void
update_ahatk(double *ahatk,double *residual,int n,int d){
    dim3 block((d+1)/THREAD_+1);
    update_ahatk_kernel<<<block,THREAD_>>>(ahatk,residual,n,d);
}

__global__
void update_lambda_2_kernel(double *lambda_2,int N_block,
                double *lambday,
                double tau2,
                double *res2,
                double *N1,
                double *yk_temp,
                double *haty,
                int nlambda){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>= nlambda)
        return;
    res2[x] =   N1[x] - yk_temp[x] + haty[x];
    double *lambda_local = lambda_2 + (N_block - 1) * nlambda;
    lambda_local[x] = lambday[x] +  tau2 * res2[x];

}

inline void
update_lambda_2(double *lambda_2,int N_block,
                double *lambday,
                double tau2,
                double *res2,
                double *N1,
                double *yk_temp,
                double *haty,
                int nlambda){

    dim3 block((nlambda)/THREAD_+1);
    update_lambda_2_kernel<<<block,THREAD_>>>(lambda_2,N_block,
                                       lambday,tau2,res2,N1,yk_temp,haty,nlambda);

}

__global__ void
update_lambda_1_kernel(double*lambda_1,
                int N_block,
                double*lambda,
                double rho,
                double*res1,
                double*beq,
                double*old_x,
                double*old_y,
                double*ahatk_temp,int nlambda){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>= nlambda)
        return;
    res1[x] =  beq[x] - old_x[x] - old_y[x] - ahatk_temp[x];
    double *lambda_local = lambda_1 + (N_block - 1) * nlambda;
    lambda_local[x] = lambda[x] +  rho * res1[x];

}

inline void
update_lambda_1(double*lambda_1,
                int N_block,
                double*lambda,
                double rho,
                double*res1,
                double*beq,
                double*old_x,
                double*old_y,
                double*ahatk_temp,int nlambda){

    dim3 block((nlambda)/THREAD_+1);
    update_lambda_1_kernel<<<block,THREAD_>>>(lambda_1,
                           N_block,
                           lambda,
                           rho,
                           res1,
                           beq,
                           old_x,
                           old_y,
                           ahatk_temp,nlambda);


}
#include<cusolverSp.h>

void admm_solver(sp_mat &AAA,sp_mat &A, sp_mat &B,sp_mat &C, sp_mat &M1,
                 sp_mat &Aeqy, vec &beqy, vec &D,
                 vec &beq, vec & N1, vec & a,vec & b,vec & S,
                  vec & LA,vec & UA,
                 int maxiter,int n,int d){

    //std::cout << A << std::endl;exit(0);

    Resources<double>  resources(1<<23);
    int k = Aeqy.n_cols;
    int ny;
    int N_block = 4;
    int nlambday = M1.n_rows;
    int nlambda = beq.size();
    int r_end = nlambda-1;


    int bbsize = B.n_cols;

    double rho = 0.01;
    double tau = 0.5;
    double tau2 = 0.01;

    double *xk = resources.allocate_resource(n+d);
    double *old_x = resources.allocate_resource(n+d);
    double *Axk = resources.allocate_resource(A.n_rows);
    double *Byk = resources.allocate_resource(B.n_rows);

    double *ahatk = resources.allocate_resource(1+d);
    double *ahatk_temp = resources.allocate_resource(1+d);

    double *old_ahat = resources.allocate_resource(1+d);
    double * r_cu = resources.allocate_resource(nlambda);
    double * temp1 = resources.allocate_resource(nlambda);
    double * temp2 = resources.allocate_resource(nlambda);

    double *yk = resources.allocate_resource(k);
    double *yk_temp = resources.allocate_resource(k);
    double *old_y = resources.allocate_resource(k);

    double *lambda_1 = resources.allocate_resource(N_block*nlambda);
    double *lambda_2 = resources.allocate_resource(N_block*nlambday);



    double *lambda = resources.allocate_resource(nlambda);
    double *lambday = resources.allocate_resource(nlambday);

    double *haty = resources.allocate_resource(nlambday);
    double * res1 = resources.allocate_resource(nlambda);
    double * res2 = resources.allocate_resource(nlambday);
    double * b1 = resources.allocate_resource(nlambda);
    double * b1_temp = resources.allocate_resource(nlambda);
    double * b2 = resources.allocate_resource(nlambday);
    double * bb = resources.allocate_resource(M1.n_cols);


    int equation_size = bbsize+beqy.n_elem;
    double * bbb = resources.allocate_resource(equation_size);
    double * yy = resources.allocate_resource(equation_size);
    double * buff = resources.allocate_resource(8*equation_size);
    double * residual = resources.allocate_resource(equation_size);
    double *bb_temp = resources.allocate_resource(M1.n_cols);

    double *a_cu=allocate_data(&resources,a);
    double *b_cu=allocate_data(&resources,b);
    double *S_cu=allocate_data(&resources,S);


    double *LA_cu=allocate_data(&resources,LA);
    double *UA_cu=allocate_data(&resources,UA);
    double *D_cu=allocate_data(&resources,D);
    double *beq_cu=allocate_data(&resources,beq);
    double *N1_cu=allocate_data(&resources,N1);
    double *beqy_cu=allocate_data(&resources,beqy);


    cublasHandle_t handle;
    cublasCreate(&handle);


     //  create sparse matrix

    COOMatrix   C_cu_(C);
    CSRMatrix   C_cu( C_cu_.n_element,C_cu_.n_rows,C_cu_.n_cols);
    C_cu.create_matrix_from_coo(&C_cu_);

    COOMatrix   A_cu_(A);
    CSRMatrix   A_cu( A_cu_.n_element,A_cu_.n_rows,A_cu_.n_cols);
    A_cu.create_matrix_from_coo(&A_cu_);

    COOMatrix   B_cu_(B);
    CSRMatrix   B_cu( B_cu_.n_element,B_cu_.n_rows,B_cu_.n_cols);
    B_cu.create_matrix_from_coo(&B_cu_);
    CSRMatrix   B_cuT( B_cu.n_element,B.n_cols,B.n_rows);
    matrixTrans(&B_cu,&B_cuT);

    COOMatrix   M1_cu_(M1);
    CSRMatrix   M1_cu( M1_cu_.n_element,M1_cu_.n_rows,M1_cu_.n_cols);
    M1_cu.create_matrix_from_coo(&M1_cu_);
    CSRMatrix   M1_cuT( M1_cu.n_element,M1_cu.n_cols,M1_cu.n_rows);
    matrixTrans(&M1_cu,&M1_cuT);

    COOMatrix   AAA_coo(AAA);
    CSRMatrix   AAA_cu( AAA_coo.n_element,AAA_coo.n_rows,AAA_coo.n_cols);
    AAA_cu.create_matrix_from_coo(&AAA_coo);
    BSRMatrix AAA_bsr_cu(&AAA_cu);


    dim3 block_lambda(nlambda/THREAD_+1);
    dim3 block_lambday(nlambday/THREAD_+1);
    dim3 block_xk((n+d)/THREAD_+1);
    dim3 block_N1((nlambday)/THREAD_+1);
    dim3 block_bb((M1.n_cols)/THREAD_+1);

    double *buff_qr;

    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    // step 2: create cusolver handle, qr info and matrix descriptor
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusparse_status = cusparseCreateMatDescr(&descrA);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO); // base-1

    cusolver_status = cusolverSpCreateCsrqrInfo(&info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    size_t size = 0;
    Qr_sp_buff(&cusolverH,&info,&descrA,
               &cusolver_status,&AAA_cu,&size);
    cudaMalloc((void**)&buff_qr,size);



    for (int iter = 1; iter < 10000; iter++){

      copy_double_device_data(old_x,xk,n+d);
      copy_double_device_data(old_y,yk,k);
      copy_double_device_data(old_ahat,ahatk,d+1);


      matrix_cslice(lambda,lambda_1,nlambda,N_block-1);
      matrix_cslice(lambday,lambda_2,nlambday,N_block-1);

      sp_matrix_times_V_ptrl(&B_cu,old_y,temp1);
      sp_matrix_times_V_ptrl(&C_cu,old_ahat,temp2);

      update_r<<<block_lambda,THREAD_>>>(r_cu,temp1,temp2,beq_cu,lambda,rho,nlambda);


      update_xk<<<block_xk,THREAD_>>>(xk,old_x,b_cu,S_cu,r_cu,a_cu,LA_cu,UA_cu,D_cu,n,d,r_end,tau,rho);


      // haty = M1_cu*old_y
      // haty = haty - N1 - lambday/tau2;
      // haty = max(haty,0);
      sp_matrix_times_V_ptrl(&M1_cu,old_y,haty);

      update_hat_y<<<block_N1,THREAD_>>>(haty,N1_cu,lambday,tau2,nlambday);

//       b1 = A*xk  b1_temp =   C*old_ahat
//       b1 = b1  + b1_temp - beq - lambda/rho; b2 =  -haty - N1 - lambday/tau2;


      sp_matrix_times_V_ptrl(&A_cu,xk,b1);
      sp_matrix_times_V_ptrl(&C_cu,old_ahat,b1_temp);
      update_b1<<<block_lambda,THREAD_>>>(b1,b1_temp,beq_cu,lambda,rho,nlambda);
      update_b2<<<block_lambday,THREAD_>>>(b2,haty,N1_cu,lambday,tau2,nlambday);


      sp_matrix_times_V_ptrl(&M1_cuT,b2,bb);    //bb = trans(M1) * b2;  bb_temp = trans(B)*b1;  bb = -(rho * bb_temp +  tau2 *bb ) ; //bbb = [-bb;beqy];
      sp_matrix_times_V_ptrl(&B_cuT,b1,bb_temp);

      update_bb<<<block_bb,THREAD_>>>(bb,bb_temp,rho,tau2,M1.n_cols);

      copy_double_device_data(bbb,bb,M1.n_cols);
      copy_double_device_data_start_length(bbb,beqy_cu,M1.n_cols,beqy.n_elem);

      Conjugate_gradient_sp_bsr(handle,&AAA_bsr_cu,bbb,yy,1e-1/((double)iter),1e3,buff,equation_size);
      //Qr_sp_bsr_buff(&AAA_cu,buff_qr,resources);
//      Qr_sp_csr(&cusolverH,&info,
//                &descrA,
//                &cusolver_status,
//                &AAA_cu,
//                bbb,yy,1e-1,1e3,buff_qr);



      copy_double_device_data(yk,yy,k);

//  A*xk = Axk   B*yk = Byk ;  residual = Axk + Byk  - beq - lambda/rho;
      sp_matrix_times_V_ptrl(&A_cu,xk,Axk);
      sp_matrix_times_V_ptrl(&B_cu,yk,Byk);
      update_residual(residual,Axk,Byk,beq_cu,lambda,rho,nlambda);
      update_ahatk(ahatk,residual,n,d);


      sp_matrix_times_V_ptrl(&C_cu,ahatk,ahatk_temp);
      update_lambda_1(lambda_1,N_block,lambda,rho,
                      res1,beq_cu,Axk,Byk,ahatk_temp,nlambda);


      sp_matrix_times_V_ptrl(&M1_cu,yk,yk_temp);
      update_lambda_2(lambda_2,N_block,lambday,tau2,res2,N1_cu,yk_temp,haty,nlambday);



      double res1_norm = 0.0;
      double res2_norm = 0.0;
      cublasDnrm2(handle,nlambda,res1,1,&res1_norm);
      cublasDnrm2(handle,nlambday,res2,1,&res2_norm);


      double residual_all = sqrt(res1_norm*res1_norm + res2_norm*res2_norm);
       if (residual_all < 1e-5)
           break;

       printf("residual = %.8f iter = %d\n",residual_all,iter);
       //exit(0);

    }


    return  ;

}



