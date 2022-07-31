#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include "admmsolver.h"
#include <stdio.h>
#include "resources.h"
#include <vector>
#include "cnic_sparsematrix.h"
#include "matrix_function.h"
#include "Conjugate_gradient.h"
#include <cmath>

#define THREAD_ 128

#define copy_double_device_data(a, b, n)                                       \
    q_ct1.memcpy(a, b, (n) * sizeof(double)).wait();
#define copy_double_device_data_start_length(a, b, start, length)              \
    q_ct1.memcpy(a + start, b, (length) * sizeof(double)).wait();

#define matrix_cslice(a, matrix, nrows, col)                                   \
    q_ct1.memcpy(a, matrix + (nrows) * (col), (nrows) * sizeof(double)).wait();

//r =  yk + ahatk - beq - lambda/rho;
void update_r(double *r,
                         double *yk,
                         double *ahatk,
                         double *beq,
                         double *lambda,
                         double rho,int nlambda, sycl::nd_item<3> item_ct1){
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    if (x>=nlambda)
        return;
    r[x] =  yk[x] + ahatk[x] - beq[x] - lambda[x]/rho;

}
void update_xk(double *xk,
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
                          double rho,
                          sycl::nd_item<3> item_ct1){
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    if (x>=n+d)
        return;

    if(x<n-1)
    {
        xk[x] = (tau*old_x[x] - b[x] - rho *  S[x]   * r[x]) /(2.0*a[x] + rho *   S[x] *S[x] + tau);
        xk[x] = sycl::min(sycl::max(xk[x], LA[x]), UA[x]);

    }
    else
    {
        if (x>n-1){

            xk[x] = (tau*old_x[x] - b[x] - rho * DD[x-n] * r[x])
                    /(2.0*a[x] + rho  * DD[x-n] * DD[x-n]  + tau);
            xk[x] = sycl::max(xk[x], 0.0);

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
            xk[n - 1] = sycl::min(sycl::max(xk[n - 1], LA[n - 1]), UA[n - 1]);
        }

    }



}


// haty = haty - N1 - lambday/tau2;
// haty = max(haty,0);

void update_hat_y(double *haty,double *N1,double *lambday,double tau2,int N1_size,
                  sycl::nd_item<3> item_ct1){
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    if (x>=N1_size)
        return;
     haty[x] = haty[x] - N1[x] - lambday[x]/tau2;
     haty[x] = sycl::max(haty[x], 0.0);
}

//b2 =  -haty - N1 - lambday/tau2;
void update_b2(double *b2,double *haty,double *N1,double *lambday,double tau2, int nlambday,
               sycl::nd_item<3> item_ct1){
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    if (x>=nlambday)
        return;
    b2[x] =  -haty[x] - N1[x] - lambday[x]/tau2;

}

void update_b1(double *b1,double*b1_temp,double*beq,double*lambda,double rho,int n,
               sycl::nd_item<3> item_ct1){
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    if (x>=n)
        return;
    b1[x] += b1_temp[x] - beq[x] - lambda[x]/rho;
}


void update_bb(double *bb,
               double *bb_temp,
               double rho,
               double tau2,
               int nn,
               sycl::nd_item<3> item_ct1){
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    if (x>=nn)
        return;

    bb[x] = -(rho * bb_temp[x] +  tau2 *bb[x] );

}
inline
double* allocate_data(Resources<double>*resources,vec &b){
    double *b_cu = resources->allocate_resource(b.n_elem);
    dpct::get_default_queue()
        .memcpy(b_cu, b.memptr(), sizeof(double) * b.n_elem)
        .wait();

    return b_cu;
}


void
update_residual_kernel(double*r,double*old_x,double*old_y,double*beq,double*lambda,
                       double rho,int nlambda, sycl::nd_item<3> item_ct1){

    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    if (x>=nlambda)
        return;
     r[x] = old_x[x] + old_y[x]  - beq[x] - lambda[x]/rho;

}
inline void
update_residual(double*r,double*old_x,double*old_y,double*beq,double*lambda,double rho,int nlambda){
    sycl::range<3> block(1, 1, nlambda / THREAD_ + 1);
   dpct::get_default_queue().parallel_for(
       sycl::nd_range<3>(block * sycl::range<3>(1, 1, THREAD_),
                         sycl::range<3>(1, 1, THREAD_)),
       [=](sycl::nd_item<3> item_ct1) {
          update_residual_kernel(r, old_x, old_y, beq, lambda, rho, nlambda,
                                 item_ct1);
       });
}

void
update_ahatk_kernel(double *ahatk,double *residual,int n,int d,
                    sycl::nd_item<3> item_ct1){
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    if (x>=(d+1))
        return;
     ahatk[x] =   -residual[n+x];
     ahatk[x] = sycl::max(ahatk[x], 0.0);
}

inline void
update_ahatk(double *ahatk,double *residual,int n,int d){
    sycl::range<3> block(1, 1, (d + 1) / THREAD_ + 1);
   dpct::get_default_queue().parallel_for(
       sycl::nd_range<3>(block * sycl::range<3>(1, 1, THREAD_),
                         sycl::range<3>(1, 1, THREAD_)),
       [=](sycl::nd_item<3> item_ct1) {
          update_ahatk_kernel(ahatk, residual, n, d, item_ct1);
       });
}


void update_lambda_2_kernel(double *lambda_2,int N_block,
                double *lambday,
                double tau2,
                double *res2,
                double *N1,
                double *yk_temp,
                double *haty,
                int nlambda, sycl::nd_item<3> item_ct1){
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
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

    sycl::range<3> block(1, 1, (nlambda) / THREAD_ + 1);
   dpct::get_default_queue().parallel_for(
       sycl::nd_range<3>(block * sycl::range<3>(1, 1, THREAD_),
                         sycl::range<3>(1, 1, THREAD_)),
       [=](sycl::nd_item<3> item_ct1) {
          update_lambda_2_kernel(lambda_2, N_block, lambday, tau2, res2, N1,
                                 yk_temp, haty, nlambda, item_ct1);
       });
}

void
update_lambda_1_kernel(double*lambda_1,
                int N_block,
                double*lambda,
                double rho,
                double*res1,
                double*beq,
                double*old_x,
                double*old_y,
                double*ahatk_temp,int nlambda, sycl::nd_item<3> item_ct1){
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
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

    sycl::range<3> block(1, 1, (nlambda) / THREAD_ + 1);
   dpct::get_default_queue().parallel_for(
       sycl::nd_range<3>(block * sycl::range<3>(1, 1, THREAD_),
                         sycl::range<3>(1, 1, THREAD_)),
       [=](sycl::nd_item<3> item_ct1) {
          update_lambda_1_kernel(lambda_1, N_block, lambda, rho, res1, beq,
                                 old_x, old_y, ahatk_temp, nlambda, item_ct1);
       });
}

void admm_solver(sp_mat &AAA, sp_mat &A, sp_mat &B, sp_mat &C, sp_mat &M1,
                 sp_mat &Aeqy, vec &beqy, vec &D, vec &beq, vec &N1, vec &a,
                 vec &b, vec &S, vec &LA, vec &UA, int maxiter, int n, int d) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

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

    sycl::queue *handle;
    handle = &q_ct1;

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

    sp_mat BT = trans(B);
    COOMatrix   B_cuT_(BT);
    CSRMatrix   B_cuT( B_cuT_.n_element,B_cuT_.n_rows,B_cuT_.n_cols);
    B_cuT.create_matrix_from_coo(&B_cuT_);

    COOMatrix   M1_cu_(M1);
    CSRMatrix   M1_cu( M1_cu_.n_element,M1_cu_.n_rows,M1_cu_.n_cols);
    M1_cu.create_matrix_from_coo(&M1_cu_);

    sp_mat M1_T = trans(M1);
    COOMatrix   M1_cuT_(M1_T);
    CSRMatrix   M1_cuT( M1_cuT_.n_element,M1_cuT_.n_rows,M1_cuT_.n_cols);
    M1_cuT.create_matrix_from_coo(&M1_cuT_);


    COOMatrix   AAA_coo(AAA);
    CSRMatrix   AAA_cu( AAA_coo.n_element,AAA_coo.n_rows,AAA_coo.n_cols);
    AAA_cu.create_matrix_from_coo(&AAA_coo);

    sycl::range<3> block_lambda(1, 1, nlambda / THREAD_ + 1);
    sycl::range<3> block_lambday(1, 1, nlambday / THREAD_ + 1);
    sycl::range<3> block_xk(1, 1, (n + d) / THREAD_ + 1);
    sycl::range<3> block_N1(1, 1, (nlambday) / THREAD_ + 1);
    sycl::range<3> block_bb(1, 1, (M1.n_cols) / THREAD_ + 1);

    for (int iter = 1; iter < 10000; iter++){

      copy_double_device_data(old_x,xk,n+d);
      copy_double_device_data(old_y,yk,k);
      copy_double_device_data(old_ahat,ahatk,d+1);

      matrix_cslice(lambda,lambda_1,nlambda,N_block-1);
      matrix_cslice(lambday,lambda_2,nlambday,N_block-1);

      sp_matrix_times_V_ptrl(&B_cu,old_y,temp1);
      sp_matrix_times_V_ptrl(&C_cu,old_ahat,temp2);

      q_ct1.parallel_for(
          sycl::nd_range<3>(block_lambda * sycl::range<3>(1, 1, THREAD_),
                            sycl::range<3>(1, 1, THREAD_)),
          [=](sycl::nd_item<3> item_ct1) {
             update_r(r_cu, temp1, temp2, beq_cu, lambda, rho, nlambda,
                      item_ct1);
          });

      q_ct1.parallel_for(
          sycl::nd_range<3>(block_xk * sycl::range<3>(1, 1, THREAD_),
                            sycl::range<3>(1, 1, THREAD_)),
          [=](sycl::nd_item<3> item_ct1) {
             update_xk(xk, old_x, b_cu, S_cu, r_cu, a_cu, LA_cu, UA_cu, D_cu, n,
                       d, r_end, tau, rho, item_ct1);
          });

      // haty = M1_cu*old_y
      // haty = haty - N1 - lambday/tau2;
      // haty = max(haty,0);
      sp_matrix_times_V_ptrl(&M1_cu,old_y,haty);

      q_ct1.parallel_for(
          sycl::nd_range<3>(block_N1 * sycl::range<3>(1, 1, THREAD_),
                            sycl::range<3>(1, 1, THREAD_)),
          [=](sycl::nd_item<3> item_ct1) {
             update_hat_y(haty, N1_cu, lambday, tau2, nlambday, item_ct1);
          });

//       b1 = A*xk  b1_temp =   C*old_ahat
//       b1 = b1  + b1_temp - beq - lambda/rho; b2 =  -haty - N1 - lambday/tau2;


      sp_matrix_times_V_ptrl(&A_cu,xk,b1);
      sp_matrix_times_V_ptrl(&C_cu,old_ahat,b1_temp);
      q_ct1.parallel_for(
          sycl::nd_range<3>(block_lambda * sycl::range<3>(1, 1, THREAD_),
                            sycl::range<3>(1, 1, THREAD_)),
          [=](sycl::nd_item<3> item_ct1) {
             update_b1(b1, b1_temp, beq_cu, lambda, rho, nlambda, item_ct1);
          });
      q_ct1.parallel_for(
          sycl::nd_range<3>(block_lambday * sycl::range<3>(1, 1, THREAD_),
                            sycl::range<3>(1, 1, THREAD_)),
          [=](sycl::nd_item<3> item_ct1) {
             update_b2(b2, haty, N1_cu, lambday, tau2, nlambday, item_ct1);
          });

      sp_matrix_times_V_ptrl(&M1_cuT,b2,bb);    //bb = trans(M1) * b2;  bb_temp = trans(B)*b1;  bb = -(rho * bb_temp +  tau2 *bb ) ; //bbb = [-bb;beqy];
      sp_matrix_times_V_ptrl(&B_cuT,b1,bb_temp);

      q_ct1.submit([&](sycl::handler &cgh) {
         auto M1_n_cols_ct4 = M1.n_cols;

         cgh.parallel_for(
             sycl::nd_range<3>(block_bb * sycl::range<3>(1, 1, THREAD_),
                               sycl::range<3>(1, 1, THREAD_)),
             [=](sycl::nd_item<3> item_ct1) {
                update_bb(bb, bb_temp, rho, tau2, M1_n_cols_ct4, item_ct1);
             });
      });

      copy_double_device_data(bbb,bb,M1.n_cols);
      copy_double_device_data_start_length(bbb,beqy_cu,M1.n_cols,beqy.n_elem);

      Conjugate_gradient_sp(handle,&AAA_cu,bbb,yy,1e-1/((double)iter),1e3,buff,equation_size);




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
      double *res_temp_ptr_ct1 = &res1_norm;
      if (sycl::get_pointer_type(&res1_norm, handle->get_context()) !=
              sycl::usm::alloc::device &&
          sycl::get_pointer_type(&res1_norm, handle->get_context()) !=
              sycl::usm::alloc::shared) {
         res_temp_ptr_ct1 =
             sycl::malloc_shared<double>(1, dpct::get_default_queue());
      }
      oneapi::mkl::blas::column_major::nrm2(*handle, nlambda, res1, 1,
                                            res_temp_ptr_ct1);
      if (sycl::get_pointer_type(&res1_norm, handle->get_context()) !=
              sycl::usm::alloc::device &&
          sycl::get_pointer_type(&res1_norm, handle->get_context()) !=
              sycl::usm::alloc::shared) {
         handle->wait();
         res1_norm = *res_temp_ptr_ct1;
         sycl::free(res_temp_ptr_ct1, dpct::get_default_queue());
      }
      double *res_temp_ptr_ct2 = &res2_norm;
      if (sycl::get_pointer_type(&res2_norm, handle->get_context()) !=
              sycl::usm::alloc::device &&
          sycl::get_pointer_type(&res2_norm, handle->get_context()) !=
              sycl::usm::alloc::shared) {
         res_temp_ptr_ct2 =
             sycl::malloc_shared<double>(1, dpct::get_default_queue());
      }
      oneapi::mkl::blas::column_major::nrm2(*handle, nlambday, res2, 1,
                                            res_temp_ptr_ct2);
      if (sycl::get_pointer_type(&res2_norm, handle->get_context()) !=
              sycl::usm::alloc::device &&
          sycl::get_pointer_type(&res2_norm, handle->get_context()) !=
              sycl::usm::alloc::shared) {
         handle->wait();
         res2_norm = *res_temp_ptr_ct2;
         sycl::free(res_temp_ptr_ct2, dpct::get_default_queue());
      }

      double residual_all = sqrt(res1_norm*res1_norm + res2_norm*res2_norm);
       if (residual_all < 1e-5)
           break;

       printf("residual = %.8f iter = %d\n",residual_all,iter);
       //exit(0);

    }


    return  ;

}



