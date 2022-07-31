#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Conjugate_gradient.h"
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

#include "matrix_function.h"
#include<stdio.h>
#include<stdlib.h>

void updateXnew(double* xOld, double*dOld,double*xNew, double alpha, int nn,
                sycl::nd_item<3> item_ct1){
    // xNew = xOld  + alpha.*dOld;
    int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (x>=nn)
        return;
    xNew[x] = xOld[x] + alpha * dOld[x];
    xOld[x] = xNew[x];


}

void update_rew(double* rnew,double *b, int nn, sycl::nd_item<3> item_ct1){
    // xNew = xOld  + alpha.*dOld;
    int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (x>=nn)
        return;
    rnew[x] = b[x] - rnew[x];


}

void updatedOld(double *rNew,double *dOld,double alpha, int nn,
                sycl::nd_item<3> item_ct1){
    //dOld = rNew + alpha* dOld;
    int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (x>=nn)
        return;
    dOld[x] = rNew[x] + alpha * dOld[x];

}
void updatedOld(double * rold, double *rNew,double *dOld,double alpha, int nn,
                sycl::nd_item<3> item_ct1){
    //dOld = rNew + alpha* dOld;
    int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (x>=nn)
        return;
    dOld[x] = rNew[x] + alpha * dOld[x];
    rold[x] = rNew[x];

}
// #include"toolbox_dongdong.h"
dpct::constant_memory<double, 1> const_number(sycl::range<1>(3),
                                              {1.0, 0.0, -1.0});

dpct::constant_memory<double, 0> yi(1.0);
dpct::constant_memory<double, 0> fuyi(-1.0);
void Conjugate_gradient(sycl::queue *handle, double *A, double *b, double *x0,
                        double tol, int itMax, double *recources, int n) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    // recources is the memeroy we allocated before for the programm
    // to use whatever it want to use
    int size_x = sizeof(double) * n;
    sycl::range<3> block(1, 1, (n) / 128 + 1);
    double *xOld = &recources[0];
    double *rOld = &recources[n];
    double *dOld = &recources[2*n];
    double *xNew = &recources[3*n];
    double *local_temp = &recources[4*n];
    double *rNew =  &recources[5*n];

    // xOld = x0;
    q_ct1.memcpy(xOld, x0, size_x).wait();

    // rOld = b-A*xOld;

    matrix_timesV(handle,A,xOld,n,n,rOld);

    q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                         sycl::range<3>(1, 1, 128)),
                       [=](sycl::nd_item<3> item_ct1) {
                           update_rew(rOld, b, n, item_ct1);
                       });

    // dOld  ;
    q_ct1.memcpy(dOld, rOld, size_x).wait();

    double bNorm = 0.0;
    double *res_temp_ptr_ct1 = &bNorm;
    if (sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::shared) {
        res_temp_ptr_ct1 =
            sycl::malloc_shared<double>(1, dpct::get_default_queue());
    }
    oneapi::mkl::blas::column_major::nrm2(*handle, n, b, 1, res_temp_ptr_ct1);
    if (sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::shared) {
        handle->wait();
        bNorm = *res_temp_ptr_ct1;
        sycl::free(res_temp_ptr_ct1, dpct::get_default_queue());
    } // bNorm;

    double alpha_ = 0.0;
    double beta_ = 0.0;
    double alpha_temp = 0.0;

    double resNorm = 0.0;

    for (int k =0; k <= 100; ++k){

        //alpha = (dOld'*rOld)/(dOld'*A*dOld);
        double *res_temp_ptr_ct2 = &alpha_temp;
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct2 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, dOld, 1, rOld, 1,
                                             res_temp_ptr_ct2);
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_temp = *res_temp_ptr_ct2;
            sycl::free(res_temp_ptr_ct2, dpct::get_default_queue());
        }

        matrix_timesV(handle,A,dOld,n,n,local_temp);
        double *res_temp_ptr_ct3 = &alpha_;
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct3 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, local_temp, 1, dOld, 1,
                                             res_temp_ptr_ct3);
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_ = *res_temp_ptr_ct3;
            sycl::free(res_temp_ptr_ct3, dpct::get_default_queue());
        }

        alpha_ = alpha_temp/alpha_;

//         printf("alpha = %.6f\n",alpha_);exit(0);


        // xNew = xOld  + alpha.*dOld;
        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {
                               updateXnew(xOld, dOld, xNew, alpha_, n,
                                          item_ct1);
                           });

        //rNew = b-A*xNew;
        matrix_timesV(handle,A,xNew,n,n,rNew);
        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {
                               update_rew(rNew, b, n, item_ct1);
                           });
        double *res_temp_ptr_ct4 = &resNorm;
        if (sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct4 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::nrm2(*handle, n, rNew, 1,
                                              res_temp_ptr_ct4);
        if (sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            resNorm = *res_temp_ptr_ct4;
            sycl::free(res_temp_ptr_ct4, dpct::get_default_queue());
        }
        if (resNorm < tol * bNorm){

            q_ct1.memcpy(x0, xNew, size_x).wait();

            return ;

        }

        double *res_temp_ptr_ct5 = &alpha_temp;
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct5 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, rNew, 1, rNew, 1,
                                             res_temp_ptr_ct5);
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_temp = *res_temp_ptr_ct5;
            sycl::free(res_temp_ptr_ct5, dpct::get_default_queue());
        }
        double *res_temp_ptr_ct6 = &alpha_;
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct6 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, rOld, 1, rOld, 1,
                                             res_temp_ptr_ct6);
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_ = *res_temp_ptr_ct6;
            sycl::free(res_temp_ptr_ct6, dpct::get_default_queue());
        }
        //beta = (rNew'*rNew)/(rOld'*rOld);
        beta_ = alpha_temp/alpha_;


        //dOld = rNew + beta* dOld;
        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {
                               updatedOld(rNew, dOld, beta_, n, item_ct1);
                           });

        //rOld = rNew;
        q_ct1.memcpy(rOld, rNew, size_x);

        //xOld = xNew;
        q_ct1.memcpy(xOld, xNew, size_x).wait();
    }
    q_ct1.memcpy(x0, xNew, size_x).wait();
}
void Conjugate_gradient_debug(sycl::queue *handle, double *A, double *b,
                              double *x0, double tol, int itMax,
                              double *recources, int n) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    // recources is the memeroy we allocated before for the programm
    // to use whatever it want to use
    int size_x = sizeof(double) * n;
    sycl::range<3> block(1, 1, (n) / 128 + 1);
    double *xOld = &recources[0];
    double *rOld = &recources[n];
    double *dOld = &recources[2*n];
    double *xNew = &recources[3*n];
    double *local_temp = &recources[4*n];
    double *rNew =  &recources[5*n];

    // xOld = x0;
    q_ct1.memcpy(xOld, x0, size_x).wait();

    // rOld = b-A*xOld;

    matrix_timesV(handle,A,xOld,n,n,rOld);

    q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                         sycl::range<3>(1, 1, 128)),
                       [=](sycl::nd_item<3> item_ct1) {
                           update_rew(rOld, b, n, item_ct1);
                       });

    // dOld  ;
    q_ct1.memcpy(dOld, rOld, size_x).wait();

    double bNorm = 0.0;
    double *res_temp_ptr_ct7 = &bNorm;
    if (sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::shared) {
        res_temp_ptr_ct7 =
            sycl::malloc_shared<double>(1, dpct::get_default_queue());
    }
    oneapi::mkl::blas::column_major::nrm2(*handle, n, b, 1, res_temp_ptr_ct7);
    if (sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::shared) {
        handle->wait();
        bNorm = *res_temp_ptr_ct7;
        sycl::free(res_temp_ptr_ct7, dpct::get_default_queue());
    } // bNorm;

    double alpha_ = 0.0;
    double beta_ = 0.0;
    double alpha_temp = 0.0;
    double resNorm = 0.0;


    for (int k =0; k <= itMax; ++k){

        //alpha = (dOld'*rOld)/(dOld'*A*dOld);
        double *res_temp_ptr_ct8 = &alpha_temp;
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct8 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, dOld, 1, rOld, 1,
                                             res_temp_ptr_ct8);
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_temp = *res_temp_ptr_ct8;
            sycl::free(res_temp_ptr_ct8, dpct::get_default_queue());
        }

        matrix_timesV(handle,A,dOld,n,n,local_temp);
        double *res_temp_ptr_ct9 = &alpha_;
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct9 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, local_temp, 1, dOld, 1,
                                             res_temp_ptr_ct9);
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_ = *res_temp_ptr_ct9;
            sycl::free(res_temp_ptr_ct9, dpct::get_default_queue());
        }

        alpha_ = alpha_temp/alpha_;

//         printf("alpha = %.6f\n",alpha_);exit(0);


        // xNew = xOld  + alpha.*dOld;
        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {
                               updateXnew(xOld, dOld, xNew, alpha_, n,
                                          item_ct1);
                           });

        //rNew = b-A*xNew;
        //cudaMemcpy(rNew,b,size_x,cudaMemcpyDeviceToDevice);
        //cublasDgemv(handle, CUBLAS_OP_N, n, n, &yi,
        //            A, n, xNew, 1, &alpha_1, rNew,1);
        //show_res(&const_number[0],2);
        matrix_timesV(handle,A,xNew,n,n,rNew);

        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {
                               update_rew(rNew, b, n, item_ct1);
                           });

        double *res_temp_ptr_ct10 = &resNorm;
        if (sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct10 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::nrm2(*handle, n, rNew, 1,
                                              res_temp_ptr_ct10);
        if (sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            resNorm = *res_temp_ptr_ct10;
            sycl::free(res_temp_ptr_ct10, dpct::get_default_queue());
        }
        printf("resnorm = %.2f\n",resNorm);exit(0);
        if (resNorm < tol * bNorm){

            q_ct1.memcpy(x0, xNew, size_x).wait();

            return ;

        }

        double *res_temp_ptr_ct11 = &alpha_temp;
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct11 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, rNew, 1, rNew, 1,
                                             res_temp_ptr_ct11);
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_temp = *res_temp_ptr_ct11;
            sycl::free(res_temp_ptr_ct11, dpct::get_default_queue());
        }
        double *res_temp_ptr_ct12 = &alpha_;
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct12 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, rOld, 1, rOld, 1,
                                             res_temp_ptr_ct12);
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_ = *res_temp_ptr_ct12;
            sycl::free(res_temp_ptr_ct12, dpct::get_default_queue());
        }
        //beta = (rNew'*rNew)/(rOld'*rOld);
        beta_ = alpha_temp/alpha_;


        //dOld = rNew + beta* dOld;
        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {
                               updatedOld(rNew, dOld, beta_, n, item_ct1);
                           });

        //rOld = rNew;
        q_ct1.memcpy(rOld, rNew, size_x);

        //xOld = xNew;
        q_ct1.memcpy(xOld, xNew, size_x).wait();
    }

}

#include"cnic_sparsematrix.h"
void Conjugate_gradient_sp(sycl::queue *handle, CSRMatrix *A, double *b,
                           double *x0, double tol, int itMax, double *recources,
                           int n) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    // recources is the memeroy we allocated before for the programm
    // to use whatever it want to use
    int size_x = sizeof(double) * n;
    sycl::range<3> block(1, 1, (n) / 128 + 1);
    double *xOld = &recources[0];
    double *rOld = &recources[n];
    double *dOld = &recources[2*n];
    double *xNew = &recources[3*n];
    double *local_temp = &recources[4*n];
    double *rNew =  &recources[5*n];

    // xOld = x0;
    q_ct1.memcpy(xOld, x0, size_x).wait();

    //matrix_timesV(handle,A,xOld,n,n,rOld);
    sp_matrix_times_V_ptrl(A,xOld,rOld);

    q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                         sycl::range<3>(1, 1, 128)),
                       [=](sycl::nd_item<3> item_ct1) {
                           update_rew(rOld, b, n, item_ct1);
                       });

    // dOld  ;
    q_ct1.memcpy(dOld, rOld, size_x).wait();

    double bNorm = 0.0;
    double *res_temp_ptr_ct13 = &bNorm;
    if (sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::shared) {
        res_temp_ptr_ct13 =
            sycl::malloc_shared<double>(1, dpct::get_default_queue());
    }
    oneapi::mkl::blas::column_major::nrm2(*handle, n, b, 1, res_temp_ptr_ct13);
    if (sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(&bNorm, handle->get_context()) !=
            sycl::usm::alloc::shared) {
        handle->wait();
        bNorm = *res_temp_ptr_ct13;
        sycl::free(res_temp_ptr_ct13, dpct::get_default_queue());
    } // bNorm;

    double alpha_ = 0.0;
    double beta_ = 0.0;
    double alpha_temp = 0.0;

    double resNorm = 0.0;

    for (int k =0; k <= itMax; ++k){

        //alpha = (dOld'*rOld)/(dOld'*A*dOld);
        double *res_temp_ptr_ct14 = &alpha_temp;
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct14 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, dOld, 1, rOld, 1,
                                             res_temp_ptr_ct14);
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_temp = *res_temp_ptr_ct14;
            sycl::free(res_temp_ptr_ct14, dpct::get_default_queue());
        }

        //matrix_timesV(handle,A,dOld,n,n,local_temp);
        sp_matrix_times_V_ptrl(A,dOld,local_temp);
        double *res_temp_ptr_ct15 = &alpha_;
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct15 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, local_temp, 1, dOld, 1,
                                             res_temp_ptr_ct15);
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_ = *res_temp_ptr_ct15;
            sycl::free(res_temp_ptr_ct15, dpct::get_default_queue());
        }

        alpha_ = alpha_temp/alpha_;

//         printf("alpha = %.6f\n",alpha_);exit(0);


        // xNew = xOld  + alpha.*dOld;
        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {
                               updateXnew(xOld, dOld, xNew, alpha_, n,
                                          item_ct1);
                           });

        //rNew = b-A*xNew;
        //matrix_timesV(handle,A,xNew,n,n,rNew);
        sp_matrix_times_V_ptrl(A,xNew,rNew);

        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {
                               update_rew(rNew, b, n, item_ct1);
                           });
        double *res_temp_ptr_ct16 = &resNorm;
        if (sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct16 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::nrm2(*handle, n, rNew, 1,
                                              res_temp_ptr_ct16);
        if (sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&resNorm, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            resNorm = *res_temp_ptr_ct16;
            sycl::free(res_temp_ptr_ct16, dpct::get_default_queue());
        }
        if (resNorm < tol * bNorm){

            q_ct1.memcpy(x0, xNew, size_x).wait();

            return ;

        }

        double *res_temp_ptr_ct17 = &alpha_temp;
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct17 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, rNew, 1, rNew, 1,
                                             res_temp_ptr_ct17);
        if (sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_temp, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_temp = *res_temp_ptr_ct17;
            sycl::free(res_temp_ptr_ct17, dpct::get_default_queue());
        }
        double *res_temp_ptr_ct18 = &alpha_;
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            res_temp_ptr_ct18 =
                sycl::malloc_shared<double>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::dot(*handle, n, rOld, 1, rOld, 1,
                                             res_temp_ptr_ct18);
        if (sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&alpha_, handle->get_context()) !=
                sycl::usm::alloc::shared) {
            handle->wait();
            alpha_ = *res_temp_ptr_ct18;
            sycl::free(res_temp_ptr_ct18, dpct::get_default_queue());
        }
        //beta = (rNew'*rNew)/(rOld'*rOld);
        beta_ = alpha_temp/alpha_;


        //dOld = rNew + beta* dOld;
        q_ct1.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                             sycl::range<3>(1, 1, 128)),
                           [=](sycl::nd_item<3> item_ct1) {
                               updatedOld(rNew, dOld, beta_, n, item_ct1);
                           });

        //rOld = rNew;
        q_ct1.memcpy(rOld, rNew, size_x);

        //xOld = xNew;
        q_ct1.memcpy(xOld, xNew, size_x).wait();
    }
    q_ct1.memcpy(x0, xNew, size_x).wait();
}


//  #include<assert.h>
// #include"resources.h"
/*
DPCT1007:5: Migration of cusolverSpXcsrqrAnalysisBatched is not supported by
    the Intel(R) DPC++ Compatibility Tool.

void Qr_sp_buff(cusolverSpHandle_t *cusolverH, csrqrInfo_t *info,
                oneapi::mkl::index_base *descrA, int *cusolver_status,
                CSRMatrix *A, size_t *size_qr) {

    int m = A->n_rows;
    int nnzA = A->n_element;
    int *d_csrRowPtrA = thrust::raw_pointer_cast(A->csr_row_ptrl.data());
    int *d_csrColIndA = thrust::raw_pointer_cast(A->cucol.data());
    double *d_csrValA = thrust::raw_pointer_cast(A->cudata.data());

    *cusolver_status = cusolverSpXcsrqrAnalysisBatched(
        *cusolverH, m, m, nnzA, *descrA, d_csrRowPtrA, d_csrColIndA, *info);
    size_t size_internal;

    int batchSize = 1;
    assert(*cusolver_status == 0);

    *cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
        *cusolverH, m, m, nnzA, *descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        batchSize, *info, &size_internal, size_qr);
    assert(*cusolver_status == 0);
    printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal);
    printf("numerical factorization needs working space %lld bytes\n", (long long)*size_qr);

}
void Qr_sp_csr(cusolverSpHandle_t *cusolverH, csrqrInfo_t *info,
               oneapi::mkl::index_base *descrA, int *cusolver_status,
               CSRMatrix *A, double *b, double *x0, double tol, int itMax,
               double *buffer_qr) {

    int batchSize = 1;
    int m = A->n_rows;
    int nnzA = A->n_element;
    int *d_csrRowPtrA = thrust::raw_pointer_cast(A->csr_row_ptrl.data());
    int *d_csrColIndA = thrust::raw_pointer_cast(A->cucol.data());
    double *d_csrValA = thrust::raw_pointer_cast(A->cudata.data());

    double *d_b = b;
    double *d_x = x0;

    *cusolver_status = cusolverSpDcsrqrsvBatched(
        *cusolverH, m, m, nnzA, *descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_b, d_x, batchSize, *info, buffer_qr);

    std::cout << *cusolver_status << std::endl;

    assert(*cusolver_status == 0);
}
*/