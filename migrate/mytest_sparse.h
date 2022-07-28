#ifndef MYTEST_SPARSE_H
#define MYTEST_SPARSE_H
#include<cuda_runtime_api.h>
#include<stdio.h>
#include<stdlib.h>
#define ARMA_ALLOW_FAKE_GCC
#include<armadillo>
using namespace arma;

template <  typename T>
int check(T *cu, T *data, int size){
     double *temp = (double *) malloc(size * sizeof(T));
     int res = 1;
     cudaMemcpy(temp,cu,size * sizeof(T),cudaMemcpyDeviceToHost);
     for (int i =  0 ;i  < size; ++i){
         if( abs(temp[i]-data[i])>1e-7){
             printf("error on indicator %d %.2f the correct data is %.2f",i,temp[i],data[i]);

             res = 0;
             break;
         }
     }

     free(temp);

     return res;
}

void run_test_for_sparse_mv();

void run_test_for_matrix_transpose(sp_mat &B);

#endif // MYTEST_SPARSE_H
