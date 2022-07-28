
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "toolbox_dongdong.h"
#include <stdio.h>
#include<stdlib.h>
#include<handler_cuda_error.h>
void show_res(double *s,int n){
    double *res = (double *)malloc(n*sizeof(double));
    CHECK(cudaMemcpy(res,s,n*sizeof(double),cudaMemcpyDeviceToHost));
    for (int i = 0 ; i< n;++i){
        printf("%.6f\t",res[i]);
    }

    free(res);
}
