#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "resources.h"
#include <stdio.h>
#include<stdlib.h>



template <typename T>
T* Resources<T>::allocate_resource(int required ){

    int nn = this->cnt;
    this->cnt += required;
    if (this->cnt  >= this->max_resources){
        printf("do not have so much resources");
        exit(0);

    }

    return  &this->resources[nn];

}
