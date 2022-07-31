#ifndef RESOURCES_H
#define RESOURCES_H
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

template <typename T>
class Resources
{
public:
    int max_resources;
    int cnt;
    T *resources;
public:

    Resources(int n){
        this->max_resources = n;
        this->cnt = 0;
        (this->resources) = (T *)sycl::malloc_device(
            sizeof(T) * this->max_resources, dpct::get_default_queue());
    }
    ~Resources(){
        sycl::free(this->resources, dpct::get_default_queue());
    }

    T* allocate_resource(int required){
        int nn = this->cnt;
        this->cnt += required;
        if (this->cnt  > this->max_resources){
            printf("do not have so much resources");
            exit(0);

        }

        return  &this->resources[nn];
    }

};

#include<stdlib.h>
template <typename T>
class Resources_cpu
{
public:
    int max_resources;
    int cnt;
    T *resources;
public:

    Resources_cpu(int n){
        this->max_resources = n;
        this->cnt = 0;
        this->resources = (T*) malloc(sizeof(T) * this->max_resources);

    }


    T* allocate_resource(int required){
        int nn = this->cnt;
        this->cnt += required;
        if (this->cnt  >= this->max_resources){
            printf("do not have so much resources");
            exit(0);

        }

        return  &this->resources[nn];
    }

};
#endif // RESOURCES_H
