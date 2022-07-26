#include<thrust_tool.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include<cuda_runtime_api.h>
#include <thrust/find.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
struct is_negative
{
  __device__ bool operator()(const int x)
  {
    return ( x < 0 );
  }
};


__global__ void set_data( int * ind, double *data, int nn){
    int x = blockIdx.x * blockDim.x  + threadIdx.x;
    if (x >= nn)
        return;

    if(data[x] > 0){
        ind[x] = x;
    }else{
        ind[x] = -1;
    }


}

// find where  A >0 and print the  indicator of the A
thrust::device_vector<int> find_positive(double*data,int n,int *new_end){

    dim3 block(n/(128 )+1);

    thrust::device_vector<int> dind(n);
    int * dv_ptr = thrust::raw_pointer_cast(dind.data());

    set_data<<<block,128>>>(dv_ptr,data,n);
    int  *new_end_cu =  thrust::remove_if(thrust::device, dv_ptr, dv_ptr+n, is_negative());

    *new_end = new_end_cu - dv_ptr;
    return dind;

}





