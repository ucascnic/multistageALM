#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <thrust_tool.h>

struct is_negative
{
  bool operator()(const int x)
  {
    return ( x < 0 );
  }
};


void set_data( int * ind, double *data, int nn, sycl::nd_item<3> item_ct1){
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    if (x >= nn)
        return;

    if(data[x] > 0){
        ind[x] = x;
    }else{
        ind[x] = -1;
    }


}

// find where  A >0 and print the  indicator of the A
dpct::device_vector<int> find_positive(double *data, int n, int *new_end) {

    sycl::range<3> block(1, 1, n / (128) + 1);

    dpct::device_vector<int> dind(n);
    int *dv_ptr = dpct::get_raw_pointer(dind.data());

   dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(block * sycl::range<3>(1, 1, 128),
                                         sycl::range<3>(1, 1, 128)),
                       [=](sycl::nd_item<3> item_ct1) {
                          set_data(dv_ptr, data, n, item_ct1);
                       });
   });
    /*
    DPCT1007:14: Migration of this CUDA API is not supported by the Intel(R)
    DPC++ Compatibility Tool.
    */
    int *new_end_cu =
        thrust::remove_if(thrust::device, dv_ptr, dv_ptr + n, is_negative());

    *new_end = new_end_cu - dv_ptr;
    return dind;

}





