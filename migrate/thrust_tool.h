#ifndef THRUST_TOOL_H
#define THRUST_TOOL_H
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include<cuda_runtime_api.h>
#include <thrust/find.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

thrust::device_vector<int> find_positive(double*data,int n,int *new_end);



#endif // THRUST_TOOL_H
