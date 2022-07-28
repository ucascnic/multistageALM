#ifndef THRUST_TOOL_H
#define THRUST_TOOL_H
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

dpct::device_vector<int> find_positive(double *data, int n, int *new_end);

#endif // THRUST_TOOL_H
