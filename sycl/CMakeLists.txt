cmake_minimum_required(VERSION 3.9)

set(CMAKE_C_COMPILER "icx")
set(CMAKE_CXX_COMPILER "dpcpp")


# project name
project (multistageALM_SYCL)

set(CMAKE_BUILD_TYPE "Debug")
 
include_directories(.)

INCLUDE_DIRECTORIES (Common)
add_subdirectory(Common)

# please change the path before running cmake .
INCLUDE_DIRECTORIES("/home/liuyun/multistageALM/armadillo-11.2.2/include")
#INCLUDE_DIRECTORIES("/opt/intel/oneapi/compiler/latest/linux/include"
#    "/opt/intel/oneapi/mkl/latest/include"
#    "/opt/intel/oneapi/compiler/latest/linux/include/sycl"
#    "/opt/intel/oneapi/dev-utilities/latest/include"
#    "/opt/intel/oneapi/dpcpp-ct/latest/include"
#    "/opt/intel/oneapi/dpl/latest/linux/include"
#    "/opt/intel/oneapi/tbb/latest/include"
#    )
LINK_DIRECTORIES("/home/liuyun/multistageALM/libs/lib")
LINK_DIRECTORIES("/home/liuyun/multistageALM/libs/lib64")
LINK_DIRECTORIES("${MKLROOT}/lib/intel64")
# INCLUDE_DIRECTORIES ("/home/chenyidong/hdf5_installed/include")

#  cuda_add_executable(ADMM_Solver   main_admm.cpp admmsolver.cu  cnic_sparsematrix.cu matrix_function.cu
#        mytest_sparse.cu Conjugate_gradient.cu  cnicsparsematrix.cpp
#        )
#    target_compile_options(ADMM_Solver PUBLIC ${LEMON_COMPILE_OPTIONS})
#    target_include_directories(ADMM_Solver PUBLIC ${LEMON_INCLUDE_DIRECTORY})
#    target_link_libraries(ADMM_Solver
#            ${LEMON_LIBRARY}
#            Common
#            -lcublas
#            -lcudart
#            -larmadillo
#            -lcusparse
#            -lcudadevrt
#            -lcusolver
#            )

  add_executable(ADMM_Solver_sycl  main_admm.cpp admmsolver.dp.cpp 
        Conjugate_gradient.dp.cpp 
        cnic_sparsematrix.dp.cpp
        cnicsparsematrix.cpp
        matrix_function.dp.cpp
        )
  target_compile_options(ADMM_Solver_sycl PUBLIC -fsycl -DMKL_ILP64  -qmkl=parallel)
  target_link_libraries(ADMM_Solver_sycl Common -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl)
