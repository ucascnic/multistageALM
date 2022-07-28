cmake_minimum_required(VERSION 3.9)

# project name
project (multistageALM)

enable_testing() 

#add_definitions(-std=c++14)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++14  -mavx -mfma")

include_directories(.)


set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES(./${CUDA_INCLUDE_DIRS})
include_directories(.)




LINK_DIRECTORIES("/usr/local/cuda/lib64")
INCLUDE_DIRECTORIES ("/usr/local/cuda/include/")

INCLUDE_DIRECTORIES ("/opt/intel/oneapi/compiler/2021.4.0/linux/include")
INCLUDE_DIRECTORIES ("/opt/intel/oneapi/compiler/2021.4.0/linux/include/sycl")


INCLUDE_DIRECTORIES (Common)

add_subdirectory(Common)

# please change the path before running cmake .
INCLUDE_DIRECTORIES("/home/liuyun/multistageALM/libs/include")
LINK_DIRECTORIES("/home/liuyun/multistageALM/libs/lib")
LINK_DIRECTORIES("/home/liuyun/multistageALM/libs/lib64")



#  cuda_add_executable(ADMM_Solver   main_admm.cpp admmsolver.cu  cnic_sparsematrix.cu matrix_function.cu
#        mytest_sparse.cu Conjugate_gradient.cu
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



cuda_add_executable(admm_test   admm_test.cpp cnic_sparsematrix.cu mytest_sparse.cu)
        target_link_libraries(admm_test
                Common
                -lcublas
                -lcudart
                -larmadillo
                -lcusparse
                -lcudadevrt
                -lcusolver
                )
    
    


