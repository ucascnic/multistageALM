cmake_minimum_required(VERSION 3.9)

# project name
project (multistageALM)

 

#add_definitions(-std=c++14)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++14  -mavx -mfma")

include_directories(.)


set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES(./${CUDA_INCLUDE_DIRS})
include_directories(.)




LINK_DIRECTORIES("/usr/local/cuda-11.1/lib64")
INCLUDE_DIRECTORIES ("/usr/local/cuda-11.1/include/")
 
INCLUDE_DIRECTORIES (Common)

add_subdirectory(Common)

# please change the path before running cmake .
INCLUDE_DIRECTORIES("/home/chenyidong/armadillo_lib/include")
LINK_DIRECTORIES("/home/chenyidong/armadillo_lib/lib")
LINK_DIRECTORIES("/home/chenyidong/hdf5_installed/lib")
INCLUDE_DIRECTORIES ("/home/chenyidong/hdf5_installed/include")



  cuda_add_executable(ADMM_Solver   main_admm.cpp admmsolver.cu  cnic_sparsematrix.cu matrix_function.cu
        mytest_sparse.cu Conjugate_gradient.cu  cnicsparsematrix.cpp
        )
    target_compile_options(ADMM_Solver PUBLIC ${LEMON_COMPILE_OPTIONS})
    target_include_directories(ADMM_Solver PUBLIC ${LEMON_INCLUDE_DIRECTORY})
    target_link_libraries(ADMM_Solver
            ${LEMON_LIBRARY}
            Common
            -lcublas
            -lcudart
            -larmadillo
            -lcusparse
            -lcudadevrt
            -lcusolver
            )





