#ifndef CNIC_SPARSEMATRIX_H
#define CNIC_SPARSEMATRIX_H
#include<vector>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include<cuda_runtime_api.h>
#include <thrust/find.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include<cuda_runtime_api.h>
#include<cusparse_v2.h>
#define ARMA_ALLOW_FAKE_GCC
#include<armadillo>
using namespace arma;



template <typename T>
void copy_data_from_vector(T *cu_data,std::vector<T> &data){

    int size = sizeof(T) * data.size();
    cudaMalloc((void **)&cu_data,size);
    cudaMemcpy(cu_data,data.data(),size,cudaMemcpyHostToDevice);
}

template <typename T>
void copy_data_to_vector(T *cu_data, int n, std::vector<T> &data){

    int size = sizeof(T) * n;
    cudaMemcpy(data.data(),cu_data,size,cudaMemcpyDeviceToHost);
}

class COOMatrix
{
public:
    COOMatrix(sp_mat &);
    COOMatrix(int m, int n,int n_ele);
    void to_GPU();
    void print_matrix();
    int n_element;
    int n_rows;
    int n_cols;


public:

    thrust::host_vector<double> data;
    thrust::host_vector<int> row;
    thrust::host_vector<int> col;

    thrust::device_vector<double> cudata;
    thrust::device_vector<int> curow;
    thrust::device_vector<int> cucol;

};


class cuVec
{
public:

    cuVec(thrust::host_vector<double>&);
    void print_vec();

    thrust::device_vector<double> cudata;
    int  n_element;



};


class CSRMatrix
{
public:
    void create_matrix_from_coo(COOMatrix * coo_matrix);
    CSRMatrix(int size,int rows,int cols);
    void print_matrix();
public:
    thrust::device_vector<double> cudata;
    thrust::device_vector<int> csr_row_ptrl;
    thrust::device_vector<int> cucol;
    int  n_element;
    int n_rows;
    int n_cols;

};

class BSRMatrix{

};

void sp_matrix_times_V(CSRMatrix*A,cuVec*b,cuVec *c);

void sp_matrix_times_V_ptrl(CSRMatrix*A, double*b, double *c);

void sp_bsr_matrix_times_V_ptrl(BSRMatrix*A, double*b, double *c);



void matrixTrans(CSRMatrix*A,CSRMatrix*B);

#endif // CNIC_SPARSEMATRIX_H
