#ifndef CNIC_SPARSEMATRIX_H
#define CNIC_SPARSEMATRIX_H
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_extras/vector.h>
#include <vector>
#include <dpct/dpl_utils.hpp>
#include <dpct/dpl_extras/vector.h>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

#define ARMA_ALLOW_FAKE_GCC
#include<armadillo>
using namespace arma;

template <typename T>
void copy_data_from_vector(T *cu_data, std::vector<T> &data) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

    int size = sizeof(T) * data.size();
    cu_data = (T *)sycl::malloc_device(size, q_ct1);
    q_ct1.memcpy(cu_data, data.data(), size).wait();
}

template <typename T>
void copy_data_to_vector(T *cu_data, int n, std::vector<T> &data){

    int size = sizeof(T) * n;
    dpct::get_default_queue().memcpy(data.data(), cu_data, size).wait();
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
    std::vector<double> data;
    std::vector<int> row;
    std::vector<int> col;

    dpct::device_vector<double> cudata;
    dpct::device_vector<int> curow;
    dpct::device_vector<int> cucol;
};


class cuVec
{
public:
    cuVec(std::vector<double> &);
    void print_vec();

    dpct::device_vector<double> cudata;
    int  n_element;
};


class CSRMatrix
{
public:
    void create_matrix_from_coo(COOMatrix * coo_matrix);
    CSRMatrix(int size,int rows,int cols);
    void print_matrix();
public:
    dpct::device_vector<double> cudata;
    dpct::device_vector<int> csr_row_ptrl;
    dpct::device_vector<int> cucol;
    int  n_element;
    int n_rows;
    int n_cols;
};

class BSRMatrix{

};

void sp_matrix_times_V(CSRMatrix*A,cuVec*b,cuVec *c);

void sp_matrix_times_V_ptrl(CSRMatrix*A, double*b, double *c);

void sp_bsr_matrix_times_V_ptrl(BSRMatrix*A, double*b, double *c);



// void matrixTrans(CSRMatrix*A,CSRMatrix*B);

#endif // CNIC_SPARSEMATRIX_H
