#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cnic_sparsematrix.h"
#include <iostream>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

#include "matrix_function.h"
#include<assert.h>

#define ARMA_ALLOW_FAKE_GCC
// #include<armadillo>

static sycl::queue *cusparse_handle = 0;
int Status_t = (cusparse_handle = &dpct::get_default_queue(), 0);
static void init_cusparse() {
  if (cusparse_handle == 0) {
    if (Status_t != 0) {
      printf("CUSPARSE Library initialization failed");
    }
  }
}
COOMatrix::COOMatrix(int m, int n,int n_ele){


    this->n_cols = n;
    this->n_rows = m;
    this->n_element = n_ele;
    // cpudata

    this->data = std::vector<double>(n_ele,0.0);
    this->col = std::vector<int>(n_ele,0);
    this->row = std::vector<int>(n_ele,0);
    //gpu data
    this->cudata = this->data;
    this->curow = this->row;
    this->cucol = this->col;


}
COOMatrix::COOMatrix(sp_mat & inputmat){

    //uvec res =  find(  inputmat );

    sp_mat::const_iterator it     = inputmat.begin();
    sp_mat::const_iterator it_end = inputmat.end();
    int n = inputmat.n_nonzero;
    this->n_element = n;
    this->data = std::vector<double>(n,0.0);
    this->col = std::vector<int>(n,0);
    this->row = std::vector<int>(n,0);
    this->n_rows = inputmat.n_rows;
    this->n_cols = inputmat.n_cols;


    int i = 0;
    for( ; it != it_end; ++it)
      {
       this->data[i] =  (*it) ;
       this->row[i] =  it.row();
        this->col[i] =  it.col();
        i++;
      }


    this->cudata = this->data;
    this->curow = this->row;
    this->cucol = this->col;

}




void CSRMatrix::print_matrix(){

//    for (int i = 0 ; i < this->n_element; ++i){
//        printf("%.4f\n",this->cucol[i]);
//    }
    show_res_T<double>((double *)dpct::get_raw_pointer(this->cudata.data()),
                       this->n_element);
}
/*
void sortcoo_and_to_csr(int rows, int cols, int N, double *val, int *row_ptrl,
                        int *col_ptrl, int *crs_row_ptrl) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

    size_t workspace_size = 0;
    Status_t = cusparseXcoosort_bufferSizeExt(
        cusparse_handle, rows, cols, N, row_ptrl, col_ptrl, &workspace_size);
    assert(0 == Status_t);
    double * buffer_for_coo_sort;
    CHECK((buffer_for_coo_sort = (double *)sycl::malloc_device(
               sizeof(char) * workspace_size, q_ct1),
           0));
    int * indptrl_cu;

    CHECK((indptrl_cu = sycl::malloc_device<int>(N, q_ct1), 0));
    int * indptrl_cu_mid = (int*) malloc( sizeof(int) *N);
    for ( int i =  0 ;i  < N ; ++i) indptrl_cu_mid[i] = i;

    q_ct1.memcpy(indptrl_cu, indptrl_cu_mid, sizeof(int) * N).wait();
    Status_t = cusparseXcoosortByRow(cusparse_handle, rows, cols, N, row_ptrl,
                                     col_ptrl, indptrl_cu, buffer_for_coo_sort);

   Status_t = cusparseDgthr(cusparse_handle, N, val, buffer_for_coo_sort,
                            indptrl_cu, oneapi::mkl::index_base::zero);

   q_ct1.memcpy(val, buffer_for_coo_sort, N * sizeof(double)).wait();
   assert(0 == Status_t);

    Status_t = cusparseXcoo2csr(cusparse_handle, row_ptrl, N, rows,
                                crs_row_ptrl, oneapi::mkl::index_base::zero);
    assert(0 == Status_t);

    sycl::free(buffer_for_coo_sort, q_ct1);
    sycl::free(indptrl_cu, q_ct1);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
*/
#include"cnicsparsematrix.h"
void CSRMatrix::create_matrix_from_coo(COOMatrix * coo_matrix){

    int n = coo_matrix->n_element;
    this->cudata = coo_matrix->cudata;
    this->cucol = coo_matrix->cucol;
    this->csr_row_ptrl = dpct::device_vector<int>(coo_matrix->n_rows + 1);

    COORowPtrl coorowptrl;
    coorowptrl.ptr.resize(n);
    for (int i =0;i<n;++i){
        coorowptrl.ptr[i] = coo_matrix->row[i];
    }


    CSRIntMatrix  csr;
    csr.col.resize(this->cucol.size());
    csr.ptr.resize(coo_matrix->n_rows+1);
    for (int i = 0 ;i < this->cucol.size();++i){
        csr.col[i]=coo_matrix->col[i];
    }


    ptrdiff_t nnz = n;
    csr.val.resize(nnz);
    for (int i = 0 ; i < nnz;++i){
        csr.val[i] = i;
    }


    qsortCOO2CSR<int>(coorowptrl.ptr.data(), csr.col.data(), csr.val.data(), 0, nnz - 1);

    compressIndices(coorowptrl.ptr.data(), csr.ptr.data(), coo_matrix->n_rows, nnz);

    std::vector<int> temp1(csr.ptr.size());

    for (int i = 0 ; i< csr.ptr.size();++i){
        temp1[i] = csr.ptr[i];

    }

    this->csr_row_ptrl = temp1;

    std::vector<int> temp2(csr.col.size());
    for (int i = 0 ; i< csr.col.size();++i){
        temp2[i] = csr.col[i];
    }
    this->cucol = temp2;

    std::vector<double> temp3 = coo_matrix->data;
    std::vector<double> temp4 = coo_matrix->data;
    for (int i = 0 ; i< temp3.size();++i){
        temp4[i] = temp3[csr.val[i]];
    }
    this->cudata = temp4;
    show_res_T<double>((double *)dpct::get_raw_pointer(this->cudata.data()),
                       10);

    return   ;

}
CSRMatrix::CSRMatrix(int size,int rows,int cols){
    this->n_element = size;
    this->n_rows = rows;
    this->n_cols = cols;

}

cuVec::cuVec(std::vector<double> &data) {
    this->n_element = data.size();
    this->cudata = data;
}

#define WARP_ 32
inline double __shfl_down_(double var, unsigned int srcLane,
                           sycl::nd_item<3> item_ct1, int width=WARP_) {
  sycl::int2 a = *reinterpret_cast<sycl::int2 *>(&var);
  a.x() = dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.x(), srcLane,
                                     width);
  a.y() = dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.y(), srcLane,
                                     width);
  return *reinterpret_cast<double*>(&a);
}
void kernal_mat_u_32_wrap(double *kernel_cudata,
                                     int *csr_row_ptrl,int *cucol,
                                     double *u_cudata,
                                     double *output_cudata,int n_rows,
                                     sycl::nd_item<3> item_ct1);
//void sp_matrix_times_V(CSRMatrix*A,cuVec*b,cuVec *c){
//    //std::cout << A->n_rows << std::endl;
//    dim3 block(A->n_rows);
//    c->cudata.resize(A->n_rows);
//    kernal_mat_u_32_wrap<<<block,WARP_>>>(thrust::raw_pointer_cast(A->cudata.data()),
//                                          thrust::raw_pointer_cast(A->csr_row_ptrl.data()),
//                                          thrust::raw_pointer_cast(A->cucol.data()),
//                                          thrust::raw_pointer_cast(b->cudata.data()),
//                                          thrust::raw_pointer_cast(c->cudata.data()),
//                                          A->n_rows);
//    c->n_element = A->n_rows;

//}

void sp_matrix_times_V_ptrl(CSRMatrix*A,double*b,double *c){

    sycl::range<3> block(1, 1, A->n_rows);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto thrust_raw_pointer_cast_A_cudata_data_ct0 =
            dpct::get_raw_pointer(A->cudata.data());
        auto thrust_raw_pointer_cast_A_csr_row_ptrl_data_ct1 =
            dpct::get_raw_pointer(A->csr_row_ptrl.data());
        auto thrust_raw_pointer_cast_A_cucol_data_ct2 =
            dpct::get_raw_pointer(A->cucol.data());
        auto A_n_rows_ct5 = A->n_rows;

        cgh.parallel_for(
            sycl::nd_range<3>(block * sycl::range<3>(1, 1, WARP_),
                              sycl::range<3>(1, 1, WARP_)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                kernal_mat_u_32_wrap(
                    thrust_raw_pointer_cast_A_cudata_data_ct0,
                    thrust_raw_pointer_cast_A_csr_row_ptrl_data_ct1,
                    thrust_raw_pointer_cast_A_cucol_data_ct2, b, c,
                    A_n_rows_ct5, item_ct1);
            });
    });
}


/*
if((row >= 404)&&(lane_id == 0)){
    printf("start ind =%d,end ind = %d\n",begin_index,end_index);
    for (int i  = 0 ; i < n_rows+1; ++i){
        printf("%d\t",csr_row_ptrl[i]);
    }
for(int i = begin_index + lane_id; i < end_index; i+=WARP_){
    if ( kernel_cudata[i] > 0){
        printf("%.2f\n",kernel_cudata[i]);
    }

}
}*/
void kernal_mat_u_32_wrap(double *kernel_cudata,
                                     int *csr_row_ptrl,int *cucol,
                                     double *u_cudata,
                                     double *output_cudata,int n_rows,
                                     sycl::nd_item<3> item_ct1){

    int thread_id = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    int vector_id = thread_id / WARP_;
    int lane_id = thread_id % WARP_;

    int row = vector_id;



    if(row <  n_rows){
        int begin_index =  csr_row_ptrl[row];
        int end_index =  csr_row_ptrl[row+1];

        double thread_sum = 0.0;
        for(int i = begin_index + lane_id; i < end_index; i+=WARP_)
            thread_sum += kernel_cudata[i] *  u_cudata [ cucol[i]];

//        if(row == 2423){
//            printf("csr_row_ptrl=%d   %d \n",csr_row_ptrl[row],csr_row_ptrl[row+1]);
//        }


        int temp = WARP_/2;
        while(temp >= 1){
            thread_sum += __shfl_down_(thread_sum, temp, item_ct1);
            temp >>= 1;
        }

        if ( lane_id == 0) {
              output_cudata[row] =  thread_sum;
        }

    }



}



void cuVec::print_vec(){
    show_res_T<double>((double *)dpct::get_raw_pointer(this->cudata.data()),
                       this->n_element);
}

//matrix transpose
/*
void matrixTrans(CSRMatrix*A,CSRMatrix*B){

    sycl::queue *handle;
    handle = &dpct::get_default_queue();
    COOMatrix matrix_temp(A->n_cols,A->n_rows,A->n_element);
    matrix_temp.cudata = A->cudata;
    matrix_temp.curow =  A->cucol;
    cusparseXcsr2coo(handle, dpct::get_raw_pointer(A->csr_row_ptrl.data()),
                     A->n_element, A->n_rows,
                     dpct::get_raw_pointer(matrix_temp.cucol.data()),
                     oneapi::mkl::index_base::zero);
    B->create_matrix_from_coo(&matrix_temp);
    handle = nullptr;
}
*/




