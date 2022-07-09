#include"cnic_sparsematrix.h"
#include<iostream>
#include<cusparse_v2.h>
#include"matrix_function.h"
#include<assert.h>
#include<thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include<cublas_v2.h>
#define ARMA_ALLOW_FAKE_GCC
#include<armadillo>

static cusparseHandle_t cusparse_handle = 0;
cusparseStatus_t Status_t = cusparseCreate(&cusparse_handle);
static void init_cusparse() {
  if (cusparse_handle == 0) {
    if (Status_t != CUSPARSE_STATUS_SUCCESS) {
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
    show_res_T<double>((double*)thrust::raw_pointer_cast(this->cudata.data()),this->n_element);
}

void sortcoo_and_to_csr(int rows,int cols,int N,
                        double *val, int * row_ptrl, int *col_ptrl,
                        int *crs_row_ptrl){



    size_t workspace_size = 0;
    Status_t = cusparseXcoosort_bufferSizeExt(
        cusparse_handle,
        rows, cols,
        N,
        row_ptrl,
        col_ptrl,
        &workspace_size);
    assert( CUSPARSE_STATUS_SUCCESS == Status_t);
    double * buffer_for_coo_sort;
    CHECK(cudaMalloc(&buffer_for_coo_sort, sizeof(char) * workspace_size));
    int * indptrl_cu;
    CHECK(cudaMalloc(&indptrl_cu, sizeof(int) * N));
    int * indptrl_cu_mid = (int*) malloc( sizeof(int) *N);
    for ( int i =  0 ;i  < N ; ++i) indptrl_cu_mid[i] = i;


    cudaMemcpy(indptrl_cu,indptrl_cu_mid,sizeof(int) * N,cudaMemcpyHostToDevice);
    Status_t = cusparseXcoosortByRow(
        cusparse_handle,
        rows, cols,
        N,
        row_ptrl,
        col_ptrl,
        indptrl_cu,
        buffer_for_coo_sort);


   Status_t = cusparseDgthr(cusparse_handle,
                          N,
                          val,
                          buffer_for_coo_sort,
                          indptrl_cu,
                          CUSPARSE_INDEX_BASE_ZERO);



   cudaMemcpy(val,buffer_for_coo_sort,
              N*sizeof(double),cudaMemcpyDeviceToDevice);
   assert( CUSPARSE_STATUS_SUCCESS == Status_t);

    Status_t = cusparseXcoo2csr(cusparse_handle,
        row_ptrl, N, rows,
        crs_row_ptrl, CUSPARSE_INDEX_BASE_ZERO);
    assert( CUSPARSE_STATUS_SUCCESS == Status_t);

    cudaFree(buffer_for_coo_sort);
    cudaFree(indptrl_cu);




}
void CSRMatrix::create_matrix_from_coo(COOMatrix * coo_matrix){

    int n = coo_matrix->n_element;
    this->cudata = coo_matrix->cudata;
    this->cucol = coo_matrix->cucol;
    this->csr_row_ptrl  = thrust::device_vector<int>(coo_matrix->n_rows + 1);


    sortcoo_and_to_csr(coo_matrix->n_rows,coo_matrix->n_cols,n,
                       thrust::raw_pointer_cast(this->cudata.data()),
                       thrust::raw_pointer_cast(coo_matrix->curow.data()),
                       thrust::raw_pointer_cast(this->cucol.data()),
                       thrust::raw_pointer_cast(this->csr_row_ptrl.data()));


}
CSRMatrix::CSRMatrix(int size,int rows,int cols){
    this->n_element = size;
    this->n_rows = rows;
    this->n_cols = cols;

}

cuVec::cuVec(thrust::host_vector<double> &data){
    this->n_element = data.size();
    this->cudata = data;
}














#include<cuda_runtime_api.h>
#define WARP_ 32
inline __device__  double __shfl_down_(double var, unsigned int srcLane, int width=WARP_) {
  int2 a = *reinterpret_cast<int2*>(&var);
  a.x = __shfl_down(a.x, srcLane, width);
  a.y = __shfl_down(a.y, srcLane, width);
  return *reinterpret_cast<double*>(&a);
}
__global__ void kernal_mat_u_32_wrap(double *kernel_cudata,
                                     int *csr_row_ptrl,int *cucol,
                                     double *u_cudata,
                                     double *output_cudata,int n_rows);
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
void sp_matrix_times_V(CSRMatrix*A,cuVec*b,cuVec *c){


    int base, nnz;
    int nnzb;
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_COLUMN;
    int m = A->n_rows;
    int n = A->n_cols;
    int blockDim = 2;
    int *bsrRowPtrC;
    cusparseMatDescr_t descrA = NULL;
    cusparseMatDescr_t descrC = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseCreateMatDescr(&descrC);

    int mb = (m + blockDim-1)/blockDim;
    int nb = (n + blockDim-1)/blockDim;
    cudaMalloc((void**)&bsrRowPtrC, sizeof(int) *(mb+1));
    cusparseXcsr2bsrNnz(handle, dirA, m, n,
     descrA, thrust::raw_pointer_cast(A->csr_row_ptrl.data()),
             thrust::raw_pointer_cast(A->cucol.data()), blockDim,
            descrC, bsrRowPtrC, &nnzb);

    int*bsrColIndC;
    double*bsrValC;
    cudaMalloc((void**)&bsrColIndC, sizeof(int)*nnzb);
    cudaMalloc((void**)&bsrValC, sizeof(double)*(blockDim*blockDim)*nnzb);
    cusparseDcsr2bsr(handle, dirA, m, n,
     descrA, thrust::raw_pointer_cast(A->cudata.data()), thrust::raw_pointer_cast(A->csr_row_ptrl.data()),
                     thrust::raw_pointer_cast(A->cucol.data()), blockDim,
     descrC, bsrValC, bsrRowPtrC, bsrColIndC);
    // step 2: allocate vector x and vector y large enough for bsrmv
    double *x;
    double *y;
    cudaMalloc((void**)&x, sizeof(double)*(nb*blockDim));
    cudaMalloc((void**)&y, sizeof(double)*(mb*blockDim));
    cudaMemcpy(x, thrust::raw_pointer_cast(b->cudata.data()), sizeof(double)*n, cudaMemcpyDeviceToDevice);
    cudaMemcpy(y, thrust::raw_pointer_cast(c->cudata.data()), sizeof(double)*m, cudaMemcpyDeviceToDevice);
    // step 3: perform bsrmv
    double alpha = 1.;
    double beta = 0.;
    cusparseDbsrmv(handle, dirA, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   mb, nb, nnzb, &alpha,
     descrC, bsrValC, bsrRowPtrC, bsrColIndC, blockDim, x, &beta, y);

    cudaMemcpy(thrust::raw_pointer_cast(c->cudata.data()),y, sizeof(double)*m, cudaMemcpyDeviceToDevice);



}
void sp_matrix_times_V_ptrl(CSRMatrix*A,double*b,double *c){

    dim3 block(A->n_rows);
    kernal_mat_u_32_wrap<<<block,WARP_>>>(thrust::raw_pointer_cast(A->cudata.data()),
                                          thrust::raw_pointer_cast(A->csr_row_ptrl.data()),
                                          thrust::raw_pointer_cast(A->cucol.data()),
                                          b,
                                          c,
                                          A->n_rows);


}
BSRMatrix::BSRMatrix(CSRMatrix *A){


    int nnzb_;
    cusparseCreate(&this->handle);
    dirA = CUSPARSE_DIRECTION_COLUMN;
    m_rows = A->n_rows;
    n_cols = A->n_cols;
    blockDim = 2;

    descrA = NULL;
    descrC = NULL;
    cusparseCreateMatDescr(&this->descrA);
    cusparseCreateMatDescr(&this->descrC);

    int mb = (m_rows + blockDim-1)/blockDim;

    this->bsrRowPtrC.reserve(mb+1);
    //cudaMalloc((void**)&bsrRowPtrC, sizeof(int) *(mb+1));
    cusparseXcsr2bsrNnz(handle, dirA, m_rows, n_cols,
     descrA, thrust::raw_pointer_cast(A->csr_row_ptrl.data()),
             thrust::raw_pointer_cast(A->cucol.data()), blockDim,
            descrC, thrust::raw_pointer_cast(this->bsrRowPtrC.data()), &nnzb_);

    bsrColIndC.reserve(nnzb_);
    bsrValC.reserve((blockDim*blockDim)*nnzb_);
    //cudaMalloc((void**)&bsrColIndC, sizeof(int)*nnzb);
    //cudaMalloc((void**)&bsrValC, sizeof(double)*(blockDim*blockDim)*nnzb);
    cusparseDcsr2bsr(handle, dirA, m_rows, n_cols,
     descrA, thrust::raw_pointer_cast(A->cudata.data()),
                     thrust::raw_pointer_cast(A->csr_row_ptrl.data()),
                     thrust::raw_pointer_cast(A->cucol.data()), blockDim,
                     descrC, thrust::raw_pointer_cast(bsrValC.data()),
                     thrust::raw_pointer_cast(bsrRowPtrC.data()),
                     thrust::raw_pointer_cast(bsrColIndC.data()));
    this->nnzb = nnzb_;

}
void sp_bsr_matrix_times_V_ptrl(BSRMatrix*A,double*b,double *c){

    int blockDim = A->blockDim;
    int m_rows = A->m_rows;
    int n_cols = A->n_cols;
    int mb = (m_rows + blockDim-1)/blockDim;
    int nb = (n_cols + blockDim-1)/blockDim;

    // step 2: allocate vector x and vector y large enough for bsrmv
    double *x;
    double *y;
    cudaMalloc((void**)&x, sizeof(double)*(nb*blockDim));
    cudaMalloc((void**)&y, sizeof(double)*(mb*blockDim));
    cudaMemcpy(x, b, sizeof(double)*n_cols, cudaMemcpyDeviceToDevice);
    cudaMemcpy(y, c, sizeof(double)*m_rows, cudaMemcpyDeviceToDevice);
    // step 3: perform bsrmv
    double alpha = 1.;
    double beta = 0.;
    cusparseDbsrmv(A->handle, A->dirA, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   mb, nb, A->nnzb, &alpha,
     A->descrC, thrust::raw_pointer_cast(A->bsrValC.data()),
                   thrust::raw_pointer_cast(A->bsrRowPtrC.data()),
                   thrust::raw_pointer_cast(A->bsrColIndC.data()),
                   blockDim, x, &beta, y);
    cudaMemcpy(c,y, sizeof(double)*m_rows, cudaMemcpyDeviceToDevice);
    cudaFree(x);
    cudaFree(y);


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
__global__ void kernal_mat_u_32_wrap(double *kernel_cudata,
                                     int *csr_row_ptrl,int *cucol,
                                     double *u_cudata,
                                     double *output_cudata,int n_rows){

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
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
            thread_sum += __shfl_down_(thread_sum, temp);
            temp >>= 1;
        }

        if ( lane_id == 0) {
              output_cudata[row] =  thread_sum;
        }

    }



}



void cuVec::print_vec(){
    show_res_T<double>((double*)thrust::raw_pointer_cast(this->cudata.data()),this->n_element);

}

//matrix transpose
void matrixTrans(CSRMatrix*A,CSRMatrix*B){

    cusparseHandle_t handle;
    cusparseCreate(&handle);
    COOMatrix matrix_temp(A->n_cols,A->n_rows,A->n_element);
    matrix_temp.cudata = A->cudata;
    matrix_temp.curow =  A->cucol;
    cusparseXcsr2coo(handle,
     thrust::raw_pointer_cast(A->csr_row_ptrl.data()),
     A->n_element,
     A->n_rows,
     thrust::raw_pointer_cast(matrix_temp.cucol.data()),
     CUSPARSE_INDEX_BASE_ZERO);
    B->create_matrix_from_coo(&matrix_temp);


    cusparseDestroy(handle);

}





