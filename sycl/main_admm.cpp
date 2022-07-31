#include<iostream>
#include"admmsolver.h"
#include<string>
#define ARMA_ALLOW_FAKE_GCC
#include<armadillo>
#include<vector>
// #include<cuda_runtime_api.h>
// #include<cnic_sparsematrix.h>
// #include<matrix_function.h>
// #include<thrust/device_ptr.h>
// #include<thrust/copy.h>
//#include"mytest_sparse.h"
#include<Common.h>
using namespace arma;

template <typename V>
std::vector<V> readFile(const char* filename) {
    std::ifstream input( filename, std::ios::binary );
    // copies all data into buffer
    std::vector<char> buffer((
            std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    int len=(int) buffer.size()/sizeof(V);
    std::vector<V> result(len);
    std::memcpy( result.data(), buffer.data(), buffer.size() );
    return result;
}




int main(int argc, char **argv){

    //run_test_for_sparse_mv();
    int n;
    if (argc <= 1)
        n = 202;
    else
        n = atoi(argv[1]);


    std::string file_path = "/home/liuyun/multistageALM/testcase/cuda_input/" + std::to_string(n) + "/";

    std::vector<std::string>  fieldes_double = {"beq","N1","lambda","xk",
                                "yk","ahatk","l1","LA","UA","a","b","DD","S","r","beqy"};
    std::vector<std::string>  fieldes_int = {"n","d"};

    std::vector<std::vector<double>>  data_double;
    std::vector<vec>  data_input_vector;
    std::vector<std::vector<int>>  data_int;

    std::cout << "reading double" << std::endl;
    for (int i = 0 ; i < fieldes_double.size(); ++i){
        std::string filename = file_path + fieldes_double[i] + ".mat";
        std::vector<double> temp  = readFile<double>(filename.c_str());

         std::cout << fieldes_double[i] << " " << temp.size() << std::endl;
        data_double.push_back(temp);
        vec t(temp.data(),temp.size());
        data_input_vector.push_back(t);
    }
    std::cout << "reading int" << std::endl;
    for (int i = 0 ; i < fieldes_int.size(); ++i){
        std::string filename = file_path + fieldes_int[i] + ".mat";
        std::vector<int> temp  = readFile<int>(filename.c_str());

        std::cout << temp[0] << std::endl;
        data_int.push_back(temp);
    }


   // read sparse matrix
    std::vector<std::string>  fieldes_sparse = {"AAA","A","B","C","M1","Aeqy"};
    std::vector<sp_mat> all_sparse_mat;
    for (int i = 0 ;i  < fieldes_sparse.size(); ++i){
        std::string file = "/home/liuyun/multistageALM/testcase/cuda_input/" +
                std::to_string(n) + "/" + fieldes_sparse[i] + ".txt";
        sp_mat A;
        A.load(file.data(),coord_ascii);
        std::cout << A.n_cols << std::endl;
        all_sparse_mat.push_back(A);

    }

    //{"beq","N1","lambda","xk",
    // "yk","ahatk","l1","LA","UA","a","b","DD"};
    int maxiter = 1000;
    n = data_int[0][0];
    int d = data_int[1][0];


    sp_mat B = all_sparse_mat[2];

   // run_test_for_matrix_transpose(B);

   // exit(0);

    admm_solver( all_sparse_mat[0],all_sparse_mat[1],all_sparse_mat[2],
            all_sparse_mat[3],all_sparse_mat[4],all_sparse_mat[5],data_input_vector[14],
                       data_input_vector[11],
                     data_input_vector[0],  data_input_vector[1],
                     data_input_vector[9], data_input_vector[10],data_input_vector[12],
                     data_input_vector[7],data_input_vector[8],
                     maxiter,n,d);

   //output.print_vec();

//    show_res_int<int>(thrust::raw_pointer_cast(csrttest.cucol.data()),0,csrttest.n_element);
//    show_res_int<int>(thrust::raw_pointer_cast(csrttest.csr_row_ptrl.data()),0,csrttest.n_rows+1);
    return 0;

}
