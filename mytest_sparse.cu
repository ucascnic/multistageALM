#include"mytest_sparse.h"
#include<string>
#include<vector>
#include<cuda_runtime_api.h>
#include<cnic_sparsematrix.h>
#include<matrix_function.h>
#include<thrust/device_ptr.h>
#include<thrust/copy.h>
#include"mytest_sparse.h"


<<<<<<< HEAD
void run_test_for_sparse_mv()
{
    int txts[7] = {10, 20, 30, 40, 50, 100, 110};
    for (int i = 0; i<7; i++){
        int n=txts[i];
        sp_mat B;
        std::cout << "run test for size = " << n  << std::endl;
        std::string file = "/home/liuyun/multistageALM/test_sparse/"+std::to_string(n)+".txt";
        B.load(file.data(),coord_ascii);

        mat  test_vec  = randn(n,1);

        thrust::host_vector<double>  x(test_vec.memptr(),test_vec.memptr()+n);

        test_vec =  B * test_vec;
        std::cout << test_vec << std::endl;
        COOMatrix   test(B);

        CSRMatrix    csrttest( test.n_element,test.n_rows,test.n_cols);
        csrttest.create_matrix_from_coo(&test);

        cuVec  cuvec(x);
        cuVec  output(x);

        //output.print_vec();
        //sp_matrix_times_V(&csrttest,&cuvec,&output);
        BSRMatrix tt(&csrttest);
        sp_bsr_matrix_times_V_ptrl(&tt,
                                   thrust::raw_pointer_cast(cuvec.cudata.data()),
                                   thrust::raw_pointer_cast(output.cudata.data()));
        //output.print_vec();
        int result = check<double>(thrust::raw_pointer_cast(output.cudata.data()),test_vec.memptr(),test_vec.size());

        std::cout << result << std::endl;
        if (result != 1){
            std::cout << "error" << std::endl;
            exit(0);
        }
    }

}
void run_test_for_matrix_transpose(sp_mat &B){


    int n = B.n_rows;

    mat  test_vec  = ones(n,1);

    thrust::host_vector<double>  x(test_vec.memptr(),test_vec.memptr()+n);

    COOMatrix   test(B);
    CSRMatrix   csrttest( test.n_element,test.n_rows,test.n_cols);
    csrttest.create_matrix_from_coo(&test);

    CSRMatrix    csrttestT( test.n_element,test.n_cols,test.n_rows);
    matrixTrans(&csrttest,&csrttestT);
    //sp_mat BB = trans(B);
    //COOMatrix   test2(BB);
    //csrttestT.create_matrix_from_coo(&test2);

    cuVec  cuvec(x);
    cuVec  output(x);
    //output.print_vec();
    sp_matrix_times_V(&csrttestT,&cuvec,&output);

    test_vec =  trans(B) * test_vec;
    std::cout<< test_vec << std::endl;
    output.print_vec();

    int result = check<double>(thrust::raw_pointer_cast(output.cudata.data()),test_vec.memptr(),test_vec.size());
    std::cout << result << std::endl;
    if (result != 1){
        std::cout << "error" << std::endl;
        exit(0);
    }


}
=======
>>>>>>> e482650b266d506d07978d670d2906c54f33f0ad
