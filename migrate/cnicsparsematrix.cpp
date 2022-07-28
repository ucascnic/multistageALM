#include "cnicsparsematrix.h"





void csr2coo(CSRIntMatrix &m,COORowPtrl &coo){
    int nrows = m.ptr.size() - 1;
    coo.ptr.resize(m.val.size());
    for (int i = 0 ; i < nrows; ++i){
        int start = m.ptr[i];
        int end = m.ptr[i+1];
        for (int j = start; j < end; ++j){
            coo.ptr[j] = i;
        }
    }
}



void matrixTrans(CSRIntMatrix &m, COORowPtrl &coo){
    m.col.swap(coo.ptr);
}


#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>
#include <omp.h>





void compressIndices(ptrdiff_t *idx, ptrdiff_t *idx_ptr, ptrdiff_t nindex, ptrdiff_t nelem)
{
    int curr_pos = 0, end_pos;
    idx_ptr[0] = 0;
    for (ptrdiff_t index = 0; index < nindex; index++)
    {
        for (end_pos = curr_pos; end_pos < nelem; end_pos++)
            if (idx[end_pos] > index) break;
        idx_ptr[index + 1] = end_pos;
        curr_pos = end_pos;
    }
    idx_ptr[nindex] = nelem;
}
