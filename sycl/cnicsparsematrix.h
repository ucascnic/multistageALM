#ifndef CNICSPARSEMATRIX_H
#define CNICSPARSEMATRIX_H

#include<vector>
#include<string>


struct CSRIntMatrix{
    std::vector<ptrdiff_t> ptr, col;
    std::vector<int> val;
};

struct COORowPtrl{
   std::vector<ptrdiff_t> ptr;
};

void csr2coo(CSRIntMatrix &m, COORowPtrl &coo);
void matrixTrans(CSRIntMatrix &m, COORowPtrl &coo);

static int cmp_pair(ptrdiff_t M1, ptrdiff_t N1, ptrdiff_t M2, ptrdiff_t N2)
{
    if (M1 == M2) return (N1 < N2);
    else return (M1 < M2);
}

template <typename T>
void qsortCOO2CSR(ptrdiff_t *row, ptrdiff_t *col, T *val, ptrdiff_t l, ptrdiff_t r)
{
    ptrdiff_t i = l, j = r, row_tmp, col_tmp;
    ptrdiff_t mid_row = row[(l + r) / 2];
    ptrdiff_t mid_col = col[(l + r) / 2];
    double val_tmp;
    while (i <= j)
    {
        while (cmp_pair(row[i], col[i], mid_row, mid_col)) i++;
        while (cmp_pair(mid_row, mid_col, row[j], col[j])) j--;
        if (i <= j)
        {
            row_tmp = row[i]; row[i] = row[j]; row[j] = row_tmp;
            col_tmp = col[i]; col[i] = col[j]; col[j] = col_tmp;
            val_tmp = val[i]; val[i] = val[j]; val[j] = val_tmp;

            i++;  j--;
        }
    }
    if (i < r) qsortCOO2CSR<T>(row, col, val, i, r);
    if (j > l) qsortCOO2CSR<T>(row, col, val, l, j);
}



void compressIndices(ptrdiff_t *idx, ptrdiff_t *idx_ptr, ptrdiff_t nindex, ptrdiff_t nelem);

#endif // CNICSPARSEMATRIX_H
