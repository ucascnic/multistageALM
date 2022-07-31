#ifndef ADMMSOLVER_H
#define ADMMSOLVER_H
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//% this function use the multiblock ADMM to solve the multi stage asset
//%  allocation model  the model is given by
//%  S   0         Gs        Gb       0              0  0  Ah           B
//%  0  -D  C   +  I  Xs  -  I  Xb +  I-Rh  Xh    +  0  0  Aend   =    Xini
//%  0   0  Z      0         0        -R             I  0            -Fmin L
//%  1   0         0         0        -r             0  1            -Fend*L
//%   w_l <=   X_{itn}^h/(sum_{i=1}^I X_{itn}^h) <= w_u
//%   LA  <=  C  <=  UA
//%   2021/07/25  chenyidong@cnic.cn
//%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#define ARMA_ALLOW_FAKE_GCC
#include<armadillo>
using namespace arma;
struct  Opt;
void admm_solver(sp_mat &AAA, sp_mat &A, sp_mat &B, sp_mat &C, sp_mat &M1,
                 sp_mat &Aeqy, vec &beqy, vec &D,
                 vec &beq, vec & N1, vec & a, vec & b, vec & S,
                 vec & LA, vec & UA,
                 int maxiter, int n, int d);

#endif // ADMMSOLVER_H
