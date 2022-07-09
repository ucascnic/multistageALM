


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//% this program  uses the multiblock ADMM to solve the multi-stage asset
//%  allocation model.  The model is given by
//%  S   0         Gs        Gb       0              0  0  Ah           B
//%  0  -D  C   +  I  Xs  -  I  Xb +  I-Rh  Xh    +  0  0  Aend   =    Xini
//%  0   0  Z      0         0        -R             I  0            -Fmin L
//%  1   0         0         0        -r             0  1            -Fend*L
//%   w_l <=   X_{itn}^h/(sum_{i=1}^I X_{itn}^h) <= w_u
//%   LA  <=  C  <=  UA
//%   2021/07/25  chenyidong@cnic.cn
//%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%













第三方依赖库：

1. Armadillo  
wget http://sourceforge.net/projects/arma/files/armadillo-11.2.2.tar.xz
tar -xvf armadillo-11.2.2.tar.xz
cd armadillo-11.2.2
./configure
make
make install 

2. hdf5
tar -xvf ./thridparty/hdf5-1.10.4.tar
cd  ./thridparty/hdf5-1.10.4
./configure
make
make install 



本程序编译方法
cmake .
make


注意
执行cmake .  之前请修改CmakeLists.txt中的第三方库的路径
'''
# please change the path before running cmake .
INCLUDE_DIRECTORIES("/home/chenyidong/armadillo/include")
LINK_DIRECTORIES("/home/chenyidong/armadillo/lib")
LINK_DIRECTORIES("/home/chenyidong/hdf5_installed/lib")
INCLUDE_DIRECTORIES ("/home/chenyidong/hdf5_installed/include")
'''


