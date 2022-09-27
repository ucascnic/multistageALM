Before compile:

1. Read the upper dir README, install armadillo and hdf5 first.
2. Modify CMakeList.txt, search "liuyun" and change the headers and libs path to your own.
3. Modify main_admm.cpp, search "liuyun" and change the input path to your own.
4. when meet compiling error about header conflict with MKL or TBB, change the armadillo header armadillo_bits/config.hpp, close the corresponding options. 


Compile: 

source /opt/intel/oneapi/setvars.sh

mkdir build; cd build
cmake ..
make

