## 大规模资产负债管理模型
 ##

$$
\min \sum _ { t = 0 } ^ { T - 1 } \left( \sum _ { n = 1 } ^ { N _ { t } } p _ { t n } cr _ { tn } \right)\lambda \sum _ { t = 1 } ^ { T } \left( \sum _ { n = 1 } ^ { N _ { t } } p _ { tn } \left( \frac { Z_{ t n } } { L_{tn} }\right)^2 \right) + \sum _ { n = 1 } ^ { N _ { t }} p _ { T n } cr _ {  n }^{end}
$$

## 本程序编译方法 ##

    cmake .
    make

## 本程序运行 ##

    ./ADMM_Solver




## 第三方依赖库： ##

## Armadillo   ##

 


	tar -xvf armadillo-11.2.2.tar.xz
	
	cd armadillo-11.2.2
	
	./configure
	
	make
	
	make install

## hdf5 ##

    wget https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.5.tar
    tar -xvf hdf5-1.10.4.tar
    ./configure
    make
    make install 	 



## 注意:执行 cmake .  之前请修改CmakeLists.txt中的第三方库的路径 ##

please change the path before running `cmake .`


	INCLUDE_DIRECTORIES("/home/chenyidong/armadillo/include")	
	LINK_DIRECTORIES("/home/chenyidong/armadillo/lib")	
	LINK_DIRECTORIES("/home/chenyidong/hdf5_installed/lib")	
	INCLUDE_DIRECTORIES ("/home/chenyidong/hdf5_installed/include")



@article{1959Receptive,
  title={Receptive fields of single neurones in the cat's striate cortex},
  author={ Hubel, D. H.  and  Wiesel, T. N. },
  journal={The Journal of Physiology},
  volume={148},
  year={1959},
}

@journal{1987Relations,
  title={Relations between the statistics of natural images and the response properties of cortical cells},
  author={Field Dj, Hayes A, Hess Rf},
  year={1987},
  journal={JOURNAL OF THE OPTICAL SOCIETY OF AMERICA A-OPTICS IMAGE SCIENCE AND VISION},
  volume={4},
}

@article{1959Receptive,
  title={Receptive fields of single neurones in the cat's striate cortex},
  author={ Hubel, D. H.  and  Wiesel, T. N. },
  journal={The Journal of Physiology},
  volume={148},
  year={1959},
}