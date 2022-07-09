











# 第三方依赖库： #

## Armadillo   ##

 


	tar -xvf armadillo-11.2.2.tar.xz

	cd armadillo-11.2.2

	./configure

	make

	make install

## hdf5 ##

    tar -xvf hdf5-1.10.4.tar
	cd  ./thridparty/hdf5-1.10.4
	./configure
	make
	make install 	 







## 本程序编译方法 ##

    cmake .
	make
 


#### 注意:执行cmake .  之前请修改CmakeLists.txt中的第三方库的路径 ####

please change the path before running `cmake .`

`
INCLUDE_DIRECTORIES("/home/chenyidong/armadillo/include")`
`

`LINK_DIRECTORIES("/home/chenyidong/armadillo/lib")`

`LINK_DIRECTORIES("/home/chenyidong/hdf5_installed/lib")`

`INCLUDE_DIRECTORIES ("/home/chenyidong/hdf5_installed/include")`



