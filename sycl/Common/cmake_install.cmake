# Install script for directory: /home/chenyidong/multiblockADMM/Common

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/chenyidong/multiblockADMM/../bin/libCommon.a")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/chenyidong/multiblockADMM/../bin" TYPE STATIC_LIBRARY FILES "/home/chenyidong/multiblockADMM/Common/libCommon.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/chenyidong/multiblockADMM/../include/Common/eps_handler.h;/home/chenyidong/multiblockADMM/../include/Common/family.h;/home/chenyidong/multiblockADMM/../include/Common/ErrorCodes.h;/home/chenyidong/multiblockADMM/../include/Common/GridTools.h;/home/chenyidong/multiblockADMM/../include/Common/MultiScaleTools.h;/home/chenyidong/multiblockADMM/../include/Common/PythonTypes.h;/home/chenyidong/multiblockADMM/../include/Common/TCostFunctionProvider.h;/home/chenyidong/multiblockADMM/../include/Common/TCouplingHandler.h;/home/chenyidong/multiblockADMM/../include/Common/TEpsScaling.h;/home/chenyidong/multiblockADMM/../include/Common/THierarchicalCostFunctionProvider.h;/home/chenyidong/multiblockADMM/../include/Common/THierarchicalPartition.h;/home/chenyidong/multiblockADMM/../include/Common/THierarchyBuilder.h;/home/chenyidong/multiblockADMM/../include/Common/Tools.h;/home/chenyidong/multiblockADMM/../include/Common/TVarListHandler.h;/home/chenyidong/multiblockADMM/../include/Common/Verbose.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/chenyidong/multiblockADMM/../include/Common" TYPE FILE FILES
    "/home/chenyidong/multiblockADMM/Common/eps_handler.h"
    "/home/chenyidong/multiblockADMM/Common/family.h"
    "/home/chenyidong/multiblockADMM/Common/ErrorCodes.h"
    "/home/chenyidong/multiblockADMM/Common/GridTools.h"
    "/home/chenyidong/multiblockADMM/Common/MultiScaleTools.h"
    "/home/chenyidong/multiblockADMM/Common/PythonTypes.h"
    "/home/chenyidong/multiblockADMM/Common/TCostFunctionProvider.h"
    "/home/chenyidong/multiblockADMM/Common/TCouplingHandler.h"
    "/home/chenyidong/multiblockADMM/Common/TEpsScaling.h"
    "/home/chenyidong/multiblockADMM/Common/THierarchicalCostFunctionProvider.h"
    "/home/chenyidong/multiblockADMM/Common/THierarchicalPartition.h"
    "/home/chenyidong/multiblockADMM/Common/THierarchyBuilder.h"
    "/home/chenyidong/multiblockADMM/Common/Tools.h"
    "/home/chenyidong/multiblockADMM/Common/TVarListHandler.h"
    "/home/chenyidong/multiblockADMM/Common/Verbose.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/chenyidong/multiblockADMM/Common/Models/cmake_install.cmake")

endif()

