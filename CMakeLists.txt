cmake_minimum_required(VERSION 3.5)
project(clpeak)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)
include(GNUInstallDirs)

# DLL Settings
if(MSVC)
  if(BUILD_SHARED_LIBS)
    add_definitions(" /DCLBLAST_DLL")
  endif()
endif(MSVC)

# C++ compiler settings
if(MSVC)
  set(FLAGS "/Ot")
  set(FLAGS "${FLAGS} /wd4715 /D_CRT_SECURE_NO_WARNINGS")
else()
  set(FLAGS "-std=c++11")
  if(VERBOSE)
    set(FLAGS "${FLAGS} -O1 -g")
  else()
    set(FLAGS "${FLAGS} -O2")
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set(FLAGS "${FLAGS} -Wall -Wno-comment -Wno-return-type -Wno-switch -Wno-missing-noreturn -Wno-unused-function")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.0)
      set(FLAGS "${FLAGS} -Wno-attributes -Wno-unused-variable")
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0.0)
      # GCC does not support attributes on template arguments
      # in particular we hit this with the alignment attributes on cl_XXX types
      # which are then used to instantiate various templates in CLBlast
      set(FLAGS "${FLAGS} -Wno-ignored-attributes")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(FLAGS "${FLAGS} -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded")
    set(FLAGS "${FLAGS} -Wno-missing-prototypes -Wno-float-equal -Wno-switch-enum -Wno-switch")
    set(FLAGS "${FLAGS} -Wno-exit-time-destructors -Wno-global-constructors -Wno-missing-noreturn")
    set(FLAGS "${FLAGS} -Wno-deprecated-declarations -Wno-unused-function")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.9.0)  # clang 4.0 or higher
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0)  # but not for AppleClang
          set(FLAGS "${FLAGS} -Wno-undefined-var-template")
        endif()
    endif()
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")


if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenCL)
FIND_PATH(HPP_FOUND CL/opencl.hpp PATHS ${OpenCL_INCLUDE_DIRS})
if(NOT OpenCL_FOUND OR NOT HPP_FOUND)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(BuildSdk)

  find_package(OpenCL REQUIRED)

  set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS} "${SDK_ROOT}/include")
  FIND_PATH(HPP_FOUND CL/opencl.hpp PATHS ${OpenCL_INCLUDE_DIRS} REQUIRED)
endif()

if(NOT OpenCL_FOUND)
  message(FATAL_ERROR "Could not find OpenCL include/libs. Set OCL_ROOT to your OpenCL SDK")
else()
  message(STATUS "Selected OpenCL includes from ${OpenCL_INCLUDE_DIRS}")
  message(STATUS "Selected OpenCL lib ${OpenCL_LIBRARIES}")
endif()

# extra libs for static library
if(UNIX AND ${OpenCL_LIBRARIES} MATCHES "libOpenCL.a")
  set(OpenCL_LIBRARIES ${OpenCL_LIBRARIES} pthread dl)
elseif(WIN32 AND ${OpenCL_LIBRARIES} MATCHES "OpenCL.lib")
  set(OpenCL_LIBRARIES ${OpenCL_LIBRARIES} cfgmgr32.lib runtimeobject.lib)
endif()

#set(CMAKE_CXX_STANDARD "11")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CLPEAK_SOURCE_FILES
    src/common.cpp
    src/entry.cpp
    src/kernel_preprocessor.cpp
)

add_executable(clpeak ${CLPEAK_SOURCE_FILES})
target_link_libraries(clpeak ${OpenCL_LIBRARIES})
target_include_directories(clpeak PRIVATE ${OpenCL_INCLUDE_DIRS} "include" "src" "src/kernel" "src/utilities")
target_compile_definitions(clpeak PRIVATE CL_TARGET_OPENCL_VERSION=120)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU"   OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(clpeak PRIVATE -fPIC -Wall -Wextra -Wno-unused-parameter)
endif()