cmake_minimum_required(VERSION 3.16)
project(bench_qnn_vs_dff LANGUAGES CXX)

option(GLADES_ENABLE_CUDA "Link against CUDA-enabled glades build" OFF)
if(GLADES_ENABLE_CUDA)
	cmake_minimum_required(VERSION 3.18)
	include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CudaGlibcCompat.cmake)
	cuda_glibc_compat_apply()
	enable_language(CUDA)
	add_definitions(-DGLADES_HAVE_CUDA=1)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(shmea REQUIRED)
find_package(glades REQUIRED)

add_executable(bench_qnn_vs_dff bench_qnn_vs_dff.cpp)
target_link_libraries(bench_qnn_vs_dff PRIVATE glades shmea)

if(GLADES_ENABLE_CUDA)
	set_target_properties(bench_qnn_vs_dff PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
