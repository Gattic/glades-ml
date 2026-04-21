# GladesCUDA static library.
#
# This is conditionally included from Networks/CMakeLists.txt only when
# GLADES_ENABLE_CUDA is ON.

cmake_minimum_required(VERSION 3.18)

set(GLADES_CUDA_SOURCES
	gpu_device.cu
	gpu_buffer.cu
	gpu_blas.cu
	gpu_kernels.cu
	gpu_atlas.cu
	gpu_vesta.cu
	gpu_helios.cu
	gpu_transformer_state.cu
	gpu_dff_state.cu
	gpu_rnn_state.cu
	gpu_cnn_kernels.cu
)

add_library(GladesCUDA STATIC ${GLADES_CUDA_SOURCES})

# CUDA needs at least C++11 for device lambdas, etc.
set_target_properties(GladesCUDA PROPERTIES
	CUDA_STANDARD 17
	CUDA_STANDARD_REQUIRED ON
	CUDA_SEPARABLE_COMPILATION OFF
	POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(GladesCUDA PRIVATE GLADES_HAVE_CUDA=1)

# Include path so gpu_*.h headers can find each other.
target_include_directories(GladesCUDA PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link cuBLAS.
find_library(CUBLAS_LIB cublas HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
if(CUBLAS_LIB)
	target_link_libraries(GladesCUDA ${CUBLAS_LIB})
else()
	target_link_libraries(GladesCUDA cublas)
endif()

# Link cuSOLVER for batched factorizations.
find_library(CUSOLVER_LIB cusolver HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
if(CUSOLVER_LIB)
	target_link_libraries(GladesCUDA ${CUSOLVER_LIB})
else()
	target_link_libraries(GladesCUDA cusolver)
endif()

# Link CUDA runtime so downstream consumers (unit tests, etc.) get it transitively.
find_library(CUDART_LIB cudart HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
if(CUDART_LIB)
	target_link_libraries(GladesCUDA ${CUDART_LIB})
else()
	target_link_libraries(GladesCUDA cudart)
endif()

# Generate code for common architectures.
# Users can override via CMAKE_CUDA_ARCHITECTURES.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
	set_target_properties(GladesCUDA PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
endif()

install(TARGETS GladesCUDA EXPORT gladesConfig
	ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR})
