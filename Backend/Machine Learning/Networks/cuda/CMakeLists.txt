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

# On Windows + Ninja the "Machine Learning" space in the source path leaks
# into the /Fd PDB flag that CMake wraps in nvcc's -Xcompiler, breaking the
# compile.  Redirect PDBs to a space-free directory under the build root.
if(MSVC)
	set_target_properties(GladesCUDA PROPERTIES
		COMPILE_PDB_NAME "GladesCUDA"
		COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pdb"
	)
endif()

target_compile_definitions(GladesCUDA PRIVATE GLADES_HAVE_CUDA=1)

# Include path so gpu_*.h headers can find each other.
target_include_directories(GladesCUDA PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link cuBLAS.
find_library(CUBLAS_LIB cublas
	HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
	      "$ENV{CUDA_PATH}/lib/x64")
if(CUBLAS_LIB)
	target_link_libraries(GladesCUDA ${CUBLAS_LIB})
else()
	target_link_libraries(GladesCUDA cublas)
endif()

# Link CUDA runtime so downstream consumers (unit tests, etc.) get it transitively.
find_library(CUDART_LIB cudart
	HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
	      "$ENV{CUDA_PATH}/lib/x64")
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
