if(NOT (${CMAKE_BINARY_DIR} STREQUAL "${CMAKE_SOURCE_DIR}/build"))
	message(FATAL_ERROR "Must be in \"build\" directory")
endif()

set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "default install path" FORCE)
file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/include)

#Init
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 98)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# CUDA GPU offloading (optional, default OFF)
option(GLADES_ENABLE_CUDA "Build with CUDA GPU offloading support" OFF)
if(GLADES_ENABLE_CUDA)
	cmake_minimum_required(VERSION 3.18) # for CUDA_ARCHITECTURES

	# Find nvcc if not already in PATH (common on Fedora/RHEL where the
	# CUDA toolkit installs to /usr/local/cuda but doesn't modify PATH).
	if(NOT CMAKE_CUDA_COMPILER)
		find_program(_NVCC_BIN nvcc
			PATHS /usr/local/cuda/bin /usr/local/cuda-12/bin
			      /opt/cuda/bin ENV CUDA_PATH
			NO_DEFAULT_PATH)
		if(_NVCC_BIN)
			set(CMAKE_CUDA_COMPILER "${_NVCC_BIN}" CACHE FILEPATH "CUDA compiler")
		endif()
	endif()

	# CMake >= 3.18 requires CMAKE_CUDA_ARCHITECTURES to be non-empty.
	# Default to "native" (detect at build time) when not explicitly set.
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
		set(CMAKE_CUDA_ARCHITECTURES native)
	endif()

	include(${CMAKE_SOURCE_DIR}/cmake/CudaGlibcCompat.cmake)
	cuda_glibc_compat_apply()

	# When the system default g++ is too new for nvcc, find an older one.
	# The ccbin flag must be visible during CMake's compiler-identification
	# step (NVCC_PREPEND_FLAGS, appended AFTER cuda_glibc_compat_apply) and
	# during the actual build (CMAKE_CUDA_FLAGS).
	#
	# Detect the default g++ major version so we only try *older* ones
	# (the default is already tried by nvcc automatically).
	execute_process(COMMAND g++ -dumpversion
		OUTPUT_VARIABLE _DEFAULT_GXX_VER OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_QUIET RESULT_VARIABLE _GXX_RC)
	if(_GXX_RC EQUAL 0)
		string(REGEX MATCH "^[0-9]+" _DEFAULT_GXX_MAJOR "${_DEFAULT_GXX_VER}")
	else()
		set(_DEFAULT_GXX_MAJOR "0")
	endif()

	set(_CUDA_HOST_CXX "")
	foreach(_ver 14 13 12 11)
		if("${_ver}" STREQUAL "${_DEFAULT_GXX_MAJOR}")
			continue()   # same as default — nvcc already tries it
		endif()
		find_program(_GCC_CANDIDATE g++-${_ver} PATHS /usr/bin NO_DEFAULT_PATH)
		if(_GCC_CANDIDATE)
			set(_CUDA_HOST_CXX "${_GCC_CANDIDATE}")
			break()
		endif()
		unset(_GCC_CANDIDATE CACHE)
	endforeach()
	if(_CUDA_HOST_CXX)
		message(STATUS "Using CUDA host compiler: ${_CUDA_HOST_CXX}")
		set(CMAKE_CUDA_HOST_COMPILER "${_CUDA_HOST_CXX}" CACHE FILEPATH "CUDA host compiler")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${_CUDA_HOST_CXX}")
		set(ENV{NVCC_PREPEND_FLAGS} "$ENV{NVCC_PREPEND_FLAGS} -ccbin ${_CUDA_HOST_CXX}")
	endif()

	enable_language(CUDA)
	add_definitions(-DGLADES_HAVE_CUDA=1)
	message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
endif()

#Compiler Flags
set(CMAKE_BUILD_TYPE Release) # -O3

if(WIN32)
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
else()
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") # its set below
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -march=native")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

#Project
project(glades)
set(G_VERSION_MAJOR 0)
set(G_VERSION_MINOR 57)

# CBLAS acceleration for ATLAS optimizer (optional, auto-detected).
# Must be after project() so FindBLAS knows the active languages.
option(GLADES_ENABLE_CBLAS "Use CBLAS for ATLAS optimizer GEMM acceleration" ON)
if(GLADES_ENABLE_CBLAS)
	find_package(BLAS QUIET)
	if(BLAS_FOUND)
		include(CheckIncludeFile)
		check_include_file("cblas.h" HAVE_CBLAS_H)
		if(HAVE_CBLAS_H)
			add_definitions(-DGLADES_HAVE_CBLAS=1)
			set(GLADES_CBLAS_LIBRARIES ${BLAS_LIBRARIES})
			message(STATUS "CBLAS enabled: ${BLAS_LIBRARIES}")
		else()
			find_path(CBLAS_INCLUDE_DIR cblas.h PATHS /usr/include/openblas)
			if(CBLAS_INCLUDE_DIR)
				add_definitions(-DGLADES_HAVE_CBLAS=1)
				include_directories(${CBLAS_INCLUDE_DIR})
				set(GLADES_CBLAS_LIBRARIES ${BLAS_LIBRARIES})
				message(STATUS "CBLAS enabled (OpenBLAS): ${BLAS_LIBRARIES}")
			else()
				message(STATUS "BLAS found but cblas.h not available — CBLAS disabled")
			endif()
		endif()
	else()
		message(STATUS "BLAS not found — ATLAS optimizer will use built-in GEMM fallback")
	endif()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#Import libs
if(EMSCRIPTEN)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    find_library(SHMEA_PATH NAMES shmea PATHS "$ENV{HOME}/.local/lib/" NO_DEFAULT_PATH)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY) # Restore cross-compilation defaults
    if(NOT SHMEA_PATH)
	message(FATAL_ERROR "Failed to find shmea. Looked in: $ENV{HOME}/.local/lib")
    else()
	message(STATUS "Found libshmea.a at: ${SHMEA_PATH}")
    endif()
else()
    find_package(shmea REQUIRED)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

#Subdirectories
include_directories(${CMAKE_SOURCE_DIR})
include_directories("include")
include_directories("services")
add_subdirectory("Backend/Machine Learning")

set(MAIN_SRC_FILES
	main.cpp
	main.h
)

if(EMSCRIPTEN)
    add_library(${PROJECT_NAME} STATIC ${MAIN_SRC_FILES})
else()
    add_library(${PROJECT_NAME} SHARED ${MAIN_SRC_FILES})
endif()

#Link libraries
target_link_libraries(${PROJECT_NAME}
	ML GNet DB pthread) #custom libs

target_include_directories(glades PRIVATE "include")
target_include_directories(glades PRIVATE "Backend")

#target_include_directories(${PROJECT_NAME} PUBLIC
#    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#    $<INSTALL_INTERFACE:include>
#    PRIVATE Backend)

install(TARGETS ${PROJECT_NAME} EXPORT gladesConfig
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---- Package config / exports ----
# Export all targets collected into the `gladesConfig` export set (glades, ML, etc).
# This generates:
# - build/gladesTargets.cmake + build/gladesConfig.cmake for in-tree consumption
# - share/glades/cmake/gladesTargets.cmake + gladesConfig*.cmake on install
set(glades_INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/glades/cmake")

export(EXPORT gladesConfig FILE "${CMAKE_CURRENT_BINARY_DIR}/gladesTargets.cmake")

# Minimal `gladesConfig.cmake` for both build-tree and install-tree usage.
# Keeping it generated from CMakeLists avoids needing a separate .in template.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/gladesConfig.cmake"
"# Auto-generated by glades/CMakeLists.txt\n"
"\n"
"include(\"\${CMAKE_CURRENT_LIST_DIR}/gladesTargets.cmake\")\n"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/gladesConfigVersion.cmake"
    VERSION "${G_VERSION_MAJOR}.${G_VERSION_MINOR}"
    COMPATIBILITY SameMajorVersion
)

install(EXPORT gladesConfig
    FILE gladesTargets.cmake
    DESTINATION "${glades_INSTALL_CMAKEDIR}")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gladesConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/gladesConfigVersion.cmake"
    DESTINATION "${glades_INSTALL_CMAKEDIR}")

# Install GMath public headers so downstream projects can #include them
install(FILES
    "Backend/Machine Learning/GMath/pca.h"
    "Backend/Machine Learning/GMath/fft.h"
    "Backend/Machine Learning/GMath/fisher_transform.h"
    "Backend/Machine Learning/GMath/kelly.h"
    "Backend/Machine Learning/GMath/qp_solver.h"
    "Backend/Machine Learning/GMath/gmath.h"
    "Backend/Machine Learning/GMath/garch.h"
    "Backend/Machine Learning/GMath/egarch.h"
    "Backend/Machine Learning/GMath/kmeans.h"
    "Backend/Machine Learning/GMath/cmatrix.h"
    "Backend/Machine Learning/GMath/OHE.h"
    "Backend/Machine Learning/GMath/garch_utils.h"
    "Backend/Machine Learning/GMath/dwt.h"
    "Backend/Machine Learning/GMath/permutation_entropy.h"
    "Backend/Machine Learning/GMath/wasserstein.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Backend/Machine Learning/GMath")

# Install the full ML public header tree (namespaced under include/glades/) so
# downstream consumers (glades-trainer) compile against the installed headers
# instead of vendored copies.  Layout preserves the "Backend/Machine Learning/…"
# include convention.
install(DIRECTORY "Backend/Machine Learning/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/glades/Backend/Machine Learning"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inc")
target_include_directories(glades INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/glades>)

#make fmt
add_custom_target(fmt
	COMMAND ./clang-format-all
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make local
add_custom_target(local
	COMMAND ${PROJECT_NAME} local
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make debug
add_custom_target(debug
	COMMAND gdb ./build/${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make ddd
add_custom_target(ddd
	COMMAND ddd ./build/${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make mem
	add_custom_target(mem
	COMMAND valgrind --leak-check=full --track-origins=yes ./build/${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make profile
	add_custom_target(profile
	COMMAND valgrind --tool=callgrind ./build/${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make uninstall
	add_custom_target(uninstall
	COMMAND xargs rm < ./build/install_manifest.txt
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
