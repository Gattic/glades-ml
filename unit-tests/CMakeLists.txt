if(NOT (${CMAKE_BINARY_DIR} STREQUAL "${CMAKE_SOURCE_DIR}/build"))
	message(FATAL_ERROR "Must be in \"build\" directory")
endif()

set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "default install path" FORCE)
file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/include)

#Init
cmake_minimum_required(VERSION 3.10)

# CUDA GPU offloading (must match main build when linking CUDA-built libraries)
option(GLADES_ENABLE_CUDA "Build with CUDA GPU offloading support" OFF)
if(GLADES_ENABLE_CUDA)
	cmake_minimum_required(VERSION 3.18)
	include(${CMAKE_SOURCE_DIR}/../cmake/CudaGlibcCompat.cmake)
	cuda_glibc_compat_apply()
	enable_language(CUDA)
	add_definitions(-DGLADES_HAVE_CUDA=1)
endif()

set(CMAKE_CXX_STANDARD 98)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

#Compiler Flags
set(CMAKE_BUILD_TYPE Debug) # -g
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

#Project
project(glades-unit-tests)
set(G_VERSION_MAJOR 0)
set(G_VERSION_MINOR 57)

set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(GNUInstallDirs)

#Import libs
find_package(shmea REQUIRED)

# Prefer the in-tree build of glades (../build) when present.
# This avoids accidentally linking unit tests against an older installed copy.
# The glades root build exports `gladesConfig.cmake` into its build directory.
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../build")
find_package(glades REQUIRED)
find_package(Freetype REQUIRED)

#Subdirectories
include_directories(${FREETYPE_INCLUDE_DIRS})
include_directories("../include")
add_subdirectory("Backend/Machine Learning")

#Executable
set(MAIN_src_files
	main.cpp
	main.h
	unit-test.cpp
	unit-test.h
)
add_executable(${PROJECT_NAME} ${MAIN_src_files})

#Link libraries
target_link_libraries(${PROJECT_NAME}
	PCATests glades shmea)#custom libs

target_include_directories(${PROJECT_NAME} PRIVATE "Backend")

# When CUDA is enabled, the final executable must run nvlink to resolve
# device symbols from separable compilation in GladesCUDA.
if(GLADES_ENABLE_CUDA)
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

#make run
add_custom_target(run
	COMMAND ${PROJECT_NAME} $(ARGS)
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make debug
add_custom_target(debug
	COMMAND gdb ./build/${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make debug ddd
add_custom_target(ddd
	COMMAND ddd ./build/${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make mem
	add_custom_target(mem
	COMMAND valgrind --leak-check=full --track-origins=yes --show-reachable=yes ./build/${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make profile
	add_custom_target(profile
	COMMAND valgrind --tool=callgrind ./build/${PROJECT_NAME} $(ARGS)
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
