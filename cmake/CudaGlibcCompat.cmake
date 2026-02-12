# CudaGlibcCompat.cmake — work around glibc >= 2.41 vs CUDA noexcept conflict.
#
# glibc 2.41+ declares cospi/sinpi/rsqrt with noexcept(true) while CUDA's
# crt/math_functions.h does not, causing cudafe++ to error out.  This module
# detects the situation at configure time and, when needed, applies a local
# header overlay that adds the missing noexcept specifiers.
#
# The overlay headers live in cmake/cuda_glibc_compat/ next to this file.
# See: https://forums.developer.nvidia.com/t/323591
#
# Usage — call BEFORE enable_language(CUDA):
#
#   include(.../cmake/CudaGlibcCompat.cmake)
#   cuda_glibc_compat_apply()
#   enable_language(CUDA)

# Capture the overlay path at include-time.
set(_CUDA_GLIBC_COMPAT_DIR "${CMAKE_CURRENT_LIST_DIR}/cuda_glibc_compat")

function(cuda_glibc_compat_apply)
    # Detect glibc version by parsing the header (no compiler / project() needed).
    if(NOT EXISTS "/usr/include/features.h")
        return()
    endif()

    file(STRINGS "/usr/include/features.h" _glibc_major
         REGEX "^#define[ \t]+__GLIBC__[ \t]+[0-9]+")
    file(STRINGS "/usr/include/features.h" _glibc_minor
         REGEX "^#define[ \t]+__GLIBC_MINOR__[ \t]+[0-9]+")

    if(NOT _glibc_major OR NOT _glibc_minor)
        return()
    endif()

    string(REGEX REPLACE ".*__GLIBC__[ \t]+([0-9]+)" "\\1" _glibc_major "${_glibc_major}")
    string(REGEX REPLACE ".*__GLIBC_MINOR__[ \t]+([0-9]+)" "\\1" _glibc_minor "${_glibc_minor}")

    if(NOT ((_glibc_major GREATER 2) OR (_glibc_major EQUAL 2 AND _glibc_minor GREATER_EQUAL 41)))
        return()
    endif()

    if(NOT EXISTS "${_CUDA_GLIBC_COMPAT_DIR}/crt/math_functions.h")
        message(WARNING "cuda_glibc_compat: overlay headers not found in ${_CUDA_GLIBC_COMPAT_DIR}")
        return()
    endif()

    message(STATUS "glibc ${_glibc_major}.${_glibc_minor} detected — applying CUDA noexcept compatibility overlay")

    # Configure-time: NVCC_PREPEND_FLAGS lets CMake's own CUDA compiler
    # identification compile succeed.
    set(ENV{NVCC_PREPEND_FLAGS} "-I${_CUDA_GLIBC_COMPAT_DIR}")

    # Build-time: CMAKE_CUDA_FLAGS is passed to every nvcc invocation.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${_CUDA_GLIBC_COMPAT_DIR}" PARENT_SCOPE)
endfunction()
