// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "main.h"
#include "Backend/Machine Learning/Networks/cuda/gpu_blas.h"
#include "Backend/Machine Learning/Networks/cuda/gpu_kernels.h"

// iter 180: force libglades.so to retain glades::gpu::set_tf32_enabled and
// get_tf32_enabled symbols.  Without this, link-time GC strips them since no
// other TU in libglades.so references them — they're only consumed by the
// glades-trainer external binary.  Holding live function pointers ensures
// the linker keeps both symbols.
namespace glades { namespace gpu {
__attribute__((used))
static void (*const _force_keep_set_tf32_enabled)(bool) = &set_tf32_enabled;
__attribute__((used))
static bool (*const _force_keep_get_tf32_enabled)() = &get_tf32_enabled;
// iter 181: same trick for ASTRA Gate-0 kernel — only the external trainer
// calls it, so GC would strip it otherwise.
__attribute__((used))
static bool (*const _force_keep_astra_update)(
    float*, const float*, float*, float, float, float, float, float,
    int, int) = &astra_update;
}}
