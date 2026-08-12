# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the AOTI ROCm backend for runtime.
#
# This backend deliberately has no runtime sources of its own. AOTInductor on
# ROCm emits a .so that calls HIP directly (hipMalloc, hipModuleLaunchKernel)
# while importing the CUDA-*named* AOTI shim symbols (aoti_torch_device_type_cuda,
# aoti_torch_create_cuda_stream_guard). Those names are AOTInductor's, identical
# on both vendors, so the CUDA backend's runtime translation units satisfy the
# contract as-is once compiled against HIP. We therefore reuse them through the
# cuda->hip compat headers in runtime/compat rather than forking ~5k lines that
# would need to be kept in sync by hand.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
find_package_torch()

find_package(hip REQUIRED)

# The CUDA backend's runtime sources, reused verbatim. Kept relative to
# EXECUTORCH_ROOT since they live outside this directory.
set(_hip_cuda_runtime_dir ${EXECUTORCH_ROOT}/backends/cuda/runtime)

# The compat directory must precede every other include path so that the
# `#include <cuda_runtime.h>` in those sources resolves to the HIP shim.
set(_hip_compat_include
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/compat>
)

# HIP marks hipSetDevice and friends [[nodiscard]] where the CUDA equivalents are
# not, so the reused sources warn on device switch/restore calls whose result is
# deliberately ignored. Suppressed here rather than by editing sources the CUDA
# backend owns.
set(_hip_cxx_compile_options -fexceptions -frtti -fPIC -Wno-unused-value)
set(_hip_export_dynamic_option -Wl,--export-dynamic)

# Platform utilities (load_library, close_library, etc.)
add_library(hip_platform STATIC ${_hip_cuda_runtime_dir}/platform/platform.cpp)

target_include_directories(
  hip_platform
  PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}> $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)

target_compile_options(
  hip_platform
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_hip_cxx_compile_options}>"
)

target_link_libraries(hip_platform PRIVATE executorch_core ${CMAKE_DL_LIBS})

install(
  TARGETS hip_platform
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# GPU architectures to generate code for. int4mm requires CDNA2 or later (it
# uses MFMA); the default targets CDNA3 (MI300). Override for other hardware,
# e.g. -DEXECUTORCH_HIP_ARCH="gfx90a;gfx942".
set(EXECUTORCH_HIP_ARCH
    "gfx942"
    CACHE STRING "AMD GPU architectures to compile device code for"
)

# int4mm.cuh selects its AMD code paths on USE_ROCM/ROCM_VERSION, matching the
# upstream ATen source it was vendored from. ROCM_VERSION is major*10000 +
# minor*100, the convention those guards compare against.
math(EXPR _hip_rocm_version "${hip_VERSION_MAJOR} * 10000 + ${hip_VERSION_MINOR} * 100")

set(_hip_device_compile_options -x hip)
foreach(_arch IN LISTS EXECUTORCH_HIP_ARCH)
  list(APPEND _hip_device_compile_options --offload-arch=${_arch})
endforeach()

# ==============================================================================
# HIP-specific AOTI shim symbols (dynamically linked)
#
# The .cu kernel sources compile unmodified against HIP: sort.cu's Thrust calls
# resolve to rocThrust, rand.cu's cuRAND Philox to the identical hipRAND
# generator, and int4mm.cuh already carries upstream ATen's USE_ROCM paths, which
# emit real MFMA instructions on CDNA.
#
# The intN_plain_mm family is deliberately excluded. Those are ExecuTorch's own
# kernels and assume a 32-lane warp throughout (MV_WARP_SIZE, __shfl_xor_sync
# reductions); a CDNA wavefront is 64 lanes, so they would compile but silently
# reduce across the wrong lanes. HipBackend advertises no fallback kernel for
# them, so an affected model fails at export rather than returning wrong numbers.
# ==============================================================================
set(_aoti_hip_shim_cxx_sources
    ${_hip_cuda_runtime_dir}/cuda_allocator.cpp
    ${_hip_cuda_runtime_dir}/shims/memory.cpp
    ${_hip_cuda_runtime_dir}/shims/cuda_guard.cpp
)
set(_aoti_hip_shim_device_sources
    ${_hip_cuda_runtime_dir}/shims/int4mm.cu ${_hip_cuda_runtime_dir}/shims/sort.cu
    ${_hip_cuda_runtime_dir}/shims/rand.cu
)
set(_aoti_hip_shim_sources ${_aoti_hip_shim_cxx_sources}
                           ${_aoti_hip_shim_device_sources}
)

add_library(aoti_hip_shims SHARED ${_aoti_hip_shim_sources})

# Compile everything through the C++ driver; the device sources additionally get
# -x hip so hipcc treats them as HIP rather than CUDA.
set_source_files_properties(
  ${_aoti_hip_shim_sources} PROPERTIES LANGUAGE CXX
)
set_source_files_properties(
  ${_aoti_hip_shim_device_sources}
  PROPERTIES COMPILE_OPTIONS "${_hip_device_compile_options}"
)

# rocThrust spells its execution policy thrust::hip::par with no thrust::cuda
# alias, so sort.cu needs the namespace alias in place before it includes Thrust.
set_source_files_properties(
  ${_hip_cuda_runtime_dir}/shims/sort.cu
  PROPERTIES
    COMPILE_OPTIONS
    "${_hip_device_compile_options};-include;${CMAKE_CURRENT_SOURCE_DIR}/runtime/compat/thrust_compat.h"
)

# CUDA_AVAILABLE selects the GPU (rather than CPU-only) SlimTensor paths in
# common_shims_slim.h; the compat headers make those paths HIP. USE_ROCM /
# ROCM_VERSION select the AMD code paths inside the vendored int4mm.
target_compile_definitions(
  aoti_hip_shims PRIVATE CUDA_AVAILABLE=1 USE_ROCM=1
                         ROCM_VERSION=${_hip_rocm_version}
)

target_include_directories(
  aoti_hip_shims
  PUBLIC ${_hip_compat_include} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
)

target_compile_options(
  aoti_hip_shims
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_hip_cxx_compile_options}>"
)

target_link_options(aoti_hip_shims PUBLIC ${_hip_export_dynamic_option})

# --whole-archive forces retention of the shim symbols, which are referenced only
# by the dlopen'd AOTI .so and so look unused at link time.
target_link_libraries(
  aoti_hip_shims
  PRIVATE hip_platform
  PUBLIC -Wl,--whole-archive
         aoti_common_shims_slim
         -Wl,--no-whole-archive
         hip::host
         extension_cuda
         ${CMAKE_DL_LIBS}
)

executorch_target_link_options_shared_lib(aoti_hip_shims)

install(
  TARGETS aoti_hip_shims
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# ==============================================================================
# ROCm backend implementation
# ==============================================================================
set(_aoti_hip_backend_sources ${_hip_cuda_runtime_dir}/cuda_backend.cpp
                               ${_hip_cuda_runtime_dir}/cuda_mutable_state.cpp
)

add_library(aoti_hip_backend STATIC ${_aoti_hip_backend_sources})

set_source_files_properties(
  ${_aoti_hip_backend_sources} PROPERTIES LANGUAGE CXX
)

# Claim the delegate id that HipPartitioner tags into the .pte. Without this the
# reused translation unit would register itself as "CudaBackend".
target_compile_definitions(
  aoti_hip_backend PRIVATE ET_AOTI_GPU_BACKEND_NAME="HipBackend"
                            CUDA_AVAILABLE=1
)

target_include_directories(
  aoti_hip_backend
  PUBLIC ${_hip_compat_include} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
)
target_compile_options(
  aoti_hip_backend
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_hip_cxx_compile_options}>"
)
target_link_options(aoti_hip_backend PUBLIC ${_hip_export_dynamic_option})

target_link_libraries(
  aoti_hip_backend
  PUBLIC hip_platform extension_tensor hip::host extension_cuda
         aoti_hip_shims ${CMAKE_DL_LIBS}
)

executorch_target_link_options_shared_lib(aoti_hip_backend)

install(
  TARGETS aoti_hip_backend
  EXPORT ExecuTorchTargets
  DESTINATION lib
)
