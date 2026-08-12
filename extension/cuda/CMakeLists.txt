# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please keep this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

# On a ROCm build this same source is compiled against HIP through the
# cuda->hip compat headers (see backends/hip/runtime/compat). The caller-stream
# API is a thin thread-local over cudaStream_t/hipStream_t, so it needs no
# ROCm-specific source of its own.
if(NOT EXECUTORCH_BUILD_HIP)
  find_package(CUDAToolkit REQUIRED)
endif()

# SHARED on purpose: the caller-stream thread-local must have a single
# definition across every shared object in the process (see export.h). A static
# copy linked into multiple shared libraries would create multiple thread-locals
# and silently break the caller-stream handshake.
add_library(extension_cuda SHARED caller_stream.cpp)
if(EXECUTORCH_BUILD_HIP)
  target_link_libraries(extension_cuda PUBLIC hip::host)
  target_include_directories(
    extension_cuda
    PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/backends/hip/runtime/compat>
  )
else()
  target_link_libraries(extension_cuda PUBLIC CUDA::cudart)
endif()
target_include_directories(extension_cuda PUBLIC ${_common_include_directories})
target_compile_options(
  extension_cuda PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_common_compile_options}>"
)
target_compile_definitions(
  extension_cuda PRIVATE EXECUTORCH_EXTENSION_CUDA_BUILDING
)

install(
  TARGETS extension_cuda
  EXPORT ExecuTorchTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
