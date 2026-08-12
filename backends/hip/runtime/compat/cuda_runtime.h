/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CUDA-runtime-to-HIP shim.
//
// The ROCm backend reuses the CUDA backend's runtime translation units rather
// than forking ~5k lines that would then have to be kept in sync by hand. Those
// sources `#include <cuda_runtime.h>`; this header is placed first on the
// include path for the ROCm targets so that include resolves here instead of to
// a real CUDA toolkit.
//
// This is sound because the CUDA surface those sources actually use is small
// (about thirty symbols, all enumerated below) and HIP defines a 1:1 equivalent
// for every one of them with identical semantics and argument order. Anything
// outside that list is deliberately NOT mapped: a source that starts using a new
// CUDA API fails to compile here rather than silently picking up a near-miss.
//
// Only two mappings are not a plain rename, and both are called out at their
// definition: cudaGraphInstantiate and the error-string helpers.

#pragma once

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// --- Error handling --------------------------------------------------------
using cudaError_t = hipError_t;
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError

// --- Device management -----------------------------------------------------
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceSynchronize hipDeviceSynchronize

using cudaDeviceProp = hipDeviceProp_t;
#define cudaGetDeviceProperties hipGetDeviceProperties

// --- Kernel attributes -----------------------------------------------------
using cudaFuncAttributes = hipFuncAttributes;
#define cudaFuncGetAttributes hipFuncGetAttributes
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
  hipFuncAttributeMaxDynamicSharedMemorySize

// --- Memory ----------------------------------------------------------------
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo

using cudaMemcpyKind = hipMemcpyKind;
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost

// hipPointerAttribute_t declares its kind as `type`, the same field name CUDA 11+
// uses on cudaPointerAttributes, so member access carries over unchanged.
using cudaPointerAttributes = hipPointerAttribute_t;
using cudaMemoryType = hipMemoryType;
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered
#define cudaMemoryTypeHost hipMemoryTypeHost
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaMemoryTypeManaged hipMemoryTypeManaged

// --- Streams and events ----------------------------------------------------
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamQuery hipStreamQuery
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamDefault hipStreamDefault
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDisableTiming hipEventDisableTiming

// --- Graphs ----------------------------------------------------------------
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaStreamCaptureMode = hipStreamCaptureMode;
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphInstantiateFlagAutoFreeOnLaunch \
  hipGraphInstantiateFlagAutoFreeOnLaunch

// The one signature that genuinely differs. CUDA 12 redefined the 3-argument
// cudaGraphInstantiate(exec, graph, flags), which is the form the reused sources
// call; HIP kept the legacy 5-argument hipGraphInstantiate and exposes the flags
// form under a separate name.
static inline hipError_t cudaGraphInstantiate(
    hipGraphExec_t* pGraphExec,
    hipGraph_t graph,
    unsigned long long flags = 0) {
  return hipGraphInstantiateWithFlags(pGraphExec, graph, flags);
}
