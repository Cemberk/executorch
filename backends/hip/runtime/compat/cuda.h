/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CUDA driver-API shim for the ROCm build. See compat/cuda_runtime.h for why
// these shims exist.
//
// The reused sources include <cuda.h> alongside <cuda_runtime.h> but do not
// call any cu* driver entry point -- everything they use (cudaError_t,
// cudaSuccess, cudaGetErrorString) comes from the runtime API. So this forwards
// to the runtime shim rather than mapping HIP's separate driver surface. If a
// source ever does need a driver call, it will fail to compile here, which is
// the intended signal to add the mapping deliberately.

#pragma once

#include "cuda_runtime.h"
