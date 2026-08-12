/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// bfloat16 shim for the HIP build. See compat/cuda_runtime.h for the rationale.
// HIP's __hip_bfloat16 is layout- and semantics-compatible with __nv_bfloat16.

#pragma once

#include <hip/hip_bf16.h>

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
