/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// fp16 shim for the HIP build. See compat/cuda_runtime.h for the rationale.
// HIP already spells the half types __half / __half2, so no aliases are needed.

#pragma once

#include <hip/hip_fp16.h>
