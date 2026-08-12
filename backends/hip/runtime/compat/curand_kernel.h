/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// cuRAND-to-hipRAND shim for the HIP build. See compat/cuda_runtime.h.
//
// Only the Philox generator used by rand.cu is mapped. hipRAND's Philox4_32_10
// is the same counter-based algorithm with the same parameters, so a given
// (seed, subsequence, offset) yields the same stream as cuRAND's.

#pragma once

#include <hiprand/hiprand_kernel.h>

using curandStatePhilox4_32_10_t = hiprandStatePhilox4_32_10_t;

#define curand_init hiprand_init
#define curand_uniform hiprand_uniform
#define curand_normal hiprand_normal
#define curand4 hiprand4
#define curand hiprand
