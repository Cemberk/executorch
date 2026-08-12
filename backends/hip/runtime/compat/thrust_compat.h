/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Thrust execution-policy shim for the HIP build.
//
// rocThrust implements the whole thrust:: API but spells its backend policy
// thrust::hip::par, with no thrust::cuda alias. A namespace alias cannot live in
// the cuda_runtime.h shim (that would pull thrust into every translation unit),
// so this header is force-included (-include) into just the sources that use a
// device execution policy. See backends/hip/CMakeLists.txt.

#pragma once

#include <thrust/system/hip/execution_policy.h>

namespace thrust {
namespace cuda = ::thrust::hip;
} // namespace thrust
