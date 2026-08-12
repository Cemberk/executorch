# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final, List

from executorch.backends.aoti.aoti_partitioner import AotiPartitioner
from executorch.backends.hip.hip_backend import HipBackend  # usort: skip
from executorch.exir._warnings import experimental
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes.propagate_device_pass import TARGET_DEVICE_COMPILE_SPEC_KEY


@final
@experimental("This API and all of hip backend related functionality are experimental.")
class HipPartitioner(AotiPartitioner):
    """
    ROCm partitioner driven by AOTInductor.

    Adds a target_device compile spec so PropagateDevicePass marks delegate
    boundary tensors as GPU-resident and inserts the host<->device copies.

    The device string is "cuda" rather than "hip" because DeviceType is a
    serialized schema enum (CPU=0, CUDA=1) shared with the .pte format, and
    ROCm reuses the CUDA slot -- consistent with PyTorch, where HIP tensors
    report device type "cuda". A HIP-only runtime build therefore claims the
    same device-allocator slot the CUDA backend would.
    """

    def __init__(
        self,
        compile_spec: List[CompileSpec],
    ) -> None:
        """
        Initialize the ROCm partitioner.

        Args:
            compile_spec: List of compile specs for the backend. To target a
                          specific AMD GPU, include a CompileSpec with key
                          "target_device" (e.g. value b"cuda:1"). Defaults to
                          "cuda:0".
        """
        has_target_device = any(
            spec.key == TARGET_DEVICE_COMPILE_SPEC_KEY for spec in compile_spec
        )
        if not has_target_device:
            compile_spec = list(compile_spec) + [
                CompileSpec(
                    TARGET_DEVICE_COMPILE_SPEC_KEY,
                    b"cuda:0",
                )
            ]
        super().__init__(HipBackend.__name__, compile_spec)
