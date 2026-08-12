# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch
from executorch.backends.hip.hip_partitioner import HipPartitioner
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import PartitionResult
from executorch.exir.passes.propagate_device_pass import TARGET_DEVICE_COMPILE_SPEC_KEY
from torch.export import export


class TestHipPartitioner(unittest.TestCase):
    """Partitioning is device-independent, so these run without an AMD GPU."""

    def _get_partition_result(
        self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]
    ) -> PartitionResult:
        exported_program = export(module, inputs, strict=True)
        return HipPartitioner([]).partition(exported_program)

    def test_delegation_spec_uses_hip_backend_id(self):
        """The .pte must carry the HipBackend delegate id, not CudaBackend."""
        partitioner = HipPartitioner([])
        self.assertEqual(partitioner.delegation_spec.backend_id, "HipBackend")

    def test_default_target_device_compile_spec(self):
        """A target_device spec must be injected so PropagateDevicePass runs.

        The value is "cuda:0" because DeviceType is a serialized schema enum in
        which ROCm reuses the CUDA slot; see HipPartitioner's docstring.
        """
        partitioner = HipPartitioner([])
        specs = {
            spec.key: spec.value for spec in partitioner.delegation_spec.compile_specs
        }
        self.assertIn(TARGET_DEVICE_COMPILE_SPEC_KEY, specs)
        self.assertEqual(specs[TARGET_DEVICE_COMPILE_SPEC_KEY], b"cuda:0")

    def test_explicit_target_device_is_preserved(self):
        """A caller-supplied target_device must not be overwritten."""
        partitioner = HipPartitioner(
            [CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:3")]
        )
        specs = [
            spec
            for spec in partitioner.delegation_spec.compile_specs
            if spec.key == TARGET_DEVICE_COMPILE_SPEC_KEY
        ]
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].value, b"cuda:3")

    def test_simple_graph_is_fully_partitioned(self):
        class AddMul(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return (x + y) * x

        result = self._get_partition_result(
            AddMul(), (torch.randn(3, 4), torch.randn(3, 4))
        )

        tags = {
            node.meta["delegation_tag"]
            for node in result.tagged_exported_program.graph.nodes
            if node.op == "call_function" and "delegation_tag" in node.meta
        }
        self.assertEqual(
            len(tags), 1, "A fully delegatable graph should form one partition"
        )


if __name__ == "__main__":
    unittest.main()
