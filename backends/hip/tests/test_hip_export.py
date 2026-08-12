# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional, Tuple

import torch
from executorch.backends.hip.hip_backend import HipBackend, is_hip_available
from executorch.backends.hip.hip_partitioner import HipPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export import export


class TestHipCompileOptions(unittest.TestCase):
    """Compile-spec parsing. Needs a real AMD GPU: get_aoti_compile_options
    refuses to produce options on a non-ROCm torch build."""

    def setUp(self):
        if not is_hip_available():
            self.skipTest("ROCm is not available")

    def test_defaults(self):
        options = HipBackend.get_aoti_compile_options([])

        self.assertTrue(options["max_autotune"])
        self.assertTrue(options["emulate_precision_casts"])
        self.assertFalse(options["aot_inductor.link_libtorch"])
        # Multi-arch fatbins are CUDA-only; ROCm embeds a single gfx object.
        self.assertFalse(options["aot_inductor.emit_multi_arch_kernel"])

    def test_emulate_precision_casts_compile_spec(self):
        options = HipBackend.get_aoti_compile_options(
            [CompileSpec(key="emulate_precision_casts", value=b"OFF")]
        )
        self.assertFalse(options["emulate_precision_casts"])

    def test_max_autotune_compile_spec(self):
        options = HipBackend.get_aoti_compile_options(
            [CompileSpec(key="max_autotune", value=b"OFF")]
        )
        self.assertFalse(options["max_autotune"])

    def test_invalid_max_autotune_compile_spec(self):
        with self.assertRaisesRegex(ValueError, "Invalid max_autotune"):
            HipBackend.get_aoti_compile_options(
                [CompileSpec(key="max_autotune", value=b"MAYBE")]
            )

    def test_non_linux_platform_rejected(self):
        with self.assertRaisesRegex(ValueError, "only supports the linux platform"):
            HipBackend.get_aoti_compile_options(
                [CompileSpec(key="platform", value=b"windows")]
            )


class TestHipBackendConfig(unittest.TestCase):
    """Configuration that holds regardless of whether a GPU is present."""

    def test_targets_cuda_torch_device(self):
        """HIP masquerades as CUDA in PyTorch, so the torch device is "cuda"."""
        self.assertEqual(HipBackend.get_device_name(), "cuda")

    def test_weights_blob_is_tagged_hip(self):
        """The external .ptd must not claim to be a CUDA blob."""
        self.assertTrue(HipBackend.save_data_externally())
        self.assertEqual(HipBackend.get_external_data_tag(), "aoti_hip_blob")

    def test_validated_fallback_kernels_advertised(self):
        """The kernels verified on AMD hardware must be offered."""
        kernels = HipBackend.get_supported_fallback_kernels()
        self.assertIn("at::_ops::sort_stable::call", kernels)
        self.assertIn("aoti_torch_cuda_randint_low_out", kernels)

    def test_unvalidated_fallback_kernels_withheld(self):
        """Kernels whose numerics are unverified on AMD must NOT be advertised,
        so a model needing one fails at export instead of returning wrong
        results. int4mm is built but unvalidated; intN_plain_mm assumes a
        32-lane warp and is not ported at all."""
        kernels = HipBackend.get_supported_fallback_kernels()
        self.assertNotIn("at::_ops::_weight_int4pack_mm::call", kernels)
        for width in (4, 5, 6, 8):
            self.assertNotIn(f"executorch_cuda::int{width}_plain_mm", kernels)

    def test_triton_replacement_is_off_by_default(self):
        """The Triton kernels are NVIDIA-tuned and must be opt-in on ROCm."""
        default_passes = HipBackend.get_custom_passes([])
        opted_in = HipBackend.get_custom_passes(
            [CompileSpec(key="triton_kernel_mode", value=b"ON")]
        )
        self.assertEqual(len(opted_in), len(default_passes) + 1)


class TestHipExport(unittest.TestCase):
    """Lower real modules through AOTInductor on an AMD GPU."""

    def setUp(self):
        if not is_hip_available():
            self.skipTest("ROCm is not available")

    def _lower(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        compile_specs: Optional[list[CompileSpec]] = None,
    ):
        exported_program = export(module, inputs, strict=True)

        if compile_specs is None:
            compile_specs = [HipBackend.generate_method_name_compile_spec("forward")]

        edge_program_manager = to_edge_transform_and_lower(
            exported_program,
            partitioner=[HipPartitioner(compile_specs)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        delegate_calls = [
            node
            for node in edge_program_manager.exported_program().graph.nodes
            if node.op == "call_function"
            and "executorch_call_delegate" in str(node.target)
        ]
        self.assertTrue(delegate_calls, "No delegate call in the lowered program")
        return edge_program_manager

    def test_simple_add(self):
        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        self._lower(AddModule().eval(), (torch.randn(3, 4), torch.randn(3, 4)))

    def test_linear_stack(self):
        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(32, 64),
                    torch.nn.GELU(),
                    torch.nn.Linear(64, 32),
                )
                self.norm = torch.nn.LayerNorm(32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(self.net(x) + x)

        self._lower(MLP().eval(), (torch.randn(8, 32),))

    def test_conv2d(self):
        class Conv2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.relu(self.conv(x))

        self._lower(Conv2dModule().eval(), (torch.randn(1, 3, 16, 16),))

    def test_serializes_pte_and_hip_weight_blob(self):
        """The full path: lower, serialize, and emit aoti_hip_blob.ptd."""

        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        import os
        import tempfile

        edge_program_manager = self._lower(
            AddModule().eval(), (torch.randn(3, 4), torch.randn(3, 4))
        )
        executorch_program = edge_program_manager.to_executorch()

        with tempfile.TemporaryDirectory() as tmpdir:
            pte_path = os.path.join(tmpdir, "model.pte")
            with open(pte_path, "wb") as f:
                executorch_program.write_to_file(f)
            executorch_program.write_tensor_data_to_file(outdir=tmpdir)

            self.assertGreater(os.path.getsize(pte_path), 0)
            self.assertIn("aoti_hip_blob.ptd", os.listdir(tmpdir))


if __name__ == "__main__":
    unittest.main()
