# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end runtime tests: export a model, execute the .pte on an AMD GPU, and
compare against eager.

These need a built ``executor_runner`` with the HIP backend linked in, so they
are opt-in. Point ``EXECUTORCH_RUNNER`` at the binary to enable them::

    cmake -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_HIP=ON \\
          -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -Bcmake-out .
    cmake --build cmake-out -j
    EXECUTORCH_RUNNER=$PWD/cmake-out/executor_runner \\
        python -m unittest executorch.backends.hip.tests.test_hip_runtime
"""

import os
import re
import subprocess
import tempfile
import unittest

import torch
from executorch.backends.hip.hip_backend import HipBackend, is_hip_available
from executorch.backends.hip.hip_partitioner import HipPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export

# executor_runner has no flag for supplying real inputs, so it fills every input
# tensor with ones (see extension/runner_util/inputs_portable.cpp). Eager is run
# on the same all-ones input to make the comparison meaningful.
_RUNNER_FILLS_INPUTS_WITH = 1.0


def _runner_path() -> str:
    return os.environ.get("EXECUTORCH_RUNNER", "")


class TestHipRuntime(unittest.TestCase):
    def setUp(self):
        if not is_hip_available():
            self.skipTest("No AMD GPU available")
        if not _runner_path() or not os.path.isfile(_runner_path()):
            self.skipTest(
                "Set EXECUTORCH_RUNNER to an executor_runner built with "
                "EXECUTORCH_BUILD_HIP=ON to run runtime tests"
            )

    def _run_on_gpu(
        self, module: torch.nn.Module, input_shapes: list[tuple[int, ...]]
    ) -> torch.Tensor:
        """Export ``module``, execute it via executor_runner, return the output."""
        inputs = tuple(
            torch.full(shape, _RUNNER_FILLS_INPUTS_WITH) for shape in input_shapes
        )

        exported = export(module, inputs, strict=True)
        lowered = to_edge_transform_and_lower(
            exported,
            partitioner=[
                HipPartitioner(
                    [HipBackend.generate_method_name_compile_spec("forward")]
                )
            ],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        program = lowered.to_executorch()

        with tempfile.TemporaryDirectory() as tmpdir:
            pte_path = os.path.join(tmpdir, "model.pte")
            with open(pte_path, "wb") as f:
                program.write_to_file(f)
            program.write_tensor_data_to_file(outdir=tmpdir)

            command = [_runner_path(), "--model_path", pte_path]
            ptd_path = os.path.join(tmpdir, "aoti_hip_blob.ptd")
            if os.path.isfile(ptd_path):
                command += ["--data_path", ptd_path]

            result = subprocess.run(
                command, capture_output=True, text=True, timeout=1800
            )

        output = result.stdout + result.stderr
        self.assertEqual(
            result.returncode, 0, f"executor_runner failed:\n{output[-2000:]}"
        )

        # executor_runner prints "OutputX 0: tensor(sizes=[1, 8], [1.5, -0.4, ...])".
        # Match the value list specifically -- scanning for bare numbers would also
        # pick up the output index and the sizes.
        payload = re.search(
            r"tensor\(sizes=\[[^\]]*\],\s*\[([^\]]*)\]\)",
            output,
        )
        self.assertIsNotNone(
            payload, f"Could not parse runner output:\n{output[-2000:]}"
        )
        # The runner elides long tensors with "...", which would silently compare
        # only a prefix. Keep test outputs small enough to print in full.
        self.assertNotIn(
            "...",
            payload.group(1),
            "executor_runner truncated the output tensor; use a model with fewer "
            "output elements so the full tensor can be compared",
        )
        return torch.tensor(
            [float(v) for v in payload.group(1).split(",") if v.strip()]
        )

    def _assert_matches_eager(self, module: torch.nn.Module, input_shapes, tol=1e-3):
        module = module.eval()
        inputs = tuple(
            torch.full(shape, _RUNNER_FILLS_INPUTS_WITH) for shape in input_shapes
        )
        with torch.no_grad():
            expected = module(*inputs).flatten()

        actual = self._run_on_gpu(module, input_shapes)

        self.assertEqual(
            actual.numel(),
            expected.numel(),
            f"element count mismatch: got {actual.numel()}, want {expected.numel()}",
        )
        max_diff = (actual - expected).abs().max().item()
        self.assertLess(
            max_diff,
            tol,
            f"GPU result diverges from eager by {max_diff:.3e}\n"
            f"  eager:  {expected.tolist()}\n"
            f"  runner: {actual.tolist()}",
        )

    def test_elementwise_add(self):
        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        self._assert_matches_eager(AddModule(), [(3, 4), (3, 4)])

    def test_mlp_with_weights_from_ptd(self):
        """Exercises the weights blob: constants must load from the .ptd."""

        torch.manual_seed(0)

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(16, 32)
                self.fc2 = torch.nn.Linear(32, 8)
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(self.fc2(torch.nn.functional.gelu(self.fc1(x))))

        self._assert_matches_eager(MLP(), [(1, 16)])

    def test_conv2d(self):
        torch.manual_seed(0)

        class ConvModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 2, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Pooled to keep the output small enough to print in full.
                return torch.nn.functional.adaptive_avg_pool2d(
                    torch.nn.functional.relu(self.conv(x)), 2
                )

        self._assert_matches_eager(ConvModule(), [(1, 3, 8, 8)])


if __name__ == "__main__":
    unittest.main()
