# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing
from typing import Any, Dict, final, List, Optional

import torch
from executorch.backends.aoti.aoti_backend import AotiBackend
from executorch.backends.cuda.passes.move_cond_predicate_to_cpu import (
    MoveCondPredicateToCpuPass,
)
from executorch.backends.cuda.passes.replace_int64_floordiv import (
    ReplaceInt64FloorDivWithFloatPass,
)
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import BackendDetails
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch._inductor.decomposition import conv1d_to_conv2d
from torch.nn.attention import SDPBackend

# ---------------------------------------------------------------------------
# Why this backend targets the "cuda" torch device
# ---------------------------------------------------------------------------
#
# On a ROCm build of PyTorch, HIP is exposed through the CUDA API surface:
# `torch.version.hip` is set, `torch.version.cuda` is None, and the device
# string is still "cuda". So the AOT path here moves the program to "cuda" and
# AOTInductor emits HIP code for it -- the generated .so calls hipMalloc /
# hipModuleLaunchKernel directly while still importing the CUDA-*named* AOTI
# shim symbols (aoti_torch_device_type_cuda, aoti_torch_create_cuda_stream_guard).
# That naming is AOTInductor's, not ours, and it is why the ROCm runtime can
# reuse the CUDA backend's shim sources compiled under HIP.


def _on_off_compile_spec_value(spec: CompileSpec) -> bool:
    value = spec.value.decode("utf-8").upper()
    if value not in ["ON", "OFF"]:
        raise ValueError(f"Invalid {spec.key}: {value}. Expected 'ON' or 'OFF'.")
    return value == "ON"


def is_hip_available() -> bool:
    """True when torch is a ROCm build with a usable HIP device."""
    return torch.version.hip is not None and torch.cuda.is_available()


def current_gfx_arch() -> Optional[str]:
    """Return the bare gfx architecture of the current device, e.g. "gfx942".

    ROCm reports the arch with feature suffixes ("gfx942:sramecc+:xnack-");
    those are stripped so the value can be compared and used in an arch list.
    """
    if not is_hip_available():
        return None
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    return arch.split(":")[0] if arch else None


@final
@experimental("This API and all of hip backend related functionality are experimental.")
class HipBackend(AotiBackend, BackendDetails):
    """
    Compiles a model to run on AMD GPUs via ROCm/HIP.

    Uses AOTInductor to generate HIP kernels libtorch-free, exactly as the CUDA
    backend does for NVIDIA. The compiled .so plus a weights blob are packed for
    the ExecuTorch runtime, which executes them through the HIP-built AOTI
    runtime (see backends/hip/runtime).
    """

    @classmethod
    def get_device_name(cls) -> str:
        # Intentionally "cuda": see the module docstring above.
        return "cuda"

    @classmethod
    def get_external_data_tag(cls) -> str:
        return "aoti_hip_blob"

    @classmethod
    def save_data_externally(cls) -> bool:
        """Emit the SO/weights blobs to aoti_hip_blob.ptd, matching CUDA."""
        return True

    @classmethod
    def get_supported_fallback_kernels(cls) -> Dict[str, Any]:
        """Fallback (non-Triton) kernels the HIP runtime can resolve.

        Only kernels whose numerics have been checked on AMD hardware are listed,
        so that anything unverified fails at export rather than silently returning
        wrong results. Two of the CUDA backend's kernels are deliberately withheld:

        * ``_weight_int4pack_mm`` is built (backends/hip/CMakeLists.txt) and runs
          on CDNA via the vendored ATen MFMA path, but its only tests assert
          shapes and error codes, never values. Add it here once its numerics are
          validated against a trusted reference.
        * The ``intN_plain_mm`` family assumes a 32-lane warp throughout, while a
          CDNA wavefront is 64 lanes, so those kernels are not ported at all.
        """
        return {
            "at::_ops::sort_stable::call": None,
            "aoti_torch_cuda_randint_low_out": None,
        }

    @classmethod
    def get_decomposition_table(cls) -> Dict[Any, Any]:
        return {
            torch.ops.aten.conv1d.default: conv1d_to_conv2d,
        }

    @classmethod
    def get_custom_passes(cls, compile_specs: List[CompileSpec]) -> List[typing.Any]:
        """Device-agnostic AOTI passes, plus optional Triton kernel replacement.

        The Triton replacement pass substitutes hand-tuned kernels written and
        autotuned against NVIDIA hardware, so it is OFF by default here and must
        be opted into with a triton_kernel_mode="ON" compile spec.
        """
        triton_kernel_mode = "OFF"
        for spec in compile_specs:
            if spec.key == "triton_kernel_mode":
                mode = spec.value.decode("utf-8").upper()
                if mode not in ["ON", "OFF"]:
                    raise ValueError(
                        f"Invalid triton_kernel_mode: {mode}. Expected 'ON' or 'OFF'."
                    )
                triton_kernel_mode = mode

        passes = [MoveCondPredicateToCpuPass(), ReplaceInt64FloorDivWithFloatPass()]
        if triton_kernel_mode == "ON":
            from executorch.backends.cuda.triton.replacement_pass import (
                ReplaceEdgeOpWithTritonOpPass,
            )

            logging.warning(
                "triton_kernel_mode=ON: the Triton kernels were tuned for NVIDIA "
                "GPUs and are not validated on ROCm. Verify numerics before use."
            )
            passes.append(ReplaceEdgeOpWithTritonOpPass())
        return passes

    @classmethod
    def get_aoti_compile_options(
        cls, compile_specs: List[CompileSpec]
    ) -> Dict[str, typing.Any]:
        """AOTI options for ROCm.

        Mirrors the CUDA backend minus everything NVIDIA-specific: there is no
        ptxas to locate and no fatbin multi-arch packaging, since AOTI embeds a
        single gfx code object selected from the compiling device.
        """
        if not is_hip_available():
            raise RuntimeError(
                "HipBackend requires a ROCm build of PyTorch with a visible AMD "
                f"GPU (torch.version.hip={torch.version.hip}, "
                f"torch.cuda.is_available()={torch.cuda.is_available()})."
            )

        options: Dict[str, typing.Any] = {
            # Disable this to support sdpa decomposition
            "loop_ordering_after_fusion": False,
            # Better model precision
            "emulate_precision_casts": True,
            # Embed the HIP code object directly into the compiled shared object
            "aot_inductor.embed_kernel_binary": True,
            # Do not link against the full PyTorch/libtorch library
            "aot_inductor.link_libtorch": False,
            # Separate weight constants from the .so file
            "aot_inductor.package": True,
            "aot_inductor.package_constants_in_so": False,
            # Store weight constants on disk in a binary blob
            "aot_inductor.package_constants_on_disk_format": "binary_blob",
            # Autotune with Triton only, so no libtorch operators are pulled in
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
            "max_autotune_conv_backends": "TRITON",
            # Multi-arch fatbins are a CUDA concept; ROCm builds one gfx target.
            "aot_inductor.emit_multi_arch_kernel": False,
        }

        emulate_precision_casts = True
        max_autotune = True
        autotune_at_compile_time = None
        for spec in compile_specs:
            if spec.key == "emulate_precision_casts":
                emulate_precision_casts = _on_off_compile_spec_value(spec)
            elif spec.key == "max_autotune":
                max_autotune = _on_off_compile_spec_value(spec)
            elif spec.key == "autotune_at_compile_time":
                autotune_at_compile_time = _on_off_compile_spec_value(spec)
            elif spec.key == "platform":
                platform = spec.value.decode("utf-8")
                if platform != "linux":
                    raise ValueError(
                        f"HipBackend only supports the linux platform, got '{platform}'."
                    )

        options["emulate_precision_casts"] = emulate_precision_casts
        options["max_autotune"] = max_autotune
        if autotune_at_compile_time is not None:
            options["triton.autotune_at_compile_time"] = autotune_at_compile_time

        logging.info("Compiling for AMD GPU arch %s", current_gfx_arch())
        return options

    @classmethod
    def get_extra_aoti_compile_context_manager(
        cls, compile_specs: Optional[List[CompileSpec]] = None
    ):
        """Force remaining SDPA ops onto the MATH backend so AOTI can lower them.

        ROCm's flash/mem-efficient SDPA kernels live in libtorch, which this
        backend does not link against.
        """
        return torch.nn.attention.sdpa_kernel([SDPBackend.MATH])

    @classmethod
    def release_moved_tensors(
        cls,
        device_edge_program,
        compile_specs: List[CompileSpec],
    ) -> None:
        """Free HIP memory held by tensors that move_to_device_pass placed on GPU,
        so the next method in a multi-method export can reuse it."""
        if not torch.cuda.is_available():
            return

        pools = []
        state_dict = getattr(device_edge_program, "state_dict", None)
        if state_dict:
            pools.append(state_dict.values())
        constants = getattr(device_edge_program, "constants", None)
        if constants:
            pools.append(constants.values())

        for pool in pools:
            for tensor in pool:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    try:
                        tensor.untyped_storage().resize_(0)
                    except Exception:
                        # Some storages are shared / non-resizable; skipping them
                        # only costs memory, so never fail the export over it.
                        pass
