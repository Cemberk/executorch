# HIP Backend (AMD GPU)

The HIP backend is the ExecuTorch solution for running models on AMD GPUs. It leverages the [AOTInductor](https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html) compiler to generate optimized HIP kernels with libtorch-free execution, and uses [Triton](https://triton-lang.org/) for high-performance GPU kernel generation. It is the AMD counterpart to the [CUDA backend](../cuda/cuda-overview.md) and shares its AOTInductor machinery.

The backend is named for HIP, the API it targets (`hipcc`, `libamdhip64`, `torch.version.hip`), which is the direct analogue of CUDA. ROCm is the surrounding software stack, and a ROCm build of PyTorch is what you need to use this backend.

> **Experimental.** Export and execution are validated on gfx942 (MI300X). See [Limitations](#limitations).

## Features

- **Optimized GPU Execution**: AOTInductor generates HIP kernels for model operators ahead of time
- **Triton Kernel Support**: Triton autotunes GEMM and convolution kernels for the target AMD architecture
- **Libtorch-free Runtime**: The compiled model runs without linking the full PyTorch library

## Target Requirements

- **Hardware**: An AMD GPU supported by ROCm. Development targets CDNA3 (gfx942); other architectures use the same code path but are not yet validated.
- **ROCm**: ROCm 6.0 or later, with `hipcc` available for the runtime build
- **Operating System**: Linux

## Development Requirements

- **Python**: Python 3.10+
- **PyTorch**: A ROCm build of PyTorch. Verify with `python -c "import torch; print(torch.version.hip)"` — this must print a version, and `torch.version.cuda` must be `None`.

## Using the HIP Backend

### Exporting models with the Python API

The backend uses the `HipBackend` and `HipPartitioner` classes:

```python
import torch
from executorch.backends.hip import HipBackend, HipPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

model, example_inputs = ..., (torch.randn(8, 128),)

exported = torch.export.export(model.eval(), example_inputs, strict=True)
lowered = to_edge_transform_and_lower(
    exported,
    partitioner=[
        HipPartitioner([HipBackend.generate_method_name_compile_spec("forward")])
    ],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)

program = lowered.to_executorch()
with open("model.pte", "wb") as f:
    program.write_to_file(f)
program.write_tensor_data_to_file(outdir=".")  # writes aoti_hip_blob.ptd
```

Weights are **not** packed into the `.pte`. Export produces two files and the runtime needs both: `model.pte` and `aoti_hip_blob.ptd`.

A ready-made script covers the bundled example models:

```bash
python -m executorch.examples.hip.scripts.export --model_name add --output_dir .
```

### Compile specs

| Key | Values | Default | Effect |
| --- | --- | --- | --- |
| `method_name` | string | — | Method being compiled; set via `generate_method_name_compile_spec` |
| `max_autotune` | `ON`/`OFF` | `ON` | Triton autotuning. `OFF` compiles much faster, runs slower |
| `emulate_precision_casts` | `ON`/`OFF` | `ON` | Match eager precision more closely |
| `autotune_at_compile_time` | `ON`/`OFF` | unset | Autotune during compilation rather than on first run |
| `triton_kernel_mode` | `ON`/`OFF` | `OFF` | Opt into the NVIDIA-tuned Triton kernels; unvalidated on AMD |
| `target_device` | e.g. `cuda:1` | `cuda:0` | Which GPU the delegate targets |

### Building and running the runtime

```bash
cmake -S . -B cmake-out -DEXECUTORCH_BUILD_HIP=ON \
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON
cmake --build cmake-out -j

./cmake-out/executor_runner --model_path model.pte --data_path aoti_hip_blob.ptd
```

Build for a different architecture with `-DEXECUTORCH_HIP_ARCH=gfx90a` (or a `;`-separated list).

`EXECUTORCH_BUILD_HIP` and `EXECUTORCH_BUILD_CUDA` are mutually exclusive: both register an AOTI delegate that claims the single `DeviceType::CUDA` device-allocator slot, so exactly one may be built. CMake enforces this.

## Kernel support

AOTInductor-generated Triton kernels cover the bulk of a model. A few hand-written kernels are shared with the CUDA backend and compile unmodified against HIP:

| Kernel | Status |
| --- | --- |
| `sort_stable` | Supported. Thrust calls resolve to rocThrust; validated against eager. |
| `randint_low_out` | Supported. cuRAND Philox maps to the identical hipRAND generator. |
| `_weight_int4pack_mm` | Built but not advertised — compiles via the vendored ATen ROCm path and emits real MFMA instructions on CDNA3, but its numerics are not yet verified on AMD. |
| `intN_plain_mm` (int4/5/6/8) | Not ported. These assume a 32-lane warp, while a CDNA wavefront is 64 lanes. |

A model that needs an unadvertised kernel fails at **export** with a clear message, rather than loading a shared library with unresolved symbols or returning wrong numbers.

## Limitations

- **Quantized models are not supported yet** — see the kernel table above.
- **Triton kernel replacement is off by default.** The kernels in `backends/cuda/triton` were written and autotuned for NVIDIA hardware. `triton_kernel_mode="ON"` enables them, but their numerics are unvalidated on AMD.
- **One architecture per build.** Unlike CUDA fatbins, a compiled `.pte` embeds code for a single gfx architecture. Export on the architecture you will run on.
- **Linux only.**

For implementation details — including why the backend targets the `cuda` torch device and how the CUDA runtime sources are reused under HIP — see [backends/hip/README.md](https://github.com/pytorch/executorch/tree/main/backends/hip).
