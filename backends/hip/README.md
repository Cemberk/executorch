# HIP Backend (AMD GPU)

Runs models on AMD GPUs by compiling them with AOTInductor, which emits HIP
kernels ahead of time. It is the AMD counterpart to `backends/cuda`, and shares
that backend's AOTI machinery: the model is lowered to a single delegate
containing a `.so` of generated kernels plus a separate weights blob.

The backend is named for HIP, the API it actually targets (`hipcc`,
`libamdhip64`, `torch.version.hip`), which is the direct analogue of CUDA. ROCm
is the surrounding software stack, and a ROCm build of PyTorch is what you need
to use this.

**Status: experimental.** Export and execution are validated on gfx942 (MI300X).
See [Limitations](#limitations) before relying on it.

## Requirements

- A ROCm build of PyTorch (`torch.version.hip` set, `torch.version.cuda` `None`).
  Check with `python -c "import torch; print(torch.version.hip)"`.
- ROCm 6.0+ with `hipcc` for the runtime build.
- A supported AMD GPU. Development targets CDNA3 (gfx942); other architectures
  go through the same code path but are not yet validated. Set
  `-DEXECUTORCH_HIP_ARCH=gfx90a` (or a `;`-separated list) to build for others.

## Exporting a model

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

Weights are **not** packed into the `.pte`. Export produces two files and the
runtime needs both: `model.pte` and `aoti_hip_blob.ptd`.

There is also a ready-made script for the bundled example models:

```bash
python -m executorch.examples.hip.scripts.export --model_name add --output_dir .
```

### Compile specs

| Key | Values | Default | Effect |
| --- | --- | --- | --- |
| `method_name` | string | — | Method being compiled; set via `generate_method_name_compile_spec` |
| `max_autotune` | `ON`/`OFF` | `ON` | Triton autotuning. `OFF` compiles much faster, runs slower |
| `emulate_precision_casts` | `ON`/`OFF` | `ON` | Match eager precision more closely |
| `autotune_at_compile_time` | `ON`/`OFF` | unset | Autotune during compilation rather than first run |
| `triton_kernel_mode` | `ON`/`OFF` | `OFF` | Opt into the NVIDIA-tuned Triton kernels; unvalidated on AMD |
| `target_device` | e.g. `cuda:1` | `cuda:0` | Which GPU the delegate targets |

## Building and running

```bash
cmake -S . -B cmake-out -DEXECUTORCH_BUILD_HIP=ON \
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON
cmake --build cmake-out -j

./cmake-out/executor_runner --model_path model.pte --data_path aoti_hip_blob.ptd
```

`EXECUTORCH_BUILD_HIP` and `EXECUTORCH_BUILD_CUDA` are mutually exclusive; see
[Device type](#device-type).

## How it works

Three facts about AOTInductor on ROCm shape this backend:

1. **The torch device is `cuda`, not `hip`.** On a ROCm build of PyTorch, HIP is
   exposed through the CUDA API surface, so the AOT path moves the program to
   `"cuda"` and `torch.cuda` calls dispatch to HIP.

2. **The generated `.so` calls HIP directly.** Its undefined symbols are
   `hipMalloc`, `hipModuleLaunchKernel`, `hipModuleLoadData` — kernel launch is
   self-contained and needs nothing from us.

3. **But it imports the CUDA-*named* AOTI shim symbols**, e.g.
   `aoti_torch_device_type_cuda` and `aoti_torch_create_cuda_stream_guard`.
   Those names are AOTInductor's own ABI and are identical on both vendors.

Together these mean the CUDA backend's runtime already satisfies the HIP
contract, provided its internal driver calls become HIP calls. The CUDA surface
those sources use is about thirty symbols, and HIP defines a 1:1 equivalent for
every one. So rather than fork ~5k lines that would then have to be kept in sync
by hand, `runtime/compat/` maps that surface to HIP and is placed first on the
include path; the CUDA translation units are compiled as-is against it. Only
`cudaGraphInstantiate` needs more than a rename (HIP keeps the legacy 5-argument
form and puts the flags version under `hipGraphInstantiateWithFlags`).

Anything outside the mapped set is deliberately left unmapped, so a source that
starts using a new CUDA API fails to compile rather than silently binding to a
near-miss.

### Why the runtime sources live under `backends/cuda`

This backend has no runtime `.cpp` of its own; `CMakeLists.txt` compiles
`backends/cuda/runtime/*` against the compat headers. Those sources are not
really CUDA-specific — they are the shared AOTI GPU runtime — but they have not
been hoisted to a vendor-neutral directory because doing so would move ~10 files
and update external includers, Buck targets, and the CUDA CI path filters. That
refactor is worth doing separately; keeping it out of this change keeps the
NVIDIA paths untouched.

### Device type

`DeviceType` is a serialized schema enum shared with the `.pte` format
(`CPU = 0`, `CUDA = 1`), and the runtime's device-allocator registry is a
fixed-size array indexed by it. HIP reuses the `CUDA` slot rather than adding a
new value, which keeps the `.pte` format unchanged and matches PyTorch, where
HIP tensors report device type `cuda`. The cost is that the CUDA and HIP
backends cannot be linked into the same binary — exactly one can claim the slot.
CMake enforces this.

## Kernel support

The AOTI-generated Triton kernels cover the bulk of a model. A handful of
hand-written kernels live in `backends/cuda/runtime/shims/*.cu`, and these
compile unmodified against HIP:

| Kernel | Status |
| --- | --- |
| `sort_stable` | **Supported.** Thrust calls resolve to rocThrust; validated against eager. |
| `randint_low_out` | **Supported.** cuRAND Philox maps to the identical hipRAND generator. |
| `_weight_int4pack_mm` | **Built but not advertised.** Compiles via the vendored ATen `USE_ROCM` path and emits real `v_mfma_f32_16x16x16_bf16` on CDNA3, but its tests assert only shapes and error codes — the numerics are unverified on AMD. Add it to `HipBackend.get_supported_fallback_kernels()` once validated. |
| `intN_plain_mm` (int4/5/6/8) | **Not ported.** These assume a 32-lane warp (`MV_WARP_SIZE`, `__shfl_xor_sync` reductions) while a CDNA wavefront is 64 lanes, so they would compile but reduce across the wrong lanes. |

A model needing an unadvertised kernel fails at **export** with a clear message,
rather than loading a `.so` with unresolved symbols or returning wrong numbers.

## Limitations

- **Quantized models are not supported yet** — see the kernel table above.
- **Triton kernel replacement is off by default.** The kernels in
  `backends/cuda/triton` were written and autotuned against NVIDIA hardware.
  `triton_kernel_mode="ON"` enables them but numerics are unvalidated on AMD.
- **Single architecture per build.** Unlike CUDA fatbins, a compiled `.pte`
  embeds code for one gfx architecture. Export on the architecture you run on.
- **Linux only.**
- Not yet wired into CI or the LLM runner presets.

## Tests

```bash
# Partitioning only; runs anywhere.
python -m unittest executorch.backends.hip.tests.test_hip_partitioner

# Compiles through AOTInductor; needs an AMD GPU (skips otherwise).
python -m unittest executorch.backends.hip.tests.test_hip_export

# Executes .pte files on the GPU and compares against eager. Needs a built
# executor_runner (skips otherwise).
EXECUTORCH_RUNNER=$PWD/cmake-out/executor_runner \
    python -m unittest executorch.backends.hip.tests.test_hip_runtime
```
