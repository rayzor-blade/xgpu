<p align="center">
  <img src="assets/xgpu-logo.png" alt="xgpu" width="320">
</p>

# xgpu

xgpu gives Caribou, Ash, and Rayzor one `gpu` API backed by wgpu. It starts
with portable WebGPU and also exposes native capabilities such as surfaces,
backend selection, pipeline caches, mesh shaders, ray tracing, and
passthrough shaders.

## When to use it

Use xgpu when you are building a runtime adapter or native library and want:

- the same GPU API and enum values across supported Haxe runtimes;
- generated bindings from a pinned WebGPU IDL instead of maintaining externs
  by hand;
- native wgpu features without giving up a future browser/Wasm path; or
- small typed handles at the language boundary instead of copied GPU objects.

Application developers normally use their runtime's xgpu adapter. They do not
need to call the generator or depend on xgpu's internal crates directly.

## Runtime support

| Runtime | What is ready | Use it through |
|---|---|---|
| Caribou | Native plugin, generated language classes, and browser-agent wire | `caribou-gpu` |
| Ash / HashLink | Adapter realignment is in progress in hlwgpu; Ash Future is available | `hlwgpu` |
| Rayzor | Native adapter, generated externs, shared buffers, and Rayzor compute extensions | `rayzor-gpu.rpkg` |

The API is named `gpu`. Caribou and Ash expose it as `gpu`; Rayzor keeps its
established `rayzor.gpu` package and `rayzor-gpu.rpkg`. The generated GPU class
names and behavior stay aligned across runtimes while each adapter follows its
host's package convention.

Rayzor adds its compiler-owned `@:shader` lowering, lazy tensor graphs, fused
and quantized kernels, and Metal/CUDA policy to this API. Those extensions ship
in `rayzor-gpu.rpkg` beside the generated portable surface. Create them over an
xgpu device with `GPUCompute.fromDevice(device)` when both APIs must share the
same device, queue, and `GpuBuffer` objects.

## Wasm host contract

xgpu generates the guest-side command wire and the worker-side JavaScript
service for browser APIs. The consuming runtime provides the harness around
that service. It must:

- run a threaded wasm build over shared `WebAssembly.Memory`;
- receive the adapter's request to start its browser service;
- stage and import the adapter's shim beside the program;
- start that shim with `{ memory, address, canvas }`; and
- serve the page with the isolation headers required by `SharedArrayBuffer`.

Caribou implements this through `host::agent`; Ash exposes the equivalent
`ash_host_agent` hook. xgpu does not own either runtime's page, wasm loader,
Worker lifecycle, side-module loader, or canvas policy.

## Versioning

xgpu is currently consumed by pinned Git revisions. Pin one revision for the
generator, core, and backend source so the declared API and backend stay in
step.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the repository layout, binding
pipeline, adapter rules, and validation commands.

xgpu began from hlwgpu; see [LICENSE.hlwgpu](LICENSE.hlwgpu).
