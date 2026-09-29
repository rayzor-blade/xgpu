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
| Haxe JavaScript | Browser-native WebGPU externs generated from the same IDL | `xgpu-js` |

The release archive includes the `xgpu-js` haxelib. Install or develop-link
that directory, then compile with `-lib xgpu-js`. Its classes live under
`gpu.js`, and `gpu.js.WebGPU.gpu` exposes the browser's `navigator.gpu`.

## GPU backends and platforms

xgpu exposes the backend names supported by its wgpu implementation:

| Backend | Platforms | Driver used |
|---|---|---|
| Metal | macOS and iOS | The system Metal driver |
| Direct3D 12 | Windows | The system D3D12 driver; this is the default Windows backend |
| Vulkan | Linux, Android, and optional Windows builds | The installed Vulkan ICD |
| OpenGL ES | Android fallback and Emscripten builds | The platform GLES driver |
| Browser WebGPU | Browser Wasm and browser-hosted WASI/Wasm threads | The browser's WebGPU implementation |

Current adapters build or package these targets:

| Platform | Architectures | Available graphics backends |
|---|---|---|
| macOS | Apple Silicon and x86-64 | Metal |
| iOS | arm64 devices and Apple Silicon simulator | Metal |
| Windows | x86-64 | Direct3D 12; Vulkan in the Vulkan build |
| Linux | x86-64 and arm64 | Vulkan |
| Android | arm64, armv7, and x86-64 | Vulkan and OpenGL ES |
| Browser | wasm32 | Browser WebGPU |
| Browser-hosted WASI | wasm32 WASI and threaded WASI | Browser WebGPU through the runtime harness |
| Emscripten | wasm32 | OpenGL ES |

Hardware support follows wgpu and the drivers installed on the target system.
xgpu does not bundle or select a vendor driver. `Noop` is reserved for internal
and headless use; it is not a graphics driver.

Rayzor additionally offers direct Metal compute and CUDA/NVRTC compute. These
are Rayzor extensions layered beside the portable xgpu API, so CUDA is not an
xgpu `Backend` value.

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

Successful `main` builds publish the generator binaries and `xgpu-js` haxelib
under the moving `nightly` release. A `v*` tag publishes the same tested
archives as a versioned release.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the repository layout, binding
pipeline, adapter rules, and validation commands.

xgpu began from hlwgpu; see [LICENSE.hlwgpu](LICENSE.hlwgpu).
