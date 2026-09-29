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
| Ash / HashLink | Conventional Haxe extern generation | `xgpu-haxe ash`; native adapter is in progress |
| Rayzor | Conventional Haxe extern generation | `xgpu-haxe rayzor`; native adapter is in progress |

The public namespace is `gpu` on every runtime. A runtime adapter may have a
different native library name, but application code keeps the same GPU class
names.

## Generate Haxe externs

Runtime maintainers can generate the full API from the repository root:

```sh
cargo run -p xgpu-bindgen --bin xgpu-haxe -- ash path/to/generated
cargo run -p xgpu-bindgen --bin xgpu-haxe -- rayzor path/to/generated
```

Promise-returning methods map to the runtime's Future type. Ash and Rayzor
still need their native completion bridges before their asynchronous GPU
adapters are complete.

## Versioning

xgpu is currently consumed by pinned Git revisions. Pin one revision for the
generator, core, and backend source so the declared API and backend stay in
step.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the repository layout, binding
pipeline, adapter rules, and validation commands.

xgpu began from hlwgpu; see [LICENSE.hlwgpu](LICENSE.hlwgpu).
