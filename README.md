# xgpu

`xgpu` is the shared wgpu implementation and binding source for Caribou, Ash,
and Rayzor. It keeps the public namespace `gpu`; WebGPU's IDL supplies the
portable starting point, while `gpu.api.rs` adds native wgpu capabilities such
as surfaces, backend selection, pipeline caches, mesh shaders, ray tracing,
and passthrough shaders.

The repository has four layers:

- `api/` is the one typed API declaration and the vendored WebGPU/canvas IDL.
- `xgpu-bindgen` parses that source and generates runtime bindings, WebGPU
  enums and dictionaries, and the Rust/JavaScript browser wire.
- `xgpu-core` owns runtime-neutral resource kinds and generational handle
  storage. GPU objects remain in Rust; language runtimes exchange small typed
  handles rather than copying wgpu objects.
- `adapters/caribou` is the complete native and browser-agent implementation
  extracted from Caribou. Its text, byte-buffer, error, root, and future code
  is the Caribou adapter around the shared API and resource core.

Caribou bindings are generated implicitly by the adapter's build script. Ash
and Rayzor can generate conventional Haxe externs from exactly the same API:

```sh
cargo run -p xgpu-bindgen --bin xgpu-haxe -- ash path/to/generated
cargo run -p xgpu-bindgen --bin xgpu-haxe -- rayzor path/to/generated
```

Promise results map to each runtime's native future type. Caribou uses
`caribou.Future<T>` and Rayzor uses `rayzor.concurrent.Future<T>`. The HashLink
emitter targets `ash.concurrent.Future<T>`; Ash still needs the externally
completable future ABI before asynchronous methods can link. The synchronous
surface and generated extern catalog do not depend on that work.

Runtime adapters follow one symbol convention generated alongside the Haxe
surface: HashLink loads `gpu_*` methods from the `xgpu` HDLL, and Rayzor loads
`xgpu_gpu_*` methods from its package. Adding an API member therefore updates
the Caribou plugin and both conventional extern sets from one declaration.

The HashLink/Ash adapter is built on
[`hl_abi`](https://github.com/rayzor-blade/hl_abi). That keeps native HDLL and
wasm side-module layouts, allocation, roots, strings, bytes, objects and
`DEFINE_PRIM` resolvers identical to other Ash libraries without linking a
second copy of the Ash runtime into the plugin.

## Development

```sh
cargo test -p xgpu-bindgen -p xgpu-core
cargo check -p xgpu-caribou
```

The implementation began from hlwgpu; see `LICENSE.hlwgpu`.
