# Contributing to xgpu

The user-facing contract is the `gpu` namespace. Keep API changes in the
shared declaration so every runtime sees the same classes, records, enums,
constants, and method names.

## Repository layout

- `api/gpu.api.rs` is the typed xgpu API declaration.
- `api/spec/` contains the pinned WebGPU and canvas IDL snapshots.
- `crates/xgpu-bindgen` generates Caribou bindings, conventional Haxe externs,
  and the Rust/JavaScript browser wire.
- `crates/xgpu-core` contains runtime-neutral resource kinds and generational
  handle storage.
- `crates/xgpu-backend` carries the shared wgpu and browser-agent operations
  that adapters compile beneath their own ABI boundary.
- Runtime adapters stay in their host repositories. Text, buffers, errors,
  roots, futures, allocation, and exported symbols follow the host runtime's
  ABI.

WebGPU is the portable baseline, not a feature ceiling. Put native wgpu
extensions in `gpu.api.rs` and mark IDL extensions explicitly so generators
can distinguish them from the browser API.

## Adapter boundary

Keep GPU resources owned by Rust and pass typed integer handles across a
language boundary. Avoid copying byte buffers when the host runtime can lend a
stable view. Runtime-specific object layouts, garbage-collector roots, errors,
and Future completion belong in the adapter.

Shared backend source uses the generated model at the adapter crate root and
imports its carrier contract from `crate::runtime`. Do not introduce a direct
dependency from that source to Caribou, HashLink, Ash, or Rayzor ABI types.

HashLink/Ash adapters use `hl_abi` for allocation, roots, strings, bytes,
objects, and `DEFINE_PRIM` exports. Rayzor adapters use
`rayzor.concurrent.Future<T>` for Promise results. Both runtimes still need an
externally completable Future bridge for asynchronous native callbacks.

## Validation

Run the generator and core tests after changing the API, IDL, wire, or handle
model:

```sh
cargo test -p xgpu-backend -p xgpu-bindgen -p xgpu-core
```

Generate both conventional Haxe surfaces when changing runtime mappings:

```sh
cargo run -p xgpu-bindgen --bin xgpu-haxe -- ash /tmp/xgpu-ash
cargo run -p xgpu-bindgen --bin xgpu-haxe -- rayzor /tmp/xgpu-rayzor
```

Caribou owns its runtime adapter in `caribou/plugins/cb_gpu`. From a sibling
Caribou checkout, validate the plugin descriptor and fixture build:

```sh
cargo build -p caribou-gpu
cargo run -p caribou-driver --bin caribou -- describe target/debug/libcaribou_gpu.dylib
cargo test -p caribou-plugin-fixtures --no-run
```

Use the platform's dynamic-library suffix in the `describe` command.
