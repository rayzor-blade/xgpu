# xgpu-backend

The shared wgpu and browser-agent implementation used by xgpu runtime
adapters. A build script calls `xgpu_backend::install(OUT_DIR)` and includes
the returned module beneath its runtime-specific ABI wrapper.

The generated API model lives at the adapter crate root. Runtime carriers are
provided by `crate::runtime`; see the repository `CONTRIBUTING.md` for that
boundary.
