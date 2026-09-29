# xgpu-backend

The shared wgpu and browser-agent implementation used by xgpu runtime
adapters. A build script calls `xgpu_backend::install(OUT_DIR)` and includes
the returned module beneath its runtime-specific ABI wrapper.

The generated API model lives at the adapter crate root. Runtime carriers are
provided by `crate::runtime`; see the repository `CONTRIBUTING.md` for that
boundary.


The installed module also exposes `extension`, which lets a host adapter clone
the wgpu device, queue, and buffer behind an xgpu handle or register a new
buffer. Runtime-owned shader and compute systems use this seam so their work
shares xgpu resources without exposing wgpu objects to the language runtime.
