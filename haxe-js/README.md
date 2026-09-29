# xgpu-js

Browser-native Haxe externs for the WebGPU API pinned by xgpu.

Add this directory as the `xgpu-js` haxelib and compile your JavaScript target
with `-lib xgpu-js`. Start with `gpu.js.WebGPU.gpu`.

Files under `src/gpu/js` are generated. Change the xgpu generator or its
vendored WebGPU IDL and regenerate them instead of editing them directly.
