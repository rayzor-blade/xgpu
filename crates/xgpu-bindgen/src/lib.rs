//! Typed WebGPU declarations and generators shared by xgpu runtime adapters.
//! Traits describe resource classes; structs describe plugin-owned records;
//! enums whose variants carry a type describe unions, which records set
//! through one setter per variant; `#[native(name)]` selects a backend
//! function. `#[idl("Name")]` imports enum values, namespace constants,
//! dictionary members or union alternatives from a vendored WebIDL source. This
//! does not infer native GPU semantics from WebIDL interfaces or generate a
//! language-specific heap layout.

pub use x_idl::{wire, haxe, web_backend, hashlink_web_backend, generate_rayzor_with_resources};
pub mod haxe_js;


/// The runtime-neutral API declaration consumed by every adapter generator.
pub const GPU_API: &str = include_str!("../../../api/gpu.api.rs");
/// The WebGPU specification snapshot used for imported types and browser wire.
pub const WEBGPU_IDL: &str = include_str!("../../../api/spec/webgpu.idl");
/// The small canvas surface appended to WebGPU for browser presentation.
pub const CANVAS_IDL: &str = include_str!("../../../api/spec/canvas.idl");

/// The browser wire schema: WebGPU plus the small OffscreenCanvas surface
/// used to size the presentation target. Runtime adapters should use this
/// whenever they generate the browser agent and guest encoder together.
pub fn browser_idl() -> String {
    format!("{WEBGPU_IDL}\n{CANVAS_IDL}")
}

/// The canonical declaration with wgpu's native feature catalog appended.
/// Keeping this here makes every runtime emitter follow the exact wgpu version
/// xgpu implements.
pub fn gpu_api() -> String {
    let variants = wgpu_types::Features::all()
        .iter_names()
        .map(|(name, _)| pascal(&name.to_ascii_lowercase().replace('_', "-")))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{GPU_API}\nenum NativeFeature {{ {variants} }}\n")
}

/// The HashLink library xgpu's primitives load from: `xgpu.hdll`, or
/// `xgpu.wasm` under Ash in a page.
pub const HASHLINK_LIBRARY: x_idl::Library<'static> = x_idl::Library("xgpu");

/// Generate xgpu's complete conventional Haxe surface for one runtime.
pub fn _haxe(runtime: haxe::Runtime) -> Result<Vec<haxe::File>, String> {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    // A file of this call's own: x-idl reads declarations from a path, and
    // builds generating at once must not share one.
    let declaration = std::env::temp_dir().join(format!(
        "gpu.api.{}.{}.rs",
        std::process::id(),
        CALLS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&declaration, gpu_api())
        .map_err(|e| format!("failed to write gpu.api.rs: {e}"))?;
    let files = match runtime {
        haxe::Runtime::HashLink => {
            HASHLINK_LIBRARY.haxe("gpu", Some(declaration.clone()), &browser_idl(), runtime)
        }
        haxe::Runtime::Rayzor => {
            haxe::generate("rayzor.gpu", Some(declaration.clone()), &browser_idl(), runtime)
        }
    };
    std::fs::remove_file(&declaration).ok();
    files
}

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

fn pascal(name: &str) -> String {
    let mut result = String::new();
    for word in name.split('-') {
        let mut chars = word.chars();
        if let Some(c) = chars.next() {
            result.extend(c.to_uppercase());
            result.extend(chars);
        }
    }
    if result.starts_with(|c: char| c.is_ascii_digit()) {
        result.insert(0, 'D');
    }
    result
}

/// Emit a self-contained set of resource wrappers, schemas and one plugin
/// table. Backends implement the selected functions with integer handles;
/// the generated ABI uses typed native objects, enums, Text and Buffer.
pub fn generate_caribou(
    namespace: &str,
    declaration: Option<PathBuf>,
    webidl: &str,
) -> Result<String, String> {
    x_idl::generate_caribou(namespace, declaration, webidl)
}

/// Emit the runtime-neutral model and exported C symbols used by Rayzor's
/// native package. The adapter supplies Text, Buffer, roots, futures, errors,
/// and the generic Enum carrier; xgpu supplies the object model and backend.
pub fn generate_rayzor(declaration: Option<PathBuf>, webidl: &str) -> Result<String, String> {
    x_idl::generate_rayzor_with_resources("gpu", declaration, webidl, &[])
}

/// Emit the runtime-neutral model and HashLink primitive resolvers used by
/// hlwgpu. Resources cross as integer handles, records as GC-finalized native
/// abstracts, and Promise results as Ash Future carriers.
pub fn generate_hashlink(declaration: Option<PathBuf>, webidl: &str) -> Result<String, String> {
    HASHLINK_LIBRARY.generate_hashlink("gpu", declaration, webidl)
}

/// Compatibility spelling for existing Caribou build scripts.
pub fn generate(
    namespace: &str,
    declaration: Option<PathBuf>,
    webidl: &str,
) -> Result<String, String> {
    generate_caribou(namespace, declaration, webidl)
}

#[cfg(test)]
mod tests {
    use std::env::temp_dir;

    use super::*;

    #[test]
    fn rayzor_model_uses_adapter_carriers_and_exports_native_symbols() {
        let api = r#"
            enum Mode { Fast, Slow }
            struct Options { label: Option<Text>, mode: Enum<Mode> }
            trait Device {
                #[native(device_open)] fn open(options: &Options) -> Box<Device>;
                #[native(device_name)] fn name(this: &Device) -> Text;
            }
        "#;
        let api_path = temp_dir().join("rayzor_api.rs");

        std::fs::write(&api_path, api).unwrap();

        let generated = generate_rayzor(Some(api_path), WEBGPU_IDL).unwrap();
        assert!(!generated.contains("caribou_abi"));
        assert!(!generated.contains("plugin !"));
        assert!(generated.contains("Rooted < Text >"));
        assert!(generated.contains("impl NativeEnum for Mode"));
        assert!(generated.contains("export_name = \"xidl_options_new\""));
        assert!(generated.contains("export_name = \"xidl_device_open\""));
        assert!(generated.contains("host :: raise (ErrorKind :: Runtime"));
        assert!(generated.contains("pub static XIDL_METHODS"));
        assert!(generated.contains("\"gpu::Device\""));
        assert!(generated.contains("__xidl_device_open as * const u8"));
        assert!(generated.contains("fn __xidl_options_new (a0 : i64)"));
        assert!(generated.contains("transmute :: < i64 , Enum < Mode > >"));
        assert!(generated.contains("param_types : [3u8 , 0u8"));
    }


    fn gpu_decl(content:&str)-> Option<PathBuf> {
        // make file unique to avoid collisions with other tests
        let file_name = format!("gpu.api.{}.rs", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros());
        let path = temp_dir().join(file_name);
        std::fs::write(&path, content).ok()?;
        Some(path)
    }

    #[test]
    fn the_complete_rayzor_model_and_registration_generate_together() {
        // copy declaration to temp dir
        let dec_dir = gpu_decl(&gpu_api()).unwrap();
        let generated = generate_rayzor(Some(dec_dir), &browser_idl()).unwrap();
        assert!(!generated.contains("caribou_abi"));
        assert!(generated.contains("xidl_gpu_device_create_buffer"));
        assert!(generated.contains("\"gpu::GpuDevice\""));
        assert!(generated.contains("pub fn xidl_runtime_symbols"));
    }

    #[test]
    fn the_complete_hashlink_model_emits_ash_future_primitives() {
        // copy declaration to temp dir
        let dec_dir = gpu_decl(&gpu_api()).unwrap();
        let generated = generate_hashlink(Some(dec_dir), &browser_idl()).unwrap();
        assert!(generated.contains("hlp_gpu_instance_request_adapter"));
        assert!(generated.contains("Xash_future_"));
        // Records are xgpu.hdll's abstracts, as the Haxe surface names them.
        assert!(generated.contains("Xxgpu_GpuDeviceDescriptor_"));
        let files = _haxe(haxe::Runtime::HashLink).unwrap();
        let device = &files.iter().find(|f| f.path == "gpu/GpuDevice.hx").unwrap().source;
        assert!(device.contains("@:hlNative(\"xgpu\", \"gpu_device_create_buffer\")"));
        assert!(generated.contains("runtime :: Managed < GpuDeviceDescriptor >"));
        assert!(generated.contains("value . into_ucs2"));
    }

    #[test]
    fn rayzor_can_supply_a_resource_wrapper_for_runtime_extensions() {
        // copy declaration to temp dir
        let dec_dir = gpu_decl(&gpu_api()).unwrap();
        std::fs::write(
            &dec_dir,
            r#"
                trait GpuBuffer {
                    #[native(buffer_destroy)] fn destroy(this: &GpuBuffer);
                }
                trait GpuDevice {
                    #[native(buffer_create)] fn buffer(this: &GpuDevice) -> Box<GpuBuffer>;
                }
            "#,
        )
        .map_err(|e| format!("failed to write gpu.api.rs: {e}"))
        .unwrap();

        let generated =
            x_idl::generate_rayzor_with_resources("gpu", Some(dec_dir), "", &["GpuBuffer"])
                .unwrap();
        assert!(!generated.contains("pub struct GpuBuffer"));
        assert!(generated.contains("impl GpuBuffer"));
        assert!(generated.contains("GpuBuffer :: from_handle (value)"));
        assert!(generated.contains("this . handle"));
    }

    #[test]
    fn a_web_backend_converts_records_to_the_wires_dictionaries() {
        let idl = r#"
          enum GPUFilterMode { "nearest", "linear" };
          enum GPUAutoLayoutMode { "auto" };
          interface GPUBuffer {};
          interface GPUSampler {};
          interface GPUPipelineLayout {};
          interface GPUDevice {
            undefined make(GPUThing descriptor);
            undefined lay(GPULaid descriptor);
          };
          typedef (GPUSampler or GPUBuffer or GPUBinding) GPUResource;
          dictionary GPUBinding { required GPUBuffer buffer; GPUSize64 size; };
          typedef [EnforceRange] unsigned long long GPUSize64;
          dictionary GPUThing {
            USVString label = "";
            required GPUSize64 size;
            GPUFilterMode filter = "nearest";
            sequence<GPUResource> resources = [];
          };
          dictionary GPULaid { required (GPUPipelineLayout or GPUAutoLayoutMode) layout; };
        "#;
        let declaration = r#"
          #[idl("GPUFilterMode")] enum Filter { #[extension] Cubic }
          #[idl("GPUBuffer")] trait Buffer {}
          #[idl("GPUSampler")] trait Sampler {}
          #[idl("GPUPipelineLayout")] trait Layout {}
          #[idl("GPUDevice")] trait Device {}
          #[idl("GPUBinding")] struct Binding {}
          #[idl("GPUResource")]
          enum Resource { Sampler(Sampler), Buffer(Buffer), Binding(Binding) }
          #[idl("GPUThing")] struct Thing { #[extension] native: Option<i32> }
          #[idl("GPULaid")] struct Laid { layout: Option<Layout> }
        "#;
        let declaration_path = temp_dir().join("gpu.api.rs");
        std::fs::write(&declaration_path, declaration)
            .map_err(|e| format!("failed to write gpu.api.rs: {e}"))
            .unwrap();
        let web = "pub fn laid_layout() {}";
        let generated = x_idl::web_backend("gpu", Some(declaration_path), idl, web).unwrap();
        syn::parse_file(&generated).unwrap();
        let flat = generated.replace(' ', "");
        for expected in [
            "implcrate::Thing{",
            "fnwire(&self)->Result<crate::wire::GPUThing,String>",
            "ifself.native.is_some(){returnErr(\"`Thing.native`isnotavailableontheweb\".to_owned());}",
            "label:match&self.label{Some(x)=>Some(x.get().as_str().to_owned()),None=>None}",
            "size:*(&self.size)asu64",
            "filter:match&self.filter{Some(x)=>Some(crate::wire::GPUFilterMode::from_index(*xasu32)",
            "crate::Resource::Buffer(x)=>crate::wire::GPUResource::GPUBuffer(crate::wire::Handle(*xasu32))",
            "crate::Resource::Binding(x)=>crate::wire::GPUResource::GPUBinding(x.wire()?)",
            "layout:crate::web::laid_layout(&self.layout)?",
        ] {
            assert!(flat.contains(expected), "{expected} in {generated}");
        }
    }
}
