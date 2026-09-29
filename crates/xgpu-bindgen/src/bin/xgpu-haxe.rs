use std::path::PathBuf;

fn main() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let target = args.next().and_then(|v| v.into_string().ok());
    let usage = "usage: xgpu-haxe <hashlink|ash|rayzor|javascript> <output-directory>";
    let root = PathBuf::from(args.next().ok_or(usage)?);
    if args.next().is_some() {
        return Err(usage.into());
    }
    let files = match target.as_deref() {
        Some("hashlink") | Some("ash") => {
            xgpu_bindgen::haxe(xgpu_bindgen::haxe::Runtime::HashLink)?
        }
        Some("rayzor") => xgpu_bindgen::haxe(xgpu_bindgen::haxe::Runtime::Rayzor)?,
        Some("javascript") | Some("js") => {
            xgpu_bindgen::haxe_js::generate(xgpu_bindgen::WEBGPU_IDL)?
        }
        _ => return Err(usage.into()),
    };
    for file in files {
        let path = root.join(file.path);
        std::fs::create_dir_all(path.parent().expect("generated file has a parent"))
            .map_err(|e| e.to_string())?;
        std::fs::write(path, file.source).map_err(|e| e.to_string())?;
    }
    Ok(())
}
