use std::path::PathBuf;

fn main() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let runtime = match args.next().and_then(|v| v.into_string().ok()).as_deref() {
        Some("hashlink") | Some("ash") => xgpu_bindgen::haxe::Runtime::HashLink,
        Some("rayzor") => xgpu_bindgen::haxe::Runtime::Rayzor,
        _ => return Err("usage: xgpu-haxe <hashlink|ash|rayzor> <output-directory>".into()),
    };
    let root = PathBuf::from(
        args.next()
            .ok_or("usage: xgpu-haxe <hashlink|ash|rayzor> <output-directory>")?,
    );
    if args.next().is_some() {
        return Err("usage: xgpu-haxe <hashlink|ash|rayzor> <output-directory>".into());
    }
    for file in xgpu_bindgen::haxe(runtime)? {
        let path = root.join(file.path);
        std::fs::create_dir_all(path.parent().expect("generated file has a parent"))
            .map_err(|e| e.to_string())?;
        std::fs::write(path, file.source).map_err(|e| e.to_string())?;
    }
    Ok(())
}
