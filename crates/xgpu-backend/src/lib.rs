//! Installs xgpu's backend implementation into a runtime adapter's build.
//!
//! The installed source expects generated GPU model types at the crate root
//! and text, buffer, error, root, value, future, and host carriers under
//! `crate::runtime`. This keeps runtime ABI code in its adapter while every
//! adapter compiles the same GPU operations.

use std::io;
use std::path::Path;

const BACKEND: &str = include_str!("template/backend.rs");
pub const WEB: &str = include_str!("template/web.rs");

const MODULES: &[(&str, &str)] = &[
    ("bundles", include_str!("template/backend/bundles.rs")),
    ("caches", include_str!("template/backend/caches.rs")),
    ("copies", include_str!("template/backend/copies.rs")),
    (
        "diagnostics",
        include_str!("template/backend/diagnostics.rs"),
    ),
    ("external", include_str!("template/backend/external.rs")),
    ("info", include_str!("template/backend/info.rs")),
    ("mesh", include_str!("template/backend/mesh.rs")),
    ("native", include_str!("template/backend/native.rs")),
    ("queries", include_str!("template/backend/queries.rs")),
    (
        "ray_tracing",
        include_str!("template/backend/ray_tracing.rs"),
    ),
    ("render", include_str!("template/backend/render.rs")),
    ("shaders", include_str!("template/backend/shaders.rs")),
    ("surfaces", include_str!("template/backend/surfaces.rs")),
];

fn include_source(source: &str) -> String {
    source
        .lines()
        .map(|line| {
            line.strip_prefix("//!")
                .map_or(line.to_owned(), |doc| format!("//{doc}"))
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

/// Write the backend bundle beneath `out`, returning its main source path.
pub fn install(out: impl AsRef<Path>) -> io::Result<std::path::PathBuf> {
    let root = out.as_ref().join("xgpu_backend");
    let modules = root.join("backend");
    std::fs::create_dir_all(&modules)?;

    let mut backend =
        include_source(BACKEND).replace("#![allow(clippy::too_many_arguments)]\n", "");
    for (name, source) in MODULES {
        let declaration = format!("mod {name};");
        let included = format!(
            "mod {name} {{ include!(concat!(env!(\"OUT_DIR\"), \"/xgpu_backend/backend/{name}.rs\")); }}"
        );
        if !backend.contains(&declaration) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("backend no longer declares {name}"),
            ));
        }
        backend = backend.replace(&declaration, &included);
        std::fs::write(modules.join(format!("{name}.rs")), include_source(source))?;
    }
    let main = root.join("backend.rs");
    std::fs::write(&main, backend)?;
    std::fs::write(root.join("web.rs"), include_source(WEB))?;
    Ok(main)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installed_backend_is_self_contained_and_runtime_neutral() {
        let out = std::env::temp_dir().join(format!("xgpu-backend-{}", std::process::id()));
        let main = install(&out).unwrap();
        let source = std::fs::read_to_string(main).unwrap();
        assert!(!source.contains("caribou_abi"));
        for (name, _) in MODULES {
            assert!(source.contains(&format!("/xgpu_backend/backend/{name}.rs")));
        }
        std::fs::remove_dir_all(out).unwrap();
    }
}
