//! Conventional Haxe externs generated from the same declaration as the
//! Caribou plugin. Runtime-specific annotations and future carriers are the
//! only differences between targets.

use std::collections::HashMap;

use quote::ToTokens;
use syn::{FnArg, GenericArgument, Item, PathArguments, ReturnType, TraitItem, Type};

/// The native binding convention an extern set targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Runtime {
    /// HashLink HDLL symbols loaded from `xgpu`; Promise results use Ash's
    /// externally completable Future carrier.
    HashLink,
    /// Rayzor package methods and `rayzor.concurrent.Future<T>`.
    Rayzor,
}

/// One generated source file, relative to a Haxe class path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct File {
    pub path: String,
    pub source: String,
}

/// Emit the conventional Haxe extern surface for a runtime.
pub fn generate(
    namespace: &str,
    declaration: &str,
    webidl: &str,
    runtime: Runtime,
) -> Result<Vec<File>, String> {
    let file = syn::parse_file(declaration).map_err(|e| e.to_string())?;
    let schema_namespace = namespace.rsplit('.').next().unwrap_or(namespace);
    let (_, _, plugin) = super::generate_parts(
        schema_namespace,
        declaration,
        webidl,
        super::RustTarget::Caribou,
        &std::collections::HashSet::new(),
    )?;
    let records: HashMap<_, _> = plugin
        .records
        .iter()
        .map(|r| (r.class.to_string(), r))
        .collect();
    let mut out = Vec::new();

    if runtime == Runtime::HashLink {
        out.push(source(
            namespace,
            "XgpuBytes",
            r#"@:noCompletion
class XgpuBytes {
	public static function take(value:hl.Abstract<"xgpu_buffer_result">):haxe.io.Bytes {
		if (value == null) return null;
		var out = haxe.io.Bytes.alloc(XgpuBytesNative.length(value));
		XgpuBytesNative.copy(value, out);
		return out;
	}
}

private extern class XgpuBytesNative {
	@:hlNative("xgpu", "buffer_result_len")
	public static function length(value:hl.Abstract<"xgpu_buffer_result">):Int;
	@:hlNative("xgpu", "buffer_result_copy")
	public static function copy(value:hl.Abstract<"xgpu_buffer_result">, out:haxe.io.Bytes):Void;
}
"#
            .to_owned(),
        ));
    }

    for item in file.items {
        match item {
            Item::Enum(item) => {
                if plugin.unions.contains_key(&item.ident.to_string()) {
                    continue;
                }
                let name = item.ident.to_string();
                let mut variants = Vec::new();
                let imported = super::idl_name(&item.attrs)?;
                if let Some(source) = imported {
                    let model = super::idl::parse(webidl)?;
                    if let Some(values) = model.enums.get(&source) {
                        variants.extend(
                            values
                                .iter()
                                .enumerate()
                                .map(|(i, v)| (super::pascal(v), Some(i as i32))),
                        );
                    } else if let Some(interface) = model.interface(&source) {
                        variants.extend(
                            interface
                                .attributes
                                .iter()
                                .filter(|a| a.readonly)
                                .enumerate()
                                .map(|(i, a)| (super::pascal(&a.name), Some(i as i32))),
                        );
                    } else {
                        return Err(format!("WebIDL enum or catalog {source} was not found"));
                    }
                }
                for variant in item.variants {
                    let value = variant
                        .discriminant
                        .as_ref()
                        .and_then(|(_, e)| super::discriminant(e));
                    variants.push((variant.ident.to_string(), value));
                }
                let mut next = 0;
                let body = variants
                    .into_iter()
                    .map(|(name, explicit)| {
                        let value = explicit.unwrap_or(next);
                        next = value.saturating_add(1);
                        format!("\tvar {name} = {value};")
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                out.push(source(
                    namespace,
                    &name,
                    format!("enum abstract {name}(Int) from Int to Int {{\n{body}\n}}\n"),
                ));
            }
            Item::Mod(item) => {
                let Some((_, items)) = item.content else {
                    continue;
                };
                let name = item.ident.to_string();
                let mut constants = Vec::new();
                if let Some(imported) = super::idl_name(&item.attrs)? {
                    let tokens = super::tokens(webidl)?;
                    let body = super::body(&tokens, "namespace", &imported)?;
                    for statement in body.split(|token| token == ";").filter(|s| !s.is_empty()) {
                        if statement.len() != 5
                            || statement[0] != "const"
                            || statement[3] != "="
                        {
                            return Err(format!("unsupported constant in {imported}"));
                        }
                        constants.push(format!(
                            "\tpublic static inline var {}:Int = {};",
                            statement[2], statement[4]
                        ));
                    }
                }
                constants.extend(items.into_iter().filter_map(|item| match item {
                    Item::Const(c) => Some(format!(
                        "\tpublic static inline var {}:Int = {};",
                        c.ident,
                        c.expr.to_token_stream()
                    )),
                    _ => None,
                }));
                let constants = constants.join("\n");
                out.push(source(
                    namespace,
                    &name,
                    format!("class {name} {{\n{constants}\n}}\n"),
                ));
            }
            Item::Struct(item) => {
                let name = item.ident.to_string();
                let record = records
                    .get(&name)
                    .ok_or_else(|| format!("record {name} was not described"))?;
                if runtime == Runtime::HashLink {
                    out.push(hashlink_record(namespace, &name, record, &plugin)?);
                    continue;
                }
                let mut required = Vec::new();
                let mut methods = Vec::new();
                for (field, ty, _) in &record.fields {
                    let field = field.to_string().trim_start_matches("r#").to_owned();
                    if let Some((key, value)) = pair(ty, "Map") {
                        let method = format!("add{}", super::pascal(&field));
                        methods.push(method_line(
                            runtime,
                            &name,
                            &method,
                            &format!(
                                "key:{}, value:{}",
                                hx_type(&key, runtime)?,
                                hx_type(&value, runtime)?
                            ),
                            "Void",
                        ));
                    } else {
                        let (container, mut value) = if let Some(value) = one(ty, "Option") {
                            ("option", value)
                        } else if let Some(value) = one(ty, "Vec") {
                            ("sequence", value)
                        } else {
                            ("required", ty.clone())
                        };
                        if container == "sequence" {
                            value = one(&value, "Option").unwrap_or(value);
                        }
                        if let Some(alternatives) =
                            plugin.unions.get(&simple_name(&value).unwrap_or_default())
                        {
                            for (variant, ty, _) in alternatives {
                                let method = if container == "sequence" {
                                    format!("add{}{variant}", super::pascal(&field))
                                } else {
                                    format!("{field}{variant}")
                                };
                                methods.push(method_line(
                                    runtime,
                                    &name,
                                    &method,
                                    &format!("value:{}", hx_type(ty, runtime)?),
                                    "Void",
                                ));
                            }
                        } else if container == "sequence" {
                            let method = format!("add{}", super::pascal(&field));
                            methods.push(method_line(
                                runtime,
                                &name,
                                &method,
                                &format!("value:{}", hx_type(&value, runtime)?),
                                "Void",
                            ));
                        } else if container == "option" {
                            methods.push(method_line(
                                runtime,
                                &name,
                                &field,
                                &format!("value:{}", hx_type(&value, runtime)?),
                                "Void",
                            ));
                        } else {
                            required.push(format!("{field}:{}", hx_type(&value, runtime)?));
                        }
                    }
                }
                let constructor = native(runtime, &name, "new");
                let mut body = format!(
                    "{}extern class {name} {{\n\t{constructor}\n\tpublic function new({});",
                    class_annotation(runtime, namespace, &name),
                    required.join(", ")
                );
                if !methods.is_empty() {
                    body.push('\n');
                    body.push_str(&methods.join("\n"));
                }
                body.push_str("\n}\n");
                out.push(source(namespace, &name, body));
            }
            Item::Trait(item) => {
                let name = item.ident.to_string();
                if runtime == Runtime::HashLink {
                    out.push(hashlink_resource(namespace, &name, &item)?);
                    continue;
                }
                let mut methods = Vec::new();
                for entry in item.items {
                    let TraitItem::Fn(method) = entry else {
                        continue;
                    };
                    let rust_name = method.sig.ident.to_string();
                    let mut args = Vec::new();
                    let mut instance = false;
                    for (at, arg) in method.sig.inputs.iter().enumerate() {
                        let FnArg::Typed(arg) = arg else { continue };
                        let syn::Pat::Ident(param) = &*arg.pat else {
                            continue;
                        };
                        if at == 0
                            && param.ident == "this"
                            && reference_name(&arg.ty).as_deref() == Some(name.as_str())
                        {
                            instance = true;
                            continue;
                        }
                        args.push(format!("{}:{}", param.ident, hx_type(&arg.ty, runtime)?));
                    }
                    let ret = match &method.sig.output {
                        ReturnType::Default => "Void".to_owned(),
                        ReturnType::Type(_, ty) => hx_type(ty, runtime)?,
                    };
                    let annotation = native(runtime, &name, &rust_name);
                    if rust_name == "new" {
                        methods.push(format!(
                            "\t{annotation}\n\tpublic function new({});",
                            args.join(", ")
                        ));
                    } else {
                        let static_ = if instance { "" } else { "static " };
                        methods.push(format!(
                            "\t{annotation}\n\tpublic {static_}function {rust_name}({}):{ret};",
                            args.join(", ")
                        ));
                    }
                }
                let body = format!(
                    "{}extern class {name} {{\n{}\n}}\n",
                    class_annotation(runtime, namespace, &name),
                    methods.join("\n")
                );
                out.push(source(namespace, &name, body));
            }
            _ => {}
        }
    }
    Ok(out)
}

fn hashlink_record(
    namespace: &str,
    name: &str,
    record: &super::convert::Record,
    plugin: &super::convert::Plugin,
) -> Result<File, String> {
    let mut required = Vec::new();
    let mut methods: Vec<(String, Vec<(String, Type)>)> = Vec::new();
    for (field, ty, _) in &record.fields {
        let field = field.to_string().trim_start_matches("r#").to_owned();
        if let Some((key, value)) = pair(ty, "Map") {
            methods.push((
                format!("add{}", super::pascal(&field)),
                vec![("key".into(), key), ("value".into(), value)],
            ));
            continue;
        }
        let (container, mut value) = if let Some(value) = one(ty, "Option") {
            ("option", value)
        } else if let Some(value) = one(ty, "Vec") {
            ("sequence", value)
        } else {
            ("required", ty.clone())
        };
        if container == "sequence" {
            value = one(&value, "Option").unwrap_or(value);
        }
        if let Some(alternatives) = plugin.unions.get(&simple_name(&value).unwrap_or_default()) {
            for (variant, ty, _) in alternatives {
                let method = if container == "sequence" {
                    format!("add{}{variant}", super::pascal(&field))
                } else {
                    format!("{field}{variant}")
                };
                methods.push((method, vec![("value".into(), ty.clone())]));
            }
        } else if container == "sequence" {
            methods.push((
                format!("add{}", super::pascal(&field)),
                vec![("value".into(), value)],
            ));
        } else if container == "option" {
            methods.push((field, vec![("value".into(), value)]));
        } else {
            required.push((field, value));
        }
    }
    let abstract_ty = format!("hl.Abstract<\"xgpu_{name}\">");
    let args = haxe_args(&required, Runtime::HashLink)?;
    let names = required
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let mut public = format!(
        "abstract {name}({abstract_ty}) {{\n\tpublic inline function new({args}) this = {name}Native.create({names});"
    );
    let mut native_class = format!(
        "private extern class {name}Native {{\n\t{}\n\tpublic static function create({args}):{abstract_ty};",
        native(Runtime::HashLink, name, "new"),
    );
    for (method, params) in methods {
        let args = haxe_args(&params, Runtime::HashLink)?;
        let names = params
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let comma = if names.is_empty() { "" } else { ", " };
        public.push_str(&format!(
            "\n\tpublic inline function {method}({args}):Void {name}Native.{method}(this{comma}{names});"
        ));
        native_class.push_str(&format!(
            "\n\t{}\n\tpublic static function {method}(self:{abstract_ty}{comma}{args}):Void;",
            native(Runtime::HashLink, name, &method),
        ));
    }
    public.push_str("\n}\n\n");
    native_class.push_str("\n}\n");
    Ok(source(namespace, name, format!("{public}{native_class}")))
}

fn hashlink_resource(namespace: &str, name: &str, item: &syn::ItemTrait) -> Result<File, String> {
    let mut public = format!("abstract {name}(Int) from Int to Int {{");
    let mut native_class = format!("private extern class {name}Native {{");
    for entry in &item.items {
        let TraitItem::Fn(method) = entry else {
            continue;
        };
        let rust_name = method.sig.ident.to_string();
        let mut params = Vec::new();
        let mut instance = false;
        for (at, arg) in method.sig.inputs.iter().enumerate() {
            let FnArg::Typed(arg) = arg else { continue };
            let syn::Pat::Ident(param) = &*arg.pat else {
                continue;
            };
            if at == 0 && param.ident == "this" && reference_name(&arg.ty).as_deref() == Some(name)
            {
                instance = true;
                continue;
            }
            params.push((param.ident.to_string(), (*arg.ty).clone()));
        }
        let args = haxe_args(&params, Runtime::HashLink)?;
        let names = params
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let ret_ty = match &method.sig.output {
            ReturnType::Default => syn::parse_quote!(()),
            ReturnType::Type(_, ty) => (**ty).clone(),
        };
        let ret = hx_type(&ret_ty, Runtime::HashLink)?;
        let native_ret = match simple_name(&ret_ty).as_deref() {
            Some("Text") => "hl.Bytes".to_owned(),
            Some("Buffer") => "hl.Abstract<\"xgpu_buffer_result\">".to_owned(),
            _ => ret.clone(),
        };
        let native_args = if instance {
            if args.is_empty() {
                "self:Int".to_owned()
            } else {
                format!("self:Int, {args}")
            }
        } else {
            args.clone()
        };
        let native_method = if rust_name == "new" {
            "create"
        } else {
            &rust_name
        };
        native_class.push_str(&format!(
            "\n\t{}\n\tpublic static function {native_method}({native_args}):{native_ret};",
            native(Runtime::HashLink, name, &rust_name),
        ));
        let call_args = if instance {
            if names.is_empty() {
                "this".to_owned()
            } else {
                format!("this, {names}")
            }
        } else {
            names
        };
        let call = format!("{name}Native.{native_method}({call_args})");
        let body = match simple_name(&ret_ty).as_deref() {
            Some("Text") => format!(
                "{{ var value = {call}; return value == null ? null : @:privateAccess String.fromUCS2(value); }}"
            ),
            Some("Buffer") => format!("return XgpuBytes.take({call})"),
            _ if matches!(&ret_ty, Type::Tuple(tuple) if tuple.elems.is_empty()) => {
                format!("{call}")
            }
            _ => format!("return {call}"),
        };
        if rust_name == "new" {
            public.push_str(&format!(
                "\n\tpublic inline function new({args}) this = {call};"
            ));
        } else {
            let static_ = if instance { "" } else { "static " };
            public.push_str(&format!(
                "\n\tpublic {static_}inline function {rust_name}({args}):{ret} {body};"
            ));
        }
    }
    public.push_str("\n}\n\n");
    native_class.push_str("\n}\n");
    Ok(source(namespace, name, format!("{public}{native_class}")))
}

fn haxe_args(args: &[(String, Type)], runtime: Runtime) -> Result<String, String> {
    args.iter()
        .map(|(name, ty)| Ok(format!("{name}:{}", hx_type(ty, runtime)?)))
        .collect::<Result<Vec<_>, String>>()
        .map(|args| args.join(", "))
}

fn source(namespace: &str, name: &str, body: String) -> File {
    File {
        path: format!("{}/{name}.hx", namespace.replace('.', "/")),
        source: format!(
            "// Generated by xgpu-bindgen. Do not edit by hand.\npackage {namespace};\n\n{body}"
        ),
    }
}

fn method_line(runtime: Runtime, class: &str, method: &str, args: &str, ret: &str) -> String {
    format!(
        "\t{}\n\tpublic function {method}({args}):{ret};",
        native(runtime, class, method)
    )
}

fn class_annotation(runtime: Runtime, namespace: &str, class: &str) -> String {
    match runtime {
        Runtime::HashLink => String::new(),
        Runtime::Rayzor => format!("@:native(\"{}::{class}\")\n", namespace.replace('.', "::")),
    }
}

fn native(runtime: Runtime, class: &str, method: &str) -> String {
    let symbol = format!("{}_{}", snake(class), snake(method));
    match runtime {
        Runtime::HashLink => format!("@:hlNative(\"xgpu\", \"{symbol}\")"),
        Runtime::Rayzor => format!("@:native(\"xgpu_{symbol}\")"),
    }
}

pub(crate) fn snake(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_uppercase() && i != 0 {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

fn hx_type(ty: &Type, runtime: Runtime) -> Result<String, String> {
    if let Type::Reference(reference) = ty {
        return hx_type(&reference.elem, runtime);
    }
    if let Type::Tuple(tuple) = ty
        && tuple.elems.is_empty()
    {
        return Ok("Void".into());
    }
    if let Some(inner) = one(ty, "Option") {
        return Ok(format!("Null<{}>", hx_type(&inner, runtime)?));
    }
    if let Some(inner) = one(ty, "Vec") {
        return Ok(format!("Array<{}>", hx_type(&inner, runtime)?));
    }
    if let Some(inner) = one(ty, "Enum").or_else(|| one(ty, "Box")) {
        return hx_type(&inner, runtime);
    }
    if let Some(inner) = one(ty, "Future") {
        let inner = hx_type(&inner, runtime)?;
        return Ok(match runtime {
            Runtime::HashLink => format!("ash.Future<{inner}>"),
            Runtime::Rayzor => format!("rayzor.concurrent.Future<{inner}>"),
        });
    }
    if let Some((key, value)) = pair(ty, "Map") {
        return Ok(format!(
            "Map<{}, {}>",
            hx_type(&key, runtime)?,
            hx_type(&value, runtime)?
        ));
    }
    let name = simple_name(ty)
        .ok_or_else(|| format!("unsupported Haxe type: {}", ty.to_token_stream()))?;
    Ok(match name.as_str() {
        "i32" | "u32" => "Int".into(),
        "i64" | "u64" => "haxe.Int64".into(),
        "f32" | "f64" => "Float".into(),
        "bool" => "Bool".into(),
        "Text" => "String".into(),
        "Buffer" | "BufferMut" => "haxe.io.Bytes".into(),
        other => other.to_owned(),
    })
}

fn one(ty: &Type, name: &str) -> Option<Type> {
    let Type::Path(path) = ty else { return None };
    let segment = path.path.segments.last()?;
    if segment.ident != name {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    match args.args.first()? {
        GenericArgument::Type(ty) if args.args.len() == 1 => Some(ty.clone()),
        _ => None,
    }
}

fn pair(ty: &Type, name: &str) -> Option<(Type, Type)> {
    let Type::Path(path) = ty else { return None };
    let segment = path.path.segments.last()?;
    if segment.ident != name {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let mut types = args.args.iter().filter_map(|arg| match arg {
        GenericArgument::Type(ty) => Some(ty.clone()),
        _ => None,
    });
    Some((types.next()?, types.next()?))
}

fn simple_name(ty: &Type) -> Option<String> {
    let Type::Path(path) = ty else { return None };
    path.path.segments.last().map(|s| s.ident.to_string())
}

fn reference_name(ty: &Type) -> Option<String> {
    let Type::Reference(reference) = ty else {
        return None;
    };
    simple_name(&reference.elem)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promises_use_each_runtimes_future() {
        let api = r#"
            trait GpuAdapter {
                #[native(device_open)]
                fn requestDevice(this: &GpuAdapter) -> Future<GpuDevice>;
            }
            trait GpuDevice { #[native(device_destroy)] fn destroy(this: &GpuDevice); }
        "#;
        let rayzor = generate("rayzor.gpu", api, "", Runtime::Rayzor).unwrap();
        let adapter = &rayzor
            .iter()
            .find(|f| f.path.ends_with("GpuAdapter.hx"))
            .unwrap()
            .source;
        assert!(adapter.contains("rayzor.concurrent.Future<GpuDevice>"));
        assert!(adapter.contains("@:native(\"xgpu_gpu_adapter_request_device\")"));
        assert!(adapter.contains("package rayzor.gpu;"));
        assert!(adapter.contains("@:native(\"rayzor::gpu::GpuAdapter\")"));
        assert!(rayzor.iter().all(|f| f.path.starts_with("rayzor/gpu/")));

        let ash = generate("gpu", api, "", Runtime::HashLink).unwrap();
        let adapter = &ash
            .iter()
            .find(|f| f.path.ends_with("GpuAdapter.hx"))
            .unwrap()
            .source;
        assert!(adapter.contains("ash.Future<GpuDevice>"));
        assert!(adapter.contains("@:hlNative(\"xgpu\", \"gpu_adapter_request_device\")"));
    }

    #[test]
    fn the_complete_gpu_surface_emits_for_both_runtimes() {
        for runtime in [Runtime::HashLink, Runtime::Rayzor] {
            let files = crate::haxe(runtime).unwrap();
            assert!(files.len() > 100);
            assert!(files.iter().all(|file| file
                .source
                .starts_with("// Generated by xgpu-bindgen. Do not edit by hand.\n")));
            let adapter = files
                .iter()
                .find(|f| f.path.ends_with("GpuAdapter.hx"))
                .unwrap();
            assert!(adapter.source.contains("Future<GpuDevice>"));
            assert!(files.iter().any(|f| f.path.ends_with("TextureFormat.hx")));
            let map_mode = files
                .iter()
                .find(|f| f.path.ends_with("MapMode.hx"))
                .unwrap();
            assert!(map_mode.source.contains("READ:Int = 0x0001"));
            assert!(map_mode.source.contains("WRITE:Int = 0x0002"));
            let usage = files
                .iter()
                .find(|f| f.path.ends_with("BufferUsage.hx"))
                .unwrap();
            assert!(usage.source.contains("MAP_READ:Int = 0x0001"));
            assert!(usage.source.contains("BLAS_INPUT:Int = 1024"));
            if runtime == Runtime::Rayzor {
                assert!(adapter.path.starts_with("rayzor/gpu/"));
                assert!(adapter.source.contains("package rayzor.gpu;"));
            }
        }
    }
}
