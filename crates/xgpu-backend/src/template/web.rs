//! The GPU plugin in a browser: each backend function a program calls is
//! encoded onto the wire to the page's WebGPU (`crate::wire`), which the
//! GPU agent decodes and runs. Functions this module does not define raise
//! that they are not available on the web (the generated `backend`).
//!
//! Commands collect in one batch, handed to the agent through the mailbox
//! when the program waits on the GPU (a request, a read) or presents a
//! frame, so a frame reaches the page whole. An upload is copied when it is
//! made and kept until its batch is sent. Objects are kept by the agent
//! under the plugin's own handles, so making one needs no round trip; the
//! GPU itself is handle 1 and the page's canvas handle 2, which no plugin
//! handle is. The program's world watches the count of replies the agent
//! has settled, and settles their futures on the program's own thread.

use std::sync::atomic::{AtomicI32, Ordering::SeqCst};
use std::sync::{LazyLock, Mutex};

use crate::runtime::{Buffer, BufferMut, ErrorKind, Future, Rooted, Text, Value, host};

use crate::handles::{Slab, kind_of};
use crate::types::Kind;
use crate::wire::{self, Handle, Mailbox};
use crate::{
    AttachmentView, GpuBindGroupDescriptor, GpuBufferDescriptor,
    GpuComputePassDescriptor, GpuComputePipelineDescriptor, GpuDeviceDescriptor, GpuExtent3D,
    GpuInstanceDescriptor, GpuRenderPassDescriptor, GpuRenderPipelineDescriptor,
    GpuRequestAdapterOptions, GpuShaderModuleDescriptor, GpuSurfaceConfiguration,
    GpuTexelCopyBufferInfo, GpuTexelCopyBufferLayout, GpuTexelCopyTextureInfo,
    GpuTextureDescriptor, GpuTextureViewDescriptor,
};

/// `navigator.gpu`, as the agent keeps it.
const GPU: Handle = Handle(1);

/// The page's canvas's WebGPU context, when the page gave the agent one.
const CANVAS: Handle = Handle(2);

/// That canvas itself, whose size is the drawing buffer's.
const CANVAS_ELEMENT: Handle = Handle(3);

/// Handles of objects the plugin makes and releases in one batch (passes,
/// command buffers, layouts it only borrows), below every plugin handle.
const TRANSIENT: u32 = 1 << 26;

/// How much a batch may hold before it goes out without a wait or a frame
/// asking for it.
const BATCH: usize = 1 << 20;

static MAILBOX: Mailbox = Mailbox::new();

/// A buffer's size and usage, which WebGPU keeps as attributes; held here
/// so reading them needs no round trip.
struct BufferEntry {
    size: i64,
    usage: i32,
}

/// An encoder, and what its next render pass attaches.
#[derive(Default)]
struct EncoderEntry {
    /// The open compute or render pass, or zero.
    compute: u32,
    render: u32,
    colour: Vec<(i32, [f64; 4])>,
    depth: Option<(i32, f64, i32)>,
}

/// The canvas's frame: its texture and the view the program draws to,
/// both released once presented.
#[derive(Default)]
struct SurfaceEntry {
    texture: u32,
    view: i32,
}

/// The formats a canvas takes, as the plugin's codes; the rest of what it
/// supports is the same for every canvas.
struct Capabilities {
    formats: Vec<i32>,
}

/// A render pipeline described step by step, as the native builder takes it.
#[derive(Default)]
struct Build {
    device: i32,
    shader: i32,
    vertex_entry: String,
    fragment_entry: String,
    layout: i32,
    buffers: Vec<wire::GPUVertexBufferLayout>,
    targets: Vec<wire::GPUColorTargetState>,
    depth: Option<wire::GPUDepthStencilState>,
    stencil: Option<(wire::GPUStencilFaceState, u32, u32)>,
    primitive: Option<wire::GPUPrimitiveState>,
}

/// What a popped error scope caught.
struct ErrorEntry {
    filter: i32,
    message: String,
}

struct State {
    commands: wire::Encoder,
    /// Uploads the batch's commands read, until it is sent.
    staged: Vec<Box<[u8]>>,
    /// Whether a frame is acquired and not yet presented.
    drawing: bool,
    /// Whether the host started the agent.
    agent: bool,
    next: u32,
    instances: Slab<()>,
    adapters: Slab<()>,
    /// Each device's queue, made once the device is.
    devices: Slab<AtomicI32>,
    queues: Slab<()>,
    buffers: Slab<BufferEntry>,
    shaders: Slab<()>,
    pipelines: Slab<()>,
    render_pipelines: Slab<()>,
    layouts: Slab<()>,
    bind_groups: Slab<()>,
    encoders: Slab<Mutex<EncoderEntry>>,
    bindings: Slab<Mutex<Vec<i32>>>,
    textures: Slab<()>,
    views: Slab<()>,
    surfaces: Slab<Mutex<SurfaceEntry>>,
    capabilities: Slab<Capabilities>,
    builders: Slab<Mutex<Build>>,
    samplers: Slab<()>,
    pipeline_layouts: Slab<()>,
    query_sets: Slab<()>,
    bundle_encoders: Slab<()>,
    bundles: Slab<()>,
    errors: Slab<ErrorEntry>,
    lost_infos: Slab<()>,
    compilations: Slab<()>,
    /// The error scopes the program has pushed, oldest first: each one's
    /// device and filter. Each device also has its own scopes beneath these
    /// (`CATCH_ALL`), which `device_take_error` reads.
    scopes: Vec<(i32, i32)>,
    /// What those catch-all scopes caught and no one has taken yet, with
    /// each one's device.
    uncaptured: Vec<(i32, String)>,
    /// Each compilation info's messages, once read.
    compiled: Vec<(i32, Vec<Message>)>,
    /// Each shader's compiler messages, read once: a browser may leave a
    /// second getCompilationInfo unanswered until the device has other work.
    shader_infos: Vec<(i32, Vec<Message>)>,
    pending: Vec<Pending>,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| {
    Mutex::new(State {
        commands: wire::Encoder::new(),
        staged: Vec::new(),
        drawing: false,
        agent: false,
        next: 2,
        instances: Slab::new(Kind::Instance),
        adapters: Slab::new(Kind::Adapter),
        devices: Slab::new(Kind::Device),
        queues: Slab::new(Kind::Queue),
        buffers: Slab::new(Kind::Buffer),
        shaders: Slab::new(Kind::Shader),
        pipelines: Slab::new(Kind::Pipeline),
        render_pipelines: Slab::new(Kind::Renderpipeline),
        layouts: Slab::new(Kind::BindGroupLayout),
        bind_groups: Slab::new(Kind::Bindgroup),
        encoders: Slab::new(Kind::Encoder),
        bindings: Slab::new(Kind::Bindings),
        textures: Slab::new(Kind::Texture),
        views: Slab::new(Kind::View),
        surfaces: Slab::new(Kind::Surface),
        capabilities: Slab::new(Kind::SurfaceCapabilities),
        builders: Slab::new(Kind::Builder),
        samplers: Slab::new(Kind::Sampler),
        pipeline_layouts: Slab::new(Kind::PipelineLayout),
        query_sets: Slab::new(Kind::QuerySet),
        bundle_encoders: Slab::new(Kind::BundleEncoder),
        bundles: Slab::new(Kind::Bundle),
        errors: Slab::new(Kind::Error),
        lost_infos: Slab::new(Kind::LostInfo),
        compilations: Slab::new(Kind::CompilationInfo),
        scopes: Vec::new(),
        uncaptured: Vec::new(),
        compiled: Vec::new(),
        shader_infos: Vec::new(),
        pending: Vec::new(),
    })
});

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl State {
    /// Hand the batch to the agent and wait until it has run it.
    fn flush(&mut self) {
        MAILBOX.send(&self.commands.bytes);
        self.commands.bytes.clear();
        self.staged.clear();
    }

    /// Send the batch if it has grown large and no frame is being drawn,
    /// whose commands must reach the agent together.
    fn flush_if_full(&mut self) {
        let drawing = self.drawing;
        let held = self.commands.bytes.len() + self.staged.iter().map(|b| b.len()).sum::<usize>();
        if held > BATCH && !drawing {
            self.flush();
        }
    }

    /// `bytes` copied until the batch is sent, as the wire reads them.
    fn stage(&mut self, bytes: &[u8]) -> wire::Bytes {
        let copy: Box<[u8]> = bytes.into();
        let staged = wire::Bytes {
            address: copy.as_ptr() as usize as u32,
            len: copy.len() as u32,
        };
        self.staged.push(copy);
        staged
    }

    fn transient(&mut self) -> Handle {
        self.next = if self.next + 1 >= TRANSIENT {
            3
        } else {
            self.next + 1
        };
        Handle(self.next)
    }

    /// A new handle of a kind the plugin keeps nothing for but the handle;
    /// 0 for any other kind, or when the kind is full.
    fn put(&mut self, kind: Kind) -> i32 {
        match kind {
            Kind::Shader => self.shaders.put(()),
            Kind::Pipeline => self.pipelines.put(()),
            Kind::Renderpipeline => self.render_pipelines.put(()),
            Kind::BindGroupLayout => self.layouts.put(()),
            Kind::Bindgroup => self.bind_groups.put(()),
            Kind::Texture => self.textures.put(()),
            Kind::View => self.views.put(()),
            Kind::Sampler => self.samplers.put(()),
            Kind::PipelineLayout => self.pipeline_layouts.put(()),
            Kind::QuerySet => self.query_sets.put(()),
            Kind::BundleEncoder => self.bundle_encoders.put(()),
            Kind::Bundle => self.bundles.put(()),
            Kind::LostInfo => self.lost_infos.put(()),
            Kind::CompilationInfo => self.compilations.put(()),
            _ => 0,
        }
    }

    /// Whether `h` is a live handle of the kind it carries.
    fn has(&self, h: i32) -> bool {
        let k = kind_of(h);
        let is = |kind: Kind| k == kind as i32;
        if is(Kind::Instance) {
            self.instances.get(h).is_some()
        } else if is(Kind::Adapter) {
            self.adapters.get(h).is_some()
        } else if is(Kind::Device) {
            self.devices.get(h).is_some()
        } else if is(Kind::Queue) {
            self.queues.get(h).is_some()
        } else if is(Kind::Buffer) {
            self.buffers.get(h).is_some()
        } else if is(Kind::Shader) {
            self.shaders.get(h).is_some()
        } else if is(Kind::Pipeline) {
            self.pipelines.get(h).is_some()
        } else if is(Kind::Renderpipeline) {
            self.render_pipelines.get(h).is_some()
        } else if is(Kind::BindGroupLayout) {
            self.layouts.get(h).is_some()
        } else if is(Kind::Bindgroup) {
            self.bind_groups.get(h).is_some()
        } else if is(Kind::Encoder) {
            self.encoders.get(h).is_some()
        } else if is(Kind::Bindings) {
            self.bindings.get(h).is_some()
        } else if is(Kind::Texture) {
            self.textures.get(h).is_some()
        } else if is(Kind::View) {
            self.views.get(h).is_some()
        } else if is(Kind::Surface) {
            self.surfaces.get(h).is_some()
        } else if is(Kind::SurfaceCapabilities) {
            self.capabilities.get(h).is_some()
        } else if is(Kind::Builder) {
            self.builders.get(h).is_some()
        } else if is(Kind::Sampler) {
            self.samplers.get(h).is_some()
        } else if is(Kind::PipelineLayout) {
            self.pipeline_layouts.get(h).is_some()
        } else if is(Kind::QuerySet) {
            self.query_sets.get(h).is_some()
        } else if is(Kind::BundleEncoder) {
            self.bundle_encoders.get(h).is_some()
        } else if is(Kind::Bundle) {
            self.bundles.get(h).is_some()
        } else if is(Kind::Error) {
            self.errors.get(h).is_some()
        } else if is(Kind::LostInfo) {
            self.lost_infos.get(h).is_some()
        } else if is(Kind::CompilationInfo) {
            self.compilations.get(h).is_some()
        } else {
            false
        }
    }

    /// Drop the plugin's handle `h` from the table of the kind it carries.
    fn remove(&mut self, h: i32) {
        let k = kind_of(h);
        let is = |kind: Kind| k == kind as i32;
        if is(Kind::Shader) {
            self.shaders.remove(h)
        } else if is(Kind::Pipeline) {
            self.pipelines.remove(h)
        } else if is(Kind::Renderpipeline) {
            self.render_pipelines.remove(h)
        } else if is(Kind::BindGroupLayout) {
            self.layouts.remove(h)
        } else if is(Kind::Bindgroup) {
            self.bind_groups.remove(h)
        } else if is(Kind::Texture) {
            self.textures.remove(h)
        } else if is(Kind::View) {
            self.views.remove(h)
        } else if is(Kind::Sampler) {
            self.samplers.remove(h)
        } else if is(Kind::PipelineLayout) {
            self.pipeline_layouts.remove(h)
        } else if is(Kind::QuerySet) {
            self.query_sets.remove(h)
        } else if is(Kind::BundleEncoder) {
            self.bundle_encoders.remove(h)
        } else if is(Kind::Bundle) {
            self.bundles.remove(h)
        } else if is(Kind::Error) {
            self.errors.remove(h)
        } else if is(Kind::LostInfo) {
            self.lost_infos.remove(h)
        } else if is(Kind::CompilationInfo) {
            self.compilations.remove(h)
        }
    }
}

fn handle(h: i32) -> Handle {
    Handle(h as u32)
}

/// A reply record: the agent's answer to one command, in the program's
/// memory, and the buffer that takes what it carries.
#[repr(C)]
struct Reply {
    state: AtomicI32,
    len: u32,
    address: u32,
    cap: u32,
}

impl Reply {
    fn new(address: *mut u8, cap: usize) -> Reply {
        Reply {
            state: AtomicI32::new(0),
            len: 0,
            address: address as usize as u32,
            cap: cap as u32,
        }
    }

    fn at(&self) -> u32 {
        self as *const Reply as usize as u32
    }
}

/// A promise's reply, with room for a rejection's message.
struct Promise {
    reply: Reply,
    message: [u8; 1024],
}

impl Promise {
    fn new() -> Box<Promise> {
        let mut promise = Box::new(Promise {
            reply: Reply::new(std::ptr::null_mut(), 0),
            message: [0; 1024],
        });
        promise.reply = Reply::new(promise.message.as_mut_ptr(), promise.message.len());
        promise
    }

    /// What a resolved promise's reply carries.
    fn value(&self) -> &[u8] {
        let len = (self.reply.len as usize).min(self.message.len());
        &self.message[..len]
    }

    fn message(&self) -> String {
        match self.reply.state.load(SeqCst) {
            3 => "the GPU's error message is too long to carry".to_owned(),
            _ => {
                let len = (self.reply.len as usize).min(self.message.len());
                String::from_utf8_lossy(&self.message[..len]).into_owned()
            }
        }
    }
}

/// A future waiting on a promise, and what it resolves with.
struct Pending {
    promise: Box<Promise>,
    waiting: Waiting,
}

enum Waiting {
    Adapter(Rooted<Future<crate::GpuAdapter>>, i32),
    Device(Rooted<Future<crate::GpuDevice>>, i32),
    Done(Rooted<Future<()>>),
    /// Run on the program's thread with whether the promise resolved.
    Then(Box<dyn FnOnce(bool, &Promise) + Send>),
}

/// Settle `waiting` from `promise` once the agent answers it.
fn wait(s: &mut State, promise: Box<Promise>, waiting: Waiting) {
    s.pending.push(Pending { promise, waiting });
}

/// The world's handler for the settled count.
unsafe extern "C" fn settled(_: *mut std::ffi::c_void) {
    settle();
}

/// Settle every future whose promise the agent has settled.
fn settle() {
    let settled: Vec<Pending> = {
        let mut s = state();
        let (done, waiting) = std::mem::take(&mut s.pending)
            .into_iter()
            .partition(|p| p.promise.reply.state.load(SeqCst) != 0);
        s.pending = waiting;
        done
    };
    for p in settled {
        let resolved = p.promise.reply.state.load(SeqCst) == 1;
        match p.waiting {
            Waiting::Adapter(future, adapter) if resolved => {
                if !future
                    .get()
                    .resolve_boxed(Box::new(crate::GpuAdapter { handle: adapter }))
                {
                    forget(|s| s.adapters.remove(adapter), adapter);
                }
            }
            Waiting::Device(future, device) if resolved => {
                // The queue came with the device; it is kept under a handle
                // of its own before the program can ask for it. The
                // catch-all scopes go in before any command can fail.
                {
                    let mut s = state();
                    let queue = s.queues.put(());
                    s.commands
                        .gpu_device_get_queue(handle(device), handle(queue));
                    if let Some(entry) = s.devices.get(device) {
                        entry.store(queue, SeqCst);
                    }
                    push_catch_all(&mut s, device);
                }
                if !future
                    .get()
                    .resolve_boxed(Box::new(crate::GpuDevice { handle: device }))
                {
                    unsafe { device_destroy(device) };
                }
            }
            Waiting::Done(future) if resolved => {
                future.get().resolve(Value::null());
            }
            Waiting::Then(then) => then(resolved, &p.promise),
            Waiting::Adapter(future, adapter) => {
                state().adapters.remove(adapter);
                future.get().reject(Text::new(&p.promise.message()).value());
            }
            Waiting::Device(future, device) => {
                state().devices.remove(device);
                future.get().reject(Text::new(&p.promise.message()).value());
            }
            Waiting::Done(future) => {
                future.get().reject(Text::new(&p.promise.message()).value());
            }
        }
    }
}

/// Drop the plugin's handle `h` and the agent's object under it.
fn forget(remove: impl FnOnce(&mut State), h: i32) {
    let mut s = state();
    remove(&mut s);
    s.commands.release(handle(h));
}

fn rejected<T>(message: &str) -> Future<T> {
    let future = Future::new();
    future.reject(Text::new(message).value());
    future
}

/// The first `len` bytes of `data`, checked.
fn bytes(data: &Buffer, len: i32) -> Option<&[u8]> {
    let Ok(len) = usize::try_from(len) else {
        host::raise(ErrorKind::Type, "negative byte length");
        return None;
    };
    if len > data.len() {
        host::raise(ErrorKind::Type, "byte length exceeds shared buffer");
        return None;
    }
    Some(unsafe { &data.as_slice()[..len] })
}

/// A descriptor as the wire's, or raised as what the web lacks.
fn converted<T>(descriptor: Result<T, String>) -> Option<T> {
    descriptor
        .map_err(|message| host::raise(ErrorKind::Runtime, &message))
        .ok()
}

fn unavailable<T>(what: &str) -> Result<T, String> {
    Err(format!("`{what}` is not available on the web"))
}

/// The plugin's power preference, whose codes are its own.
fn power_preference(power: i32) -> Option<wire::GPUPowerPreference> {
    match power {
        1 => Some(wire::GPUPowerPreference::HighPerformance),
        0 => Some(wire::GPUPowerPreference::LowPower),
        _ => None,
    }
}

/// `compatibleSurface` holds of every adapter: a page's one surface is its
/// canvas, which any WebGPU adapter presents to.
fn adapter_options(o: &GpuRequestAdapterOptions) -> wire::GPURequestAdapterOptions {
    wire::GPURequestAdapterOptions {
        feature_level: None,
        power_preference: o.powerPreference.and_then(power_preference),
        force_fallback_adapter: o.forceFallbackAdapter,
        xr_compatible: None,
    }
}

/// WebGPU's features and limits; wgpu's own, and its acceptances of
/// behaviour a browser does not have, are not.
fn device_descriptor(d: &GpuDeviceDescriptor) -> Result<wire::GPUDeviceDescriptor, String> {
    if !d.requiredNativeFeatures.is_empty() {
        return unavailable("GpuDeviceDescriptor.requiredNativeFeatures");
    }
    if !d.requiredNativeLimits.is_empty() {
        return unavailable("GpuDeviceDescriptor.requiredNativeLimits");
    }
    // Imported enums: a value's code is its index in the IDL.
    let features = d
        .requiredFeatures
        .iter()
        .map(|&f| {
            wire::GPUFeatureName::from_index(f as u32)
                .ok_or("a feature this browser's WebGPU lacks")
        })
        .collect::<Result<Vec<_>, _>>()?;
    // A limit is named as GPUSupportedLimits names its attribute.
    let limits = d
        .requiredLimits
        .iter()
        .map(|&(limit, value)| {
            let variant = crate::Limit::from_native(limit).ok_or("an unknown limit")?;
            let name = format!("{variant:?}");
            let mut chars = name.chars();
            let first = chars
                .next()
                .map(|c| c.to_ascii_lowercase())
                .unwrap_or_default();
            Ok((
                format!("{first}{}", chars.as_str()),
                wire::U64OrUndefined::U64(value.max(0) as u64),
            ))
        })
        .collect::<Result<Vec<_>, &str>>()?;
    Ok(wire::GPUDeviceDescriptor {
        label: None,
        required_features: Some(features),
        required_limits: Some(limits),
        default_queue: None,
    })
}

/// A browser checks every shader, so a check can be left on but not off.
fn shader_descriptor(
    d: &GpuShaderModuleDescriptor,
) -> Result<wire::GPUShaderModuleDescriptor, String> {
    for (check, name) in [
        (d.boundsChecks, "boundsChecks"),
        (d.forceLoopBounding, "forceLoopBounding"),
        (
            d.rayQueryInitializationTracking,
            "rayQueryInitializationTracking",
        ),
        (d.taskShaderDispatchTracking, "taskShaderDispatchTracking"),
        (
            d.meshShaderPrimitiveIndicesClamp,
            "meshShaderPrimitiveIndicesClamp",
        ),
        (d.intDivChecks, "intDivChecks"),
    ] {
        if check == Some(false) {
            return unavailable(&format!("GpuShaderModuleDescriptor.{name}(false)"));
        }
    }
    Ok(wire::GPUShaderModuleDescriptor {
        label: d.label.as_ref().map(|l| l.get().as_str().to_owned()),
        code: d.code.get().as_str().to_owned(),
        compilation_hints: None,
    })
}

// -- what generated natives call --------------------------------------------
//
// x-idl generates each native that is one WebIDL member as calls to these.
// A class is the member's WebIDL interface, or its result's.

/// How much a call answered at once may carry back.
const ANSWER: usize = 1 << 16;

/// The kind of handle the plugin keeps a WebIDL interface's objects under.
fn kind(class: &str) -> Option<Kind> {
    Some(match class {
        "GPUAdapter" => Kind::Adapter,
        "GPUDevice" => Kind::Device,
        "GPUQueue" => Kind::Queue,
        "GPUBuffer" => Kind::Buffer,
        "GPUTexture" => Kind::Texture,
        "GPUTextureView" => Kind::View,
        "GPUSampler" => Kind::Sampler,
        "GPUShaderModule" => Kind::Shader,
        "GPUBindGroup" => Kind::Bindgroup,
        "GPUBindGroupLayout" => Kind::BindGroupLayout,
        "GPUPipelineLayout" => Kind::PipelineLayout,
        "GPUComputePipeline" => Kind::Pipeline,
        "GPURenderPipeline" => Kind::Renderpipeline,
        "GPUCommandEncoder" => Kind::Encoder,
        "GPUQuerySet" => Kind::QuerySet,
        "GPURenderBundleEncoder" => Kind::BundleEncoder,
        "GPURenderBundle" => Kind::Bundle,
        "GPUError" => Kind::Error,
        "GPUDeviceLostInfo" => Kind::LostInfo,
        "GPUCompilationInfo" => Kind::CompilationInfo,
        _ => return None,
    })
}

/// Whether `h` is a live handle of `class`.
pub(crate) fn live(class: &str, h: i32) -> bool {
    kind(class).is_some_and(|k| kind_of(h) == k as i32) && state().has(h)
}

/// A new object of `class`, which `encode` makes under its handle; 0 when
/// none can be made.
pub(crate) fn make(class: &str, encode: impl FnOnce(&mut wire::Encoder, Handle)) -> i32 {
    let Some(kind) = kind(class) else {
        return 0;
    };
    let mut s = state();
    let h = s.put(kind);
    if h != 0 {
        encode(&mut s.commands, handle(h));
        s.flush_if_full();
    }
    h
}

/// A call with no result.
pub(crate) fn command(encode: impl FnOnce(&mut wire::Encoder)) {
    let mut s = state();
    encode(&mut s.commands);
    s.flush_if_full();
}

/// A call answered at once.
pub(crate) fn ask<T: wire::Decode>(encode: impl FnOnce(&mut wire::Encoder, u32)) -> Option<T> {
    answer(&mut state(), encode)
}

/// Send the batch with the call `encode` adds, and decode its reply.
fn answer<T: wire::Decode>(
    s: &mut State,
    encode: impl FnOnce(&mut wire::Encoder, u32),
) -> Option<T> {
    let mut bytes = vec![0u8; ANSWER];
    let reply = Reply::new(bytes.as_mut_ptr(), bytes.len());
    encode(&mut s.commands, reply.at());
    s.flush();
    if reply.state.load(SeqCst) != 1 {
        return None;
    }
    let len = (reply.len as usize).min(bytes.len());
    T::decode(&mut wire::Decoder::new(&bytes[..len]))
}

/// A call answered by a promise. The future settles on the program's
/// thread: `resolve` gives it the object of `class` the promise made, by
/// its handle, or it resolves null when there is no class.
pub(crate) fn promise<T: 'static>(
    class: Option<&'static str>,
    encode: impl FnOnce(&mut wire::Encoder, Option<Handle>, u32),
    resolve: fn(Future<T>, i32) -> bool,
) -> Future<T> {
    let mut s = state();
    let result = match class.map(kind) {
        None => None,
        Some(None) => return rejected("the web keeps no objects of this class"),
        Some(Some(kind)) => match s.put(kind) {
            0 => return rejected("no handle is free for the result"),
            h => Some(h),
        },
    };
    let promise = Promise::new();
    encode(&mut s.commands, result.map(handle), promise.reply.at());
    s.flush();
    let future = Future::new();
    let rooted = Rooted::new(future);
    let then = move |resolved: bool, p: &Promise| match (resolved, result) {
        (true, Some(h)) => {
            if !resolve(rooted.get(), h) {
                forget(|s| s.remove(h), h);
            }
        }
        (true, None) => {
            rooted.get().resolve(Value::null());
        }
        (false, h) => {
            if let Some(h) = h {
                forget(|s| s.remove(h), h);
            }
            rooted.get().reject(Text::new(&p.message()).value());
        }
    };
    wait(&mut s, promise, Waiting::Then(Box::new(then)));
    future
}

/// Drop `h`, a handle of `class`, and the agent's object under it.
pub(crate) fn release(class: &str, h: i32) {
    if live(class, h) {
        forget(|s| s.remove(h), h);
    }
}

/// Wait until the agent has answered each of `replies`.
fn answered(replies: &[&Reply]) {
    loop {
        let seen = MAILBOX.settled.load(SeqCst);
        if replies.iter().all(|r| r.state.load(SeqCst) != 0) {
            return;
        }
        MAILBOX.wait_settled(seen);
    }
}

// -- errors -----------------------------------------------------------------

/// The scopes under everything the program pushes, pushed in this order:
/// GPUErrorFilter's internal, out-of-memory, then validation on top.
const CATCH_ALL: [i32; 3] = [2, 1, 0];

fn push_catch_all(s: &mut State, device: i32) {
    for filter in CATCH_ALL {
        if let Some(filter) = wire::GPUErrorFilter::from_index(filter as u32) {
            s.commands
                .gpu_device_push_error_scope(handle(device), &filter);
        }
    }
}

/// Pop `device`'s innermost scope into `caught`, answered by the promise.
fn pop_scope(s: &mut State, device: i32) -> (Handle, Box<Promise>) {
    let caught = s.transient();
    let promise = Promise::new();
    s.commands
        .gpu_device_pop_error_scope(handle(device), caught, promise.reply.at());
    (caught, promise)
}

/// The message of the error a popped scope put under `caught`, which is
/// released; none when the scope caught nothing.
fn caught_message(s: &mut State, popped: &Promise, caught: Handle) -> Option<String> {
    let message = (popped.value().first() == Some(&1)).then(|| {
        answer::<String>(s, |c, at| c.gpu_error_get_message(caught, at)).unwrap_or_default()
    });
    s.commands.release(caught);
    message
}

pub unsafe fn error_scope_push(device: i32, filter: i32) {
    let Some(wired) = wire::GPUErrorFilter::from_index(filter as u32) else {
        host::raise(ErrorKind::Type, "an unknown error filter");
        return;
    };
    let mut s = state();
    if s.devices.get(device).is_none() {
        return;
    }
    s.scopes.push((device, filter));
    s.commands
        .gpu_device_push_error_scope(handle(device), &wired);
}

/// The scope's filter is the error's: WebGPU tells them apart by class.
pub unsafe fn error_scope_pop(device: i32) -> Future<crate::GpuError> {
    let mut s = state();
    if s.devices.get(device).is_none() {
        return rejected("device was destroyed");
    }
    let Some(at) = s.scopes.iter().rposition(|(owner, _)| *owner == device) else {
        return rejected("no error scope was pushed for this device");
    };
    let (_, filter) = s.scopes.remove(at);
    let (caught, promise) = pop_scope(&mut s, device);
    s.flush();
    let future = Future::new();
    let rooted = Rooted::new(future);
    let then = move |resolved: bool, p: &Promise| {
        if !resolved {
            state().commands.release(caught);
            rooted.get().reject(Text::new(&p.message()).value());
            return;
        }
        let message = caught_message(&mut state(), p, caught);
        let Some(message) = message else {
            rooted.get().resolve(Value::null());
            return;
        };
        let error = state().errors.put(ErrorEntry { filter, message });
        if !rooted
            .get()
            .resolve_boxed(Box::new(crate::GpuError { handle: error }))
        {
            state().errors.remove(error);
        }
    };
    wait(&mut s, promise, Waiting::Then(Box::new(then)));
    future
}

pub unsafe fn error_destroy(error: i32) {
    state().errors.remove(error);
}

pub unsafe fn error_filter(error: i32) -> i32 {
    state().errors.get(error).map_or(0, |e| e.filter)
}

pub unsafe fn error_message(error: i32) -> Text {
    match state().errors.get(error) {
        Some(e) => Text::new(&e.message),
        None => Text::NULL,
    }
}

/// The oldest error no scope of the program's caught, or null. The
/// device's catch-all scopes are popped, read and pushed again, which waits
/// for the GPU to have checked what was sent; they cannot be reached while
/// a scope of the program's is open on the device.
pub unsafe fn device_take_error(device: i32) -> Text {
    let mut s = state();
    if s.devices.get(device).is_none() {
        return Text::NULL;
    }
    if !s.scopes.iter().any(|(owner, _)| *owner == device) {
        let mut pops = Vec::new();
        for _ in CATCH_ALL {
            pops.push(pop_scope(&mut s, device));
        }
        push_catch_all(&mut s, device);
        s.flush();
        drop(s);
        answered(&pops.iter().map(|(_, p)| &p.reply).collect::<Vec<_>>());
        s = state();
        for (caught, p) in pops {
            if p.reply.state.load(SeqCst) != 1 {
                s.commands.release(caught);
            } else if let Some(message) = caught_message(&mut s, &p, caught) {
                s.uncaptured.push((device, message));
            }
        }
    }
    match s.uncaptured.iter().position(|(owner, _)| *owner == device) {
        Some(at) => Text::new(&s.uncaptured.remove(at).1),
        None => Text::NULL,
    }
}

// -- instance and adapter ---------------------------------------------------

pub unsafe fn instance_create() -> i32 {
    let mut s = state();
    if !s.agent {
        // Watched before any request can be answered, so none is missed.
        // The mailbox is static, and `settled` needs no context.
        let wake = unsafe {
            host::watch(
                MAILBOX.settled.as_ptr().cast_const().cast(),
                settled,
                std::ptr::null_mut(),
            )
        };
        if wake.is_null() {
            host::raise(
                ErrorKind::Runtime,
                "gpu: no world to settle the GPU's answers on",
            );
            return 0;
        }
        MAILBOX.wake.store(wake as usize as u32, SeqCst);
        s.agent = host::agent("gpu", &MAILBOX as *const Mailbox as usize);
        if !s.agent {
            host::raise(
                ErrorKind::Runtime,
                "gpu: this program's host starts no GPU agent",
            );
            return 0;
        }
    }
    s.instances.put(())
}

/// A browser has one WebGPU: there are no backends to choose between, and
/// wgpu's instance flags have nothing to act on.
pub unsafe fn instance_create_with(_descriptor: &GpuInstanceDescriptor) -> i32 {
    unsafe { instance_create() }
}

pub unsafe fn instance_destroy(instance: i32) {
    state().instances.remove(instance);
}

fn request_adapter(
    instance: i32,
    options: Option<wire::GPURequestAdapterOptions>,
) -> Future<crate::GpuAdapter> {
    let mut s = state();
    if s.instances.get(instance).is_none() {
        return rejected("instance was destroyed");
    }
    let adapter = s.adapters.put(());
    let promise = Promise::new();
    s.commands
        .gpu_request_adapter(GPU, handle(adapter), promise.reply.at(), &options);
    s.flush();
    let future = Future::new();
    wait(
        &mut s,
        promise,
        Waiting::Adapter(Rooted::new(future), adapter),
    );
    future
}

pub unsafe fn adapter_open(instance: i32, power: i32) -> Future<crate::GpuAdapter> {
    request_adapter(
        instance,
        Some(wire::GPURequestAdapterOptions {
            feature_level: None,
            power_preference: power_preference(power),
            force_fallback_adapter: None,
            xr_compatible: None,
        }),
    )
}

pub unsafe fn adapter_request_with(
    instance: i32,
    options: &GpuRequestAdapterOptions,
) -> Future<crate::GpuAdapter> {
    request_adapter(instance, Some(adapter_options(options)))
}

/// One member of the adapter's `GPUAdapterInfo`, as the agent encodes it.
fn adapter_info<T: wire::Decode>(
    adapter: i32,
    member: impl FnOnce(&mut wire::Encoder, Handle, u32),
) -> Option<T> {
    let mut s = state();
    s.adapters.get(adapter)?;
    let info = s.transient();
    let mut bytes = [0u8; 1024];
    let reply = Reply::new(bytes.as_mut_ptr(), bytes.len());
    s.commands.gpu_adapter_get_info(handle(adapter), info);
    member(&mut s.commands, info, reply.at());
    s.commands.release(info);
    s.flush();
    if reply.state.load(SeqCst) != 1 {
        return None;
    }
    let len = (reply.len as usize).min(bytes.len());
    T::decode(&mut wire::Decoder::new(&bytes[..len]))
}

/// The adapter's description, which a browser may leave empty; then its
/// device, or its vendor and architecture.
pub unsafe fn adapter_name(adapter: i32) -> Text {
    let read = |member: fn(&mut wire::Encoder, Handle, u32)| {
        adapter_info::<String>(adapter, member).unwrap_or_default()
    };
    let mut name = read(wire::Encoder::gpu_adapter_info_get_description);
    if name.is_empty() {
        name = read(wire::Encoder::gpu_adapter_info_get_device);
    }
    if name.is_empty() {
        let vendor = read(wire::Encoder::gpu_adapter_info_get_vendor);
        let architecture = read(wire::Encoder::gpu_adapter_info_get_architecture);
        name = format!("{vendor} {architecture}").trim().to_owned();
    }
    Text::new(&name)
}

/// The plugin's code for wgpu's `BrowserWebGpu`.
pub unsafe fn adapter_backend(adapter: i32) -> i32 {
    if state().adapters.get(adapter).is_some() { 5 } else { 0 }
}

pub unsafe fn adapter_subgroup_min_size(adapter: i32) -> i32 {
    adapter_info::<u32>(adapter, wire::Encoder::gpu_adapter_info_get_subgroup_min_size)
        .map_or(0, |size| size as i32)
}

pub unsafe fn adapter_subgroup_max_size(adapter: i32) -> i32 {
    adapter_info::<u32>(adapter, wire::Encoder::gpu_adapter_info_get_subgroup_max_size)
        .map_or(0, |size| size as i32)
}

pub unsafe fn adapter_destroy(adapter: i32) {
    forget(|s| s.adapters.remove(adapter), adapter);
}

// -- device -----------------------------------------------------------------

fn request_device(
    adapter: i32,
    descriptor: Option<wire::GPUDeviceDescriptor>,
) -> Future<crate::GpuDevice> {
    let mut s = state();
    if s.adapters.get(adapter).is_none() {
        return rejected("adapter was destroyed");
    }
    let device = s.devices.put(AtomicI32::new(0));
    let promise = Promise::new();
    s.commands.gpu_adapter_request_device(
        handle(adapter),
        handle(device),
        promise.reply.at(),
        &descriptor,
    );
    s.flush();
    let future = Future::new();
    wait(
        &mut s,
        promise,
        Waiting::Device(Rooted::new(future), device),
    );
    future
}

pub unsafe fn device_open(adapter: i32) -> Future<crate::GpuDevice> {
    request_device(adapter, None)
}

pub unsafe fn device_open_with(
    adapter: i32,
    descriptor: &GpuDeviceDescriptor,
) -> Future<crate::GpuDevice> {
    match device_descriptor(descriptor) {
        Ok(descriptor) => request_device(adapter, Some(descriptor)),
        Err(message) => rejected(&message),
    }
}

pub unsafe fn device_queue(device: i32) -> i32 {
    state()
        .devices
        .get(device)
        .map_or(0, |entry| entry.load(SeqCst))
}

/// Commands go out as the program needs them; this sends what waits.
pub unsafe fn device_poll(_device: i32) {
    state().flush();
}

pub unsafe fn device_destroy(device: i32) {
    let mut s = state();
    let Some(entry) = s.devices.get(device) else {
        return;
    };
    let queue = entry.load(SeqCst);
    s.devices.remove(device);
    s.queues.remove(queue);
    s.scopes.retain(|(owner, _)| *owner != device);
    s.uncaptured.retain(|(owner, _)| *owner != device);
    s.commands.gpu_device_destroy(handle(device));
    s.commands.release(handle(queue));
    s.commands.release(handle(device));
    // Sent now: what the destroy settles, the device's lost future, may be
    // awaited next, and an await sends nothing.
    s.flush();
}

pub unsafe fn queue_work_done(device: i32, queue: i32) -> Future<()> {
    let mut s = state();
    if s.devices.get(device).is_none() {
        return rejected("device was destroyed");
    }
    if s.queues.get(queue).is_none() {
        return rejected("queue was destroyed");
    }
    let promise = Promise::new();
    s.commands
        .gpu_queue_on_submitted_work_done(handle(queue), promise.reply.at());
    s.flush();
    let future = Future::new();
    wait(&mut s, promise, Waiting::Done(Rooted::new(future)));
    future
}

// -- buffers ----------------------------------------------------------------

pub unsafe fn buffer_create(device: i32, descriptor: &GpuBufferDescriptor) -> i32 {
    let Some(wired) = converted(descriptor.wire()) else {
        return 0;
    };
    let mut s = state();
    if s.devices.get(device).is_none() {
        return 0;
    }
    let buffer = s.buffers.put(BufferEntry {
        size: descriptor.size,
        usage: descriptor.usage,
    });
    s.commands
        .gpu_device_create_buffer(handle(device), handle(buffer), &wired);
    buffer
}

pub unsafe fn buffer_size(buffer: i32) -> i64 {
    state().buffers.get(buffer).map_or(0, |entry| entry.size)
}

pub unsafe fn buffer_usage(buffer: i32) -> i32 {
    state().buffers.get(buffer).map_or(0, |entry| entry.usage)
}

pub unsafe fn queue_write_buffer(queue: i32, buffer: i32, offset: i64, data: Buffer, len: i32) {
    let Some(data) = bytes(&data, len) else {
        return;
    };
    if data.is_empty() {
        return;
    }
    let mut s = state();
    if s.queues.get(queue).is_none() || s.buffers.get(buffer).is_none() {
        return;
    }
    let staged = s.stage(data);
    s.commands.gpu_queue_write_buffer(
        handle(queue),
        &handle(buffer),
        &(offset.max(0) as u64),
        &staged,
        &None,
        &None,
    );
    s.flush_if_full();
}

pub unsafe fn buffer_map_begin(device: i32, buffer: i32, offset: i64, size: i64) -> Future<()> {
    unsafe { buffer_map_with(device, buffer, 1, offset, size) }
}

pub unsafe fn buffer_map_with(
    device: i32,
    buffer: i32,
    mode: i32,
    offset: i64,
    size: i64,
) -> Future<()> {
    // GPUMapMode's READ and WRITE are the plugin's 1 and 2.
    if mode != 1 && mode != 2 {
        return rejected("a map mode is READ or WRITE");
    }
    let mut s = state();
    if s.buffers.get(buffer).is_none() {
        return rejected("buffer was destroyed");
    }
    if s.devices.get(device).is_none() {
        return rejected("device was destroyed");
    }
    let promise = Promise::new();
    s.commands.gpu_buffer_map_async(
        handle(buffer),
        promise.reply.at(),
        &(mode as u32),
        &Some(offset.max(0) as u64),
        &Some(size.max(0) as u64),
    );
    s.flush();
    let future = Future::new();
    wait(&mut s, promise, Waiting::Done(Rooted::new(future)));
    future
}

/// The mapped range, straight into `out`.
pub unsafe fn buffer_copy_out(buffer: i32, offset: i64, out: BufferMut, len: i32) -> bool {
    if len <= 0 || bytes(&out.buffer(), len).is_none() {
        return false;
    }
    let mut s = state();
    if s.buffers.get(buffer).is_none() {
        return false;
    }
    let reply = Reply::new(out.as_mut_ptr(), len as usize);
    s.commands.gpu_buffer_get_mapped_range(
        handle(buffer),
        reply.at(),
        &Some(offset.max(0) as u64),
        &Some(len as u64),
    );
    s.flush();
    reply.state.load(SeqCst) == 1 && reply.len == len as u32
}

/// The first `len` bytes of `data` into the mapped range at `offset`. False
/// when that range is not mapped for writing.
pub unsafe fn buffer_copy_in(buffer: i32, offset: i64, data: Buffer, len: i32) -> bool {
    let Some(data) = bytes(&data, len) else {
        return false;
    };
    let mut s = state();
    if s.buffers.get(buffer).is_none() {
        return false;
    }
    let staged = s.stage(data);
    let reply = Reply::new(std::ptr::null_mut(), 0);
    s.commands.gpu_buffer_get_mapped_range_write(
        handle(buffer),
        reply.at(),
        &Some(offset.max(0) as u64),
        &Some(data.len() as u64),
        &staged,
    );
    s.flush();
    reply.state.load(SeqCst) == 1
}

pub unsafe fn buffer_unmap(buffer: i32) {
    let mut s = state();
    if s.buffers.get(buffer).is_some() {
        s.commands.gpu_buffer_unmap(handle(buffer));
    }
}

pub unsafe fn buffer_destroy(buffer: i32) {
    let mut s = state();
    if s.buffers.get(buffer).is_none() {
        return;
    }
    s.buffers.remove(buffer);
    s.commands.gpu_buffer_destroy(handle(buffer));
    s.commands.release(handle(buffer));
}

// -- shaders and pipelines --------------------------------------------------

fn create_shader(device: i32, descriptor: &wire::GPUShaderModuleDescriptor) -> i32 {
    let mut s = state();
    if s.devices.get(device).is_none() {
        return 0;
    }
    let shader = s.shaders.put(());
    s.commands
        .gpu_device_create_shader_module(handle(device), handle(shader), descriptor);
    shader
}

pub unsafe fn shader_create(device: i32, wgsl: Text) -> i32 {
    create_shader(
        device,
        &wire::GPUShaderModuleDescriptor {
            label: None,
            code: wgsl.as_str().to_owned(),
            compilation_hints: None,
        },
    )
}

pub unsafe fn shader_create_with(device: i32, descriptor: &GpuShaderModuleDescriptor) -> i32 {
    converted(shader_descriptor(descriptor)).map_or(0, |d| create_shader(device, &d))
}

pub unsafe fn shader_destroy(shader: i32) {
    forget(
        |s| {
            s.shaders.remove(shader);
            s.shader_infos.retain(|(owner, _)| *owner != shader);
        },
        shader,
    );
}

/// A compute pipeline descriptor's layout: unset is "auto", as the plugin
/// declares it.
pub fn gpu_compute_pipeline_descriptor_layout(
    layout: &Option<i32>,
) -> Result<wire::GPUPipelineLayoutOrGPUAutoLayoutMode, String> {
    Ok(match layout {
        Some(layout) => {
            wire::GPUPipelineLayoutOrGPUAutoLayoutMode::GPUPipelineLayout(handle(*layout))
        }
        None => wire::GPUPipelineLayoutOrGPUAutoLayoutMode::GPUAutoLayoutMode(
            wire::GPUAutoLayoutMode::Auto,
        ),
    })
}

fn create_compute_pipeline(device: i32, descriptor: &wire::GPUComputePipelineDescriptor) -> i32 {
    let mut s = state();
    if s.devices.get(device).is_none() {
        return 0;
    }
    let pipeline = s.pipelines.put(());
    s.commands
        .gpu_device_create_compute_pipeline(handle(device), handle(pipeline), descriptor);
    pipeline
}

pub unsafe fn compute_pipeline_create(device: i32, shader: i32, entry: Text) -> i32 {
    create_compute_pipeline(
        device,
        &wire::GPUComputePipelineDescriptor {
            label: None,
            layout: wire::GPUPipelineLayoutOrGPUAutoLayoutMode::GPUAutoLayoutMode(
                wire::GPUAutoLayoutMode::Auto,
            ),
            compute: wire::GPUProgrammableStage {
                module: handle(shader),
                entry_point: Some(entry.as_str().to_owned()),
                constants: None,
            },
        },
    )
}

pub unsafe fn compute_pipeline_create_with(
    device: i32,
    descriptor: &GpuComputePipelineDescriptor,
) -> i32 {
    converted(descriptor.wire()).map_or(0, |d| create_compute_pipeline(device, &d))
}

/// A compute or a render pipeline; the handle says which.
pub unsafe fn pipeline_release(pipeline: i32) {
    forget(
        |s| {
            s.pipelines.remove(pipeline);
            s.render_pipelines.remove(pipeline);
        },
        pipeline,
    );
}

/// The layout of `pipeline`'s group `index` under `layout`, for a compute
/// or a render pipeline; false when it is neither.
fn get_bind_group_layout(s: &mut State, pipeline: i32, index: i32, layout: Handle) -> bool {
    let index = index.max(0) as u32;
    if s.render_pipelines.get(pipeline).is_some() {
        s.commands
            .gpu_render_pipeline_get_bind_group_layout(handle(pipeline), layout, &index);
    } else if s.pipelines.get(pipeline).is_some() {
        s.commands
            .gpu_compute_pipeline_get_bind_group_layout(handle(pipeline), layout, &index);
    } else {
        return false;
    }
    true
}

pub unsafe fn pipeline_bind_group_layout(pipeline: i32, index: i32) -> i32 {
    let mut s = state();
    let layout = s.layouts.put(());
    if !get_bind_group_layout(&mut s, pipeline, index, handle(layout)) {
        s.layouts.remove(layout);
        return 0;
    }
    layout
}

pub unsafe fn bind_group_layout_destroy(layout: i32) {
    forget(|s| s.layouts.remove(layout), layout);
}

// -- bind groups ------------------------------------------------------------

pub unsafe fn bindings_create() -> i32 {
    state().bindings.put(Mutex::new(Vec::new()))
}

/// Add `resource`, a buffer, view or sampler, as the next binding.
fn bind(bindings: i32, resource: i32) {
    let s = state();
    if let Some(list) = s.bindings.get(bindings)
        && s.has(resource)
    {
        list.lock().unwrap().push(resource);
    }
}

pub unsafe fn bindings_buffer(bindings: i32, buffer: i32) {
    bind(bindings, buffer);
}

pub unsafe fn bindings_view(bindings: i32, view: i32) {
    bind(bindings, view);
}

pub unsafe fn bindings_sampler(bindings: i32, sampler: i32) {
    bind(bindings, sampler);
}

pub unsafe fn bindings_destroy(bindings: i32) {
    state().bindings.remove(bindings);
}

/// A bind group over `bindings`, in order, laid out as `pipeline`'s group.
pub unsafe fn bind_group_create(device: i32, pipeline: i32, group: i32, bindings: i32) -> i32 {
    let mut s = state();
    let Some(list) = s.bindings.get(bindings) else {
        return 0;
    };
    if s.devices.get(device).is_none() {
        return 0;
    }
    let entries: Vec<_> = list
        .lock()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(binding, &resource)| wire::GPUBindGroupEntry {
            binding: binding as u32,
            resource: if kind_of(resource) == Kind::View as i32 {
                wire::GPUBindingResource::GPUTextureView(handle(resource))
            } else if kind_of(resource) == Kind::Sampler as i32 {
                wire::GPUBindingResource::GPUSampler(handle(resource))
            } else {
                wire::GPUBindingResource::GPUBuffer(handle(resource))
            },
        })
        .collect();
    let layout = s.transient();
    if !get_bind_group_layout(&mut s, pipeline, group, layout) {
        return 0;
    }
    let bind_group = s.bind_groups.put(());
    s.commands.gpu_device_create_bind_group(
        handle(device),
        handle(bind_group),
        &wire::GPUBindGroupDescriptor {
            label: None,
            layout,
            entries,
        },
    );
    s.commands.release(layout);
    bind_group
}

pub unsafe fn bind_group_create_with(device: i32, descriptor: &GpuBindGroupDescriptor) -> i32 {
    let Some(wired) = converted(descriptor.wire()) else {
        return 0;
    };
    let mut s = state();
    if s.devices.get(device).is_none() {
        return 0;
    }
    let bind_group = s.bind_groups.put(());
    s.commands
        .gpu_device_create_bind_group(handle(device), handle(bind_group), &wired);
    bind_group
}

pub unsafe fn bind_group_destroy(bind_group: i32) {
    forget(|s| s.bind_groups.remove(bind_group), bind_group);
}

// -- encoders ---------------------------------------------------------------

pub unsafe fn encoder_create(device: i32) -> i32 {
    let mut s = state();
    if s.devices.get(device).is_none() {
        return 0;
    }
    let encoder = s.encoders.put(Mutex::new(EncoderEntry::default()));
    s.commands
        .gpu_device_create_command_encoder(handle(device), handle(encoder), &None);
    encoder
}

pub unsafe fn encoder_destroy(encoder: i32) {
    forget(|s| s.encoders.remove(encoder), encoder);
}

/// Run `body` on `encoder`'s entry and the batch, or do nothing. What it
/// refuses is raised once both are unlocked: a raise does not return.
fn encoding(
    encoder: i32,
    body: impl FnOnce(&mut EncoderEntry, &mut State) -> Result<(), String>,
) {
    let refused = {
        let mut s = state();
        let Some(entry) = s.encoders.get(encoder) else {
            return;
        };
        let mut entry = entry.lock().unwrap();
        body(&mut entry, &mut s)
    };
    if let Err(message) = refused {
        host::raise(ErrorKind::Runtime, &message);
    }
}

/// The open compute pass's commands, when one is open.
fn in_compute(encoder: i32, body: impl FnOnce(Handle, &mut State)) {
    encoding(encoder, |entry, s| {
        if entry.compute != 0 {
            body(Handle(entry.compute), s);
        }
        Ok(())
    });
}

/// The open render pass's commands, when one is open.
fn in_render(encoder: i32, body: impl FnOnce(Handle, &mut State)) {
    encoding(encoder, |entry, s| {
        if entry.render != 0 {
            body(Handle(entry.render), s);
        }
        Ok(())
    });
}

/// The open compute pass's commands; raised when none is open.
fn computing(encoder: i32, body: impl FnOnce(Handle, &mut State)) {
    encoding(encoder, |entry, s| match entry.compute {
        0 => Err("gpu: no compute pass is open on this encoder".to_owned()),
        pass => {
            body(Handle(pass), s);
            Ok(())
        }
    });
}

/// The open render pass's commands; raised when none is open.
fn rendering(encoder: i32, body: impl FnOnce(Handle, &mut State)) {
    encoding(encoder, |entry, s| match entry.render {
        0 => Err("gpu: no render pass is open on this encoder".to_owned()),
        pass => {
            body(Handle(pass), s);
            Ok(())
        }
    });
}

/// The encoder's own commands; raised while a pass is open.
fn copying(encoder: i32, body: impl FnOnce(Handle, &mut State)) {
    encoding(encoder, |entry, s| {
        if entry.compute != 0 || entry.render != 0 {
            return Err("gpu: a pass is open on this encoder".to_owned());
        }
        body(handle(encoder), s);
        Ok(())
    });
}

fn pass_open(entry: &EncoderEntry) -> Result<(), String> {
    if entry.compute != 0 || entry.render != 0 {
        return Err("gpu: a pass is already open on this encoder".to_owned());
    }
    Ok(())
}

/// `value` as a count or index; raised when negative.
fn unsigned(value: i32, what: &str) -> Option<u32> {
    match u32::try_from(value) {
        Ok(value) => Some(value),
        Err(_) => {
            host::raise(ErrorKind::Type, &format!("gpu: a negative {what}"));
            None
        }
    }
}

/// `offset` and a size that is the rest of the buffer when negative.
fn range(offset: i64, size: i64) -> Option<(u64, Option<u64>)> {
    let Ok(offset) = u64::try_from(offset) else {
        host::raise(ErrorKind::Type, "gpu: a negative buffer offset");
        return None;
    };
    Some((offset, u64::try_from(size).ok()))
}

/// `count` dynamic offsets from element `start` of a shared buffer of
/// 32-bit values, as WebGPU's Uint32Array overload of setBindGroup reads
/// them.
fn dynamic_offsets(data: &Buffer, start: i64, count: i32) -> Option<Vec<u32>> {
    let (Ok(start), Ok(count)) = (usize::try_from(start), usize::try_from(count)) else {
        host::raise(ErrorKind::Type, "negative dynamic offset range");
        return None;
    };
    let end = start.checked_add(count).and_then(|end| end.checked_mul(4));
    if end.is_none_or(|end| end > data.len()) {
        host::raise(ErrorKind::Type, "dynamic offsets exceed the shared buffer");
        return None;
    }
    let bytes = unsafe { &data.as_slice()[start * 4..(start + count) * 4] };
    Some(
        bytes
            .chunks_exact(4)
            .map(|word| u32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
            .collect(),
    )
}

/// The plugin's IndexFormat, GPUIndexFormat's index.
fn index_format(format: i32) -> wire::GPUIndexFormat {
    wire::GPUIndexFormat::from_index(format as u32).unwrap_or(wire::GPUIndexFormat::Uint16)
}

pub unsafe fn compute_begin(encoder: i32) {
    encoding(encoder, |entry, s| {
        pass_open(entry)?;
        let pass = s.transient();
        s.commands
            .gpu_command_encoder_begin_compute_pass(handle(encoder), pass, &None);
        entry.compute = pass.0;
        Ok(())
    });
}

pub unsafe fn compute_pass_begin_with(encoder: i32, descriptor: &GpuComputePassDescriptor) {
    let Some(wired) = converted(descriptor.wire()) else {
        return;
    };
    encoding(encoder, |entry, s| {
        pass_open(entry)?;
        let pass = s.transient();
        s.commands
            .gpu_command_encoder_begin_compute_pass(handle(encoder), pass, &Some(wired));
        entry.compute = pass.0;
        Ok(())
    });
}

pub unsafe fn compute_set_pipeline(encoder: i32, pipeline: i32) {
    in_compute(encoder, |pass, s| {
        s.commands
            .gpu_compute_pass_encoder_set_pipeline(pass, &handle(pipeline))
    });
}

pub unsafe fn compute_set_bind_group(encoder: i32, group: i32, bind_group: i32) {
    in_compute(encoder, |pass, s| {
        s.commands.gpu_compute_pass_encoder_set_bind_group(
            pass,
            &(group.max(0) as u32),
            &Some(handle(bind_group)),
            &None,
        )
    });
}

pub unsafe fn compute_set_bind_group_offsets(
    encoder: i32,
    group: i32,
    bind_group: i32,
    offsets: Buffer,
    start: i64,
    count: i32,
) {
    let Some(offsets) = dynamic_offsets(&offsets, start, count) else {
        return;
    };
    in_compute(encoder, |pass, s| {
        s.commands.gpu_compute_pass_encoder_set_bind_group(
            pass,
            &(group.max(0) as u32),
            &Some(handle(bind_group)),
            &Some(offsets),
        )
    });
}

pub unsafe fn compute_dispatch(encoder: i32, x: i32, y: i32, z: i32) {
    in_compute(encoder, |pass, s| {
        s.commands.gpu_compute_pass_encoder_dispatch_workgroups(
            pass,
            &(x.max(0) as u32),
            &Some(y.max(0) as u32),
            &Some(z.max(0) as u32),
        )
    });
}

pub unsafe fn compute_dispatch_indirect(encoder: i32, buffer: i32, offset: i64) {
    computing(encoder, |pass, s| {
        if s.has(buffer) {
            s.commands.gpu_compute_pass_encoder_dispatch_workgroups_indirect(
                pass,
                &handle(buffer),
                &(offset.max(0) as u64),
            );
        }
    });
}

/// End whichever pass is open on the encoder.
fn end_pass(entry: &mut EncoderEntry, s: &mut State) -> Result<(), String> {
    if entry.compute != 0 {
        s.commands
            .gpu_compute_pass_encoder_end(Handle(entry.compute));
        s.commands.release(Handle(entry.compute));
        entry.compute = 0;
    }
    if entry.render != 0 {
        s.commands.gpu_render_pass_encoder_end(Handle(entry.render));
        s.commands.release(Handle(entry.render));
        entry.render = 0;
    }
    Ok(())
}

pub unsafe fn compute_end(encoder: i32) {
    encoding(encoder, end_pass);
}

pub unsafe fn encoder_compute(
    encoder: i32,
    pipeline: i32,
    bind_group: i32,
    x: i32,
    y: i32,
    z: i32,
) {
    unsafe {
        compute_begin(encoder);
        compute_set_pipeline(encoder, pipeline);
        compute_set_bind_group(encoder, 0, bind_group);
        compute_dispatch(encoder, x, y, z);
        compute_end(encoder);
    }
}

pub unsafe fn encoder_compute_indirect(
    encoder: i32,
    pipeline: i32,
    bind_group: i32,
    buffer: i32,
    offset: i64,
) {
    unsafe {
        compute_begin(encoder);
        compute_set_pipeline(encoder, pipeline);
        compute_set_bind_group(encoder, 0, bind_group);
        compute_dispatch_indirect(encoder, buffer, offset);
        compute_end(encoder);
    }
}

pub unsafe fn encoder_copy_buffer(
    encoder: i32,
    src: i32,
    src_offset: i64,
    dst: i32,
    dst_offset: i64,
    size: i64,
) {
    encoding(encoder, |_, s| {
        if s.buffers.get(src).is_none() || s.buffers.get(dst).is_none() {
            return Ok(());
        }
        s.commands.gpu_command_encoder_copy_buffer_to_buffer_2(
            handle(encoder),
            &handle(src),
            &(src_offset.max(0) as u64),
            &handle(dst),
            &(dst_offset.max(0) as u64),
            &Some(size.max(0) as u64),
        );
        Ok(())
    });
}

/// Finish the encoder and submit it; it is spent either way. The commands
/// go out with the batch.
pub unsafe fn encoder_submit(encoder: i32, queue: i32) {
    let mut s = state();
    let Some(entry) = s.encoders.get(encoder) else {
        return;
    };
    let _ = end_pass(&mut entry.lock().unwrap(), &mut s);
    s.encoders.remove(encoder);
    if s.queues.get(queue).is_some() {
        let commands = s.transient();
        s.commands
            .gpu_command_encoder_finish(handle(encoder), commands, &None);
        s.commands.gpu_queue_submit(handle(queue), &vec![commands]);
        s.commands.release(commands);
    }
    s.commands.release(handle(encoder));
    s.flush_if_full();
}

// -- copies -----------------------------------------------------------------

/// All of mip level zero of `texture`, as a copy's side.
fn whole(texture: i32) -> wire::GPUTexelCopyTextureInfo {
    wire::GPUTexelCopyTextureInfo {
        texture: handle(texture),
        mip_level: None,
        origin: None,
        aspect: None,
    }
}

/// A two-dimensional copy's extent, at least one texel each way.
fn extent(width: i32, height: i32) -> wire::GPUExtent3D {
    wire::GPUExtent3D::GPUExtent3DDict(wire::GPUExtent3DDict {
        width: width.max(1) as u32,
        height: Some(height.max(1) as u32),
        depth_or_array_layers: Some(1),
    })
}

/// Rows of `bytes_per_row` bytes, `height` of them to an image.
fn rows(bytes_per_row: i32, height: i32) -> wire::GPUTexelCopyBufferLayout {
    wire::GPUTexelCopyBufferLayout {
        offset: None,
        bytes_per_row: Some(bytes_per_row.max(0) as u32),
        rows_per_image: Some(height.max(1) as u32),
    }
}

fn buffer_side(buffer: i32, layout: wire::GPUTexelCopyBufferLayout) -> wire::GPUTexelCopyBufferInfo {
    wire::GPUTexelCopyBufferInfo {
        offset: layout.offset,
        bytes_per_row: layout.bytes_per_row,
        rows_per_image: layout.rows_per_image,
        buffer: handle(buffer),
    }
}

/// A copy's extent as the wire's.
fn extent_of(size: &GpuExtent3D) -> Option<wire::GPUExtent3D> {
    converted(size.wire()).map(wire::GPUExtent3D::GPUExtent3DDict)
}

pub unsafe fn encoder_copy_buffer_to_texture(
    encoder: i32,
    buffer: i32,
    bytes_per_row: i32,
    texture: i32,
    width: i32,
    height: i32,
) {
    copying(encoder, |this, s| {
        if s.buffers.get(buffer).is_some() && s.textures.get(texture).is_some() {
            s.commands.gpu_command_encoder_copy_buffer_to_texture(
                this,
                &buffer_side(buffer, rows(bytes_per_row, height)),
                &whole(texture),
                &extent(width, height),
            );
        }
    });
}

pub unsafe fn encoder_copy_texture_to_buffer(
    encoder: i32,
    texture: i32,
    buffer: i32,
    width: i32,
    height: i32,
    bytes_per_row: i32,
) {
    copying(encoder, |this, s| {
        if s.buffers.get(buffer).is_some() && s.textures.get(texture).is_some() {
            s.commands.gpu_command_encoder_copy_texture_to_buffer(
                this,
                &whole(texture),
                &buffer_side(buffer, rows(bytes_per_row, height)),
                &extent(width, height),
            );
        }
    });
}

pub unsafe fn encoder_copy_texture_to_texture(
    encoder: i32,
    src: i32,
    dst: i32,
    width: i32,
    height: i32,
) {
    copying(encoder, |this, s| {
        if s.textures.get(src).is_some() && s.textures.get(dst).is_some() {
            s.commands.gpu_command_encoder_copy_texture_to_texture(
                this,
                &whole(src),
                &whole(dst),
                &extent(width, height),
            );
        }
    });
}

pub unsafe fn encoder_copy_buffer_to_texture_with(
    encoder: i32,
    source: &GpuTexelCopyBufferInfo,
    destination: &GpuTexelCopyTextureInfo,
    size: &GpuExtent3D,
) {
    let (Some(source), Some(destination), Some(size)) = (
        converted(source.wire()),
        converted(destination.wire()),
        extent_of(size),
    ) else {
        return;
    };
    copying(encoder, |this, s| {
        s.commands
            .gpu_command_encoder_copy_buffer_to_texture(this, &source, &destination, &size)
    });
}

pub unsafe fn encoder_copy_texture_to_buffer_with(
    encoder: i32,
    source: &GpuTexelCopyTextureInfo,
    destination: &GpuTexelCopyBufferInfo,
    size: &GpuExtent3D,
) {
    let (Some(source), Some(destination), Some(size)) = (
        converted(source.wire()),
        converted(destination.wire()),
        extent_of(size),
    ) else {
        return;
    };
    copying(encoder, |this, s| {
        s.commands
            .gpu_command_encoder_copy_texture_to_buffer(this, &source, &destination, &size)
    });
}

pub unsafe fn encoder_copy_texture_to_texture_with(
    encoder: i32,
    source: &GpuTexelCopyTextureInfo,
    destination: &GpuTexelCopyTextureInfo,
    size: &GpuExtent3D,
) {
    let (Some(source), Some(destination), Some(size)) = (
        converted(source.wire()),
        converted(destination.wire()),
        extent_of(size),
    ) else {
        return;
    };
    copying(encoder, |this, s| {
        s.commands
            .gpu_command_encoder_copy_texture_to_texture(this, &source, &destination, &size)
    });
}

pub unsafe fn encoder_clear_buffer(encoder: i32, buffer: i32, offset: i64, size: i64) {
    copying(encoder, |this, s| {
        if s.buffers.get(buffer).is_some() {
            s.commands.gpu_command_encoder_clear_buffer(
                this,
                &handle(buffer),
                &Some(offset.max(0) as u64),
                &Some(size.max(0) as u64),
            );
        }
    });
}

pub unsafe fn encoder_resolve_query_set(
    encoder: i32,
    query_set: i32,
    first: i32,
    count: i32,
    destination: i32,
    offset: i64,
) {
    let (Some(first), Some(count)) = (unsigned(first, "first query"), unsigned(count, "query count"))
    else {
        return;
    };
    let Some((offset, _)) = range(offset, 0) else {
        return;
    };
    copying(encoder, |this, s| {
        if s.query_sets.get(query_set).is_some() && s.buffers.get(destination).is_some() {
            s.commands.gpu_command_encoder_resolve_query_set(
                this,
                &handle(query_set),
                &first,
                &count,
                &handle(destination),
                &offset,
            );
        }
    });
}

// -- debug labels -----------------------------------------------------------

/// A label belongs to whatever is recording: the open pass, or the encoder
/// when none is.
fn labelling(
    encoder: i32,
    render: impl FnOnce(&mut wire::Encoder, Handle),
    compute: impl FnOnce(&mut wire::Encoder, Handle),
    own: impl FnOnce(&mut wire::Encoder, Handle),
) {
    encoding(encoder, |entry, s| {
        if entry.render != 0 {
            render(&mut s.commands, Handle(entry.render));
        } else if entry.compute != 0 {
            compute(&mut s.commands, Handle(entry.compute));
        } else {
            own(&mut s.commands, handle(encoder));
        }
        Ok(())
    });
}

pub unsafe fn encoder_push_debug_group(encoder: i32, label: Text) {
    let label = label.as_str().to_owned();
    labelling(
        encoder,
        |c, h| c.gpu_render_pass_encoder_push_debug_group(h, &label),
        |c, h| c.gpu_compute_pass_encoder_push_debug_group(h, &label),
        |c, h| c.gpu_command_encoder_push_debug_group(h, &label),
    );
}

pub unsafe fn encoder_pop_debug_group(encoder: i32) {
    labelling(
        encoder,
        |c, h| c.gpu_render_pass_encoder_pop_debug_group(h),
        |c, h| c.gpu_compute_pass_encoder_pop_debug_group(h),
        |c, h| c.gpu_command_encoder_pop_debug_group(h),
    );
}

pub unsafe fn encoder_insert_debug_marker(encoder: i32, label: Text) {
    let label = label.as_str().to_owned();
    labelling(
        encoder,
        |c, h| c.gpu_render_pass_encoder_insert_debug_marker(h, &label),
        |c, h| c.gpu_compute_pass_encoder_insert_debug_marker(h, &label),
        |c, h| c.gpu_command_encoder_insert_debug_marker(h, &label),
    );
}

// -- render passes ----------------------------------------------------------

pub unsafe fn pass_colour(encoder: i32, view: i32, r: f64, g: f64, b: f64, a: f64) {
    encoding(encoder, |entry, _| {
        entry.colour.push((view, [r, g, b, a]));
        Ok(())
    });
}

pub unsafe fn pass_depth(encoder: i32, view: i32, clear: f64, stencil_clear: i32) {
    encoding(encoder, |entry, _| {
        entry.depth = Some((view, clear, stencil_clear));
        Ok(())
    });
}

/// Forget the attachments described for the next pass.
pub unsafe fn pass_reset(encoder: i32) {
    encoding(encoder, |entry, _| {
        entry.colour.clear();
        entry.depth = None;
        Ok(())
    });
}

/// Opens what was described: each colour attachment cleared and stored,
/// and the depth attachment, with its stencil when it was given a clear.
pub unsafe fn pass_begin(encoder: i32) {
    encoding(encoder, |entry, s| {
        pass_open(entry)?;
        let color_attachments = entry
            .colour
            .drain(..)
            .map(|(view, [r, g, b, a])| {
                Some(wire::GPURenderPassColorAttachment {
                    view: wire::GPUTextureOrGPUTextureView::GPUTextureView(handle(view)),
                    depth_slice: None,
                    resolve_target: None,
                    clear_value: Some(wire::GPUColor::GPUColorDict(wire::GPUColorDict {
                        r,
                        g,
                        b,
                        a,
                    })),
                    load_op: wire::GPULoadOp::Clear,
                    store_op: wire::GPUStoreOp::Store,
                })
            })
            .collect();
        let depth_stencil_attachment = entry.depth.take().map(|(view, clear, stencil)| {
            let stencil = stencil >= 0;
            wire::GPURenderPassDepthStencilAttachment {
                view: wire::GPUTextureOrGPUTextureView::GPUTextureView(handle(view)),
                depth_clear_value: Some(clear as f32),
                depth_load_op: Some(wire::GPULoadOp::Clear),
                depth_store_op: Some(wire::GPUStoreOp::Store),
                depth_read_only: None,
                stencil_clear_value: stencil.then_some(0),
                stencil_load_op: stencil.then_some(wire::GPULoadOp::Clear),
                stencil_store_op: stencil.then_some(wire::GPUStoreOp::Store),
                stencil_read_only: None,
            }
        });
        let pass = s.transient();
        s.commands.gpu_command_encoder_begin_render_pass(
            handle(encoder),
            pass,
            &wire::GPURenderPassDescriptor {
                label: None,
                color_attachments,
                depth_stencil_attachment,
                occlusion_query_set: None,
                timestamp_writes: None,
                max_draw_count: None,
            },
        );
        entry.render = pass.0;
        Ok(())
    });
}

pub unsafe fn render_pass_begin_with(encoder: i32, descriptor: &GpuRenderPassDescriptor) {
    let Some(wired) = converted(descriptor.wire()) else {
        return;
    };
    encoding(encoder, |entry, s| {
        pass_open(entry)?;
        let pass = s.transient();
        s.commands
            .gpu_command_encoder_begin_render_pass(handle(encoder), pass, &wired);
        entry.render = pass.0;
        Ok(())
    });
}

pub unsafe fn render_set_pipeline(encoder: i32, pipeline: i32) {
    in_render(encoder, |pass, s| {
        s.commands
            .gpu_render_pass_encoder_set_pipeline(pass, &handle(pipeline))
    });
}

pub unsafe fn render_set_vertex_buffer(encoder: i32, slot: i32, buffer: i32) {
    in_render(encoder, |pass, s| {
        s.commands.gpu_render_pass_encoder_set_vertex_buffer(
            pass,
            &(slot.max(0) as u32),
            &Some(handle(buffer)),
            &None,
            &None,
        )
    });
}

pub unsafe fn render_set_vertex_buffer_range(
    encoder: i32,
    slot: i32,
    buffer: i32,
    offset: i64,
    size: i64,
) {
    let (Some(slot), Some((offset, size))) = (unsigned(slot, "vertex buffer slot"), range(offset, size))
    else {
        return;
    };
    rendering(encoder, |pass, s| {
        if s.buffers.get(buffer).is_some() {
            s.commands.gpu_render_pass_encoder_set_vertex_buffer(
                pass,
                &slot,
                &Some(handle(buffer)),
                &Some(offset),
                &size,
            );
        }
    });
}

pub unsafe fn render_set_index_buffer(encoder: i32, buffer: i32, format: i32) {
    let format = index_format(format);
    in_render(encoder, |pass, s| {
        s.commands
            .gpu_render_pass_encoder_set_index_buffer(pass, &handle(buffer), &format, &None, &None)
    });
}

pub unsafe fn render_set_index_buffer_range(
    encoder: i32,
    buffer: i32,
    format: i32,
    offset: i64,
    size: i64,
) {
    let format = index_format(format);
    let Some((offset, size)) = range(offset, size) else {
        return;
    };
    rendering(encoder, |pass, s| {
        if s.buffers.get(buffer).is_some() {
            s.commands.gpu_render_pass_encoder_set_index_buffer(
                pass,
                &handle(buffer),
                &format,
                &Some(offset),
                &size,
            );
        }
    });
}

pub unsafe fn render_set_bind_group(encoder: i32, group: i32, bind_group: i32) {
    in_render(encoder, |pass, s| {
        s.commands.gpu_render_pass_encoder_set_bind_group(
            pass,
            &(group.max(0) as u32),
            &Some(handle(bind_group)),
            &None,
        )
    });
}

pub unsafe fn render_set_bind_group_offsets(
    encoder: i32,
    group: i32,
    bind_group: i32,
    offsets: Buffer,
    start: i64,
    count: i32,
) {
    let Some(offsets) = dynamic_offsets(&offsets, start, count) else {
        return;
    };
    in_render(encoder, |pass, s| {
        s.commands.gpu_render_pass_encoder_set_bind_group(
            pass,
            &(group.max(0) as u32),
            &Some(handle(bind_group)),
            &Some(offsets),
        )
    });
}

pub unsafe fn render_set_viewport(
    encoder: i32,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    min_depth: f64,
    max_depth: f64,
) {
    in_render(encoder, |pass, s| {
        s.commands.gpu_render_pass_encoder_set_viewport(
            pass,
            &(x as f32),
            &(y as f32),
            &(width as f32),
            &(height as f32),
            &(min_depth as f32),
            &(max_depth as f32),
        )
    });
}

pub unsafe fn render_set_scissor_rect(encoder: i32, x: i32, y: i32, width: i32, height: i32) {
    in_render(encoder, |pass, s| {
        s.commands.gpu_render_pass_encoder_set_scissor_rect(
            pass,
            &(x.max(0) as u32),
            &(y.max(0) as u32),
            &(width.max(0) as u32),
            &(height.max(0) as u32),
        )
    });
}

pub unsafe fn render_set_blend_constant(encoder: i32, r: f64, g: f64, b: f64, a: f64) {
    in_render(encoder, |pass, s| {
        s.commands.gpu_render_pass_encoder_set_blend_constant(
            pass,
            &wire::GPUColor::GPUColorDict(wire::GPUColorDict { r, g, b, a }),
        )
    });
}

pub unsafe fn render_set_stencil_reference(encoder: i32, reference: i32) {
    in_render(encoder, |pass, s| {
        s.commands
            .gpu_render_pass_encoder_set_stencil_reference(pass, &(reference.max(0) as u32))
    });
}

pub unsafe fn render_draw(encoder: i32, vertices: i32, instances: i32) {
    in_render(encoder, |pass, s| {
        s.commands.gpu_render_pass_encoder_draw(
            pass,
            &(vertices.max(0) as u32),
            &Some(instances.max(1) as u32),
            &None,
            &None,
        )
    });
}

pub unsafe fn render_draw_range(
    encoder: i32,
    vertex_count: i32,
    instance_count: i32,
    first_vertex: i32,
    first_instance: i32,
) {
    let (Some(vertices), Some(instances), Some(first), Some(first_instance)) = (
        unsigned(vertex_count, "vertex count"),
        unsigned(instance_count, "instance count"),
        unsigned(first_vertex, "first vertex"),
        unsigned(first_instance, "first instance"),
    ) else {
        return;
    };
    rendering(encoder, |pass, s| {
        s.commands.gpu_render_pass_encoder_draw(
            pass,
            &vertices,
            &Some(instances),
            &Some(first),
            &Some(first_instance),
        )
    });
}

pub unsafe fn render_draw_indexed(encoder: i32, indices: i32, instances: i32) {
    in_render(encoder, |pass, s| {
        s.commands.gpu_render_pass_encoder_draw_indexed(
            pass,
            &(indices.max(0) as u32),
            &Some(instances.max(1) as u32),
            &None,
            &None,
            &None,
        )
    });
}

pub unsafe fn render_draw_indexed_range(
    encoder: i32,
    index_count: i32,
    instance_count: i32,
    first_index: i32,
    base_vertex: i32,
    first_instance: i32,
) {
    let (Some(indices), Some(instances), Some(first), Some(first_instance)) = (
        unsigned(index_count, "index count"),
        unsigned(instance_count, "instance count"),
        unsigned(first_index, "first index"),
        unsigned(first_instance, "first instance"),
    ) else {
        return;
    };
    rendering(encoder, |pass, s| {
        s.commands.gpu_render_pass_encoder_draw_indexed(
            pass,
            &indices,
            &Some(instances),
            &Some(first),
            &Some(base_vertex),
            &Some(first_instance),
        )
    });
}

pub unsafe fn render_draw_indirect(encoder: i32, buffer: i32, offset: i64) {
    in_render(encoder, |pass, s| {
        if s.buffers.get(buffer).is_some() {
            s.commands.gpu_render_pass_encoder_draw_indirect(
                pass,
                &handle(buffer),
                &(offset.max(0) as u64),
            );
        }
    });
}

pub unsafe fn render_draw_indexed_indirect(encoder: i32, buffer: i32, offset: i64) {
    in_render(encoder, |pass, s| {
        if s.buffers.get(buffer).is_some() {
            s.commands.gpu_render_pass_encoder_draw_indexed_indirect(
                pass,
                &handle(buffer),
                &(offset.max(0) as u64),
            );
        }
    });
}

/// Counted into the query set the pass was begun with.
pub unsafe fn render_begin_occlusion_query(encoder: i32, index_of: i32) {
    let Some(query) = unsigned(index_of, "query index") else {
        return;
    };
    rendering(encoder, |pass, s| {
        s.commands
            .gpu_render_pass_encoder_begin_occlusion_query(pass, &query)
    });
}

pub unsafe fn render_end_occlusion_query(encoder: i32) {
    rendering(encoder, |pass, s| {
        s.commands.gpu_render_pass_encoder_end_occlusion_query(pass)
    });
}

pub unsafe fn render_execute_bundle(encoder: i32, bundle: i32) {
    rendering(encoder, |pass, s| {
        if s.bundles.get(bundle).is_some() {
            s.commands
                .gpu_render_pass_encoder_execute_bundles(pass, &vec![handle(bundle)]);
        }
    });
}

pub unsafe fn encoder_render_end(encoder: i32) {
    encoding(encoder, end_pass);
}

// -- render bundles ---------------------------------------------------------

/// The bundle encoder's commands, when it is live.
fn bundling(encoder: i32, body: impl FnOnce(Handle, &mut State)) {
    let mut s = state();
    if s.bundle_encoders.get(encoder).is_some() {
        body(handle(encoder), &mut s);
    }
}

pub unsafe fn bundle_set_bind_group(encoder: i32, group: i32, bind_group: i32) {
    let Some(group) = unsigned(group, "bind group index") else {
        return;
    };
    bundling(encoder, |this, s| {
        if s.bind_groups.get(bind_group).is_some() {
            s.commands.gpu_render_bundle_encoder_set_bind_group(
                this,
                &group,
                &Some(handle(bind_group)),
                &None,
            );
        }
    });
}

pub unsafe fn bundle_set_bind_group_offsets(
    encoder: i32,
    group: i32,
    bind_group: i32,
    offsets: Buffer,
    start: i64,
    count: i32,
) {
    let Some(group) = unsigned(group, "bind group index") else {
        return;
    };
    let Some(offsets) = dynamic_offsets(&offsets, start, count) else {
        return;
    };
    bundling(encoder, |this, s| {
        if s.bind_groups.get(bind_group).is_some() {
            s.commands.gpu_render_bundle_encoder_set_bind_group(
                this,
                &group,
                &Some(handle(bind_group)),
                &Some(offsets),
            );
        }
    });
}

pub unsafe fn bundle_set_vertex_buffer(
    encoder: i32,
    slot: i32,
    buffer: i32,
    offset: i64,
    size: i64,
) {
    let (Some(slot), Some((offset, size))) = (unsigned(slot, "vertex buffer slot"), range(offset, size))
    else {
        return;
    };
    bundling(encoder, |this, s| {
        if s.buffers.get(buffer).is_some() {
            s.commands.gpu_render_bundle_encoder_set_vertex_buffer(
                this,
                &slot,
                &Some(handle(buffer)),
                &Some(offset),
                &size,
            );
        }
    });
}

pub unsafe fn bundle_set_index_buffer(
    encoder: i32,
    buffer: i32,
    format: i32,
    offset: i64,
    size: i64,
) {
    let format = index_format(format);
    let Some((offset, size)) = range(offset, size) else {
        return;
    };
    bundling(encoder, |this, s| {
        if s.buffers.get(buffer).is_some() {
            s.commands.gpu_render_bundle_encoder_set_index_buffer(
                this,
                &handle(buffer),
                &format,
                &Some(offset),
                &size,
            );
        }
    });
}

pub unsafe fn bundle_encoder_destroy(encoder: i32) {
    forget(|s| s.bundle_encoders.remove(encoder), encoder);
}

pub unsafe fn bundle_destroy(bundle: i32) {
    forget(|s| s.bundles.remove(bundle), bundle);
}

// -- textures ---------------------------------------------------------------

pub unsafe fn texture_create(device: i32, descriptor: &GpuTextureDescriptor) -> i32 {
    let Some(wired) = converted(descriptor.wire()) else {
        return 0;
    };
    let mut s = state();
    if s.devices.get(device).is_none() {
        return 0;
    }
    let texture = s.textures.put(());
    s.commands
        .gpu_device_create_texture(handle(device), handle(texture), &wired);
    texture
}

pub unsafe fn texture_view(texture: i32, descriptor: &GpuTextureViewDescriptor) -> i32 {
    let Some(wired) = converted(descriptor.wire()) else {
        return 0;
    };
    let mut s = state();
    if s.textures.get(texture).is_none() {
        return 0;
    }
    let view = s.views.put(());
    s.commands
        .gpu_texture_create_view(handle(texture), handle(view), &Some(wired));
    view
}

pub unsafe fn texture_destroy(texture: i32) {
    let mut s = state();
    if s.textures.get(texture).is_none() {
        return;
    }
    s.textures.remove(texture);
    s.commands.gpu_texture_destroy(handle(texture));
    s.commands.release(handle(texture));
}

pub unsafe fn view_destroy(view: i32) {
    forget(|s| s.views.remove(view), view);
}

// -- surfaces ---------------------------------------------------------------

/// The page's canvas, whatever window handles the program passes: in a
/// browser that is the only surface there is.
pub unsafe fn surface_create(
    instance: i32,
    _platform: i32,
    _wa: i64,
    _wb: i64,
    _da: i64,
    _db: i64,
) -> i32 {
    let mut s = state();
    if s.instances.get(instance).is_none() {
        return 0;
    }
    s.surfaces.put(Mutex::new(SurfaceEntry::default()))
}

pub unsafe fn surface_destroy(surface: i32) {
    let mut s = state();
    if s.surfaces.get(surface).is_some() {
        s.surfaces.remove(surface);
        s.commands.gpu_canvas_context_unconfigure(CANVAS);
    }
}

/// The browser's preferred canvas format, as the plugin's code, which is
/// its index in the IDL.
pub unsafe fn surface_preferred_format(surface: i32, _adapter: i32) -> i32 {
    let mut s = state();
    if s.surfaces.get(surface).is_none() {
        return -1;
    }
    let mut format = [0u8; 4];
    let reply = Reply::new(format.as_mut_ptr(), format.len());
    s.commands.gpu_get_preferred_canvas_format(GPU, reply.at());
    s.flush();
    if reply.state.load(SeqCst) != 1 {
        return -1;
    }
    u32::from_le_bytes(format) as i32
}

fn configure(
    device: i32,
    surface: i32,
    size: (i32, i32),
    format: i32,
    usage: Option<i32>,
    view_formats: &[i32],
    alpha: Option<i32>,
) {
    let Some(format) = wire::GPUTextureFormat::from_index(format as u32) else {
        host::raise(
            ErrorKind::Runtime,
            "gpu: the canvas cannot take this texture format",
        );
        return;
    };
    let mut s = state();
    if s.devices.get(device).is_none() || s.surfaces.get(surface).is_none() {
        return;
    }
    let view_formats = view_formats
        .iter()
        .filter_map(|&f| wire::GPUTextureFormat::from_index(f as u32))
        .collect();
    // The plugin's AlphaMode: Opaque 1, PreMultiplied 2; the rest are the
    // browser's default.
    let alpha_mode = match alpha {
        Some(1) => Some(wire::GPUCanvasAlphaMode::Opaque),
        Some(2) => Some(wire::GPUCanvasAlphaMode::Premultiplied),
        _ => None,
    };
    // The drawing buffer takes the configuration's size, in physical pixels,
    // as a native surface does; the page keeps the canvas's CSS size.
    let (width, height) = size;
    if width > 0 && height > 0 {
        s.commands
            .offscreen_canvas_set_width(CANVAS_ELEMENT, &(width as u64));
        s.commands
            .offscreen_canvas_set_height(CANVAS_ELEMENT, &(height as u64));
    }
    s.commands.gpu_canvas_context_configure(
        CANVAS,
        &wire::GPUCanvasConfiguration {
            device: handle(device),
            format,
            usage: usage.map(|u| u as u32),
            view_formats: Some(view_formats),
            tone_mapping: None,
            alpha_mode,
        },
    );
}

pub unsafe fn surface_configure(device: i32, surface: i32, width: i32, height: i32, format: i32) {
    configure(device, surface, (width, height), format, None, &[], None);
}

pub unsafe fn surface_configure_with(
    device: i32,
    surface: i32,
    configuration: &GpuSurfaceConfiguration,
) {
    configure(
        device,
        surface,
        (configuration.width, configuration.height),
        configuration.format,
        configuration.usage,
        &configuration.viewFormats,
        configuration.alphaMode,
    );
}

/// The canvas's texture for this frame, and a view of it to draw to.
pub unsafe fn surface_acquire(surface: i32) -> i32 {
    let mut s = state();
    let Some(entry) = s.surfaces.get(surface) else {
        return 0;
    };
    let mut frame = entry.lock().unwrap();
    if frame.view != 0 {
        return frame.view;
    }
    let texture = s.transient();
    let view = s.views.put(());
    s.commands
        .gpu_canvas_context_get_current_texture(CANVAS, texture);
    s.commands
        .gpu_texture_create_view(texture, handle(view), &None);
    frame.texture = texture.0;
    frame.view = view;
    s.drawing = true;
    view
}

/// Send the frame: the agent runs it, and the page shows it.
pub unsafe fn surface_present(_queue: i32, surface: i32) {
    let mut s = state();
    let Some(entry) = s.surfaces.get(surface) else {
        return;
    };
    let mut frame = entry.lock().unwrap();
    if frame.view != 0 {
        let view = frame.view;
        s.views.remove(view);
        s.commands.release(handle(view));
        s.commands.release(Handle(frame.texture));
        frame.view = 0;
        frame.texture = 0;
    }
    s.drawing = false;
    s.flush();
}

// The plugin's codes for what every canvas presents with: PresentMode's
// Fifo, AlphaMode's Opaque and PreMultiplied, and TextureUsage's
// COPY_SRC | COPY_DST | TEXTURE_BINDING | RENDER_ATTACHMENT.
const CANVAS_PRESENT_MODES: [i32; 1] = [2];
const CANVAS_ALPHA_MODES: [i32; 2] = [1, 2];
const CANVAS_USAGES: i32 = 0x01 | 0x02 | 0x04 | 0x10;

/// What WebGPU lets a canvas be configured with: the browser's preferred
/// format first, then the others a canvas takes.
pub unsafe fn surface_capabilities(surface: i32, adapter: i32) -> i32 {
    let preferred = unsafe { surface_preferred_format(surface, adapter) };
    let mut s = state();
    if s.surfaces.get(surface).is_none() || s.adapters.get(adapter).is_none() {
        return 0;
    }
    let mut formats = Vec::new();
    for format in [
        preferred,
        wire::GPUTextureFormat::Bgra8unorm as i32,
        wire::GPUTextureFormat::Rgba8unorm as i32,
        wire::GPUTextureFormat::Rgba16float as i32,
    ] {
        if format >= 0 && !formats.contains(&format) {
            formats.push(format);
        }
    }
    s.capabilities.put(Capabilities { formats })
}

pub unsafe fn capabilities_destroy(capabilities: i32) {
    state().capabilities.remove(capabilities);
}

fn nth(values: &[i32], index_of: i32) -> i32 {
    match usize::try_from(index_of).ok().and_then(|i| values.get(i)) {
        Some(value) => *value,
        None => {
            host::raise(ErrorKind::Type, "capability index out of range");
            0
        }
    }
}

pub unsafe fn capabilities_format_count(c: i32) -> i32 {
    state().capabilities.get(c).map_or(0, |c| c.formats.len() as i32)
}

pub unsafe fn capabilities_format(c: i32, index_of: i32) -> i32 {
    let Some(c) = state().capabilities.get(c) else {
        return 0;
    };
    nth(&c.formats, index_of)
}

pub unsafe fn capabilities_present_mode_count(c: i32) -> i32 {
    state().capabilities.get(c).map_or(0, |_| CANVAS_PRESENT_MODES.len() as i32)
}

pub unsafe fn capabilities_present_mode(c: i32, index_of: i32) -> i32 {
    if state().capabilities.get(c).is_none() {
        return 0;
    }
    nth(&CANVAS_PRESENT_MODES, index_of)
}

pub unsafe fn capabilities_alpha_mode_count(c: i32) -> i32 {
    state().capabilities.get(c).map_or(0, |_| CANVAS_ALPHA_MODES.len() as i32)
}

pub unsafe fn capabilities_alpha_mode(c: i32, index_of: i32) -> i32 {
    if state().capabilities.get(c).is_none() {
        return 0;
    }
    nth(&CANVAS_ALPHA_MODES, index_of)
}

pub unsafe fn capabilities_usages(c: i32) -> i32 {
    state().capabilities.get(c).map_or(0, |_| CANVAS_USAGES)
}

// -- render pipelines -------------------------------------------------------

/// A render pipeline descriptor's layout: unset is "auto", as the plugin
/// declares it.
pub fn gpu_render_pipeline_descriptor_layout(
    layout: &Option<i32>,
) -> Result<wire::GPUPipelineLayoutOrGPUAutoLayoutMode, String> {
    gpu_compute_pipeline_descriptor_layout(layout)
}

fn create_render_pipeline(device: i32, descriptor: &wire::GPURenderPipelineDescriptor) -> i32 {
    let mut s = state();
    if s.devices.get(device).is_none() {
        return 0;
    }
    let pipeline = s.render_pipelines.put(());
    s.commands
        .gpu_device_create_render_pipeline(handle(device), handle(pipeline), descriptor);
    pipeline
}

pub unsafe fn render_pipeline_create_with(
    device: i32,
    descriptor: &GpuRenderPipelineDescriptor,
) -> i32 {
    converted(descriptor.wire()).map_or(0, |d| create_render_pipeline(device, &d))
}

pub unsafe fn pipeline_begin(device: i32) -> i32 {
    let mut s = state();
    if s.devices.get(device).is_none() {
        return 0;
    }
    s.builders.put(Mutex::new(Build {
        device,
        ..Default::default()
    }))
}

pub unsafe fn builder_destroy(builder: i32) {
    state().builders.remove(builder);
}

/// Run `body` on a builder, or do nothing.
fn building(builder: i32, body: impl FnOnce(&mut Build)) {
    let Some(entry) = state().builders.get(builder) else {
        return;
    };
    body(&mut entry.lock().unwrap());
}

pub unsafe fn pipeline_shader(builder: i32, shader: i32, vs: Text, fs: Text) {
    let (vs, fs) = (vs.as_str().to_owned(), fs.as_str().to_owned());
    building(builder, |b| {
        b.shader = shader;
        b.vertex_entry = vs;
        b.fragment_entry = fs;
    });
}

pub unsafe fn pipeline_layout(builder: i32, layout: i32) {
    building(builder, |b| b.layout = layout);
}

pub unsafe fn pipeline_vertex_buffer(builder: i32, stride: i64, step: i32) {
    building(builder, |b| {
        b.buffers.push(wire::GPUVertexBufferLayout {
            array_stride: stride.max(0) as u64,
            step_mode: wire::GPUVertexStepMode::from_index(step as u32),
            attributes: Vec::new(),
        })
    });
}

pub unsafe fn pipeline_attribute(builder: i32, format: i32, offset: i64, location: i32) {
    let Some(format) = wire::GPUVertexFormat::from_index(format as u32) else {
        return;
    };
    building(builder, |b| {
        if let Some(buffer) = b.buffers.last_mut() {
            buffer.attributes.push(wire::GPUVertexAttribute {
                format,
                offset: offset.max(0) as u64,
                shader_location: location.max(0) as u32,
            });
        }
    });
}

/// Packed against the previous attribute, at the next free location:
/// locations count across the pipeline's buffers, offsets within each.
pub unsafe fn pipeline_attribute_packed(builder: i32, format: i32) {
    let Some(format) = wire::GPUVertexFormat::from_index(format as u32) else {
        return;
    };
    building(builder, |b| {
        let location = b.buffers.iter().map(|l| l.attributes.len()).sum::<usize>() as u32;
        if let Some(buffer) = b.buffers.last_mut() {
            let offset = buffer
                .attributes
                .last()
                .map_or(0, |a| a.offset + vertex_size(a.format));
            buffer.attributes.push(wire::GPUVertexAttribute {
                format,
                offset,
                shader_location: location,
            });
        }
    });
}

/// A vertex format's size in bytes, from its name: `Float32x3` is three
/// 32-bit components, `Unorm1010102` one word.
fn vertex_size(format: wire::GPUVertexFormat) -> u64 {
    let name = format!("{format:?}");
    if name.contains("1010102") {
        return 4;
    }
    let digits = |s: &str| {
        s.chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
    };
    let start = name
        .find(|c: char| c.is_ascii_digit())
        .unwrap_or(name.len());
    let bits: u64 = digits(&name[start..]).parse().unwrap_or(32);
    let count: u64 = name
        .find('x')
        .map_or(1, |x| digits(&name[x + 1..]).parse().unwrap_or(1));
    bits / 8 * count
}

pub unsafe fn pipeline_target(builder: i32, format: i32, write_mask: i32) {
    let Some(format) = wire::GPUTextureFormat::from_index(format as u32) else {
        return;
    };
    building(builder, |b| {
        b.targets.push(wire::GPUColorTargetState {
            format,
            blend: None,
            write_mask: Some(write_mask as u32),
        })
    });
}

pub unsafe fn pipeline_blend(
    builder: i32,
    src: i32,
    dst: i32,
    op: i32,
    src_alpha: i32,
    dst_alpha: i32,
    op_alpha: i32,
) {
    let component = |src: i32, dst: i32, op: i32| wire::GPUBlendComponent {
        operation: wire::GPUBlendOperation::from_index(op as u32),
        src_factor: wire::GPUBlendFactor::from_index(src as u32),
        dst_factor: wire::GPUBlendFactor::from_index(dst as u32),
    };
    building(builder, |b| {
        if let Some(target) = b.targets.last_mut() {
            target.blend = Some(wire::GPUBlendState {
                color: component(src, dst, op),
                alpha: component(src_alpha, dst_alpha, op_alpha),
            });
        }
    });
}

pub unsafe fn pipeline_stencil(
    builder: i32,
    compare: i32,
    fail: i32,
    depth_fail: i32,
    pass_op: i32,
    read_mask: i32,
    write_mask: i32,
) {
    // Both faces the same, as the native builder has them.
    let face = wire::GPUStencilFaceState {
        compare: wire::GPUCompareFunction::from_index(compare as u32),
        fail_op: wire::GPUStencilOperation::from_index(fail as u32),
        depth_fail_op: wire::GPUStencilOperation::from_index(depth_fail as u32),
        pass_op: wire::GPUStencilOperation::from_index(pass_op as u32),
    };
    building(builder, |b| {
        b.stencil = Some((face, read_mask as u32, write_mask as u32))
    });
}

pub unsafe fn pipeline_depth(builder: i32, format: i32, write: bool, compare: i32) {
    let Some(format) = wire::GPUTextureFormat::from_index(format as u32) else {
        return;
    };
    building(builder, |b| {
        b.depth = Some(wire::GPUDepthStencilState {
            format,
            depth_write_enabled: Some(write),
            depth_compare: wire::GPUCompareFunction::from_index(compare as u32),
            stencil_front: None,
            stencil_back: None,
            stencil_read_mask: None,
            stencil_write_mask: None,
            depth_bias: None,
            depth_bias_slope_scale: None,
            depth_bias_clamp: None,
        })
    });
}

pub unsafe fn pipeline_primitive(builder: i32, topology: i32, cull: i32, front: i32) {
    building(builder, |b| {
        b.primitive = Some(wire::GPUPrimitiveState {
            topology: wire::GPUPrimitiveTopology::from_index(topology as u32),
            strip_index_format: None,
            front_face: wire::GPUFrontFace::from_index(front as u32),
            cull_mode: wire::GPUCullMode::from_index(cull as u32),
            unclipped_depth: None,
        })
    });
}

pub unsafe fn render_pipeline_build(builder: i32) -> i32 {
    let Some(entry) = state().builders.get(builder) else {
        return 0;
    };
    let build = std::mem::take(&mut *entry.lock().unwrap());
    state().builders.remove(builder);
    let layout = match build.layout {
        0 => wire::GPUPipelineLayoutOrGPUAutoLayoutMode::GPUAutoLayoutMode(
            wire::GPUAutoLayoutMode::Auto,
        ),
        layout => wire::GPUPipelineLayoutOrGPUAutoLayoutMode::GPUPipelineLayout(handle(layout)),
    };
    // A stride of zero is as wide as the attributes turned out to be.
    let buffers = build
        .buffers
        .into_iter()
        .map(|mut l| {
            if l.array_stride == 0 {
                l.array_stride = l
                    .attributes
                    .iter()
                    .map(|a| a.offset + vertex_size(a.format))
                    .max()
                    .unwrap_or(0);
            }
            Some(l)
        })
        .collect();
    let depth_stencil = build.depth.map(|mut d| {
        if let Some((face, read, write)) = build.stencil {
            d.stencil_front = Some(face.clone());
            d.stencil_back = Some(face);
            d.stencil_read_mask = Some(read);
            d.stencil_write_mask = Some(write);
        }
        d
    });
    create_render_pipeline(
        build.device,
        &wire::GPURenderPipelineDescriptor {
            label: None,
            layout,
            vertex: wire::GPUVertexState {
                module: handle(build.shader),
                entry_point: Some(build.vertex_entry),
                constants: None,
                buffers: Some(buffers),
            },
            primitive: build.primitive,
            depth_stencil,
            multisample: None,
            fragment: Some(wire::GPUFragmentState {
                module: handle(build.shader),
                entry_point: Some(build.fragment_entry),
                constants: None,
                targets: build.targets.into_iter().map(Some).collect(),
            }),
        },
    )
}

// -- texture uploads ----------------------------------------------------------

/// `height` rows of `bytes_per_row` bytes into mip level zero.
pub unsafe fn queue_write_texture(
    queue: i32,
    texture: i32,
    data: Buffer,
    width: i32,
    height: i32,
    bytes_per_row: i32,
) {
    if width <= 0 || height <= 0 || bytes_per_row <= 0 {
        host::raise(ErrorKind::Type, "texture upload dimensions must be positive");
        return;
    }
    let Some(len) = bytes_per_row.checked_mul(height) else {
        host::raise(ErrorKind::Type, "texture upload size overflow");
        return;
    };
    let Some(data) = bytes(&data, len) else {
        return;
    };
    let mut s = state();
    if s.queues.get(queue).is_none() || s.textures.get(texture).is_none() {
        return;
    }
    let staged = s.stage(data);
    s.commands.gpu_queue_write_texture(
        handle(queue),
        &whole(texture),
        &staged,
        &rows(bytes_per_row, height),
        &extent(width, height),
    );
    s.flush_if_full();
}

/// The whole shared buffer, read through `layout`.
pub unsafe fn queue_write_texture_with(
    queue: i32,
    destination: &GpuTexelCopyTextureInfo,
    data: Buffer,
    layout: &GpuTexelCopyBufferLayout,
    size: &GpuExtent3D,
) {
    let (Some(destination), Some(layout), Some(size)) = (
        converted(destination.wire()),
        converted(layout.wire()),
        extent_of(size),
    ) else {
        return;
    };
    let mut s = state();
    if s.queues.get(queue).is_none() {
        return;
    }
    let staged = s.stage(unsafe { data.as_slice() });
    s.commands
        .gpu_queue_write_texture(handle(queue), &destination, &staged, &layout, &size);
    s.flush_if_full();
}

/// WebGPU's timestamps count nanoseconds.
pub unsafe fn queue_timestamp_period(queue: i32) -> f64 {
    if state().queues.get(queue).is_some() { 1.0 } else { 0.0 }
}

// -- features and limits ----------------------------------------------------

/// The names in the GPUSupportedFeatures `get` puts under a handle.
fn feature_names(get: impl FnOnce(&mut wire::Encoder, Handle)) -> Vec<String> {
    let mut s = state();
    let features = s.transient();
    get(&mut s.commands, features);
    let names = answer::<Vec<String>>(&mut s, |c, at| {
        c.gpu_supported_features_values(features, at)
    });
    s.commands.release(features);
    names.unwrap_or_default()
}

/// Whether `names` holds the feature `which` names, the plugin's code being
/// its index in the IDL. The wire names each enum value as its string
/// spelled in PascalCase.
fn has_feature(names: &[String], which: i32) -> bool {
    let Some(feature) = wire::GPUFeatureName::from_index(which as u32) else {
        return false;
    };
    let variant = format!("{feature:?}");
    names.iter().any(|name| {
        let pascal: String = name
            .split('-')
            .map(|word| {
                let mut chars = word.chars();
                chars.next().map_or_else(String::new, |first| {
                    first.to_ascii_uppercase().to_string() + chars.as_str()
                })
            })
            .collect();
        pascal == variant
    })
}

pub unsafe fn adapter_feature(adapter: i32, which: i32) -> bool {
    if state().adapters.get(adapter).is_none() {
        return false;
    }
    let names = feature_names(|c, h| c.gpu_adapter_get_features(handle(adapter), h));
    has_feature(&names, which)
}

pub unsafe fn device_feature(device: i32, which: i32) -> bool {
    if state().devices.get(device).is_none() {
        return false;
    }
    let names = feature_names(|c, h| c.gpu_device_get_features(handle(device), h));
    has_feature(&names, which)
}

/// GPUSupportedLimits' attribute for the plugin's Limit `which`.
fn limit_getter(which: i32) -> Option<fn(&mut wire::Encoder, Handle, u32)> {
    use crate::Limit::*;
    type E = wire::Encoder;
    Some(match crate::Limit::from_native(which)? {
        MaxTextureDimension1D => E::gpu_supported_limits_get_max_texture_dimension1d,
        MaxTextureDimension2D => E::gpu_supported_limits_get_max_texture_dimension2d,
        MaxTextureDimension3D => E::gpu_supported_limits_get_max_texture_dimension3d,
        MaxTextureArrayLayers => E::gpu_supported_limits_get_max_texture_array_layers,
        MaxBindGroups => E::gpu_supported_limits_get_max_bind_groups,
        MaxBindGroupsPlusVertexBuffers => {
            E::gpu_supported_limits_get_max_bind_groups_plus_vertex_buffers
        }
        MaxImmediateSize => E::gpu_supported_limits_get_max_immediate_size,
        MaxBindingsPerBindGroup => E::gpu_supported_limits_get_max_bindings_per_bind_group,
        MaxDynamicUniformBuffersPerPipelineLayout => {
            E::gpu_supported_limits_get_max_dynamic_uniform_buffers_per_pipeline_layout
        }
        MaxDynamicStorageBuffersPerPipelineLayout => {
            E::gpu_supported_limits_get_max_dynamic_storage_buffers_per_pipeline_layout
        }
        MaxSampledTexturesPerShaderStage => {
            E::gpu_supported_limits_get_max_sampled_textures_per_shader_stage
        }
        MaxSamplersPerShaderStage => E::gpu_supported_limits_get_max_samplers_per_shader_stage,
        MaxStorageBuffersPerShaderStage => {
            E::gpu_supported_limits_get_max_storage_buffers_per_shader_stage
        }
        MaxStorageBuffersInVertexStage => {
            E::gpu_supported_limits_get_max_storage_buffers_in_vertex_stage
        }
        MaxStorageBuffersInFragmentStage => {
            E::gpu_supported_limits_get_max_storage_buffers_in_fragment_stage
        }
        MaxStorageTexturesPerShaderStage => {
            E::gpu_supported_limits_get_max_storage_textures_per_shader_stage
        }
        MaxStorageTexturesInVertexStage => {
            E::gpu_supported_limits_get_max_storage_textures_in_vertex_stage
        }
        MaxStorageTexturesInFragmentStage => {
            E::gpu_supported_limits_get_max_storage_textures_in_fragment_stage
        }
        MaxUniformBuffersPerShaderStage => {
            E::gpu_supported_limits_get_max_uniform_buffers_per_shader_stage
        }
        MaxUniformBufferBindingSize => E::gpu_supported_limits_get_max_uniform_buffer_binding_size,
        MaxStorageBufferBindingSize => E::gpu_supported_limits_get_max_storage_buffer_binding_size,
        MinUniformBufferOffsetAlignment => {
            E::gpu_supported_limits_get_min_uniform_buffer_offset_alignment
        }
        MinStorageBufferOffsetAlignment => {
            E::gpu_supported_limits_get_min_storage_buffer_offset_alignment
        }
        MaxVertexBuffers => E::gpu_supported_limits_get_max_vertex_buffers,
        MaxBufferSize => E::gpu_supported_limits_get_max_buffer_size,
        MaxVertexAttributes => E::gpu_supported_limits_get_max_vertex_attributes,
        MaxVertexBufferArrayStride => E::gpu_supported_limits_get_max_vertex_buffer_array_stride,
        MaxInterStageShaderVariables => {
            E::gpu_supported_limits_get_max_inter_stage_shader_variables
        }
        MaxColorAttachments => E::gpu_supported_limits_get_max_color_attachments,
        MaxColorAttachmentBytesPerSample => {
            E::gpu_supported_limits_get_max_color_attachment_bytes_per_sample
        }
        MaxComputeWorkgroupStorageSize => {
            E::gpu_supported_limits_get_max_compute_workgroup_storage_size
        }
        MaxComputeInvocationsPerWorkgroup => {
            E::gpu_supported_limits_get_max_compute_invocations_per_workgroup
        }
        MaxComputeWorkgroupSizeX => E::gpu_supported_limits_get_max_compute_workgroup_size_x,
        MaxComputeWorkgroupSizeY => E::gpu_supported_limits_get_max_compute_workgroup_size_y,
        MaxComputeWorkgroupSizeZ => E::gpu_supported_limits_get_max_compute_workgroup_size_z,
        MaxComputeWorkgroupsPerDimension => {
            E::gpu_supported_limits_get_max_compute_workgroups_per_dimension
        }
    })
}

/// Limit `which` of the GPUSupportedLimits `get` puts under a handle; -1
/// for a limit the plugin or the browser does not know. The wire carries an
/// unsigned long as four bytes and an unsigned long long as eight.
fn limit(which: i32, get: impl FnOnce(&mut wire::Encoder, Handle)) -> i64 {
    let Some(getter) = limit_getter(which) else {
        return -1;
    };
    let mut s = state();
    let limits = s.transient();
    get(&mut s.commands, limits);
    let mut bytes = [0u8; 8];
    let reply = Reply::new(bytes.as_mut_ptr(), bytes.len());
    getter(&mut s.commands, limits, reply.at());
    s.commands.release(limits);
    s.flush();
    match (reply.state.load(SeqCst), reply.len) {
        (1, 4) => i64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        (1, 8) => u64::from_le_bytes(bytes).min(i64::MAX as u64) as i64,
        _ => -1,
    }
}

/// Browser WebGPU is a full implementation, so no downlevel flag is missing.
pub unsafe fn adapter_downlevel(adapter: i32, _which: i32) -> bool {
    state().adapters.get(adapter).is_some()
}

pub unsafe fn adapter_limit(adapter: i32, which: i32) -> i64 {
    if state().adapters.get(adapter).is_none() {
        return 0;
    }
    limit(which, |c, h| c.gpu_adapter_get_limits(handle(adapter), h))
}

pub unsafe fn device_limit(device: i32, which: i32) -> i64 {
    if state().devices.get(device).is_none() {
        return 0;
    }
    limit(which, |c, h| c.gpu_device_get_limits(handle(device), h))
}

// -- compilation messages ---------------------------------------------------

/// One of a GPUCompilationInfo's messages. Its kind is the plugin's
/// CompilationMessageType, GPUCompilationMessageType's index.
#[derive(Clone)]
struct Message {
    text: String,
    kind: i32,
    line: i64,
    column: i64,
    offset: i64,
    length: i64,
}

/// The most objects one sequence may hand back.
const SEQUENCE: u32 = 1 << 16;

/// A message attribute that is an unsigned long long.
fn number(s: &mut State, get: fn(&mut wire::Encoder, Handle, u32), message: Handle) -> i64 {
    answer::<u64>(s, |c, at| get(c, message, at)).map_or(0, |n| n.min(i64::MAX as u64) as i64)
}

/// The messages of the GPUCompilationInfo under `info`. The agent keeps
/// them under transient handles in a row from the first, which are read and
/// released.
fn read_messages(s: &mut State, info: Handle) -> Vec<Message> {
    if s.next + SEQUENCE >= TRANSIENT {
        s.next = 2;
    }
    let first = s.transient();
    let count = answer::<u32>(s, |c, at| {
        c.gpu_compilation_info_get_messages(info, first, at)
    })
    .unwrap_or(0)
    .min(SEQUENCE - 1);
    s.next = first.0 + count;
    let mut messages = Vec::new();
    for i in 0..count {
        let m = Handle(first.0 + i);
        let text = answer::<String>(s, |c, at| c.gpu_compilation_message_get_message(m, at))
            .unwrap_or_default();
        let kind = answer::<wire::GPUCompilationMessageType>(s, |c, at| {
            c.gpu_compilation_message_get_type(m, at)
        })
        .map_or(0, |kind| kind as i32);
        messages.push(Message {
            text,
            kind,
            line: number(s, wire::Encoder::gpu_compilation_message_get_line_num, m),
            column: number(s, wire::Encoder::gpu_compilation_message_get_line_pos, m),
            offset: number(s, wire::Encoder::gpu_compilation_message_get_offset, m),
            length: number(s, wire::Encoder::gpu_compilation_message_get_length, m),
        });
        s.commands.release(m);
    }
    messages
}

/// The messages of compilation info `info`, read from the agent the first
/// time they are asked for.
fn messages(info: i32) -> Option<Vec<Message>> {
    let mut s = state();
    s.compilations.get(info)?;
    if let Some((_, read)) = s.compiled.iter().find(|(owner, _)| *owner == info) {
        return Some(read.clone());
    }
    let read = read_messages(&mut s, handle(info));
    s.compiled.push((info, read.clone()));
    Some(read)
}

/// The `index_of`th message's value, or a raised type error out of range.
fn message<T: Default>(info: i32, index_of: i32, read: impl FnOnce(&Message) -> T) -> T {
    let Some(messages) = messages(info) else {
        return T::default();
    };
    match usize::try_from(index_of).ok().and_then(|i| messages.get(i)) {
        Some(found) => read(found),
        None => {
            host::raise(ErrorKind::Type, "compilation message index out of range");
            T::default()
        }
    }
}

pub unsafe fn compilation_count(info: i32) -> i32 {
    messages(info).map_or(0, |m| m.len() as i32)
}

pub unsafe fn compilation_message(info: i32, index_of: i32) -> Text {
    match message(info, index_of, |m| Some(m.text.clone())) {
        Some(text) => Text::new(&text),
        None => Text::NULL,
    }
}

pub unsafe fn compilation_type(info: i32, index_of: i32) -> i32 {
    message(info, index_of, |m| m.kind)
}

pub unsafe fn compilation_line(info: i32, index_of: i32) -> i64 {
    message(info, index_of, |m| m.line)
}

pub unsafe fn compilation_column(info: i32, index_of: i32) -> i64 {
    message(info, index_of, |m| m.column)
}

pub unsafe fn compilation_offset(info: i32, index_of: i32) -> i64 {
    message(info, index_of, |m| m.offset)
}

pub unsafe fn compilation_length(info: i32, index_of: i32) -> i64 {
    message(info, index_of, |m| m.length)
}

/// The compiler's messages about `shader`, asked for once and kept. The
/// first ask waits for the browser to have compiled it.
fn shader_info(shader: i32) -> Option<Vec<Message>> {
    let mut s = state();
    s.shaders.get(shader)?;
    if let Some((_, kept)) = s.shader_infos.iter().find(|(owner, _)| *owner == shader) {
        return Some(kept.clone());
    }
    let info = s.transient();
    let promise = Promise::new();
    s.commands
        .gpu_shader_module_get_compilation_info(handle(shader), info, promise.reply.at());
    s.flush();
    drop(s);
    answered(&[&promise.reply]);
    let mut s = state();
    let messages = if promise.reply.state.load(SeqCst) == 1 {
        read_messages(&mut s, info)
    } else {
        Vec::new()
    };
    s.commands.release(info);
    s.shader_infos.push((shader, messages.clone()));
    Some(messages)
}

pub unsafe fn shader_compilation_info(shader: i32) -> Future<crate::GpuCompilationInfo> {
    let Some(messages) = shader_info(shader) else {
        return rejected("shader module was destroyed");
    };
    let info = {
        let mut s = state();
        let info = s.compilations.put(());
        s.compiled.push((info, messages));
        info
    };
    let future = Future::new();
    if !future.resolve_boxed(Box::new(crate::GpuCompilationInfo { handle: info })) {
        unsafe { compilation_destroy(info) };
    }
    future
}

/// One message a line, as `line N: what`, or null if the compiler said
/// nothing.
pub unsafe fn shader_messages(shader: i32) -> Text {
    let messages = shader_info(shader).unwrap_or_default();
    if messages.is_empty() {
        return Text::NULL;
    }
    let text = messages
        .iter()
        .map(|m| match m.line {
            0 => m.text.clone(),
            line => format!("line {line}: {}", m.text),
        })
        .collect::<Vec<_>>()
        .join("\n");
    Text::new(&text)
}

// -- objects WebGPU frees itself ----------------------------------------------

pub unsafe fn sampler_destroy(sampler: i32) {
    forget(|s| s.samplers.remove(sampler), sampler);
}

pub unsafe fn pipeline_layout_destroy(layout: i32) {
    forget(|s| s.pipeline_layouts.remove(layout), layout);
}

pub unsafe fn lost_destroy(info: i32) {
    forget(|s| s.lost_infos.remove(info), info);
}

pub unsafe fn compilation_destroy(info: i32) {
    forget(
        |s| {
            s.compilations.remove(info);
            s.compiled.retain(|(owner, _)| *owner != info);
        },
        info,
    );
}

// -- members the declaration holds its own way ------------------------------
//
// The declaration's AttachmentView for WebIDL's GPUTextureOrGPUTextureView,
// which the generated conversion calls these for.

fn attachment(view: &AttachmentView) -> wire::GPUTextureOrGPUTextureView {
    match view {
        AttachmentView::Texture(texture) => {
            wire::GPUTextureOrGPUTextureView::GPUTexture(handle(*texture))
        }
        AttachmentView::TextureView(view) => {
            wire::GPUTextureOrGPUTextureView::GPUTextureView(handle(*view))
        }
    }
}

pub fn gpu_render_pass_color_attachment_view(
    view: &Option<AttachmentView>,
) -> Result<wire::GPUTextureOrGPUTextureView, String> {
    view.as_ref()
        .map(attachment)
        .ok_or_else(|| "a colour attachment has no view".to_owned())
}

pub fn gpu_render_pass_color_attachment_resolve_target(
    target: &Option<AttachmentView>,
) -> Result<Option<wire::GPUTextureOrGPUTextureView>, String> {
    Ok(target.as_ref().map(attachment))
}

pub fn gpu_render_pass_depth_stencil_attachment_view(
    view: &Option<AttachmentView>,
) -> Result<wire::GPUTextureOrGPUTextureView, String> {
    view.as_ref()
        .map(attachment)
        .ok_or_else(|| "a depth attachment has no view".to_owned())
}

// -- validity ---------------------------------------------------------------

/// Whether `h` is a live handle of the kind it carries.
pub unsafe fn is_valid(h: i32) -> bool {
    state().has(h)
}
