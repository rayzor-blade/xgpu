//! Runtime-neutral pieces shared by every xgpu adapter.
//!
//! GPU resources stay in Rust and cross language boundaries as typed,
//! generational integer handles. Runtime adapters supply their own text,
//! byte-buffer, error, and future carriers around the generated API.

mod handles;
mod types;

pub use handles::{Slab, kind_of};
pub use types::Kind;
