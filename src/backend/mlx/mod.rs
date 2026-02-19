// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Apple MLX backend via the mlx-c library.
//!
//! This module provides safe Rust wrappers around the MLX C API for
//! tensor operations on Apple Silicon with Metal GPU acceleration.

pub mod ffi;
pub mod array;
pub mod stream;
pub mod ops;
pub mod io;
pub mod signal;
