use std::ops::Range;

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{Error, Result};

/// A linear bump-allocated arena mirroring the `ggml` memory model.
#[derive(Debug)]
pub struct MemoryArena {
    buffer: Vec<u8>,
    offset: usize,
    alignment: usize,
}

impl MemoryArena {
    /// Creates a new arena with the provided capacity and base alignment.
    pub fn new(capacity: usize, alignment: usize) -> Self {
        Self {
            buffer: vec![0; capacity],
            offset: 0,
            alignment: alignment.max(1),
        }
    }

    /// Returns the total arena capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the amount of memory used so far.
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Allocates a contiguous block of memory from the arena.
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Result<Range<usize>> {
        let alignment = alignment.max(self.alignment);
        if alignment.is_power_of_two() {
            let aligned_offset = align_to(self.offset, alignment);
            let end = aligned_offset.saturating_add(size);
            if end > self.buffer.len() {
                return Err(Error::OutOfMemory {
                    requested: size,
                    available: self.buffer.len().saturating_sub(self.offset),
                });
            }
            self.offset = end;
            Ok(aligned_offset..end)
        } else {
            Err(Error::InvalidAlignment(alignment))
        }
    }

    /// Writes a raw slice of bytes into the arena at the provided range.
    pub fn write_bytes(&mut self, range: &Range<usize>, data: &[u8]) {
        let dst = &mut self.buffer[range.start..range.end];
        dst.copy_from_slice(data);
    }

    /// Reads a raw slice of bytes from the arena.
    pub fn read_bytes(&self, range: &Range<usize>) -> &[u8] {
        &self.buffer[range.start..range.end]
    }

    /// Reads a `f32` slice from the arena.
    pub fn read_f32(&self, range: &Range<usize>) -> &[f32] {
        cast_slice(self.read_bytes(range))
    }

    /// Writes a `f32` slice into the arena.
    pub fn write_f32(&mut self, range: &Range<usize>, values: &[f32]) {
        let dst = &mut self.buffer[range.start..range.end];
        let dst_f32 = cast_slice_mut(dst);
        dst_f32.copy_from_slice(values);
    }

    /// Resets the arena so subsequent allocations start from the beginning.
    pub fn reset(&mut self) {
        self.offset = 0;
        for byte in &mut self.buffer {
            *byte = 0;
        }
    }
}

fn align_to(value: usize, alignment: usize) -> usize {
    let mask = alignment - 1;
    (value + mask) & !mask
}
