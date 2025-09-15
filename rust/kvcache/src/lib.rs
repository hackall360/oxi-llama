use std::collections::HashMap;

/// Minimal tensor representation backed by a contiguous `Vec` of
/// `f32` values. This keeps memory usage compact without depending on
/// an external tensor library.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Self {
        Self { data, shape: shape.to_vec() }
    }

    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        Self::new(data.to_vec(), shape)
    }

    pub fn floats(&self) -> Vec<f32> {
        self.data.clone()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Batch of token positions for a forward pass.
#[derive(Clone, Debug)]
pub struct Batch {
    pub positions: Vec<i32>,
}

/// Causal cache storing key/value history for autoregressive models.
///
/// The implementation keeps keys and values in contiguous vectors for
/// memory efficiency. Positions and chunk starts are tracked separately
/// to avoid per-token allocations.
#[derive(Clone, Debug)]
pub struct Causal {
    swa_window_size: i32,
    swa_memory_size: i32,
    chunk_size: i32,

    // Stored history
    keys: Vec<f32>,
    values: Vec<f32>,
    positions: Vec<i32>,
    chunk_starts: Vec<i32>,

    // Current forward pass positions
    cur_positions: Vec<i32>,

    // Tensor dimensions
    embed: usize,
    heads: usize,
}

impl Causal {
    pub fn new() -> Self {
        Self {
            swa_window_size: i32::MAX,
            swa_memory_size: i32::MAX,
            chunk_size: 0,
            keys: Vec::new(),
            values: Vec::new(),
            positions: Vec::new(),
            chunk_starts: Vec::new(),
            cur_positions: Vec::new(),
            embed: 0,
            heads: 0,
        }
    }
    pub fn new_swa(window: i32) -> Self { Self { swa_window_size: window, ..Self::new() } }
    pub fn new_swa_mem(window: i32, mem: i32) -> Self { Self { swa_window_size: window, swa_memory_size: mem, ..Self::new() } }
    pub fn new_chunked(chunk: i32) -> Self { Self { chunk_size: chunk, ..Self::new() } }

    pub fn init(&mut self, _dtype: DType, _max_seq: usize, _capacity: usize, _max_batch: usize) {}

    pub fn start_forward(&mut self, batch: Batch, _reserve: bool) {
        self.cur_positions = batch.positions;
    }

    pub fn set_layer(&mut self, _layer: usize) {}

    pub fn put(&mut self, key: &Tensor, value: &Tensor) {
        self.embed = key.shape[0];
        self.heads = key.shape[1];
        let batch = key.shape[2];
        let block = self.embed * self.heads;

        // Determine current chunk state
        let mut chunk_start = if self.chunk_size > 0 {
            *self.chunk_starts.last().unwrap_or(&0)
        } else {
            0
        };
        let mut chunk_len = if self.chunk_size > 0 {
            self.positions
                .last()
                .map(|p| p - chunk_start + 1)
                .unwrap_or(0)
        } else {
            0
        };

        let had_tokens = self.positions.len();
        for b in 0..batch {
            let pos = self.cur_positions[b];
            if self.chunk_size > 0 {
                if chunk_len >= self.chunk_size || pos != chunk_start + chunk_len {
                    chunk_start = pos;
                    chunk_len = 0;
                }
            }

            let start = b * block;
            let end = start + block;
            self.keys.extend_from_slice(&key.data[start..end]);
            self.values.extend_from_slice(&value.data[start..end]);
            self.positions.push(pos);
            self.chunk_starts.push(chunk_start);
            if self.chunk_size > 0 {
                chunk_len += 1;
            }
        }

        if self.swa_memory_size != i32::MAX && had_tokens > 0 {
            while self.positions.len() as i32 > self.swa_memory_size {
                self.positions.remove(0);
                self.chunk_starts.remove(0);
                self.keys.drain(0..block);
                self.values.drain(0..block);
            }
        }
    }

    pub fn get(&self) -> (Tensor, Tensor, Tensor) {
        let history = self.positions.len();
        if history == 0 {
            return (
                Tensor::new(vec![], &[self.embed, self.heads, 0]),
                Tensor::new(vec![], &[self.embed, self.heads, 0]),
                Tensor::new(vec![], &[0, 0]),
            );
        }

        let out = Tensor::new(self.keys.clone(), &[self.embed, self.heads, history]);
        let val = Tensor::new(self.values.clone(), &[self.embed, self.heads, history]);

        let batch = self.cur_positions.len();
        let mut mask = vec![f32::NEG_INFINITY; batch * history];
        for (row, pos) in self.cur_positions.iter().enumerate() {
            for col in 0..history {
                let tok_pos = self.positions[col];
                if tok_pos > *pos {
                    continue;
                }
                if tok_pos < pos - self.swa_window_size {
                    continue;
                }
                if self.chunk_size > 0 {
                    let start = self.chunk_starts[col];
                    if *pos < start || tok_pos < start || tok_pos > *pos {
                        continue;
                    }
                    if *pos - start >= self.chunk_size {
                        continue;
                    }
                }
                mask[row * history + col] = 0.0;
            }
        }
        let m = Tensor::new(mask, &[batch, history]);
        (out, val, m)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DType {
    F16,
    F32,
    Other,
}

/// Simple encoder cache that stores a single key/value tensor per layer.
/// The tensors are position independent and the returned mask is always
/// empty. This mirrors the behaviour of the Go implementation used for
/// vision encoders.
#[derive(Clone, Debug)]
pub struct EncoderCache {
    cur_layer: usize,
    cur_pos: i32,
    cur_reserve: bool,

    encoder_cached: bool,
    encoder_pos: i32,

    keys: HashMap<usize, Tensor>,
    values: HashMap<usize, Tensor>,
}

impl EncoderCache {
    pub fn new() -> Self {
        Self {
            cur_layer: 0,
            cur_pos: 0,
            cur_reserve: false,
            encoder_cached: false,
            encoder_pos: 0,
            keys: HashMap::new(),
            values: HashMap::new(),
        }
    }

    pub fn init(&mut self, _dtype: DType, _max_seq: usize, _capacity: usize, _max_batch: usize) {}

    pub fn start_forward(&mut self, batch: Batch, reserve: bool) {
        self.cur_pos = batch.positions.last().copied().unwrap_or(0);
        self.cur_reserve = reserve;
    }

    pub fn set_layer(&mut self, layer: usize) {
        self.cur_layer = layer;
    }

    pub fn encoder_cached(&self) -> bool {
        self.encoder_cached
    }

    pub fn get(&self) -> (Tensor, Tensor, Tensor) {
        let k = self
            .keys
            .get(&self.cur_layer)
            .cloned()
            .unwrap_or_else(|| Tensor::new(vec![], &[]));
        let v = self
            .values
            .get(&self.cur_layer)
            .cloned()
            .unwrap_or_else(|| Tensor::new(vec![], &[]));
        (k, v, Tensor::new(vec![], &[0, 0]))
    }

    pub fn put(&mut self, key: &Tensor, value: &Tensor) {
        if !self.cur_reserve {
            self.encoder_pos = self.cur_pos;
            self.encoder_cached = true;
        }
        self.keys.insert(self.cur_layer, key.clone());
        self.values.insert(self.cur_layer, value.clone());
    }

    pub fn copy_prefix(&mut self, _src: usize, _dst: usize, _len: i32) {}

    pub fn can_resume(&self, _seq: usize, _pos: i32) -> bool {
        true
    }

    pub fn remove(&mut self, _seq: usize, begin: i32, end: i32) {
        if self.encoder_cached && self.encoder_pos >= begin && self.encoder_pos < end {
            self.encoder_cached = false;
        }
    }
}

