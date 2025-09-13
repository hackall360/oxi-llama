use std::f32;

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
    pub fn floats(&self) -> Vec<f32> { self.data.clone() }
    pub fn shape(&self) -> &[usize] { &self.shape }
}

#[derive(Clone, Debug)]
struct Token {
    pos: i32,
    data: Vec<f32>,
    chunk_start: i32,
}

#[derive(Clone, Debug)]
pub struct Batch {
    pub positions: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct Causal {
    swa_window_size: i32,
    swa_memory_size: i32,
    chunk_size: i32,

    tokens: Vec<Token>,
    cur_positions: Vec<i32>,
    embed: usize,
    heads: usize,
}

impl Causal {
    pub fn new() -> Self {
        Self {
            swa_window_size: i32::MAX,
            swa_memory_size: i32::MAX,
            chunk_size: 0,
            tokens: Vec::new(),
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

    pub fn put(&mut self, key: &Tensor, _value: &Tensor) {
        self.embed = key.shape[0];
        self.heads = key.shape[1];
        let batch = key.shape[2];
        let mut chunk_start = if self.chunk_size > 0 {
            self.tokens.last().map(|t| t.chunk_start).unwrap_or(0)
        } else { 0 };
        let mut chunk_len = if self.chunk_size > 0 {
            self.tokens
                .last()
                .map(|t| t.pos - chunk_start + 1)
                .unwrap_or(0)
        } else { 0 };
        let block = self.embed * self.heads;
        let had_tokens = self.tokens.len();
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
            let slice = key.data[start..end].to_vec();
            self.tokens.push(Token { pos, data: slice, chunk_start });
            if self.chunk_size > 0 {
                chunk_len += 1;
            }
        }
        if self.swa_memory_size != i32::MAX && had_tokens > 0 {
            while self.tokens.len() as i32 > self.swa_memory_size {
                self.tokens.remove(0);
            }
        }
    }

    pub fn get(&self) -> (Tensor, Tensor, Tensor) {
        let history = self.tokens.len();
        if history == 0 {
            return (
                Tensor::new(vec![], &[self.embed, self.heads, 0]),
                Tensor::new(vec![], &[self.embed, self.heads, 0]),
                Tensor::new(vec![], &[0, 0]),
            );
        }
        let mut data = Vec::with_capacity(self.embed * self.heads * history);
        for tok in &self.tokens {
            data.extend_from_slice(&tok.data);
        }
        let out = Tensor::new(data.clone(), &[self.embed, self.heads, history]);
        let val = Tensor::new(data, &[self.embed, self.heads, history]);

        let batch = self.cur_positions.len();
        let mut mask = vec![f32::NEG_INFINITY; batch * history];
        for (row, pos) in self.cur_positions.iter().enumerate() {
            for (col, tok) in self.tokens.iter().enumerate() {
                if tok.pos > *pos { continue; }
                if tok.pos < pos - self.swa_window_size { continue; }
                if self.chunk_size > 0 {
                    let start = tok.chunk_start;
                    if *pos < start { continue; }
                    if tok.pos < start { continue; }
                    if tok.pos > *pos { continue; }
                    if *pos - start >= self.chunk_size { continue; }
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

