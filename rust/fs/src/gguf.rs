use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::Result;

use crate::ggml::Tensor;

const MAGIC: &[u8; 4] = b"GGUF";
const VERSION: u32 = 3;
const TENSOR_TYPE_F32: u32 = 0;
const ALIGNMENT: u64 = 32;

/// Representation of a GGUF key-value value.
#[derive(Clone, Debug)]
pub enum Value {
    Uint32(u32),
    Float32(f32),
    Bool(bool),
    String(String),
    Int32Array(Vec<i32>),
    StringArray(Vec<String>),
    Float32Array(Vec<f32>),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::Uint32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Value::StringArray(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            Value::Float32Array(v) => Some(v),
            _ => None,
        }
    }
}

/// Tensor metadata read from a GGUF file.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub kind: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn num_bytes(&self) -> u64 {
        // Only f32 tensors are used in the tests.
        self.shape.iter().product::<u64>() * 4
    }
}

pub struct GgufFile {
    file: File,
    kv: HashMap<String, Value>,
    tensors: Vec<TensorInfo>,
    data_offset: u64,
}

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic").into());
        }
        let version = read_u32(&mut f)?;
        if version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad version").into());
        }
        let n_tensors = read_u64(&mut f)? as usize;
        let n_kv = read_u64(&mut f)? as usize;

        let mut kv = HashMap::new();
        for _ in 0..n_kv {
            let key = read_string(&mut f)?;
            let ty = read_u32(&mut f)?;
            let val = match ty {
                4 => Value::Uint32(read_u32(&mut f)?),
                6 => Value::Float32(read_f32(&mut f)?),
                7 => Value::Bool(read_u8(&mut f)? != 0),
                8 => Value::String(read_string(&mut f)?),
                9 => {
                    let elem_ty = read_u32(&mut f)?;
                    let n = read_u64(&mut f)? as usize;
                    match elem_ty {
                        5 => {
                            let mut v = Vec::with_capacity(n);
                            for _ in 0..n {
                                v.push(read_i32(&mut f)?);
                            }
                            Value::Int32Array(v)
                        }
                        6 => {
                            let mut v = Vec::with_capacity(n);
                            for _ in 0..n {
                                v.push(read_f32(&mut f)?);
                            }
                            Value::Float32Array(v)
                        }
                        8 => {
                            let mut v = Vec::with_capacity(n);
                            for _ in 0..n {
                                v.push(read_string(&mut f)?);
                            }
                            Value::StringArray(v)
                        }
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "unsupported array type",
                            )
                            .into())
                        }
                    }
                }
                _ => {
                    return Err(
                        io::Error::new(io::ErrorKind::InvalidData, "unsupported type").into(),
                    )
                }
            };
            kv.insert(key, val);
        }

        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(&mut f)?;
            let ndims = read_u32(&mut f)? as usize;
            let mut shape = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                shape.push(read_u64(&mut f)?);
            }
            let kind = read_u32(&mut f)?;
            let offset = read_u64(&mut f)?;
            tensors.push(TensorInfo {
                name,
                shape,
                kind,
                offset,
            });
        }

        let pos = f.seek(SeekFrom::Current(0))?;
        let padding = (ALIGNMENT - (pos % ALIGNMENT)) % ALIGNMENT;
        f.seek(SeekFrom::Current(padding as i64))?;
        let data_offset = pos + padding;

        Ok(GgufFile {
            file: f,
            kv,
            tensors,
            data_offset,
        })
    }

    pub fn key_value(&self, key: &str) -> Option<&Value> {
        self.kv.get(key)
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn num_key_values(&self) -> usize {
        self.kv.len()
    }

    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor_reader(&mut self, name: &str) -> io::Result<(TensorInfo, Vec<u8>)> {
        let ti = self
            .tensor_info(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "tensor"))?
            .clone();
        self.file
            .seek(SeekFrom::Start(self.data_offset + ti.offset))?;
        let mut data = vec![0u8; ti.num_bytes() as usize];
        self.file.read_exact(&mut data)?;
        Ok((ti, data))
    }
}

pub fn write_gguf<P: AsRef<Path>>(
    path: P,
    kv: &HashMap<String, Value>,
    tensors: &[Tensor],
) -> Result<()> {
    let mut f = File::create(path)?;

    // header
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&(tensors.len() as u64).to_le_bytes())?;
    f.write_all(&(kv.len() as u64).to_le_bytes())?;

    // key-values
    for (k, v) in kv {
        let kbytes = k.as_bytes();
        f.write_all(&(kbytes.len() as u64).to_le_bytes())?;
        f.write_all(kbytes)?;
        write_value(&mut f, v)?;
    }

    // tensor infos
    let mut offset = 0u64;
    for t in tensors {
        let name_bytes = t.name.as_bytes();
        f.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
        f.write_all(name_bytes)?;
        f.write_all(&(t.shape.len() as u32).to_le_bytes())?;
        for &dim in &t.shape {
            f.write_all(&dim.to_le_bytes())?;
        }
        f.write_all(&t.kind.to_le_bytes())?;
        f.write_all(&offset.to_le_bytes())?;

        offset += t.num_bytes();
        let pad = (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
        offset += pad;
    }

    // align to data section
    let pos = f.seek(SeekFrom::Current(0))?;
    let pad = (ALIGNMENT - (pos % ALIGNMENT)) % ALIGNMENT;
    if pad > 0 {
        f.write_all(&vec![0u8; pad as usize])?;
    }

    // tensor data
    let mut written = 0u64;
    for t in tensors {
        f.write_all(&t.data)?;
        written += t.num_bytes();
        let pad = (ALIGNMENT - (written % ALIGNMENT)) % ALIGNMENT;
        if pad > 0 {
            f.write_all(&vec![0u8; pad as usize])?;
            written += pad;
        }
    }

    Ok(())
}

fn write_value<W: Write>(w: &mut W, v: &Value) -> io::Result<()> {
    match v {
        Value::Uint32(n) => {
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&n.to_le_bytes())
        }
        Value::Float32(fv) => {
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&fv.to_le_bytes())
        }
        Value::Bool(b) => {
            w.write_all(&7u32.to_le_bytes())?;
            w.write_all(&[*b as u8])
        }
        Value::String(s) => {
            w.write_all(&8u32.to_le_bytes())?;
            let bytes = s.as_bytes();
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(bytes)
        }
        Value::Int32Array(vs) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&5u32.to_le_bytes())?; // element type
            w.write_all(&(vs.len() as u64).to_le_bytes())?;
            for v in vs {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        }
        Value::StringArray(vs) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&8u32.to_le_bytes())?;
            w.write_all(&(vs.len() as u64).to_le_bytes())?;
            for s in vs {
                let bytes = s.as_bytes();
                w.write_all(&(bytes.len() as u64).to_le_bytes())?;
                w.write_all(bytes)?;
            }
            Ok(())
        }
        Value::Float32Array(vs) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&(vs.len() as u64).to_le_bytes())?;
            for v in vs {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        }
    }
}

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let n = read_u64(r)? as usize;
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf).unwrap_or_default())
}
