use std::{fs::File, io::Write};

use convert::tensor::{BaseTensor, Tensor, TensorKind};
use convert::reader::{FsReader, ModelReader};

#[test]
fn tensor_kind_from_shape() {
    let t = BaseTensor::new("w", vec![2, 2], vec![0.0; 4]);
    assert_eq!(t.kind(), TensorKind::F16);
    let t = BaseTensor::new("b", vec![4], vec![0.0; 4]);
    assert_eq!(t.kind(), TensorKind::F32);
    let t = BaseTensor::new("x.bias", vec![2, 2], vec![0.0; 4]);
    assert_eq!(t.kind(), TensorKind::F32);
}

#[test]
fn repacker_invoked_on_write() {
    let mut t = BaseTensor::new("w", vec![2], vec![1.0, 2.0]);
    t.set_repacker(|_, data, _| Ok(data.iter().map(|v| v * 2.0).collect()));
    let mut buf = Vec::new();
    t.write_to(&mut buf).unwrap();
    let mut out = [0u8; 8];
    out.copy_from_slice(&buf);
    let v1 = f32::from_le_bytes(out[0..4].try_into().unwrap());
    let v2 = f32::from_le_bytes(out[4..8].try_into().unwrap());
    assert_eq!((v1, v2), (2.0, 4.0));
}

#[test]
fn fs_reader_loads_tensors() {
    let tmp = tempfile::tempdir().unwrap();
    let mut f = File::create(tmp.path().join("a.tensor")).unwrap();
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    let reader = FsReader::new(tmp.path());
    let tensors = reader.read_tensors().unwrap();
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].shape(), &[1]);
}
