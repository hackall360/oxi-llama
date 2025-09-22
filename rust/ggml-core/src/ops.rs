use smallvec::{smallvec, SmallVec};

use crate::{
    arena::MemoryArena,
    dtype::DType,
    error::{Error, Result},
    tensor::{TensorData, TensorId},
};

pub(crate) trait OpKernel {
    fn eval(
        &self,
        tensors: &mut [TensorData],
        arena: &mut MemoryArena,
        id: TensorId,
        inputs: &[TensorId],
    ) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct Operation {
    kind: OperationKind,
    inputs: SmallVec<[TensorId; 4]>,
}

impl Operation {
    pub fn constant() -> Self {
        Self {
            kind: OperationKind::Constant,
            inputs: SmallVec::new(),
        }
    }

    pub fn parameter() -> Self {
        Self {
            kind: OperationKind::Parameter,
            inputs: SmallVec::new(),
        }
    }

    pub fn unary(kind: UnaryOpKind, input: TensorId) -> Self {
        Self {
            kind: OperationKind::Unary(kind),
            inputs: smallvec![input],
        }
    }

    pub fn binary(kind: BinaryOpKind, lhs: TensorId, rhs: TensorId) -> Self {
        Self {
            kind: OperationKind::Binary(kind),
            inputs: smallvec![lhs, rhs],
        }
    }

    pub fn matmul(lhs: TensorId, rhs: TensorId) -> Self {
        Self {
            kind: OperationKind::MatMul,
            inputs: smallvec![lhs, rhs],
        }
    }

    pub fn kind(&self) -> &OperationKind {
        &self.kind
    }

    pub fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }
}

#[derive(Clone, Debug)]
pub enum OperationKind {
    Constant,
    Parameter,
    Unary(UnaryOpKind),
    Binary(BinaryOpKind),
    MatMul,
}

#[derive(Clone, Copy, Debug)]
pub enum UnaryOpKind {
    Identity,
    Neg,
    Exp,
    Relu,
}

#[derive(Clone, Copy, Debug)]
pub enum BinaryOpKind {
    Add,
    Mul,
}

impl OpKernel for OperationKind {
    fn eval(
        &self,
        tensors: &mut [TensorData],
        arena: &mut MemoryArena,
        id: TensorId,
        inputs: &[TensorId],
    ) -> Result<()> {
        match self {
            OperationKind::Constant => {
                tensors[id.index()].mark_computed();
                Ok(())
            }
            OperationKind::Parameter => {
                if tensors[id.index()].is_computed() {
                    Ok(())
                } else {
                    Err(Error::UninitializedTensor(id))
                }
            }
            OperationKind::Unary(kind) => kind.eval(tensors, arena, id, inputs),
            OperationKind::Binary(kind) => kind.eval(tensors, arena, id, inputs),
            OperationKind::MatMul => matmul_eval(tensors, arena, id, inputs),
        }
    }
}

impl UnaryOpKind {
    fn eval(
        &self,
        tensors: &mut [TensorData],
        arena: &mut MemoryArena,
        id: TensorId,
        inputs: &[TensorId],
    ) -> Result<()> {
        let input_id = inputs
            .first()
            .copied()
            .ok_or_else(|| Error::Operation("unary op missing input".into()))?;
        let (input_range, input_dtype) = {
            let tensors_ref: &[_] = &*tensors;
            let input = &tensors_ref[input_id.index()];
            (input.storage(), input.dtype)
        };
        let (output_range, output_dtype) = {
            let tensors_ref: &[_] = &*tensors;
            let output = &tensors_ref[id.index()];
            (output.storage(), output.dtype)
        };
        if input_dtype != DType::F32 {
            return Err(Error::dtype_mismatch(DType::F32, input_dtype));
        }
        if output_dtype != DType::F32 {
            return Err(Error::dtype_mismatch(DType::F32, output_dtype));
        }

        let input_values = {
            let view = arena.read_f32(&input_range);
            view.to_vec()
        };
        let mut out = vec![0.0f32; input_values.len()];
        match self {
            UnaryOpKind::Identity => out.copy_from_slice(&input_values),
            UnaryOpKind::Neg => {
                for (o, v) in out.iter_mut().zip(input_values.iter()) {
                    *o = -*v;
                }
            }
            UnaryOpKind::Exp => {
                for (o, v) in out.iter_mut().zip(input_values.iter()) {
                    *o = v.exp();
                }
            }
            UnaryOpKind::Relu => {
                for (o, v) in out.iter_mut().zip(input_values.iter()) {
                    *o = v.max(0.0);
                }
            }
        }
        arena.write_f32(&output_range, &out);
        tensors[id.index()].mark_computed();
        Ok(())
    }
}

impl BinaryOpKind {
    fn eval(
        &self,
        tensors: &mut [TensorData],
        arena: &mut MemoryArena,
        id: TensorId,
        inputs: &[TensorId],
    ) -> Result<()> {
        if inputs.len() != 2 {
            return Err(Error::Operation("binary op expects two inputs".into()));
        }
        let lhs_id = inputs[0];
        let rhs_id = inputs[1];
        let (lhs_range, lhs_dtype, rhs_range, rhs_dtype, output_range) = {
            let tensors_ref: &[_] = &*tensors;
            let lhs = &tensors_ref[lhs_id.index()];
            let rhs = &tensors_ref[rhs_id.index()];
            let out = &tensors_ref[id.index()];
            (
                lhs.storage(),
                lhs.dtype,
                rhs.storage(),
                rhs.dtype,
                out.storage(),
            )
        };
        let out_dtype = tensors[id.index()].dtype;
        if lhs_dtype != DType::F32 {
            return Err(Error::dtype_mismatch(DType::F32, lhs_dtype));
        }
        if rhs_dtype != DType::F32 {
            return Err(Error::dtype_mismatch(DType::F32, rhs_dtype));
        }
        if out_dtype != DType::F32 {
            return Err(Error::dtype_mismatch(DType::F32, out_dtype));
        }
        let lhs_values = {
            let view = arena.read_f32(&lhs_range);
            view.to_vec()
        };
        let rhs_values = {
            let view = arena.read_f32(&rhs_range);
            view.to_vec()
        };
        let mut out = vec![0.0f32; lhs_values.len()];
        match self {
            BinaryOpKind::Add => {
                for i in 0..out.len() {
                    out[i] = lhs_values[i] + rhs_values[i];
                }
            }
            BinaryOpKind::Mul => {
                for i in 0..out.len() {
                    out[i] = lhs_values[i] * rhs_values[i];
                }
            }
        }
        arena.write_f32(&output_range, &out);
        tensors[id.index()].mark_computed();
        Ok(())
    }
}

fn matmul_eval(
    tensors: &mut [TensorData],
    arena: &mut MemoryArena,
    id: TensorId,
    inputs: &[TensorId],
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Operation("matmul expects two inputs".into()));
    }
    let lhs_id = inputs[0];
    let rhs_id = inputs[1];
    let (lhs_range, lhs_shape, rhs_range, rhs_shape, out_range) = {
        let tensors_ref: &[_] = &*tensors;
        let lhs = &tensors_ref[lhs_id.index()];
        let rhs = &tensors_ref[rhs_id.index()];
        let out = &tensors_ref[id.index()];
        (
            lhs.storage(),
            lhs.shape.clone(),
            rhs.storage(),
            rhs.shape.clone(),
            out.storage(),
        )
    };
    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(Error::Operation("matmul only supports 2D tensors".into()));
    }
    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let k_rhs = rhs_shape[0];
    let n = rhs_shape[1];
    if k != k_rhs {
        return Err(Error::Operation("matmul inner dimensions mismatch".into()));
    }
    let lhs_values = {
        let view = arena.read_f32(&lhs_range);
        view.to_vec()
    };
    let rhs_values = {
        let view = arena.read_f32(&rhs_range);
        view.to_vec()
    };
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                let lhs_index = row * k + inner;
                let rhs_index = inner * n + col;
                acc += lhs_values[lhs_index] * rhs_values[rhs_index];
            }
            out[row * n + col] = acc;
        }
    }
    arena.write_f32(&out_range, &out);
    tensors[id.index()].mark_computed();
    Ok(())
}
