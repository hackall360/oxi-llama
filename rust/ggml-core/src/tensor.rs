use std::{fmt, ops::Range};

use smallvec::SmallVec;

use crate::{
    context::Context,
    dtype::DType,
    error::Result,
    graph::ComputationGraph,
    ops::{BinaryOpKind, Operation, UnaryOpKind},
};

pub type Shape = SmallVec<[usize; 4]>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub(crate) usize);

impl TensorId {
    pub(crate) fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor#{}", self.0)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TensorData {
    pub(crate) name: Option<String>,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) operation: Operation,
    pub(crate) storage: Range<usize>,
    pub(crate) computed: bool,
}

impl TensorData {
    pub(crate) fn new(
        name: Option<String>,
        shape: Shape,
        dtype: DType,
        operation: Operation,
        storage: Range<usize>,
        computed: bool,
    ) -> Self {
        Self {
            name,
            shape,
            dtype,
            operation,
            storage,
            computed,
        }
    }

    pub(crate) fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub(crate) fn storage(&self) -> Range<usize> {
        self.storage.clone()
    }

    pub(crate) fn mark_computed(&mut self) {
        self.computed = true;
    }

    pub(crate) fn mark_pending(&mut self) {
        self.computed = false;
    }

    pub(crate) fn is_computed(&self) -> bool {
        self.computed
    }

    pub(crate) fn operation(&self) -> &Operation {
        &self.operation
    }
}

#[derive(Clone)]
pub struct Tensor {
    pub(crate) ctx: Context,
    pub(crate) id: TensorId,
}

impl Tensor {
    pub(crate) fn new(ctx: Context, id: TensorId) -> Self {
        Self { ctx, id }
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn context(&self) -> &Context {
        &self.ctx
    }

    pub fn dtype(&self) -> DType {
        self.ctx.tensor_dtype(self.id)
    }

    pub fn shape(&self) -> Shape {
        self.ctx.tensor_shape(self.id)
    }

    pub fn elements(&self) -> usize {
        self.ctx.tensor_elements(self.id)
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        self.ctx.tensor_to_vec(self.id)
    }

    pub fn set_f32(&self, data: &[f32]) -> Result<()> {
        self.ctx.tensor_set_f32(self.id, data)
    }

    pub fn set_name<S: Into<String>>(&self, name: S) -> Result<()> {
        self.ctx.tensor_set_name(self.id, name.into())
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.ctx.binary_op(self, other, BinaryOpKind::Add)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        self.ctx.binary_op(self, other, BinaryOpKind::Mul)
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        self.ctx.matmul(self, other)
    }

    pub fn apply_unary(&self, kind: UnaryOpKind) -> Result<Tensor> {
        self.ctx.unary_op(self, kind)
    }

    pub fn graph(&self) -> ComputationGraph {
        let mut graph = ComputationGraph::new(self.ctx.clone());
        graph.add(self);
        graph
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.id)
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish()
    }
}
