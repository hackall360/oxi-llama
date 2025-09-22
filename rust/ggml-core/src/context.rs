use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use smallvec::SmallVec;

use crate::{
    arena::MemoryArena,
    dtype::DType,
    error::{Error, Result},
    ops::{BinaryOpKind, OpKernel, Operation, OperationKind, UnaryOpKind},
    tensor::{Shape, Tensor, TensorData, TensorId},
};

#[derive(Clone, Debug)]
pub struct Context {
    pub(crate) inner: Rc<ContextInner>,
}

#[derive(Debug)]
pub(crate) struct ContextInner {
    pub(crate) arena: RefCell<MemoryArena>,
    pub(crate) tensors: RefCell<Vec<TensorData>>,
    next_id: Cell<usize>,
}

#[derive(Clone, Debug)]
pub struct ContextParams {
    pub memory_size: usize,
    pub memory_alignment: usize,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            memory_size: 16 * 1024 * 1024,
            memory_alignment: 32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ContextBuilder {
    params: ContextParams,
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self {
            params: ContextParams::default(),
        }
    }
}

impl ContextBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn memory_size(mut self, bytes: usize) -> Self {
        self.params.memory_size = bytes;
        self
    }

    pub fn memory_alignment(mut self, alignment: usize) -> Self {
        self.params.memory_alignment = alignment;
        self
    }

    pub fn build(self) -> Context {
        Context::new(self.params)
    }
}

impl Context {
    pub fn new(params: ContextParams) -> Self {
        let arena = MemoryArena::new(params.memory_size, params.memory_alignment);
        Self {
            inner: Rc::new(ContextInner {
                arena: RefCell::new(arena),
                tensors: RefCell::new(Vec::new()),
                next_id: Cell::new(0),
            }),
        }
    }

    pub fn builder() -> ContextBuilder {
        ContextBuilder::default()
    }

    pub fn tensor_from_f32(&self, shape: &[usize], data: &[f32]) -> Result<Tensor> {
        let shape_vec = shape_from_slice(shape)?;
        let expected = shape_vec.iter().product::<usize>();
        if expected != data.len() {
            return Err(Error::ShapeMismatch {
                expected: shape_vec.iter().copied().collect(),
                actual: vec![data.len()],
            });
        }
        let bytes = data.len() * std::mem::size_of::<f32>();
        let storage = {
            let mut arena = self.inner.arena.borrow_mut();
            let range = arena.allocate(bytes, DType::F32.alignment())?;
            arena.write_f32(&range, data);
            range
        };
        Ok(self.register_tensor(shape_vec, DType::F32, Operation::constant(), storage, true))
    }

    pub fn tensor_zeroed(&self, shape: &[usize]) -> Result<Tensor> {
        let shape_vec = shape_from_slice(shape)?;
        let numel = shape_vec.iter().product::<usize>();
        let zeros = vec![0.0f32; numel];
        self.tensor_from_f32(shape, &zeros)
    }

    pub fn parameter(&self, shape: &[usize]) -> Result<Tensor> {
        let shape_vec = shape_from_slice(shape)?;
        let numel = shape_vec.iter().product::<usize>();
        let bytes = numel * DType::F32.size_in_bytes();
        let storage = {
            let mut arena = self.inner.arena.borrow_mut();
            arena.allocate(bytes, DType::F32.alignment())?
        };
        Ok(self.register_tensor(
            shape_vec,
            DType::F32,
            Operation::parameter(),
            storage,
            false,
        ))
    }

    pub(crate) fn binary_op(
        &self,
        lhs: &Tensor,
        rhs: &Tensor,
        kind: BinaryOpKind,
    ) -> Result<Tensor> {
        self.assert_same_context(lhs)?;
        self.assert_same_context(rhs)?;
        let lhs_shape = self.tensor_shape(lhs.id);
        let rhs_shape = self.tensor_shape(rhs.id);
        if lhs_shape != rhs_shape {
            return Err(Error::ShapeMismatch {
                expected: lhs_shape.iter().copied().collect(),
                actual: rhs_shape.iter().copied().collect(),
            });
        }
        let dtype = self.tensor_dtype(lhs.id);
        let rhs_dtype = self.tensor_dtype(rhs.id);
        if dtype != DType::F32 {
            return Err(Error::dtype_mismatch(DType::F32, dtype));
        }
        if rhs_dtype != DType::F32 {
            return Err(Error::dtype_mismatch(DType::F32, rhs_dtype));
        }
        let numel = lhs_shape.iter().product::<usize>();
        let bytes = numel * dtype.size_in_bytes();
        let storage = {
            let mut arena = self.inner.arena.borrow_mut();
            arena.allocate(bytes, dtype.alignment())?
        };
        Ok(self.register_tensor(
            lhs_shape,
            dtype,
            Operation::binary(kind, lhs.id, rhs.id),
            storage,
            false,
        ))
    }

    pub(crate) fn unary_op(&self, input: &Tensor, kind: UnaryOpKind) -> Result<Tensor> {
        self.assert_same_context(input)?;
        let shape = self.tensor_shape(input.id);
        let dtype = self.tensor_dtype(input.id);
        if dtype != DType::F32 {
            return Err(Error::dtype_mismatch(DType::F32, dtype));
        }
        let numel = shape.iter().product::<usize>();
        let bytes = numel * dtype.size_in_bytes();
        let storage = {
            let mut arena = self.inner.arena.borrow_mut();
            arena.allocate(bytes, dtype.alignment())?
        };
        Ok(self.register_tensor(
            shape,
            dtype,
            Operation::unary(kind, input.id),
            storage,
            false,
        ))
    }

    pub(crate) fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        self.assert_same_context(lhs)?;
        self.assert_same_context(rhs)?;
        let lhs_shape = self.tensor_shape(lhs.id);
        let rhs_shape = self.tensor_shape(rhs.id);
        if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
            return Err(Error::Operation("matmul expects 2D tensors".into()));
        }
        if lhs_shape[1] != rhs_shape[0] {
            return Err(Error::Operation("matmul inner dimensions mismatch".into()));
        }
        let out_shape: Shape = SmallVec::from_slice(&[lhs_shape[0], rhs_shape[1]]);
        let dtype = self.tensor_dtype(lhs.id);
        let rhs_dtype = self.tensor_dtype(rhs.id);
        if dtype != DType::F32 {
            return Err(Error::dtype_mismatch(DType::F32, dtype));
        }
        if rhs_dtype != DType::F32 {
            return Err(Error::dtype_mismatch(DType::F32, rhs_dtype));
        }
        let numel = out_shape.iter().product::<usize>();
        let bytes = numel * dtype.size_in_bytes();
        let storage = {
            let mut arena = self.inner.arena.borrow_mut();
            arena.allocate(bytes, dtype.alignment())?
        };
        Ok(self.register_tensor(
            out_shape,
            dtype,
            Operation::matmul(lhs.id, rhs.id),
            storage,
            false,
        ))
    }

    pub(crate) fn tensor_dtype(&self, id: TensorId) -> DType {
        let tensors = self.inner.tensors.borrow();
        tensors[id.index()].dtype
    }

    pub(crate) fn tensor_shape(&self, id: TensorId) -> Shape {
        let tensors = self.inner.tensors.borrow();
        tensors[id.index()].shape.clone()
    }

    pub(crate) fn tensor_elements(&self, id: TensorId) -> usize {
        let tensors = self.inner.tensors.borrow();
        tensors[id.index()].elements()
    }

    pub(crate) fn tensor_to_vec(&self, id: TensorId) -> Result<Vec<f32>> {
        let storage = {
            let tensors = self.inner.tensors.borrow();
            let tensor = &tensors[id.index()];
            if tensor.dtype != DType::F32 {
                return Err(Error::dtype_mismatch(DType::F32, tensor.dtype));
            }
            tensor.storage()
        };
        let arena = self.inner.arena.borrow();
        Ok(arena.read_f32(&storage).to_vec())
    }

    pub(crate) fn tensor_set_f32(&self, id: TensorId, data: &[f32]) -> Result<()> {
        let (storage, shape) = {
            let tensors = self.inner.tensors.borrow();
            let tensor = &tensors[id.index()];
            if tensor.dtype != DType::F32 {
                return Err(Error::dtype_mismatch(DType::F32, tensor.dtype));
            }
            (tensor.storage(), tensor.shape.clone())
        };
        let expected = shape.iter().product::<usize>();
        if expected != data.len() {
            return Err(Error::ShapeMismatch {
                expected: shape.iter().copied().collect(),
                actual: vec![data.len()],
            });
        }
        {
            let mut arena = self.inner.arena.borrow_mut();
            arena.write_f32(&storage, data);
        }
        {
            let mut tensors = self.inner.tensors.borrow_mut();
            tensors[id.index()].mark_computed();
        }
        Ok(())
    }

    pub(crate) fn tensor_set_name(&self, id: TensorId, name: String) -> Result<()> {
        let mut tensors = self.inner.tensors.borrow_mut();
        tensors[id.index()].name = Some(name);
        Ok(())
    }

    pub(crate) fn operation_inputs(&self, id: TensorId) -> SmallVec<[TensorId; 4]> {
        let tensors = self.inner.tensors.borrow();
        SmallVec::from_slice(tensors[id.index()].operation().inputs())
    }

    pub(crate) fn mark_pending(&self, ids: &[TensorId]) {
        let mut tensors = self.inner.tensors.borrow_mut();
        for id in ids {
            let kind = tensors[id.index()].operation().kind().clone();
            if matches!(kind, OperationKind::Parameter) {
                continue;
            }
            tensors[id.index()].mark_pending();
        }
    }

    pub(crate) fn evaluate(&self, id: TensorId) -> Result<()> {
        let (operation, inputs) = {
            let tensors = self.inner.tensors.borrow();
            let tensor = &tensors[id.index()];
            if tensor.is_computed() {
                return Ok(());
            }
            (
                tensor.operation().clone(),
                tensor.operation().inputs().to_vec(),
            )
        };
        let mut tensors = self.inner.tensors.borrow_mut();
        let mut arena = self.inner.arena.borrow_mut();
        operation.kind().eval(&mut tensors, &mut arena, id, &inputs)
    }

    pub(crate) fn evaluate_order(&self, order: &[TensorId]) -> Result<()> {
        for id in order {
            self.evaluate(*id)?;
        }
        Ok(())
    }

    fn assert_same_context(&self, tensor: &Tensor) -> Result<()> {
        if Rc::ptr_eq(&self.inner, &tensor.ctx.inner) {
            Ok(())
        } else {
            Err(Error::ContextMismatch)
        }
    }

    fn register_tensor(
        &self,
        shape: Shape,
        dtype: DType,
        operation: Operation,
        storage: std::ops::Range<usize>,
        computed: bool,
    ) -> Tensor {
        let id = self.next_id();
        let data = TensorData::new(None, shape, dtype, operation, storage, computed);
        self.inner.tensors.borrow_mut().push(data);
        Tensor::new(self.clone(), id)
    }

    fn next_id(&self) -> TensorId {
        let id = self.inner.next_id.get();
        self.inner.next_id.set(id + 1);
        TensorId(id)
    }
}

fn shape_from_slice(shape: &[usize]) -> Result<Shape> {
    if shape.iter().any(|&dim| dim == 0) {
        return Err(Error::InvalidShape(shape.to_vec()));
    }
    Ok(Shape::from_iter(shape.iter().copied()))
}
