use std::collections::HashMap;

use crate::{
    context::Context,
    error::{Error, Result},
    tensor::{Tensor, TensorId},
};

#[derive(Clone, Debug)]
pub struct ComputationGraph {
    ctx: Context,
    outputs: Vec<TensorId>,
}

impl ComputationGraph {
    pub fn new(ctx: Context) -> Self {
        Self {
            ctx,
            outputs: Vec::new(),
        }
    }

    pub fn add(&mut self, tensor: &Tensor) {
        if !self.outputs.iter().any(|id| id == &tensor.id) {
            self.outputs.push(tensor.id);
        }
    }

    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }

    pub fn compile(&self) -> Result<GraphExecutor> {
        let order = topo_sort(&self.ctx, &self.outputs)?;
        Ok(GraphExecutor {
            ctx: self.ctx.clone(),
            order,
        })
    }

    pub fn compute(&self) -> Result<()> {
        let executor = self.compile()?;
        executor.run()
    }
}

#[derive(Clone, Debug)]
pub struct GraphExecutor {
    ctx: Context,
    order: Vec<TensorId>,
}

impl GraphExecutor {
    pub fn run(&self) -> Result<()> {
        self.ctx.mark_pending(&self.order);
        self.ctx.evaluate_order(&self.order)
    }

    pub fn order(&self) -> &[TensorId] {
        &self.order
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VisitState {
    Temporary,
    Permanent,
}

fn topo_sort(ctx: &Context, outputs: &[TensorId]) -> Result<Vec<TensorId>> {
    let mut order = Vec::new();
    let mut states = HashMap::new();
    for &id in outputs {
        dfs(ctx, id, &mut states, &mut order)?;
    }
    Ok(order)
}

fn dfs(
    ctx: &Context,
    id: TensorId,
    states: &mut HashMap<TensorId, VisitState>,
    order: &mut Vec<TensorId>,
) -> Result<()> {
    match states.get(&id) {
        Some(VisitState::Permanent) => return Ok(()),
        Some(VisitState::Temporary) => return Err(Error::GraphCycle),
        None => {}
    }
    states.insert(id, VisitState::Temporary);
    for input in ctx.operation_inputs(id) {
        dfs(ctx, input, states, order)?;
    }
    states.insert(id, VisitState::Permanent);
    order.push(id);
    Ok(())
}
