// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
pub mod behavior;
pub mod builder;
pub mod concurrency;
pub mod core;
pub mod diagnostics;
pub mod fsm_api;
pub mod model;
pub mod resource;
pub mod types;
pub mod validator;

pub use crate::core::{
    CoreContextProvider, CoreDispatcher, CoreEngine, CoreExecutor, EngineError, Event, ExecutorExt,
    FsmContext, FsmDispatcher, FsmEngine, FsmExecutor, FsmLifecycleController, LifecycleController,
    PoisonErrorConverter, StateHandle, TransitionResult,
};

pub use validator::{
    FsmRecoveryCoordinator, FsmRuleEngine, FsmRuntimeValidator, FsmStructuralValidator,
    FsmValidator, ModelTypes, RecoveryCoordinator, RuleEngine, RuntimeValidator,
    StructuralValidator, ValidationError, ValidationResult, Validator,
};
