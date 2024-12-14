use crate::behavior::Behavior;
use crate::concurrency::Concurrency;
use crate::core::CoreEngine;
use crate::diagnostics::Diagnostics;
use crate::model::ModelDefinition;
use crate::resource::ResourceManager;
use crate::validation::Validator;

pub trait FsmApi: Send + Sync {
    type Engine: CoreEngine;
    type Model: ModelDefinition;
    type Behavior: Behavior<TransitionDef = <Self::Model as ModelDefinition>::TransitionDefinition>;
    type Concurrency: Concurrency<Event = <Self::Engine as CoreEngine>::Event>;
    type Diagnostics: Diagnostics;
    type ResourceManager: ResourceManager;
    type Validator: Validator<Model = Self::Model>;
    type Error;

    fn create_fsm(
        &self,
        model: Self::Model,
        concurrency: Self::Concurrency,
        behavior: Self::Behavior,
        diagnostics: Self::Diagnostics,
        resource_manager: Self::ResourceManager,
        validator: Self::Validator,
    ) -> Result<Self::Engine, Self::Error>;

    fn get_state_info(&self, engine: &Self::Engine) -> <Self::Engine as CoreEngine>::StateHandle;

    fn dispatch_event(
        &self,
        engine: &Self::Engine,
        event: <Self::Engine as CoreEngine>::Event,
    ) -> Result<<Self::Engine as CoreEngine>::TransitionResult, Self::Error>;
}
