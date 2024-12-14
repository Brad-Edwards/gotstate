use crate::diagnostics::Diagnostics;
use crate::resource::ResourceManager;

pub trait CoreContextProvider: Send + Sync {
    type StateHandle: Send + Sync;
    type Diagnostics: Diagnostics;
    type ResourceManager: ResourceManager;

    fn state_handle(&self) -> Self::StateHandle;
    fn diagnostics(&self) -> &Self::Diagnostics;
    fn resource_manager(&self) -> &Self::ResourceManager;
}
