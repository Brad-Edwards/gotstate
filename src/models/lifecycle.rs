use crate::LifecycleController;

#[derive(Debug)]
pub struct FsmLifecycleController<E: Send + Sync> {
    stopped: std::sync::Mutex<bool>,
    _phantom_e: std::marker::PhantomData<E>,
}

impl<E: Send + Sync> FsmLifecycleController<E> {
    pub fn new() -> Self {
        Self {
            stopped: std::sync::Mutex::new(false),
            _phantom_e: std::marker::PhantomData,
        }
    }
}

impl<E: Send + Sync + From<super::EngineError>> LifecycleController for FsmLifecycleController<E> {
    type Error = E;

    fn start(&self) -> Result<(), Self::Error> {
        let mut s = self.stopped.lock().map_err(|_| super::EngineError::UnknownError("Poisoned mutex".into()))?;
        *s = false;
        Ok(())
    }

    fn stop(&self) -> Result<(), Self::Error> {
        let mut s = self.stopped.lock().map_err(|_| super::EngineError::UnknownError("Poisoned mutex".into()))?;
        *s = true;
        Ok(())
    }

    fn cleanup(&self) -> Result<(), Self::Error> {
        // Perform resource cleanup if needed
        Ok(())
    }
}
