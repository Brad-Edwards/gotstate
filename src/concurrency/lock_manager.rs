pub trait LockManager: Send + Sync {
    type LockHandle: Send;
    type Error;

    fn acquire_lock(&self) -> Result<Self::LockHandle, Self::Error>;
    fn release_lock(&self, handle: Self::LockHandle);
}
