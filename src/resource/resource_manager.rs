pub trait ResourceManager: Send + Sync {
    type ResourceHandle: Send + Sync;
    type ResourceType: Send + Sync;
    type ResourceConfig: Send + Sync;
    type Error;

    fn allocate_resource(
        &self,
        rtype: &Self::ResourceType,
        config: &Self::ResourceConfig,
    ) -> Result<Self::ResourceHandle, Self::Error>;
    fn release_resource(&self, handle: Self::ResourceHandle) -> Result<(), Self::Error>;
    fn cleanup(&self) -> Result<(), Self::Error>;
}
