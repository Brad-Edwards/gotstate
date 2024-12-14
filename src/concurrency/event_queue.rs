pub trait EventQueue: Send + Sync {
    type Event: Send;
    type Error;

    fn enqueue(&self, event: Self::Event) -> Result<(), Self::Error>;
    fn dequeue(&self) -> Option<Self::Event>;
    fn is_empty(&self) -> bool;
}
