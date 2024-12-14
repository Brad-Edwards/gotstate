pub trait Logger: Send + Sync {
    type LogLevel: Send + Sync;
    fn log(&self, message: &str, level: &Self::LogLevel);
}
