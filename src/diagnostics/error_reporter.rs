pub trait ErrorReporter: Send + Sync {
    type ErrorInfo: Send + Sync;
    fn report_error(&self, error: &Self::ErrorInfo);
}
