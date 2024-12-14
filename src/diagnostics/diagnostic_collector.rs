pub trait DiagnosticCollector: Send + Sync {
    type DiagnosticData: Send + Sync;
    fn collect_data(&self) -> Self::DiagnosticData;
}
