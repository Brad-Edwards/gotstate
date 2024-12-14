mod diagnostic_collector;
mod error_reporter;
mod logger;

pub use diagnostic_collector::DiagnosticCollector;
pub use error_reporter::ErrorReporter;
pub use logger::Logger;

pub trait Diagnostics: Send + Sync {
    type ErrorInfo: Send + Sync;
    type LogLevel: Send + Sync;
    type DiagnosticData: Send + Sync;

    fn log(&self, message: &str, level: &Self::LogLevel);
    fn report_error(&self, error: &Self::ErrorInfo);
    fn get_diagnostic_data(&self) -> Self::DiagnosticData;
}
