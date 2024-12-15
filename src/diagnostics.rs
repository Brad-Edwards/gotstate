// SPDX-License-Identifier: MIT OR Apache-2.0

//! Diagnostics module.
use std::collections::HashMap;

/// Statistics for state transitions
#[derive(Debug, Default)]
pub struct TransitionStats {
    pub attempts: u64,
    pub successes: u64,
}
/// `Diagnostics` provides logging, error reporting, and metric collection.
///
/// **Key Considerations:**
/// - Allows integrated logging and error reporting for FSM operations.
/// - `DiagnosticData` can be polled to retrieve current diagnostic information.
pub trait Diagnostics: Send + Sync {
    type ErrorInfo: Send + Sync;
    type LogLevel: Send + Sync;
    type DiagnosticData: Send + Sync;

    fn log(&self, message: &str, level: &Self::LogLevel);
    fn report_error(&self, error: &Self::ErrorInfo);
    fn get_diagnostic_data(&self) -> Self::DiagnosticData;
    fn log_transition_attempt(&self, from: &str, to: &str, success: bool);
    fn get_transition_statistics(&self) -> HashMap<String, TransitionStats>;
}

/// `Logger` provides a simplified logging interface.
///
/// **Key Considerations:**
/// - Allows flexible backend implementations.
/// - Minimizes overhead if logging is disabled or minimized in production.
pub trait Logger: Send + Sync {
    type LogLevel: Send + Sync;

    fn log(&self, message: &str, level: &Self::LogLevel);
}

/// `ErrorReporter` handles reporting errors in a standardized manner.
///
/// **Key Considerations:**
/// - May integrate with external error tracking systems (e.g., Sentry).
pub trait ErrorReporter: Send + Sync {
    type ErrorInfo: Send + Sync;

    fn report_error(&self, error: &Self::ErrorInfo);
}

/// `DiagnosticCollector` aggregates diagnostic data.
///
/// **Key Considerations:**
/// - Useful for performance metrics or state-machine insights.
/// - Data can be used for runtime debugging or analytics.
pub trait DiagnosticCollector: Send + Sync {
    type DiagnosticData: Send + Sync;

    fn collect_data(&self) -> Self::DiagnosticData;
}
