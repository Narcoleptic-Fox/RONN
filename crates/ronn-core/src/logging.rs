//! Structured logging configuration for RONN.
//!
//! This module provides centralized logging setup using the `tracing` crate
//! for structured, contextual logging throughout the RONN runtime.

use tracing::Level;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

/// Logging configuration for RONN runtime.
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Minimum log level to display
    pub level: LogLevel,
    /// Whether to include timestamps
    pub with_timestamps: bool,
    /// Whether to include thread IDs
    pub with_thread_ids: bool,
    /// Whether to include source code locations
    pub with_source_location: bool,
    /// Whether to log span events (enter/exit)
    pub with_span_events: bool,
    /// Whether to output in JSON format
    pub json_format: bool,
}

/// Log level configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Trace-level logging (most verbose)
    Trace,
    /// Debug-level logging
    Debug,
    /// Info-level logging
    Info,
    /// Warn-level logging
    Warn,
    /// Error-level logging (least verbose)
    Error,
}

impl LogLevel {
    /// Convert to tracing::Level
    fn to_tracing_level(&self) -> Level {
        match self {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            with_timestamps: true,
            with_thread_ids: false,
            with_source_location: false,
            with_span_events: false,
            json_format: false,
        }
    }
}

impl LoggingConfig {
    /// Create a new logging configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum log level.
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.level = level;
        self
    }

    /// Enable or disable timestamps.
    pub fn with_timestamps(mut self, enable: bool) -> Self {
        self.with_timestamps = enable;
        self
    }

    /// Enable or disable thread IDs.
    pub fn with_thread_ids(mut self, enable: bool) -> Self {
        self.with_thread_ids = enable;
        self
    }

    /// Enable or disable source code locations.
    pub fn with_source_location(mut self, enable: bool) -> Self {
        self.with_source_location = enable;
        self
    }

    /// Enable or disable span event logging.
    pub fn with_span_events(mut self, enable: bool) -> Self {
        self.with_span_events = enable;
        self
    }

    /// Enable or disable JSON output format.
    pub fn with_json_format(mut self, enable: bool) -> Self {
        self.json_format = enable;
        self
    }

    /// Create a development-friendly configuration (verbose).
    pub fn development() -> Self {
        Self {
            level: LogLevel::Debug,
            with_timestamps: true,
            with_thread_ids: true,
            with_source_location: true,
            with_span_events: true,
            json_format: false,
        }
    }

    /// Create a production-friendly configuration (minimal).
    pub fn production() -> Self {
        Self {
            level: LogLevel::Info,
            with_timestamps: true,
            with_thread_ids: false,
            with_source_location: false,
            with_span_events: false,
            json_format: true, // JSON for log aggregation
        }
    }
}

/// Initialize the global logger with the given configuration.
///
/// This should be called once at the start of the application.
///
/// # Example
///
/// ```no_run
/// use ronn_core::logging::{init_logging, LoggingConfig};
///
/// init_logging(LoggingConfig::development());
/// ```
pub fn init_logging(config: LoggingConfig) {
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(config.level.to_tracing_level().as_str()))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let span_events = if config.with_span_events {
        FmtSpan::ENTER | FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    };

    if config.json_format {
        // JSON format for production
        let fmt_layer = fmt::layer()
            .json()
            .with_span_events(span_events)
            .with_current_span(true)
            .with_thread_ids(config.with_thread_ids)
            .with_file(config.with_source_location)
            .with_line_number(config.with_source_location);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();
    } else {
        // Human-readable format for development
        let fmt_layer = fmt::layer()
            .with_span_events(span_events)
            .with_thread_ids(config.with_thread_ids)
            .with_file(config.with_source_location)
            .with_line_number(config.with_source_location)
            .with_target(config.with_source_location);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();
    }
}

/// Initialize logging with default configuration.
///
/// Convenience function that uses `LoggingConfig::default()`.
pub fn init_default_logging() {
    init_logging(LoggingConfig::default());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, LogLevel::Info);
        assert!(config.with_timestamps);
        assert!(!config.with_thread_ids);
    }

    #[test]
    fn test_logging_config_development() {
        let config = LoggingConfig::development();
        assert_eq!(config.level, LogLevel::Debug);
        assert!(config.with_span_events);
        assert!(!config.json_format);
    }

    #[test]
    fn test_logging_config_production() {
        let config = LoggingConfig::production();
        assert_eq!(config.level, LogLevel::Info);
        assert!(config.json_format);
        assert!(!config.with_source_location);
    }

    #[test]
    fn test_logging_config_builder() {
        let config = LoggingConfig::new()
            .with_level(LogLevel::Trace)
            .with_timestamps(false)
            .with_thread_ids(true)
            .with_json_format(true);

        assert_eq!(config.level, LogLevel::Trace);
        assert!(!config.with_timestamps);
        assert!(config.with_thread_ids);
        assert!(config.json_format);
    }
}
