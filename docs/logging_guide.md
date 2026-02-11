# RONN Logging Guide

This guide explains how to use RONN's structured logging system based on the `tracing` crate.

## Quick Start

### Initialize Logging

```rust
use ronn_core::logging::{init_logging, LoggingConfig};

fn main() {
    // Option 1: Default configuration (Info level, human-readable)
    init_logging(LoggingConfig::default());

    // Option 2: Development configuration (verbose)
    init_logging(LoggingConfig::development());

    // Option 3: Production configuration (JSON, minimal)
    init_logging(LoggingConfig::production());

    // Option 4: Custom configuration
    init_logging(
        LoggingConfig::new()
            .with_level(LogLevel::Debug)
            .with_timestamps(true)
            .with_thread_ids(true)
            .with_json_format(false)
    );
}
```

### Using Logging in Your Code

```rust
use tracing::{info, debug, warn, error, trace};
use tracing::instrument;

#[instrument]
pub fn load_model(path: &str) -> Result<Model> {
    info!("Loading model from {}", path);

    match std::fs::read(path) {
        Ok(data) => {
            debug!("Read {} bytes from model file", data.len());
            parse_model(data)
        }
        Err(e) => {
            error!("Failed to read model file: {}", e);
            Err(e.into())
        }
    }
}

#[instrument(skip(tensor))]
pub fn execute_inference(tensor: &Tensor) -> Result<Tensor> {
    let start = std::time::Instant::now();

    trace!("Input tensor shape: {:?}", tensor.shape());

    let result = process_tensor(tensor)?;

    let elapsed = start.elapsed();
    info!("Inference completed in {:?}", elapsed);

    Ok(result)
}
```

## Log Levels

| Level | Purpose | Example Use |
|-------|---------|-------------|
| **TRACE** | Very detailed | Function entry/exit, loop iterations |
| **DEBUG** | Debugging info | Intermediate values, algorithm steps |
| **INFO** | General info | Model loaded, inference started |
| **WARN** | Warnings | Deprecated features, performance issues |
| **ERROR** | Errors | Failed operations, exceptions |

## Configuration Options

### LoggingConfig

```rust
pub struct LoggingConfig {
    pub level: LogLevel,              // Minimum log level
    pub with_timestamps: bool,        // Include timestamps
    pub with_thread_ids: bool,        // Include thread IDs
    pub with_source_location: bool,   // Include file:line
    pub with_span_events: bool,       // Log function enter/exit
    pub json_format: bool,            // JSON vs human-readable
}
```

### Preset Configurations

#### Development (Verbose, Human-Readable)

```rust
LoggingConfig::development()
// - Level: Debug
// - Timestamps: Yes
// - Thread IDs: Yes
// - Source locations: Yes
// - Span events: Yes
// - Format: Human-readable
```

**Output Example**:
```
2025-10-25T12:34:56.789Z DEBUG [thread:12345] ronn_core::session src/session.rs:42 Creating new session
2025-10-25T12:34:56.790Z  INFO [thread:12345] ronn_core::session src/session.rs:45 Session created successfully id=550e8400-e29b-41d4-a716-446655440000
```

#### Production (Minimal, JSON)

```rust
LoggingConfig::production()
// - Level: Info
// - Timestamps: Yes
// - Thread IDs: No
// - Source locations: No
// - Span events: No
// - Format: JSON
```

**Output Example**:
```json
{"timestamp":"2025-10-25T12:34:56.789Z","level":"INFO","target":"ronn_core::session","fields":{"message":"Session created","id":"550e8400-e29b-41d4-a716-446655440000"}}
```

## Environment Variables

Control logging via environment variables:

```bash
# Set log level
export RUST_LOG=debug

# Module-specific levels
export RUST_LOG=ronn_core=debug,ronn_providers=info

# Regex filters
export RUST_LOG="ronn_core::session=trace,ronn_providers=warn"
```

## Structured Fields

Add structured context to logs:

```rust
use tracing::{info, info_span};

pub fn process_batch(batch_id: u64, size: usize) {
    let _span = info_span!("process_batch", batch_id, size).entered();

    info!("Processing batch");
    // Logs will include batch_id and size fields
}
```

## Span Instrumentation

Automatically log function entry/exit:

```rust
use tracing::instrument;

#[instrument]
pub fn complex_operation(param: i32) -> Result<String> {
    // Function entry/exit logged automatically
    // with param value
    Ok(format!("Result: {}", param))
}

#[instrument(skip(large_data))]
pub fn process_data(id: u64, large_data: &[u8]) -> Result<()> {
    // Skip large_data from logs
    // But still log id
    Ok(())
}
```

## Performance Logging

Log performance metrics:

```rust
use tracing::{info, debug};
use std::time::Instant;

pub fn benchmark_inference(model: &Model, iterations: usize) {
    let start = Instant::now();

    for i in 0..iterations {
        if i % 100 == 0 {
            debug!("Completed {} iterations", i);
        }
        model.run();
    }

    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_millis() / iterations as u128;

    info!(
        iterations,
        total_ms = elapsed.as_millis(),
        avg_ms,
        "Benchmark completed"
    );
}
```

## Error Logging

Log errors with context:

```rust
use tracing::{error, warn};

pub fn load_weights(path: &Path) -> Result<Weights> {
    match std::fs::read(path) {
        Ok(data) => parse_weights(data),
        Err(e) => {
            error!(
                path = %path.display(),
                error = %e,
                "Failed to load weights"
            );
            Err(WeightsError::IoError(e))
        }
    }
}

pub fn optimize_graph(graph: &mut Graph) -> Result<()> {
    let initial_nodes = graph.node_count();

    graph.remove_dead_code();

    let final_nodes = graph.node_count();
    let removed = initial_nodes - final_nodes;

    if removed > 0 {
        info!(initial_nodes, final_nodes, removed, "Removed dead nodes");
    } else {
        debug!("No dead nodes found");
    }

    Ok(())
}
```

## Integration with Monitoring Systems

### JSON Format for Log Aggregation

```rust
// Production: use JSON for Elasticsearch, Splunk, etc.
init_logging(LoggingConfig::production());

// Logs will be in JSON format for easy parsing
```

### Metrics Integration

```rust
use tracing::{info, info_span};

pub fn record_inference_metrics(latency_ms: f64, throughput: f64) {
    info!(
        latency_ms,
        throughput,
        "inference_metrics"
    );
    // Parse in log aggregation system for metrics
}
```

## Best Practices

### DO ✅

1. **Use structured fields**:
   ```rust
   info!(user_id = 42, action = "login", "User action");
   ```

2. **Log at appropriate levels**:
   - ERROR: Operation failures
   - WARN: Potential issues
   - INFO: Significant events
   - DEBUG: Detailed state
   - TRACE: Very verbose

3. **Use `#[instrument]` for functions**:
   ```rust
   #[instrument]
   pub fn important_function(param: i32) -> Result<()> { ... }
   ```

4. **Skip sensitive data**:
   ```rust
   #[instrument(skip(password))]
   pub fn authenticate(username: &str, password: &str) { ... }
   ```

5. **Log performance-critical operations**:
   ```rust
   let _span = info_span!("expensive_operation").entered();
   ```

### DON'T ❌

1. **Don't log in tight loops without guards**:
   ```rust
   // Bad
   for item in items {
       debug!("Processing {}", item); // Too verbose!
   }

   // Good
   for (i, item) in items.iter().enumerate() {
       if i % 1000 == 0 {
           debug!("Processed {} items", i);
       }
   }
   ```

2. **Don't log sensitive information**:
   ```rust
   // Bad
   info!("Password: {}", password); // Security issue!

   // Good
   info!("Authentication attempt for user: {}", username);
   ```

3. **Don't format strings unnecessarily**:
   ```rust
   // Bad (formats even if debug disabled)
   debug!("{}", format!("Value: {:?}", expensive_debug()));

   // Good (lazy evaluation)
   debug!("Value: {:?}", expensive_debug());
   ```

## Testing with Logging

In tests, you might want to see logs:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber;

    #[test]
    fn test_with_logging() {
        // Initialize logging for this test
        let _ = tracing_subscriber::fmt()
            .with_test_writer()
            .try_init();

        // Your test code
        my_function();
    }
}
```

## Examples

See these files for logging usage:
- `crates/ronn-core/src/session.rs` - Session lifecycle logging
- `crates/ronn-providers/src/cpu/provider.rs` - Provider execution logging
- `crates/ronn-hrm/src/router.rs` - Routing decision logging
- `examples/brain-features/src/main.rs` - Application-level logging

## Further Reading

- [tracing crate documentation](https://docs.rs/tracing)
- [tracing-subscriber documentation](https://docs.rs/tracing-subscriber)
- [Tokio tracing guide](https://tokio.rs/tokio/topics/tracing)

---

**Last Updated**: 2026-02-01
