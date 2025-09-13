use std::{env, path::Path};

pub use tracing::{Level, debug, error, info, trace, warn};

use tracing::Subscriber;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields, MakeWriter, format::Writer};
use tracing_subscriber::registry::LookupSpan;

struct LogFormat;

impl<S, N> FormatEvent<S, N> for LogFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        let meta = event.metadata();
        let level = meta.level().as_str().to_uppercase();
        let file = meta.file().map(|f| {
            Path::new(f)
                .file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default()
        });
        let line = meta.line().unwrap_or(0);
        if let Some(f) = file {
            write!(writer, "{level} {f}:{line} ")?;
        } else {
            write!(writer, "{level} ")?;
        }
        ctx.format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

/// Parse a string into a [`Level`].
pub fn parse_level(s: &str) -> Option<Level> {
    match s.to_ascii_lowercase().as_str() {
        "trace" => Some(Level::TRACE),
        "debug" => Some(Level::DEBUG),
        "info" => Some(Level::INFO),
        "warn" | "warning" => Some(Level::WARN),
        "error" => Some(Level::ERROR),
        _ => None,
    }
}

/// Initialize logging using `OLLAMA_LOG_LEVEL` or default to INFO.
pub fn init() {
    init_with_env("OLLAMA_LOG_LEVEL");
}

/// Initialize logging reading level from the given environment variable.
pub fn init_with_env(var: &str) {
    let level = env::var(var)
        .ok()
        .and_then(|v| parse_level(&v))
        .unwrap_or(Level::INFO);

    let filter = EnvFilter::builder()
        .with_default_directive(level.into())
        .from_env_lossy();

    tracing_subscriber::fmt()
        .event_format(LogFormat)
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

/// Build a subscriber with the given level and writer. Mainly used for tests.
pub fn subscriber_with_writer<W>(level: Level, writer: W) -> impl Subscriber + Send + Sync + 'static
where
    W: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    let filter = EnvFilter::new(level.as_str().to_ascii_lowercase());
    tracing_subscriber::fmt()
        .event_format(LogFormat)
        .with_env_filter(filter)
        .with_writer(writer)
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tracing::subscriber::with_default;

    #[derive(Clone)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl<'a> MakeWriter<'a> for Buffer {
        type Writer = Buffer;
        fn make_writer(&'a self) -> Self::Writer {
            Buffer(self.0.clone())
        }
    }

    impl std::io::Write for Buffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn level_parser() {
        assert_eq!(parse_level("TRACE"), Some(Level::TRACE));
        assert_eq!(parse_level("debug"), Some(Level::DEBUG));
        assert_eq!(parse_level("Info"), Some(Level::INFO));
        assert_eq!(parse_level("warn"), Some(Level::WARN));
        assert_eq!(parse_level("error"), Some(Level::ERROR));
        assert_eq!(parse_level("unknown"), None);
    }

    #[test]
    fn output_format() {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let writer = Buffer(buf.clone());
        let subscriber = subscriber_with_writer(Level::INFO, writer);
        with_default(subscriber, || {
            info!("hello world");
        });
        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(out.contains("INFO"), "{out}");
        assert!(out.contains("lib.rs"), "{out}");
        assert!(out.contains("hello world"), "{out}");
    }
}
