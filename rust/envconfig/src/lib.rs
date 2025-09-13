use std::env;
use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;
use log::warn;

/// Returns the scheme and host. Host can be configured via the OLLAMA_HOST environment variable.
/// Default is scheme "http" and host "127.0.0.1:11434".
pub fn host() -> String {
    let mut default_port = "11434".to_string();

    let s = var("OLLAMA_HOST").trim().to_string();
    let (scheme, rest) = match s.split_once("://") {
        Some((sch, hp)) => {
            if sch == "http" { default_port = "80".into(); }
            else if sch == "https" { default_port = "443".into(); }
            (sch.to_string(), hp.to_string())
        }
        None => ("http".into(), s),
    };

    let (hostport, path) = match rest.split_once('/') {
        Some((hp, p)) => (hp.to_string(), p.to_string()),
        None => (rest, String::new()),
    };

    let mut host = "127.0.0.1".to_string();
    let mut port = default_port.clone();

    if let Some((h, p)) = split_host_port(&hostport) {
        host = h;
        port = p;
    } else {
        let trimmed = hostport.trim_matches(['[', ']']);
        if let Ok(ip) = trimmed.parse::<IpAddr>() {
            host = ip.to_string();
        } else if !hostport.is_empty() {
            host = hostport;
        }
    }

    if let Ok(n) = port.parse::<i64>() {
        if n < 0 || n > 65535 {
            warn!("invalid port, using default {}", default_port);
            port = default_port.clone();
        }
    } else {
        warn!("invalid port, using default {}", default_port);
        port = default_port.clone();
    }

    let hostport = join_host_port(&host, &port);
    let mut url_str = format!("{}://{}", scheme, hostport);
    if !path.is_empty() {
        url_str.push('/');
        url_str.push_str(&path);
    }
    url_str
}

/// AllowedOrigins returns a list of allowed origins. AllowedOrigins can be configured via the OLLAMA_ORIGINS environment variable.
pub fn allowed_origins() -> Vec<String> {
    let mut origins: Vec<String> = Vec::new();
    let s = var("OLLAMA_ORIGINS");
    if !s.is_empty() {
        origins.extend(s.split(',').map(|s| s.to_string()));
    }

    for origin in ["localhost", "127.0.0.1", "0.0.0.0"].iter() {
        origins.push(format!("http://{}", origin));
        origins.push(format!("https://{}", origin));
        origins.push(format!("http://{}:*", origin));
        origins.push(format!("https://{}:*", origin));
    }

    origins.extend([
        "app://*".to_string(),
        "file://*".to_string(),
        "tauri://*".to_string(),
        "vscode-webview://*".to_string(),
        "vscode-file://*".to_string(),
    ]);

    origins
}

/// Models returns the path to the models directory. Models directory can be configured via the OLLAMA_MODELS environment variable.
/// Default is $HOME/.ollama/models
pub fn models() -> PathBuf {
    if let Ok(s) = env::var("OLLAMA_MODELS") {
        if !s.is_empty() {
            return PathBuf::from(s);
        }
    }
    let home = dirs::home_dir().expect("home directory not found");
    home.join(".ollama").join("models")
}

/// KeepAlive returns the duration that models stay loaded in memory. KeepAlive can be configured via the OLLAMA_KEEP_ALIVE environment variable.
/// Negative values are treated as infinite. Zero is treated as no keep alive.
/// Default is 5 minutes.
pub fn keep_alive() -> Duration {
    let secs = parse_duration_env("OLLAMA_KEEP_ALIVE").unwrap_or(300);
    if secs < 0 {
        Duration::MAX
    } else {
        Duration::from_secs(secs as u64)
    }
}

/// LoadTimeout returns the duration for stall detection during model loads. LoadTimeout can be configured via the OLLAMA_LOAD_TIMEOUT environment variable.
/// Zero or Negative values are treated as infinite.
/// Default is 5 minutes.
pub fn load_timeout() -> Duration {
    let secs = parse_duration_env("OLLAMA_LOAD_TIMEOUT").unwrap_or(300);
    if secs <= 0 {
        Duration::MAX
    } else {
        Duration::from_secs(secs as u64)
    }
}

/// Var returns an environment variable stripped of leading and trailing quotes or spaces
pub fn var(key: &str) -> String {
    env::var(key)
        .unwrap_or_default()
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

fn split_host_port(s: &str) -> Option<(String, String)> {
    if s.starts_with('[') {
        if let Some(end) = s.find(']') {
            let host = &s[1..end];
            let rest = &s[end + 1..];
            if rest.starts_with(':') {
                return Some((host.to_string(), rest[1..].to_string()));
            }
            return None;
        }
        return None;
    }
    if let Some(idx) = s.rfind(':') {
        if s[..idx].contains(':') {
            return None;
        }
        let host = &s[..idx];
        let port = &s[idx + 1..];
        return Some((host.to_string(), port.to_string()));
    }
    None
}

fn join_host_port(host: &str, port: &str) -> String {
    if host.contains(':') && !(host.starts_with('[') && host.ends_with(']')) {
        format!("[{}]:{}", host, port)
    } else {
        format!("{}:{}", host, port)
    }
}

fn parse_duration_env(key: &str) -> Option<i64> {
    let s = var(key);
    if s.is_empty() {
        return None;
    }
    let negative = s.trim_start().starts_with('-');
    let s_trim = s.trim_start_matches('-');
    if s_trim.contains('d') || s_trim.contains('w') || s_trim.contains('y') {
        return None;
    }
    if let Ok(d) = humantime::parse_duration(s_trim) {
        let secs = d.as_secs() as i64;
        return Some(if negative { -secs } else { secs });
    }
    if let Ok(n) = s_trim.parse::<i64>() {
        return Some(if negative { -n } else { n });
    }
    None
}
