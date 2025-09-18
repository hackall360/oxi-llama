use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::progress::State;

pub struct Spinner {
    message: Arc<RwLock<String>>,
    pub message_width: isize,
    parts: Vec<&'static str>,
    value: Arc<AtomicUsize>,
    ticker_stop: Arc<AtomicBool>,
    started: Instant,
    stopped: Arc<AtomicBool>,
}

impl Spinner {
    pub fn new(message: impl Into<String>) -> Self {
        let spinner = Spinner {
            message: Arc::new(RwLock::new(message.into())),
            message_width: -1,
            parts: vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            value: Arc::new(AtomicUsize::new(0)),
            ticker_stop: Arc::new(AtomicBool::new(false)),
            started: Instant::now(),
            stopped: Arc::new(AtomicBool::new(false)),
        };
        spinner.start();
        spinner
    }

    pub fn set_message(&self, message: impl Into<String>) {
        let mut lock = self.message.write().unwrap();
        *lock = message.into();
    }

    fn start(&self) {
        let stop = self.ticker_stop.clone();
        let value = self.value.clone();
        thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(100));
                let v = (value.load(Ordering::SeqCst) + 1) % 10;
                value.store(v, Ordering::SeqCst);
            }
        });
    }

    pub fn stop(&self) {
        if !self.stopped.swap(true, Ordering::SeqCst) {
            self.ticker_stop.store(true, Ordering::SeqCst);
        }
    }

    pub fn to_string(&self) -> String {
        let mut sb = String::new();
        let message = self.message.read().unwrap().clone();
        if !message.is_empty() {
            let mut message = message.trim().to_string();
            if self.message_width > 0 && message.len() > self.message_width as usize {
                message.truncate(self.message_width as usize);
            }
            sb.push_str(&message);
            if self.message_width > 0 {
                let pad = self.message_width as usize - message.len();
                if pad > 0 {
                    sb.push_str(&" ".repeat(pad));
                }
            }
            sb.push(' ');
        }
        if !self.stopped.load(Ordering::SeqCst) {
            let parts_index = self.value.load(Ordering::SeqCst) % self.parts.len();
            sb.push_str(self.parts[parts_index]);
            sb.push(' ');
        }
        sb
    }
}

impl State for Spinner {
    fn render(&self) -> String {
        self.to_string()
    }

    fn stop(&self) {
        Spinner::stop(self);
    }
}
