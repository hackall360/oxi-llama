use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use console::Term;

pub trait State: Send + Sync {
    fn render(&self) -> String;
    fn stop(&self) {}
}

pub struct Progress {
    term: Term,
    states: Arc<Mutex<Vec<Arc<dyn State>>>>,
    ticker_stop: Arc<AtomicBool>,
    handle: Mutex<Option<thread::JoinHandle<()>>>,
}

impl Progress {
    pub fn new() -> Self {
        let term = Term::stderr();
        let states: Arc<Mutex<Vec<Arc<dyn State>>>> = Arc::new(Mutex::new(Vec::new()));
        let stop_flag = Arc::new(AtomicBool::new(false));
        let states_clone = states.clone();
        let term_clone = term.clone();
        let stop_clone = stop_flag.clone();
        let handle = thread::spawn(move || {
            while !stop_clone.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(100));
                render_inner(&term_clone, &states_clone);
            }
        });
        Progress { term, states, ticker_stop: stop_flag, handle: Mutex::new(Some(handle)) }
    }

    pub fn add<S>(&self, state: Arc<S>)
    where
        S: State + 'static,
    {
        let mut states = self.states.lock().unwrap();
        states.push(state as Arc<dyn State>);
    }

    pub fn stop(&self) -> bool {
        if self.ticker_stop.swap(true, Ordering::SeqCst) {
            return false;
        }
        if let Some(handle) = self.handle.lock().unwrap().take() {
            handle.join().ok();
        }
        {
            let states = self.states.lock().unwrap();
            for s in states.iter() {
                s.stop();
            }
        }
        render_inner(&self.term, &self.states);
        let _ = self.term.write_line("");
        true
    }
}

fn render_inner(term: &Term, states: &Arc<Mutex<Vec<Arc<dyn State>>>>) {
    let states_guard = states.lock().unwrap();
    if states_guard.is_empty() { return; }
    let _ = term.clear_last_lines(states_guard.len());
    for s in states_guard.iter() {
        let _ = term.write_line(&s.render());
    }
}
