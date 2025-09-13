use super::*;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn bar_formatting_contains_percent() {
    let bar = Bar::new("download", 100, 0);
    bar.set(50);
    let out = bar.to_string();
    assert!(out.contains("50%"));
    assert!(out.contains("download"));
}

#[test]
fn progress_concurrent_usage() {
    let progress = Progress::new();
    let bar = Arc::new(Bar::new("task", 100, 0));
    let spinner = Arc::new(Spinner::new("spin"));
    progress.add(bar.clone());
    progress.add(spinner.clone());

    let updater = {
        let bar = bar.clone();
        thread::spawn(move || {
            for i in 0..=100 {
                bar.set(i);
                thread::sleep(Duration::from_millis(5));
            }
        })
    };

    thread::sleep(Duration::from_millis(200));
    assert!(progress.stop());
    assert!(!progress.stop());
    updater.join().unwrap();

    let spin_output = spinner.to_string();
    for part in ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] {
        assert!(!spin_output.contains(part));
    }
}
