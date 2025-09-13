use assert_cmd::Command;
use predicates::str::contains;
use std::fs::File;

#[test]
fn create_missing_modelfile() {
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.arg("create").arg("testmodel")
        .assert()
        .failure()
        .stderr(contains("specified Modelfile wasn't found"));
}

#[test]
fn create_with_modelfile() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("Modelfile");
    File::create(&path).unwrap();
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.current_dir(dir.path())
        .arg("create")
        .arg("testmodel")
        .assert()
        .success();
}

#[test]
fn unicode_model_dir() {
    let dir = tempfile::Builder::new().prefix("ollama_埃").tempdir().unwrap();
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.current_dir(dir.path())
        .arg("create")
        .arg("testmodel")
        .assert()
        .failure()
        .stderr(contains("specified Modelfile wasn't found"));
}
