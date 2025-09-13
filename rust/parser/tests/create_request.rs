use parser::{parse_file};
use std::io::Cursor;
use std::path::PathBuf;

#[test]
fn create_request_basic() {
    let input = "FROM test\nTEMPLATE some template\nLICENSE MIT\nPARAMETER temperature 0.5\nMESSAGE user Hello\n";
    let mf = parse_file(Cursor::new(input)).unwrap();
    let req = mf.create_request(PathBuf::from(".").as_path()).unwrap();
    assert_eq!(req.template, "some template");
    assert_eq!(req.license.unwrap(), serde_json::json!(["MIT"]));
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.parameters.get("temperature").unwrap(), &serde_json::json!(0.5));
}
