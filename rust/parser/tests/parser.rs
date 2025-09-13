use parser::{parse_file, Command};
use parser::modelfile::ParserError;
use std::io::Cursor;

#[test]
fn parse_trim_space() {
    let input = "FROM \" model \"\nPARAMETER a b\n";
    let mf = parse_file(Cursor::new(input)).unwrap();
    assert_eq!(mf.commands[0].args, " model ");
}

#[test]
fn parse_from_cases() {
    let cases = vec![
        ("FROM foo", vec![Command { name: "model".into(), args: "foo".into() }], None),
        ("", vec![], Some(ParserError::MissingFrom)),
    ];
    for (inp, expected_cmds, err) in cases {
        let res = parse_file(Cursor::new(inp));
        if let Some(e) = err {
            let perr = res.unwrap_err().downcast::<ParserError>().unwrap();
            assert_eq!(perr, e);
        } else {
            assert_eq!(res.unwrap().commands, expected_cmds);
        }
    }
}

#[test]
fn parse_bad_command() {
    let input = "FROM foo\nBAD what\n";
    let err = parse_file(Cursor::new(input)).unwrap_err();
    assert!(err.downcast_ref::<ParserError>().is_some());
}

#[test]
fn parse_messages() {
    let input = "FROM foo\nMESSAGE user Hello\n";
    let mf = parse_file(Cursor::new(input)).unwrap();
    assert_eq!(mf.commands[1].args, "user: Hello");
}

#[test]
fn parse_quoted() {
    let input = "FROM foo\nSYSTEM \"\"\"\nHello\n\"\"\"\n";
    let mf = parse_file(Cursor::new(input)).unwrap();
    assert_eq!(mf.commands[1].args, "\nHello\n");
}
