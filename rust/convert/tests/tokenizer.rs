use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

use convert::{parse_tokenizer, Tokenizer, Vocabulary};

fn create_tokenizer_fs(dir: &Path, files: &[(&str, &str)]) {
    for (name, contents) in files {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = File::create(path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
    }
}

#[test]
fn string_chat_template() {
    let temp = tempfile::tempdir().unwrap();
    create_tokenizer_fs(
        temp.path(),
        &[
            ("tokenizer.json", "{}"),
            (
                "tokenizer_config.json",
                "{\n  \"chat_template\": \"<default template>\"\n}",
            ),
        ],
    );

    let got = parse_tokenizer(temp.path(), &[]).unwrap();
    let want = Tokenizer {
        vocabulary: Vocabulary {
            model: "gpt2".into(),
            tokens: vec![],
            scores: vec![],
            types: vec![],
        },
        special_vocabulary: vec![],
        merges: vec![],
        pre: "default".into(),
        template: "<default template>".into(),
    };
    assert_eq!(got, want);
}

#[test]
fn list_chat_template() {
    let temp = tempfile::tempdir().unwrap();
    create_tokenizer_fs(temp.path(), &[
        ("tokenizer.json", "{}"),
        ("tokenizer_config.json", "{\n  \"chat_template\": [{\n    \"name\": \"default\",\n    \"template\": \"<default template>\"\n  }, {\n    \"name\": \"tools\",\n    \"template\": \"<tools template>\"\n  }]\n}")
    ]);

    let got = parse_tokenizer(temp.path(), &[]).unwrap();
    let want = Tokenizer {
        vocabulary: Vocabulary {
            model: "gpt2".into(),
            tokens: vec![],
            scores: vec![],
            types: vec![],
        },
        special_vocabulary: vec![],
        merges: vec![],
        pre: "default".into(),
        template: "<default template>".into(),
    };
    assert_eq!(got, want);
}

#[test]
fn added_tokens() {
    let temp = tempfile::tempdir().unwrap();
    create_tokenizer_fs(temp.path(), &[
        ("tokenizer.json", "{\n  \"added_tokens\": [{\n    \"id\": 999,\n    \"content\": \"<unused999>\",\n    \"special\": false\n  }]\n}")
    ]);

    let got = parse_tokenizer(temp.path(), &[]).unwrap();
    let want = Tokenizer {
        vocabulary: Vocabulary {
            model: "gpt2".into(),
            tokens: vec!["<unused999>".into()],
            scores: vec![999.0],
            types: vec![4],
        },
        special_vocabulary: vec![],
        merges: vec![],
        pre: "default".into(),
        template: String::new(),
    };
    assert_eq!(got, want);
}
