use std::io::{Cursor, Read, Seek, SeekFrom};

use fs::util::bufioutil::BufferedSeeker;

#[test]
fn buffered_seeker_behaves_like_go_version() {
    const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";
    let mut seeker = BufferedSeeker::new(Cursor::new(ALPHABET.as_bytes()), 0);

    let mut buf = [0u8; 5];
    seeker.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"abcde");

    seeker.seek(SeekFrom::Start(0)).unwrap();
    let mut one = [0u8; 1];
    seeker.read_exact(&mut one).unwrap();
    assert_eq!(&one, b"a");
    assert!(seeker.buffered() > 0);

    seeker.seek(SeekFrom::Current(1)).unwrap();
    seeker.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"cdefg");

    seeker.seek(SeekFrom::Start(0)).unwrap();
    seeker.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"abcde");

    seeker.seek(SeekFrom::End(-5)).unwrap();
    seeker.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"vwxyz");
}
