use std::io::{self, BufReader, Read, Seek, SeekFrom};

pub struct BufferedSeeker<R> {
    reader: BufReader<R>,
}

impl<R: Read + Seek> BufferedSeeker<R> {
    pub fn new(inner: R, size: usize) -> Self {
        let reader = if size > 0 {
            BufReader::with_capacity(size, inner)
        } else {
            BufReader::new(inner)
        };
        Self { reader }
    }

    pub fn buffered(&self) -> usize {
        self.reader.buffer().len()
    }

    pub fn into_inner(self) -> R {
        self.reader.into_inner()
    }
}

impl<R: Read + Seek> Read for BufferedSeeker<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.reader.read(buf)
    }
}

impl<R: Read + Seek> Seek for BufferedSeeker<R> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.reader.seek(pos)
    }
}
