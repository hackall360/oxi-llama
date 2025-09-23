#[cfg(feature = "amx")]
pub fn matmul_i8_tile(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut out = vec![0_i32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0_i32;
            for inner in 0..k {
                let av = a[row * k + inner] as i32;
                let bv = b[inner * n + col] as i32;
                acc += av * bv;
            }
            out[row * n + col] = acc;
        }
    }
    out
}

#[cfg(not(feature = "amx"))]
pub fn matmul_i8_tile(_a: &[i8], _b: &[i8], _m: usize, _k: usize, _n: usize) -> Vec<i32> {
    panic!("AMX support not enabled in this build");
}
