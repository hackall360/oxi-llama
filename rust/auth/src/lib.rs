use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine;
use ed25519_dalek::{Signer, SigningKey, Signature};
use rand::RngCore;
use ssh_key::PrivateKey;
use std::{env, fs, path::PathBuf};
use tracing::info;

const DEFAULT_PRIVATE_KEY: &str = "id_ed25519";

fn key_path() -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let home = env::var("HOME")?;
    Ok(PathBuf::from(home).join(".ollama").join(DEFAULT_PRIVATE_KEY))
}

pub fn get_public_key() -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let key_path = key_path()?;
    let private_key_file = match fs::read(&key_path) {
        Ok(k) => k,
        Err(e) => {
            info!("Failed to load private key: {}", e);
            return Err(Box::new(e));
        }
    };
    let private_key = PrivateKey::from_openssh(private_key_file)?;
    let public_key = private_key.public_key().to_openssh()?.trim().to_string();
    Ok(public_key)
}

pub fn new_nonce(rng: &mut impl RngCore, length: usize) -> Result<String, rand::Error> {
    let mut nonce = vec![0u8; length];
    rng.try_fill_bytes(&mut nonce)?;
    Ok(URL_SAFE_NO_PAD.encode(&nonce))
}

pub fn sign(bts: &[u8]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let key_path = key_path()?;
    let private_key_file = match fs::read(&key_path) {
        Ok(k) => k,
        Err(e) => {
            info!("Failed to load private key: {}", e);
            return Err(Box::new(e));
        }
    };
    let private_key = PrivateKey::from_openssh(private_key_file)?;
    let public_key = private_key.public_key().to_openssh()?;
    let parts: Vec<&str> = public_key.split(' ').collect();
    if parts.len() < 2 {
        return Err(Box::<dyn std::error::Error + Send + Sync>::from("malformed public key"));
    }
    // get signing key
    let keypair = private_key
        .key_data()
        .ed25519()
        .ok_or_else(|| "expected ed25519 key".to_string())?;
    let signing_key: SigningKey = (&keypair.private).into();
    let sig: Signature = signing_key.sign(bts);
    let signature_b64 = STANDARD.encode(sig.to_bytes());
    Ok(format!("{}:{}", parts[1].trim(), signature_b64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;
    use std::{fs, sync::Mutex};
    use tempfile::TempDir;
    use ssh_key::Algorithm;
    use ed25519_dalek::{Signature, VerifyingKey, Verifier};

    static LOCK: Mutex<()> = Mutex::new(());

    fn setup_key() -> (TempDir, String, VerifyingKey) {
        let _guard = LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        env::set_var("HOME", dir.path());
        fs::create_dir_all(dir.path().join(".ollama")).unwrap();
        let mut rng = OsRng;
        let private = PrivateKey::random(&mut rng, Algorithm::Ed25519).unwrap();
        let priv_str = private.to_openssh(Default::default()).unwrap();
        fs::write(dir.path().join(".ollama").join(DEFAULT_PRIVATE_KEY), priv_str).unwrap();
        let pub_key_str = private.public_key().to_openssh().unwrap();
        let keypair = private.key_data().ed25519().unwrap();
        let verifying = VerifyingKey::from_bytes(keypair.public.as_ref()).unwrap();
        (dir, pub_key_str.trim().to_string(), verifying)
    }

    #[test]
    fn test_get_public_key_success() {
        let (_dir, pub_key, _) = setup_key();
        let res = get_public_key().unwrap();
        assert_eq!(res, pub_key);
    }

    #[test]
    fn test_get_public_key_missing() {
        let _guard = LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        env::set_var("HOME", dir.path());
        assert!(get_public_key().is_err());
    }

    #[test]
    fn test_new_nonce_success() {
        let mut rng = OsRng;
        let nonce = new_nonce(&mut rng, 16).unwrap();
        let decoded = URL_SAFE_NO_PAD.decode(nonce.as_bytes()).unwrap();
        assert_eq!(decoded.len(), 16);
    }

    struct FailRng;
    impl RngCore for FailRng {
        fn next_u32(&mut self) -> u32 { 0 }
        fn next_u64(&mut self) -> u64 { 0 }
        fn fill_bytes(&mut self, _dest: &mut [u8]) {}
        fn try_fill_bytes(&mut self, _dest: &mut [u8]) -> Result<(), rand::Error> { Err(rand::Error::new("fail")) }
    }

    #[test]
    fn test_new_nonce_error() {
        let mut rng = FailRng;
        assert!(new_nonce(&mut rng, 16).is_err());
    }

    #[test]
    fn test_sign_success() {
        let (_dir, pub_key, verifying) = setup_key();
        let msg = b"hello";
        let signed = sign(msg).unwrap();
        let parts: Vec<&str> = signed.split(':').collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], pub_key.split(' ').nth(1).unwrap());
        let sig_bytes = base64::engine::general_purpose::STANDARD.decode(parts[1]).unwrap();
        let signature = Signature::from_bytes(&sig_bytes.try_into().unwrap());
        verifying.verify(msg, &signature).unwrap();
    }

    #[test]
    fn test_sign_missing() {
        let _guard = LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        env::set_var("HOME", dir.path());
        assert!(sign(b"abc").is_err());
    }
}

