use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::paths::Paths;

#[derive(Debug, Serialize, Deserialize)]
struct StoreData {
    id: String,
    #[serde(rename = "first-time-run")]
    first_time_run: bool,
}

impl Default for StoreData {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            first_time_run: false,
        }
    }
}

#[derive(Clone)]
pub struct Store {
    inner: Arc<Mutex<StoreInner>>,
}

struct StoreInner {
    data: StoreData,
    path: PathBuf,
}

impl Store {
    pub fn open(paths: &Paths) -> Result<Self> {
        let store_path = paths.store_file()?;
        if let Some(parent) = store_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create store directory {}", parent.display())
            })?;
        }

        let data = if store_path.exists() {
            let mut file = OpenOptions::new()
                .read(true)
                .open(&store_path)
                .with_context(|| format!("failed to open store {}", store_path.display()))?;
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)
                .with_context(|| format!("failed to read store {}", store_path.display()))?;
            serde_json::from_slice(&buf).unwrap_or_default()
        } else {
            StoreData::default()
        };

        let inner = StoreInner {
            data,
            path: store_path,
        };
        let store = Store {
            inner: Arc::new(Mutex::new(inner)),
        };
        store.persist_if_missing()?;
        Ok(store)
    }

    pub fn id(&self) -> String {
        self.inner.lock().data.id.clone()
    }

    pub fn first_time_run(&self) -> bool {
        self.inner.lock().data.first_time_run
    }

    pub fn set_first_time_run(&self, value: bool) -> Result<()> {
        let mut inner = self.inner.lock();
        if inner.data.first_time_run != value {
            inner.data.first_time_run = value;
            inner.persist()?;
        }
        Ok(())
    }

    fn persist_if_missing(&self) -> Result<()> {
        let mut inner = self.inner.lock();
        if inner.data.id.is_empty() {
            inner.data.id = Uuid::new_v4().to_string();
            inner.persist()?;
        } else if !inner.path.exists() {
            inner.persist()?;
        }
        Ok(())
    }
}

impl StoreInner {
    fn persist(&self) -> Result<()> {
        let payload = serde_json::to_vec_pretty(&self.data)?;
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)
            .with_context(|| format!("failed to write store {}", self.path.display()))?;
        file.write_all(&payload)?;
        file.flush()?;
        Ok(())
    }
}
