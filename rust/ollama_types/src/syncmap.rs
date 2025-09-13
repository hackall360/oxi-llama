use std::collections::HashMap;
use std::hash::Hash;
use std::sync::RwLock;

#[derive(Debug)]
pub struct SyncMap<K, V> {
    inner: RwLock<HashMap<K, V>>,
}

impl<K, V> Default for SyncMap<K, V>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> SyncMap<K, V>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    pub fn load(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.inner.read().unwrap().get(key).cloned()
    }

    pub fn store(&self, key: K, value: V) {
        self.inner.write().unwrap().insert(key, value);
    }

    pub fn items(&self) -> HashMap<K, V>
    where
        K: Clone,
        V: Clone,
    {
        self.inner.read().unwrap().clone()
    }
}
