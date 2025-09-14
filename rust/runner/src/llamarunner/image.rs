use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::SystemTime;

use crate::MultimodalSupport;

const IMAGE_CACHE_SIZE: usize = 4;

#[derive(Clone)]
struct ImageCache {
    key: u64,
    val: Vec<Vec<f32>>,
    last_used: SystemTime,
}

pub struct ImageContext<E: MultimodalSupport> {
    pub(crate) embedder: E,
    images: Vec<ImageCache>,
}

impl<E: MultimodalSupport> ImageContext<E> {
    pub fn new(embedder: E) -> Self {
        Self {
            embedder,
            images: (0..IMAGE_CACHE_SIZE)
                .map(|_| ImageCache {
                    key: 0,
                    val: Vec::new(),
                    last_used: SystemTime::UNIX_EPOCH,
                })
                .collect(),
        }
    }

    fn hash_image(&self, data: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }

    fn find_image(&mut self, hash: u64) -> Option<Vec<Vec<f32>>> {
        for img in &mut self.images {
            if img.key == hash {
                img.last_used = SystemTime::now();
                return Some(img.val.clone());
            }
        }
        None
    }

    fn add_image(&mut self, hash: u64, embed: Vec<Vec<f32>>) {
        let mut best_idx = 0usize;
        let mut best_time = self.images[0].last_used;
        for (i, img) in self.images.iter().enumerate() {
            if img.key == hash {
                best_idx = i;
                break;
            }
            if img.last_used < best_time {
                best_time = img.last_used;
                best_idx = i;
            }
        }
        self.images[best_idx] = ImageCache {
            key: hash,
            val: embed,
            last_used: SystemTime::now(),
        };
    }

    pub fn new_embed(&mut self, data: &[u8]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if data.is_empty() {
            return Err("received zero length image".into());
        }
        let hash = self.hash_image(data);
        if let Some(embed) = self.find_image(hash) {
            return Ok(embed);
        }
        let embed = self.embedder.embed_image(data)?;
        self.add_image(hash, embed.clone());
        Ok(embed)
    }

    pub fn batch_size(&self, configured_batch_size: usize) -> usize {
        self.embedder.batch_size(configured_batch_size)
    }

    pub fn embed_size(&self) -> usize {
        self.embedder.embed_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyEmbedder;

    impl MultimodalSupport for DummyEmbedder {
        fn embed_image(
            &mut self,
            _data: &[u8],
        ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
            Ok(vec![])
        }
    }

    #[test]
    fn test_image_cache() {
        let mut cache = ImageContext::new(DummyEmbedder);

        let val_a = vec![vec![0.1, 0.2], vec![0.3]];
        let val_b = vec![vec![0.4], vec![0.5], vec![0.6]];
        let val_c = vec![vec![0.7]];
        let val_d = vec![vec![0.8]];
        let val_e = vec![vec![0.9]];

        let hash_a = cache.hash_image(b"a");
        let hash_b = cache.hash_image(b"b");
        let hash_b2 = cache.hash_image(b"c");
        let hash_d = cache.hash_image(b"d");
        let hash_e = cache.hash_image(b"e");

        // Empty cache
        assert!(cache.find_image(hash_a).is_none());

        // Insert A
        cache.add_image(hash_a, val_a.clone());
        assert_eq!(cache.find_image(hash_a).unwrap(), val_a);

        // Insert B
        cache.add_image(hash_b, val_b.clone());
        assert_eq!(cache.find_image(hash_a).unwrap(), val_a);
        assert_eq!(cache.find_image(hash_b).unwrap(), val_b);

        // Replace B with C
        cache.add_image(hash_b, val_c.clone());
        assert_eq!(cache.find_image(hash_a).unwrap(), val_a);
        assert_eq!(cache.find_image(hash_b).unwrap(), val_c);

        // Evict A by adding B', D and E
        cache.add_image(hash_b2, val_b.clone());
        cache.add_image(hash_d, val_d.clone());
        cache.add_image(hash_e, val_e.clone());

        assert!(cache.find_image(hash_a).is_none());
        assert_eq!(cache.find_image(hash_b).unwrap(), val_c);
        assert_eq!(cache.find_image(hash_b2).unwrap(), val_b);
        assert_eq!(cache.find_image(hash_d).unwrap(), val_d);
        assert_eq!(cache.find_image(hash_e).unwrap(), val_e);
    }
}
