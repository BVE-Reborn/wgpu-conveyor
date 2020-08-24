#![allow(clippy::eval_order_dependence)] // Tons of false positives

use crate::{AutomatedBuffer, BeltBufferId, IdBuffer};
use std::{hash::Hash, sync::Arc};
use wgpu::*;

pub trait AutomatedBufferSet<'buf> {
    type Key: Hash + Eq + Clone;
    type Value;
    fn get(self) -> Self::Value;
    fn value_to_key(value: &Self::Value) -> Self::Key;
}

impl<'buf> AutomatedBufferSet<'buf> for &'buf AutomatedBuffer {
    type Key = BeltBufferId;
    type Value = Arc<IdBuffer>;
    fn get(self) -> Self::Value {
        self.get_current_inner()
    }

    fn value_to_key(value: &Self::Value) -> Self::Key {
        value.id
    }
}

impl<'buf> AutomatedBufferSet<'buf> for (&'buf AutomatedBuffer,) {
    type Key = BeltBufferId;
    type Value = Arc<IdBuffer>;
    fn get(self) -> Self::Value {
        self.0.get_current_inner()
    }

    fn value_to_key(value: &Self::Value) -> Self::Key {
        value.id
    }
}

impl<'buf> AutomatedBufferSet<'buf> for (&'buf AutomatedBuffer, &'buf AutomatedBuffer) {
    type Key = (BeltBufferId, BeltBufferId);
    type Value = (Arc<IdBuffer>, Arc<IdBuffer>);
    fn get(self) -> Self::Value {
        (self.0.get_current_inner(), self.1.get_current_inner())
    }

    fn value_to_key(value: &Self::Value) -> Self::Key {
        (value.0.id, value.1.id)
    }
}

impl<'buf> AutomatedBufferSet<'buf> for (&'buf AutomatedBuffer, &'buf AutomatedBuffer, &'buf AutomatedBuffer) {
    type Key = (BeltBufferId, BeltBufferId, BeltBufferId);
    type Value = (Arc<IdBuffer>, Arc<IdBuffer>, Arc<IdBuffer>);
    fn get(self) -> Self::Value {
        (
            self.0.get_current_inner(),
            self.1.get_current_inner(),
            self.2.get_current_inner(),
        )
    }

    fn value_to_key(value: &Self::Value) -> Self::Key {
        (value.0.id, value.1.id, value.2.id)
    }
}

impl<'buf> AutomatedBufferSet<'buf>
    for (
        &'buf AutomatedBuffer,
        &'buf AutomatedBuffer,
        &'buf AutomatedBuffer,
        &'buf AutomatedBuffer,
    )
{
    type Key = (BeltBufferId, BeltBufferId, BeltBufferId, BeltBufferId);
    type Value = (Arc<IdBuffer>, Arc<IdBuffer>, Arc<IdBuffer>, Arc<IdBuffer>);
    fn get(self) -> Self::Value {
        (
            self.0.get_current_inner(),
            self.1.get_current_inner(),
            self.2.get_current_inner(),
            self.3.get_current_inner(),
        )
    }

    fn value_to_key(value: &Self::Value) -> Self::Key {
        (value.0.id, value.1.id, value.2.id, value.3.id)
    }
}

pub struct BindGroupCache<Key: Hash + Eq + Clone> {
    cache: lru::LruCache<Key, BindGroup>,
}
impl<Key: Hash + Eq + Clone> BindGroupCache<Key> {
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(4)
    }

    #[must_use]
    pub fn with_capacity(size: usize) -> Self {
        Self {
            cache: lru::LruCache::new(size),
        }
    }

    pub fn create_bind_group<'a, Set, BindGroupFn>(
        &mut self,
        buffers: Set,
        use_cache: bool,
        bind_group_fn: BindGroupFn,
    ) -> Key
    where
        Set: AutomatedBufferSet<'a, Key = Key>,
        BindGroupFn: FnOnce(&Set::Value) -> BindGroup,
    {
        let value = buffers.get();
        let key = Set::value_to_key(&value);
        if self.cache.contains(&key) && use_cache {
            return key;
        }
        // Bumps LRU-ness
        self.cache.put(key.clone(), bind_group_fn(&value));
        key
    }

    pub fn get(&self, key: &Key) -> Option<&BindGroup> {
        self.cache.peek(key)
    }
}

impl<Key: Hash + Eq + Clone> Default for BindGroupCache<Key> {
    fn default() -> Self {
        Self::new()
    }
}
