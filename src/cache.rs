#![allow(clippy::eval_order_dependence)] // Tons of false positives

use crate::{AutomatedBuffer, BeltBufferId, IdBuffer};
use std::{hash::Hash, sync::Arc};
use wgpu::*;

/// Values that can be used with [`BindGroupCache`].
///
/// Implemented for &AutomatedBuffer and 1, 2, 3, and 4-tuples thereof.
pub trait AutomatedBufferSet<'buf> {
    /// Key type corresponding to this buffer type.
    ///
    /// [`BufferCache1`], [`BufferCache2`], [`BufferCache3`], and [`BufferCache4`]
    /// are aliases for this value for the corresponding value, 2, 3, and 4-tuple.
    type Key: Hash + Eq + Clone;
    /// Underlying buffer type.
    type Value;
    /// Get the buffer values
    fn get(self) -> Self::Value;
    /// Translate a value into a key
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

/// Key type for a single buffer.
pub type BufferCache1 = BeltBufferId;
/// Key type for two buffers.
pub type BufferCache2 = (BeltBufferId, BeltBufferId);
/// Key type for three buffers.
pub type BufferCache3 = (BeltBufferId, BeltBufferId, BeltBufferId);
/// Key type for four buffers.
pub type BufferCache4 = (BeltBufferId, BeltBufferId, BeltBufferId, BeltBufferId);

/// Bind group cache. Corresponds to a single bind group.
///
/// If you use multiple bind groups in a single cache, you will likely have cache
/// misses constantly.
pub struct BindGroupCache<Key: Hash + Eq + Clone> {
    cache: lru::LruCache<Key, BindGroup>,
}
impl<Key: Hash + Eq + Clone> BindGroupCache<Key> {
    /// Create a bind group cache with default size 4.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(4)
    }

    /// Create a bind group cache with given size.
    #[must_use]
    pub fn with_capacity(size: usize) -> Self {
        Self {
            cache: lru::LruCache::new(size),
        }
    }

    /// Empty cache to force repopulation
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Using the set of [`AutomatedBuffer`](AutomatedBuffer)s provided
    /// in `buffers`, create or retrieve a bind group and return the key
    /// to call `get` with.
    ///
    /// If a bind group is not found, `bind_group_fn` will be called with
    /// the resolved [`BeltBufferId`](BeltBufferId) for the given buffers.
    ///
    /// This result will then be cached.
    ///
    /// `use_cache`, if false, will always call the function and overwrite
    /// the value (if any) in the cache. This is useful if bind groups
    /// are changing every invocation.
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
