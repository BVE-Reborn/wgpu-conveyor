// Rust warnings
#![forbid(unsafe_code)]
#![deny(future_incompatible)]
#![deny(nonstandard_style)]
#![deny(rust_2018_idioms)]
// Rustdoc Warnings
#![deny(broken_intra_doc_links)]

//! Buffer belt abstraction for wgpu supporting UMA optimization, automatic resizing, and a bind group cache.
//!
//! ## Example
//!
//! ```
//! use wgpu_conveyor::{AutomatedBuffer, AutomatedBufferManager, UploadStyle, BindGroupCache};
//! use wgpu::*;
//!
//! // Create wgpu instance, adapter, device, queue, and bind_group_layout.
//!
//! # let instance = Instance::new(BackendBit::PRIMARY);
//! # let adapter_opt = RequestAdapterOptions { compatible_surface: None, power_preference: PowerPreference::HighPerformance };
//! # let adapter = match pollster::block_on(instance.request_adapter(&adapter_opt)) {
//! #     Some(adapter) => adapter,
//! #     None => { eprintln!("no adapter found, skipping functional test"); return; },
//! # };
//! # let device_opt = DeviceDescriptor { label: None, features: Features::MAPPABLE_PRIMARY_BUFFERS, limits: Limits::default() };
//! # let (device, queue) = pollster::block_on(adapter.request_device(&device_opt, None)).unwrap();
//! # let entry = BindGroupLayoutEntry{ binding: 0, visibility: ShaderStage::VERTEX, ty: BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None};
//! # let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: None, entries: &[entry]});
//! let device_type = adapter.get_info().device_type;
//!
//! // Create a single buffer manager.
//! let mut manager = AutomatedBufferManager::new(UploadStyle::from_device_type(&device_type));
//!
//! // Create a buffer from that manager
//! let mut buffer = manager.create_new_buffer(&device, 128, BufferUsage::UNIFORM, Some("label"));
//!
//! /////////////////////////////////////
//! // -- Below happens every frame -- //
//! /////////////////////////////////////
//!
//! // Write to that buffer
//! let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
//! buffer.write_to_buffer(
//!     &device,
//!     &mut command_encoder,
//!     128,
//!     |_encoder: &mut CommandEncoder, buffer: &mut [u8]| {
//!         for (idx, byte) in buffer.iter_mut().enumerate() {
//!             *byte = idx as u8;
//!         }
//!     }
//! );
//!
//! // Use buffer in bind group
//! let mut bind_group_cache = BindGroupCache::new();
//! let bind_group_key = bind_group_cache.create_bind_group(&buffer, true, |raw_buf| {
//!     device.create_bind_group(&BindGroupDescriptor {
//!         label: None,
//!         layout: &bind_group_layout,
//!         entries: &[BindGroupEntry {
//!             binding: 0,
//!             resource: raw_buf.inner.as_entire_binding()
//!         }]
//!     })
//! });
//!
//! # let mut renderpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
//! // Use bind group
//! renderpass.set_bind_group(0, bind_group_cache.get(&bind_group_key).unwrap(), &[]);
//!
//! # drop(renderpass);
//! // Submit copies
//! queue.submit(Some(command_encoder.finish()));
//!
//! // Pump buffers
//! let futures = manager.pump();
//!
//! # fn spawn<T>(_: T) {}
//! // Run futures async
//! for fut in futures {
//!     spawn(fut);
//! }
//!
//! // Loop back to beginning of frame
//! ```
//!
//! ## MSRV
//!
//! Rust 1.48

use arrayvec::ArrayVec;
pub use cache::*;
use parking_lot::Mutex;
use std::{
    borrow::Borrow,
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
};
use wgpu::*;

mod cache;

pub type BeltBufferId = usize;

/// Method of upload used by all automated buffers.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum UploadStyle {
    /// Maps the buffer directly. Efficient on UMA systems. Requires
    /// [`wgpu::Features::MAPPABLE_PRIMARY_BUFFERS`] enabled during device creation.
    Mapping,
    /// Maps a staging buffer then copies to buffer. Efficent on NUMA systems.
    Staging,
}
impl UploadStyle {
    /// Chooses most efficient style for the device type.
    #[must_use]
    pub fn from_device_type(ty: &DeviceType) -> Self {
        match ty {
            DeviceType::IntegratedGpu | DeviceType::Cpu => Self::Mapping,
            DeviceType::DiscreteGpu | DeviceType::VirtualGpu | DeviceType::Other => Self::Staging,
        }
    }
}

/// Creates and manages all running [`AutomatedBuffer`]s.
///
/// Responsible for making sure the belts are properly pumped.
///
/// [`pump`](AutomatedBufferManager::pump) must be called after all
/// buffers are written to, and it's futures spawned on an executor
/// that will run them concurrently.
pub struct AutomatedBufferManager {
    belts: Vec<Weak<Mutex<Belt>>>,
    style: UploadStyle,
}
impl AutomatedBufferManager {
    #[must_use]
    pub const fn new(style: UploadStyle) -> Self {
        Self {
            belts: Vec::new(),
            style,
        }
    }

    pub fn create_new_buffer(
        &mut self,
        device: &Device,
        size: BufferAddress,
        usage: BufferUsage,
        label: Option<impl Into<String> + Borrow<str>>,
    ) -> AutomatedBuffer {
        let buffer = AutomatedBuffer::new(device, size, usage, label, self.style);
        self.belts.push(Arc::downgrade(&buffer.belt));
        buffer
    }

    /// Must be called after all buffers are written to and the returned futures must be spawned
    /// on an executor that will run them concurrently.
    ///
    /// If they are not polled, the belts will just constantly leak memory as the futures
    /// allow the belts to reuse buffers.
    pub fn pump(&mut self) -> Vec<impl Future<Output = ()>> {
        let mut valid = Vec::with_capacity(self.belts.len());
        let mut futures = Vec::with_capacity(self.belts.len());
        for belt in &self.belts {
            if let Some(belt) = belt.upgrade() {
                if let Some(future) = Belt::pump(belt) {
                    futures.push(future);
                }
                valid.push(true);
            } else {
                valid.push(false);
            }
        }
        let mut valid_iter = valid.into_iter();
        self.belts.retain(|_| valid_iter.next().unwrap_or(false));
        futures
    }
}

fn check_should_resize(current: BufferAddress, desired: BufferAddress) -> Option<BufferAddress> {
    assert!(current.is_power_of_two());
    if current == 16 && desired <= 16 {
        return None;
    }
    let lower_bound = current / 4;
    if desired <= lower_bound || current < desired {
        Some((desired + 1).next_power_of_two())
    } else {
        None
    }
}

/// A buffer plus an id, internal size, and dirty flag.
///
/// If you muck with this manually, AutomatedBelts will likely break, but they're there
/// in case they are needed.
pub struct IdBuffer {
    /// Underlying buffer.
    pub inner: Buffer,
    /// Hashable id that is unique within the AutomatedBuffer.
    pub id: BeltBufferId,
    /// Size of the buffer. _Not_ the requested size.
    pub size: BufferAddress,
    /// Buffer has been written to and must be pumped.
    pub dirty: AtomicBool,
}

/// Internal representation of a belt
struct Belt {
    usable: ArrayVec<[Arc<IdBuffer>; 2]>,
    usage: BufferUsage,
    current_id: usize,
    live_buffers: usize,
}
impl Belt {
    fn new(usage: BufferUsage) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            usable: ArrayVec::new(),
            usage,
            current_id: 0,
            live_buffers: 0,
        }))
    }

    fn create_buffer(&mut self, device: &Device, size: BufferAddress) {
        let raw_buffer = device.create_buffer(&BufferDescriptor {
            usage: self.usage,
            size,
            mapped_at_creation: true,
            label: None,
        });
        let buffer_id = self.current_id;
        self.current_id += 1;
        self.usable.insert(
            0,
            Arc::new(IdBuffer {
                inner: raw_buffer,
                id: buffer_id,
                size,
                dirty: AtomicBool::new(false),
            }),
        );
    }

    /// Ensures that `self.usable` contains at least one usable buffer for `size`.
    fn ensure_buffer(&mut self, device: &Device, size: BufferAddress) {
        if self.usable.is_empty() {
            let new_size = size.next_power_of_two().max(16);
            tracing::debug!("No buffers in belt, creating new buffer of size {}", new_size);
            self.create_buffer(device, new_size);
            self.live_buffers += 1;
        } else {
            let old = &self.usable[0];
            if let Some(new_size) = check_should_resize(old.size, size) {
                tracing::debug!(
                    "Resizing to {} from {} due to desired size {}",
                    new_size,
                    old.size,
                    size
                );
                self.usable.remove(0);
                self.create_buffer(device, new_size);
            }
        }
    }

    /// Get the active buffer
    fn get_buffer(&self) -> &IdBuffer {
        self.get_buffer_arc()
    }

    /// Get the active buffer as an Arc
    fn get_buffer_arc(&self) -> &Arc<IdBuffer> {
        self.usable
            .get(0)
            .expect("Cannot call get_buffer without calling ensure_buffer first")
    }

    /// Pump the belt and return a future which must be polled to
    /// recall the buffer.
    fn pump(lockable: Arc<Mutex<Self>>) -> Option<impl Future<Output = ()>> {
        let mut inner = lockable.lock();

        if inner.usable.is_empty() {
            return None;
        }

        let buffer_ref = &inner.usable[0];
        let buffer = if buffer_ref.dirty.load(Ordering::Relaxed) {
            inner.usable.remove(0)
        } else {
            return None;
        };

        drop(inner);

        let mapping = buffer.inner.slice(..).map_async(MapMode::Write);
        Some(async move {
            mapping.await.expect("Could not map buffer");
            let mut inner = lockable.lock();
            buffer.dirty.store(false, Ordering::Relaxed);
            if inner.usable.is_full() {
                inner.usable.remove(0);
                inner.live_buffers -= 1;
            }
            inner.usable.push(buffer);
        })
    }
}

/// Statistics about current buffers.
#[derive(Debug, Copy, Clone)]
pub struct AutomatedBufferStats {
    /// ID of the current buffer. Each new buffer gets a sequentially higher number, so
    /// can be used to figure out how many buffers were made in total.
    pub current_id: BeltBufferId,
    /// Total number of buffers that are in the queue or currently in flight.
    pub live_buffers: usize,
    /// Current size of the underlying buffer.
    pub current_size: Option<BufferAddress>,
}

enum UpstreamBuffer {
    Mapping,
    Staging {
        inner: Arc<IdBuffer>,
        usage: BufferUsage,
        label: Option<String>,
    },
}

/// A buffer which automatically uses either staging buffers or direct mapping to write to its
/// internal buffer based on the provided [`UploadStyle`].
pub struct AutomatedBuffer {
    belt: Arc<Mutex<Belt>>,
    upstream: UpstreamBuffer,
}
impl AutomatedBuffer {
    fn new(
        device: &Device,
        initial_size: BufferAddress,
        usage: BufferUsage,
        label: Option<impl Into<String> + Borrow<str>>,
        style: UploadStyle,
    ) -> Self {
        let initial_size = initial_size.next_power_of_two().max(16);
        let (upstream, belt_usage) = if style == UploadStyle::Staging {
            let upstream_usage = BufferUsage::COPY_DST | usage;
            let belt_usage = BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC;
            let upstream = UpstreamBuffer::Staging {
                inner: Arc::new(IdBuffer {
                    inner: device.create_buffer(&BufferDescriptor {
                        size: initial_size,
                        usage: upstream_usage,
                        label: label.as_ref().map(|v| v.borrow()),
                        mapped_at_creation: false,
                    }),
                    id: 0,
                    dirty: AtomicBool::new(false),
                    size: initial_size,
                }),
                usage: upstream_usage,
                label: label.map(Into::into),
            };
            (upstream, belt_usage)
        } else {
            let belt_usage = BufferUsage::MAP_WRITE | usage;
            let upstream = UpstreamBuffer::Mapping;
            (upstream, belt_usage)
        };

        Self {
            belt: Belt::new(belt_usage),
            upstream,
        }
    }

    pub fn stats(&self) -> AutomatedBufferStats {
        let guard = self.belt.lock();
        AutomatedBufferStats {
            current_id: guard.current_id,
            live_buffers: guard.live_buffers,
            current_size: guard.usable.get(0).map(|v| v.size),
        }
    }

    /// Buffer that should be used in bind groups to access the data in the buffer.
    ///
    /// Every single time [`write_to_buffer`](AutomatedBuffer::write_to_buffer) is called,
    /// this could change and the bind group needs to be remade. The [`BindGroupCache`] can
    /// help streamline this process and re-use bind groups.
    pub fn get_current_inner(&self) -> Arc<IdBuffer> {
        match self.upstream {
            UpstreamBuffer::Mapping => Arc::clone(self.belt.lock().get_buffer_arc()),
            UpstreamBuffer::Staging { inner: ref arc, .. } => Arc::clone(arc),
        }
    }

    /// Ensure there's a valid upstream buffer of the given size.
    ///
    /// no-op when mapping.
    fn ensure_upstream(&mut self, device: &Device, size: BufferAddress) {
        let size = size.max(16);
        if let UpstreamBuffer::Staging {
            ref mut inner,
            usage,
            ref label,
        } = self.upstream
        {
            if let Some(new_size) = check_should_resize(inner.size, size) {
                let new_buffer = device.create_buffer(&BufferDescriptor {
                    size: new_size,
                    label: label.as_deref(),
                    mapped_at_creation: false,
                    usage,
                });
                *inner = Arc::new(IdBuffer {
                    inner: new_buffer,
                    size: new_size,
                    dirty: AtomicBool::new(false),
                    id: inner.id + 1,
                })
            }
        }
    }

    /// Writes to the underlying buffer using the proper write style.
    ///
    /// The buffer will be resized to the given `size`.
    ///
    /// All needed copy operations will be recorded onto `encoder`.
    ///
    /// Once the buffer is mapped and ready to be written to, the slice of exactly
    /// `size` bytes will be provided to `data_fn` to be written in.
    pub fn write_to_buffer<DataFn>(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        size: BufferAddress,
        data_fn: DataFn,
    ) where
        DataFn: FnOnce(&mut CommandEncoder, &mut [u8]),
    {
        self.ensure_upstream(device, size);
        let mut inner = self.belt.lock();
        inner.ensure_buffer(device, size);
        let buffer = inner.get_buffer();
        let slice = buffer.inner.slice(0..size);
        let mut mapping = slice.get_mapped_range_mut();
        data_fn(encoder, &mut mapping[0..size as usize]);
        drop(mapping);
        buffer.dirty.store(true, Ordering::Relaxed);
        buffer.inner.unmap();

        if let UpstreamBuffer::Staging { ref inner, .. } = self.upstream {
            encoder.copy_buffer_to_buffer(&buffer.inner, 0, &inner.inner, 0, size as BufferAddress);
        }
    }
}

/// Write to a single [`AutomatedBuffer`].
pub fn write_to_buffer1<DataFn>(
    device: &Device,
    encoder: &mut CommandEncoder,
    buffer0: &mut AutomatedBuffer,
    size0: BufferAddress,
    data_fn: DataFn,
) where
    DataFn: FnOnce(&mut CommandEncoder, &mut [u8]),
{
    buffer0.write_to_buffer(device, encoder, size0, data_fn)
}

/// Write to a two [`AutomatedBuffer`]s at the same time.
pub fn write_to_buffer2<DataFn>(
    device: &Device,
    encoder: &mut CommandEncoder,
    buffer0: &mut AutomatedBuffer,
    size0: BufferAddress,
    buffer1: &mut AutomatedBuffer,
    size1: BufferAddress,
    data_fn: DataFn,
) where
    DataFn: FnOnce(&mut CommandEncoder, &mut [u8], &mut [u8]),
{
    buffer0.write_to_buffer(device, encoder, size0, |encoder, data0| {
        buffer1.write_to_buffer(device, encoder, size1, |encoder, data1| {
            data_fn(encoder, data0, data1);
        })
    })
}

/// Write to a three [`AutomatedBuffer`]s at the same time.
#[allow(clippy::too_many_arguments)]
pub fn write_to_buffer3<DataFn>(
    device: &Device,
    encoder: &mut CommandEncoder,
    buffer0: &mut AutomatedBuffer,
    size0: BufferAddress,
    buffer1: &mut AutomatedBuffer,
    size1: BufferAddress,
    buffer2: &mut AutomatedBuffer,
    size2: BufferAddress,
    data_fn: DataFn,
) where
    DataFn: FnOnce(&mut CommandEncoder, &mut [u8], &mut [u8], &mut [u8]),
{
    buffer0.write_to_buffer(device, encoder, size0, |encoder, data0| {
        buffer1.write_to_buffer(device, encoder, size1, |encoder, data1| {
            buffer2.write_to_buffer(device, encoder, size2, |encoder, data2| {
                data_fn(encoder, data0, data1, data2);
            })
        })
    })
}

/// Write to a four [`AutomatedBuffer`]s at the same time.
#[allow(clippy::too_many_arguments)]
pub fn write_to_buffer4<DataFn>(
    device: &Device,
    encoder: &mut CommandEncoder,
    buffer0: &mut AutomatedBuffer,
    size0: BufferAddress,
    buffer1: &mut AutomatedBuffer,
    size1: BufferAddress,
    buffer2: &mut AutomatedBuffer,
    size2: BufferAddress,
    buffer3: &mut AutomatedBuffer,
    size3: BufferAddress,
    data_fn: DataFn,
) where
    DataFn: FnOnce(&mut CommandEncoder, &mut [u8], &mut [u8], &mut [u8], &mut [u8]),
{
    buffer0.write_to_buffer(device, encoder, size0, |encoder, data0| {
        buffer1.write_to_buffer(device, encoder, size1, |encoder, data1| {
            buffer2.write_to_buffer(device, encoder, size2, |encoder, data2| {
                buffer3.write_to_buffer(device, encoder, size3, |encoder, data3| {
                    data_fn(encoder, data0, data1, data2, data3);
                })
            })
        })
    })
}

#[cfg(test)]
mod test {
    use crate::check_should_resize;

    #[test]
    fn automated_buffer_resize() {
        assert_eq!(check_should_resize(64, 128), Some(256));
        assert_eq!(check_should_resize(128, 128), None);
        assert_eq!(check_should_resize(256, 128), None);

        assert_eq!(check_should_resize(64, 64), None);
        assert_eq!(check_should_resize(128, 64), None);
        assert_eq!(check_should_resize(256, 65), None);
        assert_eq!(check_should_resize(256, 64), Some(128));
        assert_eq!(check_should_resize(256, 63), Some(64));

        assert_eq!(check_should_resize(16, 16), None);
        assert_eq!(check_should_resize(16, 8), None);
        assert_eq!(check_should_resize(16, 4), None);
    }
}
