# wgpu-conveyor

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/BVE-Reborn/wgpu-conveyor/CI)
[![Crates.io](https://img.shields.io/crates/v/wgpu-conveyor)](https://crates.io/crates/wgpu-conveyor)
[![Documentation](https://docs.rs/wgpu-conveyor/badge.svg)](https://docs.rs/wgpu-conveyor)
![License](https://img.shields.io/crates/l/wgpu-conveyor)

Buffer belt abstraction for wgpu supporting UMA optimization, automatic resizing, and a bind group cache.

### Example

```rust
use wgpu_conveyor::{AutomatedBuffer, AutomatedBufferManager, UploadStyle, BindGroupCache};
use wgpu::*;

// Create wgpu instance, adapter, device, queue, and bind_group_layout.

let device_type = adapter.get_info().device_type;

// Create a single buffer manager.
let mut manager = AutomatedBufferManager::new(UploadStyle::from_device_type(&device_type));

// Create a buffer from that manager
let mut buffer = manager.create_new_buffer(&device, 128, BufferUsage::UNIFORM, Some("label"));

/////////////////////////////////////
// -- Below happens every frame -- //
/////////////////////////////////////

// Write to that buffer
let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
buffer.write_to_buffer(
    &device,
    &mut command_encoder,
    128,
    |_encoder: &mut CommandEncoder, buffer: &mut [u8]| {
        for (idx, byte) in buffer.iter_mut().enumerate() {
            *byte = idx as u8;
        }
    }
);

// Use buffer in bind group
let mut bind_group_cache = BindGroupCache::new();
let bind_group_key = bind_group_cache.create_bind_group(&buffer, true, |raw_buf| {
    device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: raw_buf.inner.as_entire_binding()
        }]
    })
});

// Use bind group
renderpass.set_bind_group(0, bind_group_cache.get(&bind_group_key).unwrap(), &[]);

// Submit copies
queue.submit(Some(command_encoder.finish()));

// Pump buffers
let futures = manager.pump();

// Run futures async
for fut in futures {
    spawn(fut);
}

// Loop back to beginning of frame
```

### MSRV

Rust 1.47

License: MIT OR Apache-2.0 OR Zlib
