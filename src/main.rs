use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use serde::{Serialize};
use chrono::Utc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano::VulkanLibrary;

// --- Binary Protocol Specification (Phase 2.5) ---
// Total Size: 40 bytes (8-byte aligned)
#[repr(C, packed)]
struct TelemetryHeader {
    magic: u32,          // 0xDEADBEEF
    padding_a: u32,      // 8-byte alignment
    tick: u64,           // Current simulation step
    timestamp: i64,      // Microseconds since epoch
    compute_ms: f32,     // GPU dispatch time
    tps: f32,            // Ticks Per Second
    width: u16,          // 128
    height: u16,         // 128
    padding_b: u32,      // Pad to 40 bytes to maintain 8-byte alignment for payload
}

// --- Voxel Schema ---
#[derive(Copy, Clone, Serialize, Debug, Default, vulkano::buffer::BufferContents)]
#[repr(C)]
struct Voxel {
    packed: u32,
}

impl Voxel {
    fn new(id: u8, state: u8, thermal: u8, light: u8) -> Self {
        Voxel {
            packed: (id as u32) | ((state as u32) << 8) | ((thermal as u32) << 16) | ((light as u32) << 24)
        }
    }
}

// Struct for uniform data
#[derive(Copy, Clone, vulkano::buffer::BufferContents)]
#[repr(C)]
struct TimeInfo {
    u_time: u32,
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/growth.glsl",
    }
}

fn main() {
    println!("--- Genesis Core: Phase 2.5 HIFI Engine Starting ---");

    // 0. Initialize ZMQ
    let zmq_context = zmq::Context::new();
    let publisher = zmq_context.socket(zmq::PUB).unwrap();
    
    // Applying ZMQ_CONFLATE and SNDHWM for lag mitigation
    publisher.set_conflate(true).expect("Failed to set ZMQ_CONFLATE");
    publisher.set_sndhwm(1).expect("Failed to set ZMQ_SNDHWM");
    
    publisher.bind("tcp://0.0.0.0:5555").expect("Could not bind ZMQ publisher");

    // 1. Initialize Vulkan
    let library = VulkanLibrary::new().expect("no local Vulkan library/driver found");
    let instance = Instance::new(library, InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        ..Default::default()
    }).expect("failed to create instance");

    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .find(|p| p.properties().device_name.contains("5060"))
        .expect("RTX 5060 Ti not found");

    // 3. Create Logical Device
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_i, q)| q.queue_flags.contains(QueueFlags::COMPUTE))
        .expect("couldn't find a compute queue family") as u32;

    let (device, mut queues) = Device::new(physical_device, DeviceCreateInfo {
        queue_create_infos: vec![QueueCreateInfo {
            queue_family_index,
            ..Default::default()
        }],
        ..Default::default()
    }).expect("failed to create device");

    let queue = queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

    // 5. Initialize Voxel Grid (128x128x32)
    let width = 128;
    let height = 128;
    let depth = 32;
    let data_size = width * height * depth;
    
    // Fill with Dirt (ID 1)
    let grid_data = vec![Voxel::new(1, 0, 20, 255); data_size]; 

    let grid_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        grid_data,
    ).expect("failed to create buffer");

    let time_buffer = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        TimeInfo { u_time: 0 }
    ).expect("failed to create time buffer");

    let shader = cs::load(device.clone()).expect("failed to create shader module");
    let entry_point = shader.entry_point("main").expect("main entry point not found");
    let stage = PipelineShaderStageCreateInfo::new(entry_point);

    let pipeline = {
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        ).unwrap();
        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        ).expect("failed to create compute pipeline")
    };

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    
    println!("Simulation Loop Active...");
    let mut tick: u64 = 0;
    let mut last_tps_check = Instant::now();
    let mut last_tick_for_tps = 0;
    let mut current_tps = 0.0;

    loop {
        let loop_start = Instant::now();
        
        // Update time uniform
        time_buffer.write().unwrap().u_time = tick as u32;

        let set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, grid_buffer.clone()),
                WriteDescriptorSet::buffer(1, time_buffer.clone()),
            ],
            [],
        ).unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, set.clone())
            .unwrap()
            .dispatch([ (data_size as u32 / 256) + 1, 1, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();
        
        // --- Profile Compute Dispatch ---
        let compute_start = Instant::now();
        vulkano::sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        let compute_ms = compute_start.elapsed().as_secs_f32() * 1000.0;

        // --- TPS Calculation (Every 60 ticks) ---
        if tick % 60 == 0 && tick > 0 {
            let elapsed = last_tps_check.elapsed().as_secs_f32();
            current_tps = (tick - last_tick_for_tps) as f32 / elapsed;
            last_tps_check = Instant::now();
            last_tick_for_tps = tick;
        }

        // --- Prepare Binary Packet ---
        let header = TelemetryHeader {
            magic: 0xDEADBEEF,
            padding_a: 0,
            tick,
            timestamp: Utc::now().timestamp_micros(),
            compute_ms,
            tps: current_tps,
            width: width as u16,
            height: height as u16,
            padding_b: 0,
        };

        // Read top slice (layer 31) for HIFI visualization
        let content = grid_buffer.read().unwrap();
        let slice_start = (depth - 1) * width * height;
        let slice_end = slice_start + (width * height);
        let voxel_data = &content[slice_start..slice_end];

        // Construct Packet: Header (40 bytes) + Voxel Data (65536 bytes)
        let mut packet = Vec::with_capacity(40 + (width * height * 4));
        
        // Unsafe cast header to bytes
        unsafe {
            let header_ptr = &header as *const TelemetryHeader as *const u8;
            let header_bytes = std::slice::from_raw_parts(header_ptr, std::mem::size_of::<TelemetryHeader>());
            packet.extend_from_slice(header_bytes);
            
            let voxel_ptr = voxel_data.as_ptr() as *const u8;
            let voxel_bytes = std::slice::from_raw_parts(voxel_ptr, width * height * 4);
            packet.extend_from_slice(voxel_bytes);
        }

        publisher.send(packet, 0).unwrap();

        if tick % 60 == 0 {
            println!("[Tick {}] Binary Packet Sent. TPS: {:.2} | Compute: {:.2}ms", tick, current_tps, compute_ms);
        }

        tick += 1;
        
        // Target 60Hz
        let elapsed = loop_start.elapsed();
        let target = Duration::from_millis(16);
        if elapsed < target {
            thread::sleep(target - elapsed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<TelemetryHeader>(), 40);
    }

    #[test]
    fn test_voxel_packing() {
        let v = Voxel::new(1, 2, 3, 4);
        // ID (1) | State (2 << 8) | Thermal (3 << 16) | Light (4 << 24)
        // 1 | 512 | 196608 | 67108864 = 67305985
        assert_eq!(v.packed, 67305985);
    }

    #[test]
    fn test_magic_byte_order() {
        let header = TelemetryHeader {
            magic: 0xDEADBEEF,
            padding_a: 0,
            tick: 0,
            timestamp: 0,
            compute_ms: 0.0,
            tps: 0.0,
            width: 0,
            height: 0,
            padding_b: 0,
        };
        unsafe {
            let ptr = &header as *const TelemetryHeader as *const u8;
            let bytes = std::slice::from_raw_parts(ptr, 4);
            // Little Endian: EF BE AD DE
            assert_eq!(bytes, &[0xEF, 0xBE, 0xAD, 0xDE]);
        }
    }
}