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

// --- Binary Protocol Specification (Phase 3.0) ---
// Total Size: 48 bytes (8-byte aligned)
#[repr(C, packed)]
struct TelemetryHeader {
    magic: u32,          // 0xDEADBEEF
    padding_a: u32,      // Bridge Status (Injected by bridge)
    tick: u64,           // Current simulation step
    timestamp: i64,      // Microseconds since epoch
    compute_ms: f32,     // GPU dispatch time
    tps: f32,            // Ticks Per Second
    width: u16,          // 128
    height: u16,         // 128
    agent_count: u16,    // 100
    padding_b: u8,       // Padding
    padding_c: u8,       // Padding
    padding_d: u8,       // Padding
    padding_e: u8,       // Padding
    padding_f: u32,      // Padding
    padding_g: u16,      // Pad to 48 bytes to maintain 8-byte alignment for payload
}

// --- Agent Schema (64-byte Aligned) ---
#[derive(Copy, Clone, Debug, Default, vulkano::buffer::BufferContents)]
#[repr(C)]
struct Agent {
    pos: [f32; 3],       // 12 bytes
    vel: [f32; 3],       // 12 bytes
    rotation: f32,       // 4 bytes
    vitals: u32,         // 4 bytes (Hunger, Health, etc)
    brain_id: u64,       // 8 bytes
    padding: [u8; 24],   // 24 bytes = 64 total
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

// Struct for uniform data (SimInfo)
#[derive(Copy, Clone, vulkano::buffer::BufferContents)]
#[repr(C)]
struct SimInfo {
    u_time: u32,
    world_width: u32,
    world_height: u32,
    world_depth: u32,
}

mod cs_growth {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/growth.glsl",
    }
}

mod cs_agents {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/agents.glsl",
    }
}

fn main() {
    println!("--- Genesis Core: Phase 3 Life Engine Starting ---");

    // 0. Initialize ZMQ
    let zmq_context = zmq::Context::new();
    let publisher = zmq_context.socket(zmq::PUB).unwrap();
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

    // 5. Initialize World Dimensions
    let width = 128;
    let height = 128;
    let depth = 32;
    let data_size = width * height * depth;
    let agent_count = 100;
    
    // Fill with Dirt (ID 1) with randomized initial state
    let mut grid_data = Vec::with_capacity(data_size);
    let mut seed: u32 = 12345;
    for _ in 0..data_size {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let initial_state = (seed >> 16) as u8;
        grid_data.push(Voxel::new(1, initial_state, 20, 255));
    }

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
    ).expect("failed to create voxel buffer");

    // Initialize Agents
    let mut initial_agents = Vec::with_capacity(agent_count);
    for i in 0..agent_count {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let rx = (seed % (width as u32)) as f32;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let ry = (seed % (height as u32)) as f32;
        
        initial_agents.push(Agent {
            pos: [rx, ry, (depth - 1) as f32],
            vel: [0.05, 0.02, 0.0], // Slow initial drift
            rotation: 0.0,
            vitals: 0,
            brain_id: i as u64,
            ..Default::default()
        });
    }

    let agent_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        initial_agents,
    ).expect("failed to create agent buffer");

    let sim_info_buffer = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        SimInfo { 
            u_time: 0,
            world_width: width as u32,
            world_height: height as u32,
            world_depth: depth as u32,
        }
    ).expect("failed to create sim info buffer");

    // 6. Pipelines
    let shader_growth = cs_growth::load(device.clone()).expect("failed to create growth shader");
    let shader_agents = cs_agents::load(device.clone()).expect("failed to create agent shader");

    let pipeline_growth = {
        let stage = PipelineShaderStageCreateInfo::new(shader_growth.entry_point("main").unwrap());
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        ).unwrap();
        ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(stage, layout)).unwrap()
    };

    let pipeline_agents = {
        let stage = PipelineShaderStageCreateInfo::new(shader_agents.entry_point("main").unwrap());
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        ).unwrap();
        ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(stage, layout)).unwrap()
    };

    println!("Simulation Loop Active (Phase 3)...");
    let mut tick: u64 = 0;
    let mut last_tps_check = Instant::now();
    let mut last_tick_for_tps = 0;
    let mut current_tps = 0.0;

    loop {
        let loop_start = Instant::now();
        
        sim_info_buffer.write().unwrap().u_time = tick as u32;

        let set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            pipeline_agents.layout().set_layouts().get(0).unwrap().clone(),
            [
                WriteDescriptorSet::buffer(0, grid_buffer.clone()),
                WriteDescriptorSet::buffer(1, agent_buffer.clone()),
                WriteDescriptorSet::buffer(2, sim_info_buffer.clone()),
            ],
            [],
        ).unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        // Dispatch Growth & Agents
        builder
            .bind_pipeline_compute(pipeline_growth.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_growth.layout().clone(), 0, set.clone())
            .unwrap()
            .dispatch([ (data_size as u32 / 256) + 1, 1, 1])
            .unwrap()
            .bind_pipeline_compute(pipeline_agents.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_agents.layout().clone(), 0, set.clone())
            .unwrap()
            .dispatch([ (agent_count as u32 / 64) + 1, 1, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();
        
        let compute_start = Instant::now();
        vulkano::sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        let compute_ms = compute_start.elapsed().as_secs_f32() * 1000.0;

        if tick % 60 == 0 && tick > 0 {
            let elapsed = last_tps_check.elapsed().as_secs_f32();
            current_tps = (tick - last_tick_for_tps) as f32 / elapsed;
            last_tps_check = Instant::now();
            last_tick_for_tps = tick;
        }

        // --- Prepare Binary Packet v3 ---
        let header = TelemetryHeader {
            magic: 0xDEADBEEF,
            padding_a: 0,
            tick,
            timestamp: Utc::now().timestamp_micros(),
            compute_ms,
            tps: current_tps,
            width: width as u16,
            height: height as u16,
            agent_count: agent_count as u16,
            padding_b: 0, padding_c: 0, padding_d: 0, padding_e: 0,
            padding_f: 0,
            padding_g: 0,
        };

        let voxel_content = grid_buffer.read().unwrap();
        let slice_start = (depth - 1) * width * height;
        let slice_end = slice_start + (width * height);
        let voxel_data = &voxel_content[slice_start..slice_end];

        let agent_content = agent_buffer.read().unwrap();

        let mut packet = Vec::with_capacity(48 + (width * height * 4) + (agent_count * 64));
        
        unsafe {
            let header_ptr = &header as *const TelemetryHeader as *const u8;
            packet.extend_from_slice(std::slice::from_raw_parts(header_ptr, 48));
            
            let voxel_ptr = voxel_data.as_ptr() as *const u8;
            packet.extend_from_slice(std::slice::from_raw_parts(voxel_ptr, width * height * 4));
            
            let agent_ptr = agent_content.as_ptr() as *const u8;
            packet.extend_from_slice(std::slice::from_raw_parts(agent_ptr, agent_count * 64));
        }

        publisher.send(packet, 0).unwrap();

        if tick % 60 == 0 {
            println!("[Tick {}] Life Engine Active. Agents: {} | TPS: {:.1}", tick, agent_count, current_tps);
        }

        tick += 1;
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
        assert_eq!(std::mem::size_of::<TelemetryHeader>(), 48);
    }

    #[test]
    fn test_agent_size() {
        assert_eq!(std::mem::size_of::<Agent>(), 64);
    }
}
