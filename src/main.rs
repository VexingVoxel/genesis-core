use std::sync::Arc;
use std::thread;
use std::time::Duration;
use serde::{Serialize};
use chrono::Utc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
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

#[derive(Serialize)]
struct Heartbeat {
    tick: u64,
    node_name: String,
    status: String,
    timestamp: i64,
    voxel_slice: Vec<u32>, // The 32x32 visualization slice
}

// NEW: Add a struct for our uniform data
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
    println!("--- Genesis Core: Phase 2 Engine Starting ---");

    // 0. Initialize ZMQ
    let zmq_context = zmq::Context::new();
    let publisher = zmq_context.socket(zmq::PUB).unwrap();
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

    // NEW: Create buffer for time uniform
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
    loop {
        // NEW: Update the time buffer and create a new descriptor set for this frame
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
        
        vulkano::sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        // --- Read Voxel Slice for Visualization ---
        let mut voxel_slice = Vec::with_capacity(1024); // 32x32
        
        if tick % 10 == 0 {
            let content = grid_buffer.read().unwrap();
            for y in 48..80 {
                for x in 48..80 {
                    let idx = (31 * width * height) + (y * width) + x;
                    voxel_slice.push(content[idx].packed);
                }
            }
        }

        // Send Heartbeat
        let heartbeat = Heartbeat {
            tick,
            node_name: "genesis-compute".to_string(),
            status: "SIMULATING".to_string(),
            timestamp: Utc::now().timestamp(),
            voxel_slice,
        };
        
        let json = serde_json::to_string(&heartbeat).unwrap();
        publisher.send(&json, 0).unwrap();

        if tick % 60 == 0 {
            println!("[Tick {}] Heartbeat + Voxel Data sent.", tick);
        }

        tick += 1;
        thread::sleep(Duration::from_millis(16));
    }
}