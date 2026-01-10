use std::sync::Arc;
use std::thread;
use std::time::Duration;
use serde::{Serialize};

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

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/growth.glsl",
    }
}

fn main() {
    println!("--- Genesis Core: GPU Engine Starting ---");

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

    println!("Using GPU: {}", physical_device.properties().device_name);

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

    let width = 128;
    let height = 128;
    let depth = 32;
    let data_size = width * height * depth;
    let grid_data = vec![Voxel::new(1, 0, 20, 255); data_size]; 

    let grid_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        grid_data,
    ).expect("failed to create buffer");

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
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, grid_buffer.clone())],
        [],
    ).unwrap();

    println!("Simulation Loop Active...");
    let mut tick = 0;
    loop {
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

        if tick % 60 == 0 {
            println!("[Tick {}] GPU simulation step complete.", tick);
        }

        tick += 1;
        thread::sleep(Duration::from_millis(16));
    }
}