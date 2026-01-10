use std::thread;
use std::time::Duration;
use serde::{Serialize};
use chrono::Utc;

#[derive(Copy, Clone, Serialize)]
struct Voxel(u32);

impl Voxel {
    fn new(id: u8, state: u8, thermal: u8, light: u8) -> Self {
        let packed = (id as u32) | 
                     ((state as u32) << 8) | 
                     ((thermal as u32) << 16) | 
                     ((light as u32) << 24);
        Voxel(packed)
    }
}

#[derive(Serialize)]
struct Heartbeat {
    tick: u64,
    node_name: String,
    status: String,
    voxel_count: usize,
    timestamp: i64,
}

fn main() {
    let context = zmq::Context::new();
    let publisher = context.socket(zmq::PUB).unwrap();
    publisher.bind("tcp://*:5555").expect("Could not bind publisher");
    
    println!("--- Genesis Core Phase 2 Starting ---");
    
    let width = 128;
    let height = 128;
    let depth = 32;
    let grid = vec![Voxel::new(0, 0, 20, 255); width * height * depth];
    let voxel_count = grid.len();
    
    println!("Voxel Grid Initialized: {} voxels", voxel_count);

    let mut tick = 0;
    loop {
        let heartbeat = Heartbeat {
            tick,
            node_name: "genesis-compute".to_string(),
            status: "SIMULATING".to_string(),
            voxel_count,
            timestamp: Utc::now().timestamp(),
        };

        let json = serde_json::to_string(&heartbeat).unwrap();
        publisher.send(&json, 0).unwrap();

        if tick % 60 == 0 {
            println!("Tick {}: Simulating {} voxels", tick, voxel_count);
        }

        tick += 1;
        thread::sleep(Duration::from_millis(16)); 
    }
}
