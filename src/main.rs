use std::thread;
use std::time::Duration;
use serde::{Serialize};
use chrono::Utc;

#[derive(Serialize)]
struct Heartbeat {
    tick: u64,
    node_name: String,
    status: String,
    timestamp: i64,
}

fn main() {
    let context = zmq::Context::new();
    let publisher = context.socket(zmq::PUB).unwrap();
    
    // Bind to all interfaces on port 5555
    publisher.bind("tcp://*:5555").expect("Could not bind publisher");
    
    println!("--- Genesis Core Heartbeat Emitter Starting ---");
    println!("Broadcasting on tcp://*:5555");

    let mut tick = 0;
    let node_name = "genesis-compute".to_string();

    loop {
        let heartbeat = Heartbeat {
            tick,
            node_name: node_name.clone(),
            status: "OK".to_string(),
            timestamp: Utc::now().timestamp(),
        };

        let json = serde_json::to_string(&heartbeat).unwrap();
        publisher.send(&json, 0).unwrap();

        if tick % 10 == 0 {
            println!("Sent Heartbeat: {}", json);
        }

        tick += 1;
        thread::sleep(Duration::from_secs(1));
    }
}
