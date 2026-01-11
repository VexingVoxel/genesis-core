#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Agent {
    float pos_x, pos_y, pos_z;
    float vel_x, vel_y, vel_z;
    float rotation;
    uint vitals;
    uint brain_id_lo;
    uint brain_id_hi;
    uint padding[6]; // Pad to 64 bytes
};

layout(set = 0, binding = 0, std430) buffer VoxelGrid {
    uint voxels[];
} grid;

layout(set = 0, binding = 1, std430) buffer AgentBuffer {
    Agent agents[];
} agents;

layout(set = 0, binding = 2) uniform SimInfo {
    uint u_time;
    uint world_width;
    uint world_height;
    uint world_depth;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= agents.agents.length()) return;

    Agent a = agents.agents[idx];

    // 1. Movement (Simple Euler Integration)
    a.pos_x += a.vel_x;
    a.pos_y += a.vel_y;
    a.pos_z += a.vel_z;

    // 2. World Boundaries (Wrapping X/Y, Clamping Z)
    if (a.pos_x < 0.0) a.pos_x += float(world_width);
    if (a.pos_x >= float(world_width)) a.pos_x -= float(world_width);
    if (a.pos_y < 0.0) a.pos_y += float(world_height);
    if (a.pos_y >= float(world_height)) a.pos_y -= float(world_height);
    
    // Clamp Z to the top layer for now
    if (a.pos_z < 0.0) a.pos_z = 0.0;
    if (a.pos_z >= float(world_depth)) a.pos_z = float(world_depth - 1);

    // 3. Grazing Logic
    // Map floating point position to voxel index (Top Layer)
    uint vx = uint(a.pos_x);
    uint vy = uint(a.pos_y);
    uint vz = uint(a.pos_z);
    
    uint v_idx = (vz * world_width * world_height) + (vy * world_width) + vx;
    
    if (v_idx < grid.voxels.length()) {
        uint v = grid.voxels[v_idx];
        uint id = v & 0xFF;
        
        if (id == 2) { // Grass
            // Convert to Dirt (1)
            grid.voxels[v_idx] = (v & 0xFFFFFF00) | 1;
            
            // Reset Hunger (lowest 8 bits of vitals)
            a.vitals = (a.vitals & 0xFFFFFF00) | 0;
        } else {
            // Increase Hunger every 10 ticks
            if (u_time % 10 == 0) {
                uint hunger = a.vitals & 0xFF;
                if (hunger < 255) hunger++;
                a.vitals = (a.vitals & 0xFFFFFF00) | hunger;
            }
        }
    }

    // Write back
    agents.agents[idx] = a;
}
