#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Agent {
    float pos_x, pos_y, pos_z;
    float vel_x, vel_y, vel_z;
    float rotation;
    uint vitals;
    uint brain_id_lo;
    uint brain_id_hi;
    uint padding[6];
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

    // 1. Movement
    a.pos_x += a.vel_x;
    a.pos_y += a.vel_y;
    a.pos_z += a.vel_z;

    // 2. Wrap boundaries
    if (a.pos_x < 0.0) a.pos_x += float(world_width);
    if (a.pos_x >= float(world_width)) a.pos_x -= float(world_width);
    if (a.pos_y < 0.0) a.pos_y += float(world_height);
    if (a.pos_y >= float(world_height)) a.pos_y -= float(world_height);

    // 3. Dynamic Rotation (Face direction of velocity)
    // Adding + PI/2 (1.5707) because our triangle mesh faces 'Up' by default
    a.rotation = atan(a.vel_y, a.vel_x) + 1.57079;

    // 4. Grazing Logic (Now aligned with Godot coordinate space)
    uint vx = uint(a.pos_x);
    uint vy = uint(a.pos_y);
    uint vz = uint(a.pos_z);
    
    uint v_idx = (vz * world_width * world_height) + (vy * world_width) + vx;
    
    if (v_idx < grid.voxels.length()) {
        uint v = grid.voxels[v_idx];
        uint id = v & 0xFF;
        
        if (id == 2) { // Grass
            grid.voxels[v_idx] = (v & 0xFFFFFF00) | 1; // Change to Dirt
            a.vitals = (a.vitals & 0xFFFFFF00) | 0;    // Reset Hunger
        }
    }

    agents.agents[idx] = a;
}