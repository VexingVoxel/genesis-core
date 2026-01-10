#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer VoxelGrid {
    uint voxels[];
} grid;

layout(set = 0, binding = 1) uniform TimeInfo {
    uint u_time; // This is the tick count, ~60 per second
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= grid.voxels.length()) return;

    uint v = grid.voxels[idx];
    uint id = v & 0xFF;
    uint state = (v >> 8) & 0xFF;
    
    // Logic: If Dirt (1), and if the tick is a multiple of 60 (i.e., once per second)
    if (id == 1 && u_time > 0 && u_time % 60 == 0) {
        state += 10; // Increase state by a noticeable amount each second
        
        if (state >= 255) {
            id = 2; // Transition to Grass (ID 2)
            state = 0;
        }
    }
    
    // Pack back
    grid.voxels[idx] = (v & 0xFFFF0000) | (state << 8) | id;
}