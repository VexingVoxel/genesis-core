#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer VoxelGrid {
    uint voxels[];
} grid;

layout(set = 0, binding = 2) uniform SimInfo {
    uint u_time;
    uint world_width;
    uint world_height;
    uint world_depth;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= grid.voxels.length()) return;

    uint v = grid.voxels[idx];
    uint id = v & 0xFF;
    uint state = (v >> 8) & 0xFF;
    
    // Logic: If Dirt (1), and if the tick is a multiple of 60
    if (id == 1 && u_time > 0 && u_time % 60 == 0) {
        state += 10;
        
        if (state >= 255) {
            id = 2; // Transition to Grass (ID 2)
            state = 0;
        }
    }
    
    // Pack back
    grid.voxels[idx] = (v & 0xFFFF0000) | (state << 8) | id;
}