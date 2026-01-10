#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer VoxelGrid {
    uint voxels[];
} grid;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= grid.voxels.length()) return;

    uint v = grid.voxels[idx];
    uint id = v & 0xFF;
    uint state = (v >> 8) & 0xFF;
    
    // Logic: If Dirt (1), increase nutrient state
    if (id == 1) {
        state += 1;
        if (state >= 255) {
            id = 2; // Transition to Grass (ID 2)
            state = 0;
        }
    }
    
    // Pack back (Keep thermal/light bits [16-31], update state [8-15] and ID [0-7])
    grid.voxels[idx] = (v & 0xFFFF0000) | (state << 8) | id;
}
