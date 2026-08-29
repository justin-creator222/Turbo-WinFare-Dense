// Shader to probe subgroup / wave size at runtime on Vulkan 1.3
RWStructuredBuffer<uint> out_buf : register(u0);

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x == 0) {
        out_buf[0] = WaveGetLaneCount();
        out_buf[1] = WaveIsFirstLane() ? 1 : 0;
    }
}
