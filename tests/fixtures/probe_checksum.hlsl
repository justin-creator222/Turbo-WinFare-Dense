// Shader to checksum a large host-visible buffer on Vulkan 1.3
ByteAddressBuffer in_buf : register(t0, space0);
RWStructuredBuffer<uint> out_buf : register(u1, space0);

struct PushParams {
    uint num_words;
};
[[vk::push_constant]] PushParams params;

groupshared uint s_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID) {
    uint local_sum = 0;
    uint stride = 256 * 256;
    for (uint i = dtid.x; i < params.num_words; i += stride) {
        local_sum += in_buf.Load(i * 4);
    }
    s_sum[gtid.x] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (gtid.x < s) {
            s_sum[gtid.x] += s_sum[gtid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gtid.x == 0) {
        InterlockedAdd(out_buf[0], s_sum[0]);
    }
}
