#pragma once

#include "g4dense/streamer.hpp"
#include <vector>
#include <memory>

namespace g4dense {

class LayerCacheManager {
public:
    explicit LayerCacheManager(std::shared_ptr<LayerStreamer> streamer);
    ~LayerCacheManager() = default;

    // Prefetch next layers in pipeline (depth-3 pipeline)
    void prefetch_layer(int layer_id);

    // Get mapped host pointer for layer
    void* get_layer_host_ptr(int layer_id);

private:
    std::shared_ptr<LayerStreamer> streamer_;
};

} // namespace g4dense
