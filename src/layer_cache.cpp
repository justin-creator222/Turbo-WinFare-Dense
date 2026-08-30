#include "g4dense/layer_cache.hpp"
#include <stdexcept>

namespace g4dense {

LayerCacheManager::LayerCacheManager(std::shared_ptr<LayerStreamer> streamer)
    : streamer_(std::move(streamer)) {
    if (!streamer_) {
        throw std::invalid_argument("LayerCacheManager: null streamer provided");
    }
}

void LayerCacheManager::prefetch_layer(int layer_id) {
    if (!streamer_ || layer_id < 0) return;
    auto plan = streamer_->plan_layers({layer_id});
    streamer_->fetch_misses(plan);
    streamer_->release_plan(plan);
}

void* LayerCacheManager::get_layer_host_ptr(int layer_id) {
    if (!streamer_ || layer_id < 0) return nullptr;
    auto plan = streamer_->plan_layers({layer_id});
    streamer_->fetch_misses(plan);
    void* ptr = plan.slots.empty() ? nullptr : plan.slots[0]->host_ptr;
    streamer_->release_plan(plan);
    return ptr;
}

} // namespace g4dense
