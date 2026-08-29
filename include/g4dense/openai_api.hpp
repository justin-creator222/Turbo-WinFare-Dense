#pragma once

#include "g4dense/json.hpp"
#include "g4dense/sampling.hpp"
#include <string>
#include <vector>
#include <optional>
#include <deque>
#include <mutex>
#include <condition_variable>

namespace g4dense {

enum class StopReason {
    Eos,
    Length,
    Cancelled
};

struct GenerationResult {
    std::string text;
    int prompt_tokens{0};
    int completion_tokens{0};
    double duration_ms{0.0};
    double tps{0.0};
    StopReason stop_reason{StopReason::Eos};
};

struct OpenAIServerConfig {
    std::string default_model{"gemma-4-31b-dense"};
    int default_max_tokens{512};
    int max_context_tokens{8192};
};

struct ValidatedChatRequest {
    std::string model;
    std::vector<std::pair<std::string, std::string>> messages; // role -> content
    int max_tokens{512};
    SamplingParams sampling{};
    bool stream{false};
};

struct ApiError {
    int http_status{400};
    std::string type{"invalid_request_error"};
    std::string message;
    std::string param;
};

bool validate_chat_request(const JsonValue& body, const OpenAIServerConfig& cfg,
                           ValidatedChatRequest& out, ApiError& err);

std::string render_error(const ApiError& err);
std::string render_completion(const std::string& id, const std::string& model,
                              const GenerationResult& result, int64_t created);
std::string render_chunk(const std::string& id, const std::string& model, int64_t created,
                         const std::string& delta, bool role_only);
std::string render_final_chunk(const std::string& id, const std::string& model, int64_t created,
                               StopReason reason);
std::string render_usage_chunk(const std::string& id, const std::string& model, int64_t created,
                               const GenerationResult& result);
std::string render_models_list(const std::string& model_id);

const char* finish_reason_for(StopReason reason);

class RequestCoordinator {
public:
    explicit RequestCoordinator(int queue_limit = 4) : queue_limit_(queue_limit) {}

    class Lease {
    public:
        Lease() = default;
        explicit Lease(RequestCoordinator* c) : owner_(c) {}
        ~Lease() { if (owner_) owner_->release(); }
        Lease(Lease&& o) noexcept : owner_(o.owner_) { o.owner_ = nullptr; }
        Lease& operator=(Lease&& o) noexcept {
            if (this != &o) { if (owner_) owner_->release(); owner_ = o.owner_; o.owner_ = nullptr; }
            return *this;
        }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
    private:
        RequestCoordinator* owner_{nullptr};
    };

    std::optional<Lease> acquire();
    int waiting() const {
        std::lock_guard<std::mutex> g(mutex_);
        return static_cast<int>(queue_.size());
    }

private:
    void release();
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<uint64_t> queue_;
    uint64_t next_ticket_{0};
    bool busy_{false};
    int queue_limit_{4};
};

} // namespace g4dense
