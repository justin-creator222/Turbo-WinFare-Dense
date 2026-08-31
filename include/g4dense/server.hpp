#pragma once

#include "g4dense/runner.hpp"
#include "g4dense/openai_api.hpp"
#include "g4dense/http.hpp"
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>

namespace g4dense {

struct ServerConfig {
    float temperature{0.2f};
    float top_p{0.95f};
    int top_k{64};
    // The GUI has had sliders for these two since it was written, and posted them to
    // /api/config, which silently ignored both -- a control that parses and does
    // nothing, which this project's own contributing rules forbid.
    float repetition_penalty{1.0f};
    uint32_t draft_k{8};
    bool speculative_enabled{true};
    // 0 = no fixed seed. Exposed so a run can be reproduced from the UI.
    bool has_seed{false};
    uint64_t seed{0};
    int max_tokens{512};
    uint32_t active_tier_id{1};
    // The engine's default, not an aspiration. This said 8192 while the runner initialised
    // 4096, so /api/config reported a context the model was not running.
    int context_len{4096};
};

class HTTPServer {
public:
    HTTPServer();
    ~HTTPServer();

    void start(uint16_t port, std::shared_ptr<ForwardRunner> runner, std::shared_ptr<VulkanContext> ctx = nullptr);
    void stop();
    bool is_running() const { return is_running_; }

    void set_context(std::shared_ptr<VulkanContext> ctx);
    std::shared_ptr<ForwardRunner> runner() const;

    void set_load_error(const std::string& message);
    void set_openai_config(const OpenAIServerConfig& cfg);
    OpenAIServerConfig openai_config() const;

    void set_initial_engine_config(int context_len, uint32_t tier_id);

private:
    void listen_loop(uint16_t port);
    void handle_client(uintptr_t client_socket);

    bool handle_openai(uintptr_t client, const HttpRequest& req);
    void handle_chat_completion(uintptr_t client, const HttpRequest& req);

    std::shared_ptr<ForwardRunner> current_runner() const;
    std::shared_ptr<VulkanContext> current_ctx() const;
    bool swap_runner(const std::string& container_path, bool load, std::string& error);

    std::shared_ptr<ForwardRunner> runner_;
    std::shared_ptr<VulkanContext> ctx_;

    ServerConfig config_;
    mutable std::mutex config_mutex_;
    std::mutex generate_mutex_;
    mutable std::mutex runner_mutex_;

    OpenAIServerConfig openai_cfg_;
    std::unique_ptr<RequestCoordinator> coordinator_;

    std::string load_error_;

    // Set by POST /api/stop, passed to ForwardRunner::generate as its cancel_flag, and
    // cleared at the start of each generation. Without it the GUI's Stop button had nothing
    // to talk to -- the endpoint returned 404 and a slow generation could not be abandoned.
    std::atomic<bool> cancel_generation_{false};

    std::atomic<bool> is_running_{false};
    std::thread server_thread_;
    uintptr_t listen_socket_{0};
};

} // namespace g4dense
