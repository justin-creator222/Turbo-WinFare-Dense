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

// A model download + conversion running in the background.
//
// A new user has no .g4dense container and no way to get one without leaving the UI for a
// 40-minute command line job. This is that job, driven from the browser. The work itself is
// tools/fetch_model.py -- the conversion is MLX INT4 repacking and has no C++ equivalent -- so
// the server supervises a child process rather than doing it directly.
struct DownloadJob {
    enum class State { Idle, Running, Done, Failed, Cancelled };

    State state{State::Idle};
    std::string model;      // "e2b" | "31b"
    std::string stage;      // download | convert | verify
    std::string message;
    std::string container;  // set on success
    double percent{0.0};
    int64_t started_ms{0};
};

class HTTPServer {
public:
    HTTPServer();
    ~HTTPServer();

    void start(uint16_t port, std::shared_ptr<ForwardRunner> runner, std::shared_ptr<VulkanContext> ctx = nullptr);
    void stop();
    bool is_running() const { return is_running_; }

    void set_context(std::shared_ptr<VulkanContext> ctx);

    // Model provisioning, for a machine that has no containers yet.
    bool start_download(const std::string& model, std::string& error);
    void cancel_download();
    DownloadJob download_status() const;
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

    // Download job state. The worker thread writes it; /api/download_status reads it.
    DownloadJob job_;
    mutable std::mutex job_mutex_;
    std::thread job_thread_;
    // Set by cancel_download(); the worker terminates the child process and exits.
    std::atomic<bool> job_cancel_{false};
    // The child process handle, so cancellation can actually stop a 40-minute download rather
    // than only setting a flag the worker checks between lines.
    std::atomic<void*> job_process_{nullptr};

    void run_download_job(std::string model);

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
