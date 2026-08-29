#include <winsock2.h>
#include <ws2tcpip.h>
#include "g4dense/server.hpp"
#include "g4dense/manifest.hpp"
#include "g4dense/json.hpp"
#include "g4dense/http.hpp"
#include "g4dense/telemetry.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <chrono>

#pragma comment(lib, "ws2_32.lib")
namespace fs = std::filesystem;

namespace g4dense {

HTTPServer::HTTPServer() {
    coordinator_ = std::make_unique<RequestCoordinator>(4);
}

HTTPServer::~HTTPServer() {
    stop();
}

std::shared_ptr<ForwardRunner> HTTPServer::current_runner() const {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    return runner_;
}

std::shared_ptr<VulkanContext> HTTPServer::current_ctx() const {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    return ctx_;
}

void HTTPServer::set_context(std::shared_ptr<VulkanContext> ctx) {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    ctx_ = ctx;
}

void HTTPServer::set_load_error(const std::string& message) {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    load_error_ = message;
}

void HTTPServer::set_openai_config(const OpenAIServerConfig& cfg) {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    openai_cfg_ = cfg;
}

OpenAIServerConfig HTTPServer::openai_config() const {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    return openai_cfg_;
}

void HTTPServer::set_initial_engine_config(int context_len, uint32_t tier_id) {
    std::lock_guard<std::mutex> guard(config_mutex_);
    config_.context_len = context_len;
    config_.active_tier_id = tier_id;
}

bool HTTPServer::swap_runner(const std::string& container_path, bool load, std::string& error) {
    std::unique_lock<std::mutex> gen(generate_mutex_, std::try_to_lock);
    if (!gen.owns_lock()) {
        error = "BUSY: A generation is in progress.";
        return false;
    }

    if (!load) {
        std::lock_guard<std::mutex> guard(runner_mutex_);
        runner_ = nullptr;
        return true;
    }

    std::shared_ptr<VulkanContext> ctx = current_ctx();
    if (!ctx) {
        ctx = std::make_shared<VulkanContext>();
        ctx->initialize();
        std::lock_guard<std::mutex> guard(runner_mutex_);
        ctx_ = ctx;
    }

    try {
        std::string resolved = resolve_bundle_path(container_path);
        auto tok = std::make_shared<Tokenizer>();
        tok->load_vocabulary();

        auto next = std::make_shared<ForwardRunner>(ctx, tok, resolved);
        next->initialize();

        {
            std::lock_guard<std::mutex> guard(runner_mutex_);
            runner_ = std::move(next);
            load_error_.clear();
        }
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        std::lock_guard<std::mutex> guard(runner_mutex_);
        load_error_ = error;
        return false;
    }
}

void HTTPServer::start(uint16_t port, std::shared_ptr<ForwardRunner> runner, std::shared_ptr<VulkanContext> ctx) {
    {
        std::lock_guard<std::mutex> guard(runner_mutex_);
        runner_ = runner;
        ctx_ = ctx;
    }

    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    listen_socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    int opt = 1;
    setsockopt(listen_socket_, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    if (bind(listen_socket_, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        closesocket(listen_socket_);
        throw G4DenseFormatError("HTTPServer: failed to bind port " + std::to_string(port));
    }

    if (listen(listen_socket_, SOMAXCONN) == SOCKET_ERROR) {
        closesocket(listen_socket_);
        throw G4DenseFormatError("HTTPServer: failed to listen on socket");
    }

    is_running_ = true;
    server_thread_ = std::thread(&HTTPServer::listen_loop, this, port);
}

void HTTPServer::stop() {
    if (is_running_.exchange(false)) {
        if (listen_socket_ != 0 && listen_socket_ != INVALID_SOCKET) {
            closesocket(listen_socket_);
            listen_socket_ = 0;
        }
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        WSACleanup();
    }
}

void HTTPServer::listen_loop(uint16_t port) {
    while (is_running_) {
        sockaddr_in client_addr{};
        int addr_len = sizeof(client_addr);
        SOCKET client_sock = accept(listen_socket_, (sockaddr*)&client_addr, &addr_len);
        if (client_sock == INVALID_SOCKET) {
            if (!is_running_) break;
            continue;
        }

        std::thread([this, client_sock]() {
            handle_client(client_sock);
            closesocket(client_sock);
        }).detach();
    }
}

void HTTPServer::handle_client(uintptr_t client_socket) {
    HttpRequest req;
    HttpReadResult res = read_http_request(client_socket, req, 16 * 1024 * 1024, 10000);
    if (res != HttpReadResult::Ok) {
        send_http_response(client_socket, 400, "text/plain", "Bad Request", false);
        return;
    }

    // CORS preflight
    if (req.method == "OPTIONS") {
        std::vector<std::pair<std::string, std::string>> cors_headers = {
            {"Access-Control-Allow-Origin", "*"},
            {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
            {"Access-Control-Allow-Headers", "Content-Type, Authorization"}
        };
        send_http_response(client_socket, 200, "text/plain", "", false, cors_headers);
        return;
    }

    // OpenAI routing
    if (req.path.rfind("/v1/", 0) == 0) {
        handle_openai(client_socket, req);
        return;
    }

    // API Telemetry
    if (req.path == "/api/telemetry") {
        auto r = current_runner();
        std::string json = r ? r->get_latest_telemetry().to_json_string() : "{}";
        send_http_response(client_socket, 200, "application/json", json, false);
        return;
    }

    // API Switch Tier
    if (req.path == "/api/switch_tier" && req.method == "POST") {
        try {
            JsonValue root = JsonValue::parse(req.body);
            uint32_t tier_id = root.at("tier_id", "body").as_uint32("tier_id");
            auto r = current_runner();
            if (r) {
                r->switch_memory_tier(tier_id);
            }
            send_http_response(client_socket, 200, "application/json", "{\"status\": \"ok\"}", false);
        } catch (const std::exception& e) {
            send_http_response(client_socket, 400, "application/json", "{\"error\": \"invalid request\"}", false);
        }
        return;
    }

    // Static Web GUI files
    std::string file_path = "gui" + req.path;
    if (req.path == "/" || req.path.empty()) {
        file_path = fs::exists("gui/index.html") ? "gui/index.html" : "../gui/index.html";
    } else if (!fs::exists(file_path)) {
        file_path = "../gui" + req.path;
    }

    if (fs::exists(file_path) && !fs::is_directory(file_path)) {
        std::ifstream f(file_path, std::ios::binary);
        std::ostringstream ss;
        ss << f.rdbuf();
        std::string content = ss.str();

        std::string mime = "text/plain";
        if (file_path.ends_with(".html")) mime = "text/html";
        else if (file_path.ends_with(".css")) mime = "text/css";
        else if (file_path.ends_with(".js")) mime = "application/javascript";
        else if (file_path.ends_with(".svg")) mime = "image/svg+xml";
        else if (file_path.ends_with(".png")) mime = "image/png";

        send_http_response(client_socket, 200, mime, content, false);
        return;
    }

    send_http_response(client_socket, 404, "text/plain", "Not Found", false);
}

bool HTTPServer::handle_openai(uintptr_t client, const HttpRequest& req) {
    if (req.path == "/v1/models") {
        std::string json = render_models_list("gemma-4-31b-dense");
        send_http_response(client, 200, "application/json", json, false);
        return true;
    }

    if (req.path == "/v1/chat/completions") {
        handle_chat_completion(client, req);
        return true;
    }

    send_http_response(client, 404, "application/json", "{\"error\": \"endpoint not found\"}", false);
    return true;
}

void HTTPServer::handle_chat_completion(uintptr_t client, const HttpRequest& req) {
    JsonValue body;
    try {
        body = JsonValue::parse(req.body, "chat_body");
    } catch (...) {
        send_http_response(client, 400, "application/json", "{\"error\": \"invalid JSON\"}", false);
        return;
    }

    ValidatedChatRequest chat_req;
    ApiError err;
    if (!validate_chat_request(body, openai_cfg_, chat_req, err)) {
        send_http_response(client, err.http_status, "application/json", render_error(err), false);
        return;
    }

    auto lease = coordinator_->acquire();
    if (!lease) {
        send_http_response(client, 429, "application/json", "{\"error\": \"queue full\"}", false);
        return;
    }

    std::unique_lock<std::mutex> gen_lock(generate_mutex_);
    auto r = current_runner();
    if (!r) {
        send_http_response(client, 503, "application/json", "{\"error\": \"no model loaded\"}", false);
        return;
    }

    std::string id = "chatcmpl-g4dense";
    int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    GenerationOptions gen_opts;
    gen_opts.max_tokens = chat_req.max_tokens;
    gen_opts.sampling = chat_req.sampling;

    if (chat_req.stream) {
        send_http_headers(client, 200, "text/event-stream");
        std::string role_chunk = render_chunk(id, chat_req.model, created, "", true);
        send_raw(client, role_chunk);

        r->generate(req.body, gen_opts, [&](uint32_t, const std::string& piece) {
            std::string sse_chunk = render_chunk(id, chat_req.model, created, piece, false);
            return send_raw(client, sse_chunk);
        });

        std::string finish_chunk = render_final_chunk(id, chat_req.model, created, StopReason::Eos);
        send_raw(client, finish_chunk);
        send_raw(client, "data: [DONE]\n\n");
    } else {
        std::string full_response;
        r->generate(req.body, gen_opts, [&](uint32_t, const std::string& piece) {
            full_response += piece;
            return true;
        });

        GenerationResult res;
        res.text = full_response;
        res.stop_reason = StopReason::Eos;
        std::string json = render_completion(id, chat_req.model, res, created);
        send_http_response(client, 200, "application/json", json, false);
    }
}

} // namespace g4dense
