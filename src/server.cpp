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

    // ---- Endpoints the GUI calls that previously returned 404 -----------------------
    //
    // gui/app.js calls nine endpoints; only telemetry, switch_tier and the /v1 pair existed,
    // so six 404'd. The visible symptom was a sidebar showing hardcoded HTML placeholders
    // instead of the engine's resolved values, and inert Stop / model-management controls.

    // GET  /api/config -- the engine's RESOLVED configuration, not the HTML defaults.
    // POST /api/config -- apply what can change live; report what needs a reload.
    if (req.path == "/api/config") {
        auto r = current_runner();
        if (req.method == "GET") {
            std::lock_guard<std::mutex> guard(config_mutex_);
            std::ostringstream js;
            js << "{\"config\":{"
               << "\"temperature\":" << config_.temperature
               << ",\"top_p\":" << config_.top_p
               << ",\"top_k\":" << config_.top_k
               << ",\"max_tokens\":" << config_.max_tokens
               << ",\"slots\":" << (r ? 4 : 0)
               << ",\"eviction_policy\":\"LRU\""
               << ",\"context_len\":" << config_.context_len
               // Bounded by ATTN_MAX_SPAN: full-attention layers stage their whole score span
               // in groupshared, so the GUI must not offer a context the kernel cannot serve.
               << ",\"context_max\":" << ForwardRunner::kAttentionMaxSpan
               << ",\"tier\":" << (r ? r->active_tier_id() : config_.active_tier_id)
               << "}}";
            send_http_response(client_socket, 200, "application/json", js.str(), false);
            return;
        }
        if (req.method == "POST") {
            try {
                JsonValue root = JsonValue::parse(req.body);
                bool requires_reload = false;
                {
                    std::lock_guard<std::mutex> guard(config_mutex_);
                    if (root.has("temperature")) config_.temperature = static_cast<float>(root.at("temperature", "body").as_double("temperature"));
                    if (root.has("top_p"))       config_.top_p = static_cast<float>(root.at("top_p", "body").as_double("top_p"));
                    if (root.has("top_k"))       config_.top_k = static_cast<int>(root.at("top_k", "body").as_uint32("top_k"));
                    if (root.has("max_tokens"))  config_.max_tokens = static_cast<int>(root.at("max_tokens", "body").as_uint32("max_tokens"));
                    // Slots and context are fixed at initialize(); saying so is what stops the
                    // GUI from appearing to apply a change that did nothing.
                    if (root.has("context_len")) {
                        int want = static_cast<int>(root.at("context_len", "body").as_uint32("context_len"));
                        if (want != config_.context_len) requires_reload = true;
                    }
                    if (root.has("slots")) requires_reload = true;
                }
                send_http_response(client_socket, 200, "application/json",
                                   std::string("{\"requires_reload\":") + (requires_reload ? "true" : "false") + "}", false);
            } catch (const std::exception&) {
                send_http_response(client_socket, 400, "application/json", "{\"error\":\"invalid config body\"}", false);
            }
            return;
        }
    }

    // GET /api/models -- only containers that actually PARSE, across every search root.
    // An existence check is not enough: a truncated or stale file would be offered and then
    // fail at load.
    if (req.path == "/api/models" && req.method == "GET") {
        auto r = current_runner();
        std::string active = r ? r->container_path() : std::string{};
        std::error_code ec;
        std::ostringstream js;
        js << "{\"models\":[";
        bool first = true;
        for (const auto& root : bundle_search_roots()) {
            if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) continue;
            for (const auto& entry : fs::directory_iterator(root, ec)) {
                if (ec) break;
                const auto& p = entry.path();
                if (p.extension() != ".g4dense") continue;
                if (!bundle_loads(p.string())) continue;
                fs::path canon_a = fs::weakly_canonical(p, ec);
                fs::path canon_b = active.empty() ? fs::path{} : fs::weakly_canonical(fs::path(active), ec);
                if (!first) js << ",";
                first = false;
                js << "{\"path\":\"";
                for (char c : p.string()) { if (c == '\\') js << "\\\\"; else js << c; }
                js << "\",\"name\":\"" << p.filename().string() << "\""
                   << ",\"is_active\":" << ((!active.empty() && canon_a == canon_b) ? "true" : "false")
                   << "}";
            }
        }
        js << "]}";
        send_http_response(client_socket, 200, "application/json", js.str(), false);
        return;
    }

    // POST /api/load_model, /api/unload_model -- swap_runner() already validates and
    // initializes the new container BEFORE releasing the old one, so a bad path cannot
    // destroy a working model.
    if ((req.path == "/api/load_model" || req.path == "/api/unload_model") && req.method == "POST") {
        const bool load = (req.path == "/api/load_model");
        std::string path;
        if (load) {
            try {
                JsonValue root = JsonValue::parse(req.body);
                path = root.at("path", "body").as_string("path");
            } catch (const std::exception&) {
                send_http_response(client_socket, 400, "application/json", "{\"error\":\"missing path\"}", false);
                return;
            }
        }
        std::string error;
        if (swap_runner(path, load, error)) {
            send_http_response(client_socket, 200, "application/json", "{\"status\":\"ok\"}", false);
        } else {
            std::ostringstream js;
            js << "{\"error\":\"";
            for (char c : error) { if (c == '"' || c == '\\') js << '\\'; js << c; }
            js << "\"}";
            send_http_response(client_socket, 503, "application/json", js.str(), false);
        }
        return;
    }

    // POST /api/clear_cache -- drop the layer streamer's resident slots.
    if (req.path == "/api/clear_cache" && req.method == "POST") {
        auto r = current_runner();
        if (r) r->clear_layer_cache();
        send_http_response(client_socket, 200, "application/json", "{\"status\":\"ok\"}", false);
        return;
    }

    // POST /api/stop -- generate() polls this flag between tokens.
    if (req.path == "/api/stop" && req.method == "POST") {
        cancel_generation_ = true;
        send_http_response(client_socket, 200, "application/json", "{\"status\":\"stopping\"}", false);
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

    // Render the parsed conversation through the Gemma 4 turn template.
    //
    // This used to pass `req.body` -- the raw HTTP JSON payload -- straight in as the prompt,
    // so the model was fed `{"model":...,"messages":[...]}` as literal text and never saw the
    // user's question at all. Rendering here also preserves multi-turn history, which a
    // single-string prompt cannot.
    std::vector<Tokenizer::ChatMessage> tmpl_messages;
    tmpl_messages.reserve(chat_req.messages.size());
    for (const auto& [role, content] : chat_req.messages) {
        tmpl_messages.push_back(Tokenizer::ChatMessage{role, content});
    }
    std::string chat_prompt = r->tokenizer()
                                  ? r->tokenizer()->apply_chat_template(tmpl_messages)
                                  : std::string{};
    // generate() would otherwise wrap it a second time.
    gen_opts.use_chat_template = false;

    // A fresh generation clears any pending stop left over from the previous one.
    cancel_generation_ = false;

    if (chat_req.stream) {
        send_http_headers(client, 200, "text/event-stream");
        std::string role_chunk = render_chunk(id, chat_req.model, created, "", true);
        send_raw(client, role_chunk);

        r->generate(chat_prompt, gen_opts, [&](uint32_t, const std::string& piece) {
            std::string sse_chunk = render_chunk(id, chat_req.model, created, piece, false);
            return send_raw(client, sse_chunk);
        }, &cancel_generation_);

        std::string finish_chunk = render_final_chunk(id, chat_req.model, created, StopReason::Eos);
        send_raw(client, finish_chunk);
        send_raw(client, "data: [DONE]\n\n");
    } else {
        std::string full_response;
        int emitted = 0;
        r->generate(chat_prompt, gen_opts, [&](uint32_t, const std::string& piece) {
            full_response += piece;
            ++emitted;
            return true;
        }, &cancel_generation_);

        GenerationResult res;
        res.text = full_response;
        res.stop_reason = StopReason::Eos;
        // These were left at zero, so every response reported usage 0/0/0.
        res.completion_tokens = emitted;
        if (auto tk = r->tokenizer()) {
            res.prompt_tokens = static_cast<int>(tk->encode(chat_prompt, false).size());
        }
        std::string json = render_completion(id, chat_req.model, res, created);
        send_http_response(client, 200, "application/json", json, false);
    }
}

} // namespace g4dense
