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
#include <iomanip>

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
        // The KV cache is sized inside initialize(), so a context change can only take effect
        // here. Without this the config endpoint answered requires_reload: true and then the
        // reload quietly ignored the new value.
        {
            std::lock_guard<std::mutex> guard(config_mutex_);
            if (config_.context_len > 0) {
                next->set_max_context(static_cast<uint32_t>(config_.context_len));
            }
        }
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
    // Three endpoints the GUI has always called and the server never implemented, so the
    // model metadata panel and the tier matrix rendered defaults and the "Reset KV Cache"
    // button reported success while doing nothing.
    if (req.path == "/api/model_info" && req.method == "GET") {
        auto r = current_runner();
        std::ostringstream js;
        if (!r) {
            js << "{\"loaded\":false}";
        } else {
            const auto& h = r->header();
            js << std::fixed << std::setprecision(2);
            js << "{\"loaded\":true"
               << ",\"name\":\"" << std::filesystem::path(r->container_path()).stem().string() << "\""
               << ",\"num_layers\":" << h.num_layers
               << ",\"d_model\":" << h.d_model
               << ",\"d_ff\":" << h.d_ff
               << ",\"num_q_heads\":" << h.num_q_heads
               << ",\"num_kv_heads\":" << h.num_kv_heads
               << ",\"head_dim\":" << h.head_dim
               << ",\"global_head_dim\":" << h.global_head_dim
               << ",\"global_kv_heads\":" << h.global_kv_heads
               << ",\"vocab_size\":" << h.vocab_size
               << ",\"sliding_window\":" << h.sliding_window
               << ",\"softcapping\":" << h.final_logit_softcapping
               << ",\"lm_head_size_mb\":" << (static_cast<double>(h.lm_head_size) / (1024.0 * 1024.0))
               << ",\"quant_type\":\"MLX INT4 (Group " << h.quant_group_size << ")\""
               << ",\"scale_dtype\":\"BF16\""
               << ",\"device_name\":\"" << r->device_name() << "\""
               << ",\"has_draft\":" << (r->has_draft_model() ? "true" : "false");

            // WHICH layers stream, not just how many. They are evenly spaced across the stack,
            // so a layer map that shaded the first N cells was drawing the wrong ones.
            js << ",\"streamed_layers\":[";
            const auto& streamed = r->streamed_layers();
            for (size_t i = 0; i < streamed.size(); ++i) {
                js << (i ? "," : "") << streamed[i];
            }
            js << "]";

            js << ",\"global_layers\":[";
            bool first_g = true;
            for (uint32_t l = 0; l < h.num_layers; ++l) {
                if ((h.global_layer_mask >> l) & 1ull) {
                    js << (first_g ? "" : ",") << l;
                    first_g = false;
                }
            }
            js << "]}";
        }
        send_http_response(client_socket, 200, "application/json", js.str(), false);
        return;
    }

    if (req.path == "/api/reset_kv" && req.method == "POST") {
        auto r = current_runner();
        if (r) r->reset_kv_cache();
        send_http_response(client_socket, 200, "application/json",
                           std::string("{\"status\":\"") + (r ? "ok" : "no model") + "\"}", false);
        return;
    }

    if (req.path == "/api/tiers" && req.method == "GET") {
        // Served from config/tiers.json so the UI cannot drift from the planning artifact.
        // If it is missing, say so rather than inventing a table.
        std::string body = "{\"tiers\":[]}";
        try {
            const std::string tiers_path = resolve_resource_path("config/tiers.json");
            std::ifstream tf(tiers_path, std::ios::binary);
            if (tf) {
                std::ostringstream buf;
                buf << tf.rdbuf();
                body = buf.str();
            }
        } catch (const std::exception&) {
            // fall through to the empty table
        }
        send_http_response(client_socket, 200, "application/json", body, false);
        return;
    }

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
               << ",\"repetition_penalty\":" << config_.repetition_penalty
               << ",\"draft_k\":" << config_.draft_k
               << ",\"speculative_enabled\":" << (config_.speculative_enabled ? "true" : "false")
               << ",\"has_draft\":" << ((r && r->has_draft_model()) ? "true" : "false")
               << ",\"draft_k_max\":" << kGemmMaxBatch
               << ",\"seed\":" << config_.seed
               << ",\"has_seed\":" << (config_.has_seed ? "true" : "false")
               << ",\"max_tokens\":" << config_.max_tokens
               << ",\"slots\":" << (r ? 4 : 0)
               << ",\"eviction_policy\":\"LRU\""
               // What the model is ACTUALLY running, when one is loaded. The configured value
               // only takes effect at the next load.
               << ",\"context_len\":" << (r ? r->max_context() : static_cast<uint32_t>(config_.context_len))
               << ",\"context_len_pending\":" << config_.context_len
               // No longer a kernel limit. Attention is an online softmax and nothing in it
               // scales with the attended span; what a larger context costs is KV cache, about
               // 336 MB more at 8192, or roughly one resident layer. So the GUI may offer it,
               // and it takes effect on reload.
               << ",\"context_max\":" << 8192u
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
                    if (root.has("repetition_penalty")) config_.repetition_penalty = static_cast<float>(root.at("repetition_penalty", "body").as_double("repetition_penalty"));
                    if (root.has("draft_k")) {
                        // Clamped, not rejected: the verify batch is draft_k wide, so a larger
                        // value would silently be truncated by the kernel anyway.
                        uint32_t k = root.at("draft_k", "body").as_uint32("draft_k");
                        config_.draft_k = k < 1u ? 1u : (k > kGemmMaxBatch ? kGemmMaxBatch : k);
                    }
                    if (root.has("speculative_enabled")) config_.speculative_enabled = root.at("speculative_enabled", "body").as_bool("speculative_enabled");
                    if (root.has("seed")) {
                        uint64_t sd = root.at("seed", "body").as_uint64("seed");
                        config_.seed = sd;
                        config_.has_seed = (sd != 0);
                    }
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
    // Server-side settings the OpenAI request body has no field for. A per-request value wins
    // where one was supplied; otherwise the configured value applies -- which is what makes the
    // GUI's sliders take effect at all.
    if (chat_req.sampling.repetition_penalty == 1.0f) {
        gen_opts.sampling.repetition_penalty = config_.repetition_penalty;
    }
    if (!gen_opts.sampling.has_seed && config_.has_seed) {
        gen_opts.sampling.has_seed = true;
        gen_opts.sampling.seed = config_.seed;
    }
    gen_opts.draft_k = config_.draft_k;
    gen_opts.speculative_enabled = config_.speculative_enabled;

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
