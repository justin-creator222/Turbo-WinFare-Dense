#define G4DENSE_BUILD_DLL
#include "g4dense/c_api.h"
#include "g4dense/runner.hpp"
#include "g4dense/manifest.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/json.hpp"

#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <filesystem>
#include <cstring>

struct G4DenseEngineContext {
    std::shared_ptr<g4dense::VulkanContext> ctx;
    std::shared_ptr<g4dense::Tokenizer> tokenizer;
    std::shared_ptr<g4dense::ForwardRunner> runner;
    std::string container_path;
    std::string last_response;
    std::string last_telemetry;
    int last_stop_reason{0};
};

namespace {
thread_local std::string g_last_error;

void set_error(const std::string& msg) { g_last_error = msg; }
void clear_error() { g_last_error.clear(); }
} // namespace

extern "C" {

G4DENSE_API const char* g4dense_engine_last_error(void) {
    return g_last_error.c_str();
}

G4DENSE_API void g4dense_options_default(G4DenseGenerationOptions* out) {
    if (!out) return;
    out->max_tokens = 512;
    out->temperature = 0.2f;
    out->top_p = 0.95f;
    out->top_k = 64;
    out->repetition_penalty = 1.0f;
    out->has_seed = 0;
    out->seed = 0;
    out->system_prompt = nullptr;
    out->stop_strings = nullptr;
    out->stop_count = 0;
    out->tier_id = 1;
}

G4DENSE_API void* g4dense_engine_create(const char* container_path) {
    clear_error();
    try {
        auto e_ctx = std::make_unique<G4DenseEngineContext>();
        e_ctx->ctx = std::make_shared<g4dense::VulkanContext>();
        e_ctx->ctx->initialize();

        std::string c_path = (container_path && strlen(container_path) > 0) ? container_path : "gemma-4-31b-dense.g4dense";
        c_path = g4dense::resolve_bundle_path(c_path);
        e_ctx->container_path = c_path;

        e_ctx->tokenizer = std::make_shared<g4dense::Tokenizer>();
        e_ctx->tokenizer->load_vocabulary();

        e_ctx->runner = std::make_shared<g4dense::ForwardRunner>(e_ctx->ctx, e_ctx->tokenizer, c_path);
        e_ctx->runner->initialize();

        return e_ctx.release();
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

G4DENSE_API void g4dense_engine_destroy(void* handle) {
    if (!handle) return;
    delete static_cast<G4DenseEngineContext*>(handle);
}

G4DENSE_API const char* g4dense_engine_generate(void* handle, const char* prompt, int max_tokens) {
    clear_error();
    if (!handle || !prompt) {
        set_error("Invalid handle or null prompt");
        return nullptr;
    }

    auto* e_ctx = static_cast<G4DenseEngineContext*>(handle);
    try {
        g4dense::GenerationOptions opts;
        opts.max_tokens = max_tokens > 0 ? max_tokens : 512;
        opts.sampling.temperature = 0.2f;
        opts.sampling.top_p = 0.95f;
        opts.sampling.top_k = 64;

        e_ctx->last_response.clear();
        e_ctx->runner->generate(
            prompt, opts,
            [&](uint32_t, const std::string& piece) {
                e_ctx->last_response += piece;
                return true;
            }
        );
        return e_ctx->last_response.c_str();
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

G4DENSE_API const char* g4dense_engine_get_telemetry(void* handle) {
    if (!handle) return "{}";
    auto* e_ctx = static_cast<G4DenseEngineContext*>(handle);
    e_ctx->last_telemetry = e_ctx->runner->get_latest_telemetry().to_json_string();
    return e_ctx->last_telemetry.c_str();
}

G4DENSE_API const char* g4dense_engine_generate_ex(void* handle, const char* messages_json,
                                                   const G4DenseGenerationOptions* opts,
                                                   G4DenseTokenCallback on_token, void* user) {
    clear_error();
    if (!handle || !messages_json) {
        set_error("Invalid handle or null messages");
        return nullptr;
    }

    auto* e_ctx = static_cast<G4DenseEngineContext*>(handle);
    try {
        g4dense::GenerationOptions gen_opts;
        if (opts) {
            gen_opts.max_tokens = opts->max_tokens;
            gen_opts.sampling.temperature = opts->temperature;
            gen_opts.sampling.top_p = opts->top_p;
            gen_opts.sampling.top_k = opts->top_k;
            gen_opts.sampling.repetition_penalty = opts->repetition_penalty;
            gen_opts.sampling.has_seed = (opts->has_seed != 0);
            gen_opts.sampling.seed = opts->seed;
            gen_opts.active_tier_id = opts->tier_id;
        }

        e_ctx->last_response.clear();
        int token_idx = 0;
        e_ctx->runner->generate(
            messages_json, gen_opts,
            [&](uint32_t token, const std::string& piece) {
                e_ctx->last_response += piece;
                if (on_token) {
                    int cont = on_token(user, token_idx++, token, piece.c_str());
                    return cont != 0;
                }
                return true;
            }
        );
        return e_ctx->last_response.c_str();
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

G4DENSE_API int g4dense_engine_last_stop_reason(void* handle) {
    if (!handle) return 0;
    auto* e_ctx = static_cast<G4DenseEngineContext*>(handle);
    return e_ctx->last_stop_reason;
}

G4DENSE_API int g4dense_engine_switch_tier(void* handle, int tier_id) {
    clear_error();
    if (!handle) {
        set_error("Null handle");
        return -1;
    }
    auto* e_ctx = static_cast<G4DenseEngineContext*>(handle);
    try {
        e_ctx->runner->switch_memory_tier(tier_id);
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    }
}

G4DENSE_API void g4dense_engine_stop(void* handle) {
    // Runner cancellation
}

} // extern "C"
