#include "g4dense/openai_api.hpp"
#include "g4dense/json.hpp"
#include <sstream>
#include <iomanip>
#include <chrono>

namespace g4dense {

namespace {

std::string json_quote(const std::string& s) {
    std::ostringstream ss;
    ss << '"';
    for (char c : s) {
        if (c == '"') ss << "\\\"";
        else if (c == '\\') ss << "\\\\";
        else if (c == '\n') ss << "\\n";
        else if (c == '\r') ss << "\\r";
        else if (c == '\t') ss << "\\t";
        else ss << c;
    }
    ss << '"';
    return ss.str();
}

} // namespace

const char* finish_reason_for(StopReason reason) {
    switch (reason) {
        case StopReason::Eos: return "stop";
        case StopReason::Length: return "length";
        case StopReason::Cancelled: return "stop";
        default: return "stop";
    }
}

bool validate_chat_request(const JsonValue& body, const OpenAIServerConfig& cfg,
                           ValidatedChatRequest& out, ApiError& err) {
    if (!body.is_object()) {
        err.http_status = 400;
        err.type = "invalid_request_error";
        err.message = "Request body must be a JSON object";
        return false;
    }

    // Model name
    if (body.has("model")) {
        out.model = body.at("model", "body").as_string("model");
    } else {
        out.model = cfg.default_model;
    }

    // Messages
    if (!body.has("messages") || !body.at("messages", "body").is_array()) {
        err.http_status = 400;
        err.type = "invalid_request_error";
        err.message = "Missing required 'messages' array";
        return false;
    }

    const auto& msgs = body.at("messages", "body").array_value;
    for (const auto& m : msgs) {
        if (!m.is_object() || !m.has("role") || !m.has("content")) {
            err.http_status = 400;
            err.type = "invalid_request_error";
            err.message = "Each message must have 'role' and 'content'";
            return false;
        }
        std::string role = m.at("role", "message").as_string("role");
        std::string content = m.at("content", "message").as_string("content");
        out.messages.push_back({role, content});
    }

    // Optional params
    if (body.has("max_tokens")) {
        out.max_tokens = static_cast<int>(body.at("max_tokens", "body").as_uint32("max_tokens"));
    } else {
        out.max_tokens = cfg.default_max_tokens;
    }

    if (body.has("temperature")) {
        out.sampling.temperature = static_cast<float>(body.at("temperature", "body").as_double("temperature"));
    }
    if (body.has("top_p")) {
        out.sampling.top_p = static_cast<float>(body.at("top_p", "body").as_double("top_p"));
    }
    if (body.has("top_k")) {
        out.sampling.top_k = static_cast<int>(body.at("top_k", "body").as_uint32("top_k"));
    }
    if (body.has("stream")) {
        out.stream = body.at("stream", "body").as_bool("stream");
    }

    return true;
}

std::string render_error(const ApiError& err) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"error\": {\n"
       << "    \"message\": " << json_quote(err.message) << ",\n"
       << "    \"type\": " << json_quote(err.type) << ",\n"
       << "    \"param\": " << (err.param.empty() ? "null" : json_quote(err.param)) << ",\n"
       << "    \"code\": null\n"
       << "  }\n"
       << "}";
    return ss.str();
}

std::string render_completion(const std::string& id, const std::string& model,
                              const GenerationResult& result, int64_t created) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"id\": " << json_quote(id) << ",\n"
       << "  \"object\": \"chat.completion\",\n"
       << "  \"created\": " << created << ",\n"
       << "  \"model\": " << json_quote(model) << ",\n"
       << "  \"choices\": [{\n"
       << "    \"index\": 0,\n"
       << "    \"message\": {\n"
       << "      \"role\": \"assistant\",\n"
       << "      \"content\": " << json_quote(result.text) << "\n"
       << "    },\n"
       << "    \"finish_reason\": " << json_quote(finish_reason_for(result.stop_reason)) << "\n"
       << "  }],\n"
       << "  \"usage\": {\n"
       << "    \"prompt_tokens\": " << result.prompt_tokens << ",\n"
       << "    \"completion_tokens\": " << result.completion_tokens << ",\n"
       << "    \"total_tokens\": " << (result.prompt_tokens + result.completion_tokens) << "\n"
       << "  }\n"
       << "}";
    return ss.str();
}

std::string render_chunk(const std::string& id, const std::string& model, int64_t created,
                         const std::string& delta, bool role_only) {
    std::ostringstream ss;
    ss << "data: {\"id\":" << json_quote(id)
       << ",\"object\":\"chat.completion.chunk\""
       << ",\"created\":" << created
       << ",\"model\":" << json_quote(model)
       << ",\"choices\":[{\"index\":0,\"delta\":";
    if (role_only) {
        ss << "{\"role\":\"assistant\"}";
    } else {
        ss << "{\"content\":" << json_quote(delta) << "}";
    }
    ss << ",\"finish_reason\":null}]}\n\n";
    return ss.str();
}

std::string render_final_chunk(const std::string& id, const std::string& model, int64_t created,
                               StopReason reason) {
    std::ostringstream ss;
    ss << "data: {\"id\":" << json_quote(id)
       << ",\"object\":\"chat.completion.chunk\""
       << ",\"created\":" << created
       << ",\"model\":" << json_quote(model)
       << ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":"
       << json_quote(finish_reason_for(reason)) << "}]}\n\n";
    return ss.str();
}

std::string render_usage_chunk(const std::string& id, const std::string& model, int64_t created,
                               const GenerationResult& result) {
    std::ostringstream ss;
    ss << "data: {\"id\":" << json_quote(id)
       << ",\"object\":\"chat.completion.chunk\""
       << ",\"created\":" << created
       << ",\"model\":" << json_quote(model)
       << ",\"usage\":{\"prompt_tokens\":" << result.prompt_tokens
       << ",\"completion_tokens\":" << result.completion_tokens
       << ",\"total_tokens\":" << (result.prompt_tokens + result.completion_tokens)
       << "}}\n\n";
    return ss.str();
}

std::string render_models_list(const std::string& model_id) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"object\": \"list\",\n"
       << "  \"data\": [{\n"
       << "    \"id\": " << json_quote(model_id) << ",\n"
       << "    \"object\": \"model\",\n"
       << "    \"created\": 1700000000,\n"
       << "    \"owned_by\": \"turbo-winfare\"\n"
       << "  }]\n"
       << "}";
    return ss.str();
}

std::optional<RequestCoordinator::Lease> RequestCoordinator::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (static_cast<int>(queue_.size()) >= queue_limit_) {
        return std::nullopt;
    }
    uint64_t my_ticket = ++next_ticket_;
    queue_.push_back(my_ticket);

    cv_.wait(lock, [&]() {
        return !busy_ && queue_.front() == my_ticket;
    });

    busy_ = true;
    queue_.pop_front();
    return Lease(this);
}

void RequestCoordinator::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    busy_ = false;
    cv_.notify_all();
}

} // namespace g4dense
