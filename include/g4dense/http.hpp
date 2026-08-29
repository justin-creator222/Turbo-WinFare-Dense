#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace g4dense {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string query;
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;
    bool keep_alive{false};

    std::string header(std::string_view name) const;
};

enum class HttpReadResult {
    Ok,
    Closed,
    Malformed,
    TooLarge,
    UnsupportedMediaType,
    Timeout,
};

HttpReadResult read_http_request(uintptr_t socket, HttpRequest& out,
                                 size_t max_body_bytes, int timeout_ms);

bool send_http_response(uintptr_t socket, int status, std::string_view content_type,
                        std::string_view body, bool keep_alive,
                        const std::vector<std::pair<std::string, std::string>>& extra = {});

bool send_http_headers(uintptr_t socket, int status, std::string_view content_type,
                       const std::vector<std::pair<std::string, std::string>>& extra = {});

bool send_raw(uintptr_t socket, std::string_view data);

const char* http_reason_phrase(int status);

} // namespace g4dense
