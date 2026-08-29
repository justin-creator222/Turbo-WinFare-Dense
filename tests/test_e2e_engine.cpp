#include <winsock2.h>
#include <ws2tcpip.h>
#include <filesystem>
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>

#include "g4dense/runner.hpp"
#include "g4dense/server.hpp"
#include "g4dense/tokenizer.hpp"
#include "g4dense/vk_context.hpp"
#include "g4dense/telemetry.hpp"

using namespace g4dense;

std::string http_get(uint16_t port, const std::string& path) {
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(s, (sockaddr*)&addr, sizeof(addr)) != 0) {
        closesocket(s);
        return "";
    }

    std::string req = "GET " + path + " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    send(s, req.c_str(), static_cast<int>(req.size()), 0);

    std::string resp;
    char buf[4096];
    int n;
    while ((n = recv(s, buf, sizeof(buf) - 1, 0)) > 0) {
        buf[n] = '\0';
        resp += buf;
    }
    closesocket(s);
    return resp;
}

std::string http_post(uint16_t port, const std::string& path, const std::string& json_body) {
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(s, (sockaddr*)&addr, sizeof(addr)) != 0) {
        closesocket(s);
        return "";
    }

    std::string req = "POST " + path + " HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " +
                      std::to_string(json_body.size()) + "\r\nConnection: close\r\n\r\n" + json_body;
    send(s, req.c_str(), static_cast<int>(req.size()), 0);

    std::string resp;
    char buf[4096];
    int n;
    while ((n = recv(s, buf, sizeof(buf) - 1, 0)) > 0) {
        buf[n] = '\0';
        resp += buf;
    }
    closesocket(s);
    return resp;
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Turbo-WinFare Dense End-to-End Engine Tests   " << std::endl;
    std::cout << "================================================" << std::endl;

    // 1. Initialize VulkanContext & Tokenizer
    std::cout << "1. Initializing Vulkan 1.3 Context..." << std::endl;
    auto ctx = std::make_shared<VulkanContext>();
    ctx->initialize();
    std::cout << "   GPU Device: " << ctx->device_name() << std::endl;
    std::cout << "   Subgroup Size: " << ctx->subgroup_size() << std::endl;

    auto tok = std::make_shared<Tokenizer>();
    try {
        tok->load_vocabulary("tests/fixtures/tokenizer.json");
    } catch (...) {
        try {
            tok->load_vocabulary("../tests/fixtures/tokenizer.json");
        } catch (...) {}
    }

    // 2. Initialize ForwardRunner
    std::cout << "2. Initializing ForwardRunner on tests/fixtures/tiny.g4dense..." << std::endl;
    std::string model_path = "tests/fixtures/tiny.g4dense";
    if (!std::filesystem::exists(model_path)) {
        model_path = "../tests/fixtures/tiny.g4dense";
    }

    auto runner = std::make_shared<ForwardRunner>(ctx, tok, model_path);
    runner->initialize();
    std::cout << "   Model loaded successfully." << std::endl;

    // 3. Multi-Tier Dynamic Live Switching (Gate 3)
    std::cout << "3. Testing Live Tier Dynamic Switching (Gate 3)..." << std::endl;
    runner->switch_memory_tier(1);
    assert(runner->active_tier_id() == 1);
    std::cout << "   Tier 1 (6.0 GB Baseline) active." << std::endl;

    runner->switch_memory_tier(2);
    assert(runner->active_tier_id() == 2);
    std::cout << "   Tier 2 (10.0 GB Balanced) active." << std::endl;

    runner->switch_memory_tier(3);
    assert(runner->active_tier_id() == 3);
    std::cout << "   Tier 3 (16.0 GB High-Perf) active." << std::endl;
    std::cout << "   [PASS] Gate 3 Multi-Tier dynamic switching verified." << std::endl;

    // 4. Speculative Generation Forward Pass
    std::cout << "4. Testing Speculative Generation Loop..." << std::endl;
    GenerationOptions opts;
    opts.max_tokens = 8;
    opts.speculative_enabled = true;
    opts.draft_k = 4;
    opts.sampling.temperature = 0.0f; // Greedy

    std::string generated_text;
    int token_count = 0;

    runner->generate("Hello", opts, [&](uint32_t token, const std::string& piece) {
        generated_text += piece;
        token_count++;
        std::cout << "   Token " << token_count << " (id=" << token << "): \"" << piece << "\"" << std::endl;
        return true;
    });

    assert(token_count > 0);
    std::cout << "   Generated " << token_count << " tokens." << std::endl;

    TelemetrySnapshot tele = runner->get_latest_telemetry();
    std::cout << "   Telemetry: RAM footprint=" << tele.ram_footprint_mb << " MB, total RAM="
              << tele.ram_total_mb << " MB, TPS=" << tele.tps << std::endl;
    assert(tele.ram_footprint_mb > 0.0);

    // 5. Embedded HTTP Server & API Verification (Gate 4)
    std::cout << "5. Testing HTTP Server, Web GUI, and OpenAI API (Gate 4)..." << std::endl;
    uint16_t port = 18080;
    HTTPServer server;
    server.start(port, runner, ctx);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Test GET /
    std::string index_resp = http_get(port, "/");
    assert(index_resp.find("200 OK") != std::string::npos);
    assert(index_resp.find("TURBO-WINFARE") != std::string::npos);
    std::cout << "   [PASS] GET / serves Web GUI HTML." << std::endl;

    // Test GET /api/telemetry
    std::string tele_resp = http_get(port, "/api/telemetry");
    assert(tele_resp.find("200 OK") != std::string::npos);
    assert(tele_resp.find("\"ram_footprint_mb\"") != std::string::npos);
    std::cout << "   [PASS] GET /api/telemetry returns telemetry JSON." << std::endl;

    // Test POST /api/switch_tier
    std::string tier_resp = http_post(port, "/api/switch_tier", "{\"tier_id\": 2}");
    assert(tier_resp.find("200 OK") != std::string::npos);
    assert(runner->active_tier_id() == 2);
    std::cout << "   [PASS] POST /api/switch_tier switched to Tier 2." << std::endl;

    // Test POST /v1/chat/completions
    std::string chat_body = "{\"model\": \"gemma-4-31b-dense\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 4}";
    std::string chat_resp = http_post(port, "/v1/chat/completions", chat_body);
    assert(chat_resp.find("200 OK") != std::string::npos);
    assert(chat_resp.find("chat.completion") != std::string::npos);
    std::cout << "   [PASS] POST /v1/chat/completions returned completion." << std::endl;

    server.stop();

    std::cout << "\n>>> ALL END-TO-END ENGINE & GATE VERIFICATIONS PASSED! <<<" << std::endl;
    return 0;
}
