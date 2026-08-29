#pragma once

#include "g4dense/tokenizer.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

namespace g4dense {

class IncrementalDetokenizer {
public:
    explicit IncrementalDetokenizer(const Tokenizer& tok, bool skip_special = true)
        : tok_(tok), skip_special_(skip_special) {}

    // Returns text that is now definitely complete.
    std::string push(uint32_t token);

    // Flushes anything still buffered.
    std::string finish();

    void reset() { pending_.clear(); }

private:
    std::string drain(bool flush_all);

    const Tokenizer& tok_;
    bool skip_special_{true};
    std::string pending_;
};

class StreamingStopMatcher {
public:
    StreamingStopMatcher() = default;
    explicit StreamingStopMatcher(std::vector<std::string> stops);

    std::string push(std::string_view text);
    std::string finish();

    bool stopped() const { return stopped_; }
    const std::string& matched() const { return matched_; }
    bool empty() const { return stops_.empty(); }

private:
    std::vector<std::string> stops_;
    size_t max_stop_len_{0};
    std::string buffer_;
    std::string matched_;
    bool stopped_{false};
};

} // namespace g4dense
