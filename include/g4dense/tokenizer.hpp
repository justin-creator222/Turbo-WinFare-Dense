#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace g4dense {

// Appends the valid UTF-8 prefix of `in` to `out`, substituting U+FFFD for bytes that cannot
// form a character, and returns how many bytes of `in` were consumed.
//
// With `flush_all == false` an incomplete trailing sequence is left unconsumed, so a caller
// streaming byte-fallback tokens can hold it until the rest arrives. With `flush_all == true`
// the tail is terminal and becomes a replacement character.
//
// Both the batch decoder (Tokenizer::decode) and the streaming one (IncrementalDetokenizer)
// go through this, so streamed output is byte-identical to non-streamed output for the same
// token sequence. They diverged before: batch emitted raw bytes while streaming repaired
// them, which meant an SSE response and a blocking response disagreed -- and raw invalid
// UTF-8 cannot be encoded into a JSON body at all.
size_t utf8_repair(std::string_view in, bool flush_all, std::string& out);

enum class PieceType : uint8_t {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Byte = 5,
    Unused = 6
};

struct TokenPiece {
    std::string piece;
    float score{0.0f};
    PieceType type{PieceType::Normal};
    uint32_t id{0};
};

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    bool load_vocabulary(const std::string& vocab_file = "");

    std::vector<uint32_t> encode(const std::string& text, bool add_bos = true) const;
    std::string decode(const std::vector<uint32_t>& tokens, bool skip_special = true) const;
    std::string decode_single(uint32_t token, bool skip_special = true) const;

    bool is_byte_fallback(uint32_t token, uint8_t& value) const;
    bool is_special_token(uint32_t token) const {
        return token < is_special_.size() && is_special_[token];
    }

    struct ChatMessage {
        std::string role;     // "user", "model", "system"
        std::string content;
    };

    std::string apply_chat_template(const std::vector<ChatMessage>& messages) const;

    uint32_t bos_id() const { return bos_id_; }
    uint32_t eos_id() const { return eos_id_; }
    uint32_t pad_id() const { return pad_id_; }
    uint32_t unk_id() const { return unk_id_; }
    uint32_t end_of_turn_id() const { return end_of_turn_id_; }
    uint32_t vocab_size() const { return vocab_size_; }
    bool is_loaded() const { return loaded_; }

    const std::vector<uint32_t>& stop_token_ids() const { return stop_token_ids_; }

private:
    uint32_t bos_id_{2};
    uint32_t eos_id_{1};
    uint32_t pad_id_{0};
    uint32_t unk_id_{3};
    uint32_t turn_start_id_{105};    // <|turn>
    uint32_t end_of_turn_id_{106};   // <turn|>
    uint32_t tool_response_id_{50};  // <|tool_response>
    uint32_t vocab_size_{262144};
    bool loaded_{false};

    std::vector<uint32_t> stop_token_ids_{1, 106, 50};

    std::vector<std::string> pieces_;
    std::unordered_map<std::string, uint32_t> piece_map_;
    std::unordered_map<uint8_t, uint32_t> byte_to_id_;
    std::unordered_map<std::string, uint32_t> merge_ranks_;
    std::vector<std::pair<std::string, uint32_t>> added_tokens_;
    std::vector<bool> is_special_;

    bool parse_tokenizer_json(const std::string& filepath);
    void bpe_segment(const std::string& text, std::vector<uint32_t>& out) const;
    void emit_byte_fallback(const std::string& piece, std::vector<uint32_t>& out) const;
};

} // namespace g4dense
