#pragma once

// Strict, dependency-free JSON parser and serializer for G4Dense.

#include "g4dense/format.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <sstream>

namespace g4dense {

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type{Type::Null};
    bool bool_value{false};
    double number_value{0.0};
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::map<std::string, JsonValue> object_value;

    bool is_null()   const { return type == Type::Null; }
    bool is_bool()   const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_object() const { return type == Type::Object; }
    bool is_array()  const { return type == Type::Array; }

    bool has(const std::string& key) const {
        return type == Type::Object && object_value.count(key) > 0;
    }

    const JsonValue& at(const std::string& key, const std::string& context) const {
        if (type != Type::Object) {
            throw G4DenseFormatError(context + "." + key + ": parent is not an object");
        }
        auto it = object_value.find(key);
        if (it == object_value.end()) {
            throw G4DenseFormatError(context + ": missing required field '" + key + "'");
        }
        return it->second;
    }

    std::string as_string(const std::string& context) const {
        if (type != Type::String) {
            throw G4DenseFormatError(context + ": expected string");
        }
        return string_value;
    }

    uint64_t as_uint64(const std::string& context) const {
        if (type != Type::Number || number_value < 0.0) {
            throw G4DenseFormatError(context + ": expected non-negative number");
        }
        return static_cast<uint64_t>(number_value);
    }

    uint32_t as_uint32(const std::string& context) const {
        uint64_t v = as_uint64(context);
        if (v > UINT32_MAX) {
            throw G4DenseFormatError(context + ": value exceeds uint32 range");
        }
        return static_cast<uint32_t>(v);
    }

    double as_double(const std::string& context) const {
        if (type != Type::Number) {
            throw G4DenseFormatError(context + ": expected number");
        }
        return number_value;
    }

    bool as_bool(const std::string& context) const {
        if (type != Type::Bool) {
            throw G4DenseFormatError(context + ": expected bool");
        }
        return bool_value;
    }

    static JsonValue parse(std::string_view text, const std::string& context = "json");
    std::string serialize() const;
};

// Parser helper implementation
inline JsonValue JsonValue::parse(std::string_view text, const std::string& context) {
    size_t idx = 0;

    auto skip_whitespace = [&]() {
        while (idx < text.size() && (text[idx] == ' ' || text[idx] == '\t' || text[idx] == '\r' || text[idx] == '\n')) {
            idx++;
        }
    };

    auto parse_string = [&]() -> std::string {
        idx++; // skip opening quote
        std::string res;
        while (idx < text.size()) {
            char c = text[idx++];
            if (c == '"') return res;
            if (c == '\\') {
                if (idx >= text.size()) throw G4DenseFormatError(context + ": truncated escape sequence");
                char esc = text[idx++];
                if (esc == '"') res.push_back('"');
                else if (esc == '\\') res.push_back('\\');
                else if (esc == '/') res.push_back('/');
                else if (esc == 'n') res.push_back('\n');
                else if (esc == 'r') res.push_back('\r');
                else if (esc == 't') res.push_back('\t');
                else if (esc == 'u') {
                    if (idx + 4 > text.size()) throw G4DenseFormatError(context + ": truncated unicode escape");
                    idx += 4; // skip 4 hex digits (basic support)
                    res.push_back('?');
                } else res.push_back(esc);
            } else {
                res.push_back(c);
            }
        }
        throw G4DenseFormatError(context + ": unterminated string literal");
    };

    auto parse_number = [&]() -> double {
        size_t start = idx;
        if (idx < text.size() && text[idx] == '-') idx++;
        while (idx < text.size() && (std::isdigit((unsigned char)text[idx]) || text[idx] == '.' || text[idx] == 'e' || text[idx] == 'E' || text[idx] == '+' || text[idx] == '-')) {
            idx++;
        }
        std::string s(text.substr(start, idx - start));
        try {
            return std::stod(s);
        } catch (...) {
            throw G4DenseFormatError(context + ": invalid number format: " + s);
        }
    };

    auto parse_val = [&](auto& self) -> JsonValue {
        skip_whitespace();
        if (idx >= text.size()) throw G4DenseFormatError(context + ": unexpected end of input");

        char c = text[idx];
        if (c == '"') {
            JsonValue val;
            val.type = Type::String;
            val.string_value = parse_string();
            return val;
        } else if (c == '{') {
            idx++;
            JsonValue val;
            val.type = Type::Object;
            skip_whitespace();
            if (idx < text.size() && text[idx] == '}') {
                idx++;
                return val;
            }
            while (idx < text.size()) {
                skip_whitespace();
                if (idx >= text.size() || text[idx] != '"') throw G4DenseFormatError(context + ": expected object string key");
                std::string key = parse_string();
                skip_whitespace();
                if (idx >= text.size() || text[idx] != ':') throw G4DenseFormatError(context + ": expected ':' after key '" + key + "'");
                idx++;
                val.object_value[key] = self(self);
                skip_whitespace();
                if (idx < text.size() && text[idx] == '}') {
                    idx++;
                    return val;
                }
                if (idx < text.size() && text[idx] == ',') {
                    idx++;
                } else {
                    throw G4DenseFormatError(context + ": expected ',' or '}' in object");
                }
            }
            throw G4DenseFormatError(context + ": unterminated object");
        } else if (c == '[') {
            idx++;
            JsonValue val;
            val.type = Type::Array;
            skip_whitespace();
            if (idx < text.size() && text[idx] == ']') {
                idx++;
                return val;
            }
            while (idx < text.size()) {
                val.array_value.push_back(self(self));
                skip_whitespace();
                if (idx < text.size() && text[idx] == ']') {
                    idx++;
                    return val;
                }
                if (idx < text.size() && text[idx] == ',') {
                    idx++;
                } else {
                    throw G4DenseFormatError(context + ": expected ',' or ']' in array");
                }
            }
            throw G4DenseFormatError(context + ": unterminated array");
        } else if (text.substr(idx, 4) == "true") {
            idx += 4;
            JsonValue val;
            val.type = Type::Bool;
            val.bool_value = true;
            return val;
        } else if (text.substr(idx, 5) == "false") {
            idx += 5;
            JsonValue val;
            val.type = Type::Bool;
            val.bool_value = false;
            return val;
        } else if (text.substr(idx, 4) == "null") {
            idx += 4;
            JsonValue val;
            val.type = Type::Null;
            return val;
        } else if (c == '-' || std::isdigit((unsigned char)c)) {
            JsonValue val;
            val.type = Type::Number;
            val.number_value = parse_number();
            return val;
        }
        throw G4DenseFormatError(context + ": unexpected character: " + std::string(1, c));
    };

    JsonValue root = parse_val(parse_val);
    skip_whitespace();
    return root;
}

inline std::string JsonValue::serialize() const {
    std::stringstream ss;
    switch (type) {
        case Type::Null: ss << "null"; break;
        case Type::Bool: ss << (bool_value ? "true" : "false"); break;
        case Type::Number: ss << number_value; break;
        case Type::String: {
            ss << '"';
            for (char c : string_value) {
                if (c == '"') ss << "\\\"";
                else if (c == '\\') ss << "\\\\";
                else if (c == '\n') ss << "\\n";
                else if (c == '\r') ss << "\\r";
                else if (c == '\t') ss << "\\t";
                else ss << c;
            }
            ss << '"';
            break;
        }
        case Type::Array: {
            ss << '[';
            for (size_t i = 0; i < array_value.size(); ++i) {
                ss << array_value[i].serialize();
                if (i + 1 < array_value.size()) ss << ", ";
            }
            ss << ']';
            break;
        }
        case Type::Object: {
            ss << '{';
            size_t count = 0;
            for (const auto& [k, v] : object_value) {
                ss << '"' << k << "\": " << v.serialize();
                if (++count < object_value.size()) ss << ", ";
            }
            ss << '}';
            break;
        }
    }
    return ss.str();
}

} // namespace g4dense
