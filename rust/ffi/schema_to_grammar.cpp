#include "schema_to_grammar.h"
#include "../../llama/llama.cpp/common/json-schema-to-grammar.h"
#include <nlohmann/json.hpp>
#include <cstring>
#include <string>

extern "C" int schema_to_grammar(const char *json_schema, char *grammar, size_t max_len) {
    try {
        nlohmann::ordered_json schema = nlohmann::ordered_json::parse(json_schema);
        std::string grammar_str = json_schema_to_grammar(schema);
        size_t len = grammar_str.length();
        if (len >= max_len) {
            len = max_len - 1;
        }
        std::memcpy(grammar, grammar_str.c_str(), len);
        grammar[len] = '\0';
        return static_cast<int>(len);
    } catch (...) {
        if (max_len > 0) {
            grammar[0] = '\0';
        }
        return 0;
    }
}
