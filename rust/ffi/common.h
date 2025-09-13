#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <cstddef>
inline std::string string_repeat(const std::string & str, size_t n) {
    std::string result;
    result.reserve(str.size() * n);
    for (size_t i = 0; i < n; ++i) {
        result += str;
    }
    return result;
}
inline std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) result << separator;
        result << values[i];
    }
    return result.str();
}
inline std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);
    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    parts.push_back(str.substr(start));
    return parts;
}
