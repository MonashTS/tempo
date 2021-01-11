#pragma once

#include <algorithm>
#include <cerrno>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace tempo::reader {

    // --- --- ---
    // --- --- --- istream
    // --- --- ---

    /** Test if a char is a whitespace (excluding a newline). */
    inline bool is_white(int c) {
        static auto w = std::string(" \f\r\t\v");
        return w.find(c) != std::string::npos;
    }

    /** Skip while whitespaces (excluding newline) are read. */
    inline void skip_white(std::istream &input) {
        int c{EOF};
        while ((c = input.peek()) != EOF && is_white(c)) { input.ignore(1); }
    }

    /** Skip the current line (read until newline) */
    inline void skip_line(std::istream &input) {
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    /** Read a word in the buffer (read until a whitespace, including newline, is found).
     *  Return the char that ended the read, keeping it in the stream. */
    inline int read_word(std::istream &input, std::string &buffer) {
        int c = EOF;
        while ((c = input.peek()) != EOF && !std::isspace(c)) { buffer.push_back(input.get()); }
        return c;
    }

    /** Read a word in the buffer (read until a whitespace, including newline, is found), converting to lower case.
     *  Return the char that ended the read, keeping it in the stream. */
    inline int read_word_to_lower(std::istream &input, std::string &buffer) {
        int c{EOF};
        while ((c = input.peek()) != EOF && !std::isspace(c)) { buffer.push_back(std::tolower(input.get())); }
        return c;
    }

    /** Test if a char is a delimiter ',' or ':' or '\n' */
    inline bool is_delim(int c) {
        return c == ',' || c == ':' || c == '\n';
    }

    /** Read into buffer until a delimiter is found; returns that delimiter (taken out opf the stream). */
    inline int read_until_delim(std::istream &input, std::string &buffer) {
        int c{EOF};
        while ((c = input.peek()) != EOF && !is_delim(c)) { buffer.push_back(input.get()); }
        if (c != EOF) { input.ignore(1); }
        return c;
    }

    // --- --- ---
    // --- --- --- string/istringstream
    // --- --- ---

    /** String splitter on a delimiter. Accept a istringstream */
    inline std::vector<std::string> split(std::istringstream &&input, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        while (std::getline(input, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }

    /** trim from start (in place) */
    inline void ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
            return !std::isspace(ch);
        }));
    }

    /** trim from end (in place) */
    inline void rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
            return !std::isspace(ch);
        }).base(), s.end());
    }

    /** trim from both ends (in place) */
    inline void trim(std::string &s) {
        ltrim(s);
        rtrim(s);
    }

    /** Attempt to convert a string into a bool */
    inline std::optional<bool> as_bool(const std::string &str) {
        if (str == "true") {
            return {true};
        } else if (str == "false") {
            return {false};
        } else {
            return {};
        }
    }

    /** Attempt to convert a string into an integer */
    inline std::optional<int> as_int(const std::string &str) {
        try {
            int i = std::stoi(str);
            return {i};
        } catch (...) {
            return {};
        }
    }

    /** Attempt to convert a string into an size_t */
    inline std::optional<size_t> as_size_t(const std::string &str) {
        try {
            size_t i = std::stoul(str);
            return {i};
        } catch (...) {
            return {};
        }
    }

    /** Attempt to convert a string into an double */
    inline std::optional<double> as_double(const std::string &str) {
        errno = 0;
        double d = strtod(str.c_str(), 0);
        if (errno != 0) { return {}; }
        else { return {d}; }
    }

} // End of namespace tempo::reader

