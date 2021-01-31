#pragma once

#include <list>
#include <filesystem>
#include <optional>
#include <variant>
#include <string>
#include <iostream>
#include <fstream>
#include <optional>

#include "../src/tempo/reader/readingtools.hpp"

namespace fs = std::filesystem;

enum struct DISTANCE {
    DTW,
    CDTW,
    WDTW,
    ERP,
    LCSS,
    MSM,
    SQED,
    TWE
};

enum struct TRANSFORM {
    DERIVATIVE
};

enum struct DTWLB {
    NONE,
    KEOGH,
    KEOGH2,
    WEBB
};

struct CMDArgs {
    std::variant<std::tuple<fs::path, std::string>, std::tuple<fs::path, fs::path>> ucr_traintest_path {};
    std::list<TRANSFORM> transforms {};
    std::optional<fs::path> outpath{};
    DISTANCE distance {};
    // Per distance argument
    union {
        struct {DTWLB lb;} dtw;
        struct {DTWLB lb; bool wint; double wratio;} cdtw;
        struct {double weight_factor;} wdtw;
        struct {double gv; bool wint; double wratio;} erp;
        struct {double epsilon; bool wint; double wratio;} lcss;
        struct {double cost;} msm;
        struct {double nu; double lambda;} twe;
    } distargs;
};

std::string to_string(DISTANCE dist);
std::string to_string(TRANSFORM tr);
std::string to_string(DTWLB lb);

std::string dist_to_JSON(const CMDArgs& args);

void print_usage(const std::string &execname, std::ostream &out);

/// Print an error followed by usage, then exit
[[noreturn]] void inline print_error_exit(const std::string& invoc_string, const std::string& msg, int exit_code) {
    std::cerr << "Error: " << msg << std::endl;
    print_usage(invoc_string, std::cerr);
    exit(exit_code);
}

CMDArgs read_args(int argc, char*argv);