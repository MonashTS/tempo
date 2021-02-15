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

#include <tempo/reader/ts/ts.hpp>
#include <tempo/tseries/dataset.hpp>
#include <tempo/utils/utils/timing.hpp>

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
    NONE,
    DERIVATIVE
};

enum struct DTWLB_Kind {
    NONE,
    KEOGH,
    KEOGH2,
    ENHANCED,
    WEBB
};

struct DTWLB {
    DTWLB_Kind kind;
    union{
        struct{size_t v;} enhanced;
    } lb_param;
};


struct CMDArgs {
    std::variant<std::tuple<fs::path, std::string>, std::tuple<fs::path, fs::path>> ucr_traintest_path {};
    std::optional<fs::path> outpath{};

    TRANSFORM transforms {TRANSFORM::NONE};
    // Per transform argument
    union {
        struct {} none;
        struct {int rank;} derivative;
    } transargs ;

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
std::string to_string(DTWLB_Kind lb);

std::string dist_to_JSON(const CMDArgs& args);

void print_usage(const std::string &execname, std::ostream &out);

/// Print an error followed by usage, then exit
[[noreturn]] void inline print_error_exit(const std::string& invoc_string, const std::string& msg, int exit_code) {
    std::cerr << "Error: " << msg << std::endl;
    print_usage(invoc_string, std::cerr);
    exit(exit_code);
}

CMDArgs read_args(int argc, char** argv);


std::variant<std::string, tempo::Dataset<double, std::string>> read_data(std::ostream &log, fs::path& dataset_path);