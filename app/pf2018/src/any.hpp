#pragma once

#include <tempo/reader/ts/ts.hpp>
#include <tempo/tseries/dataset.hpp>
#include <tempo/utils/utils/timing.hpp>

#include <filesystem>
#include <variant>
#include <optional>
#include <tuple>
#include <string>
#include <iostream>


struct CMDArgs{
  std::variant<
    std::tuple<std::filesystem::path, std::string>,
    std::tuple<std::filesystem::path, std::filesystem::path>> ucr_traintest_path {};

  std::optional<std::filesystem::path> outpath{};

  size_t nb_candidates{5};
  size_t nb_trees{100};
  size_t random_seed{0};
  size_t nb_thread{1};
};


CMDArgs read_args(int argc, char** argv);

void print_usage(const std::string &execname, std::ostream &out);

/// Print an error followed by usage, then exit
[[noreturn]] void inline print_error_exit(const std::string& invoc_string, const std::string& msg, int exit_code) {
  std::cerr << "Error: " << msg << std::endl;
  print_usage(invoc_string, std::cerr);
  exit(exit_code);
}

std::variant<std::string, std::shared_ptr<tempo::Dataset<double, std::string>>>
  read_data(std::ostream &log, std::filesystem::path& dataset_path);
