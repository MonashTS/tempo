#pragma once

#include "utils.hpp"
#include "readingtools.hpp"

namespace tempo::scli {

  /// Helper type: reading function. Try to read a string
  template<typename T>
  using fread_t = std::function<std::optional<T>(std::string const&)>;

  /// Helper type: prefix extracting function.
  template<typename T>
  using fpfxext_t = std::function<std::optional<T>(std::string const&, std::string const&)>;

  /// Get a parameter with a fread function. Return the default value 'defval' if not found.
  template<typename T>
  T get_parameter(std::vector<std::string> const& argv, fread_t<T> read, T defval) {
    for (auto const& a : argv) {
      auto oa = read(a);
      if (oa) { return oa.value(); }
    }
    return defval;
  }

  /// Get a parameter with a fread function. Return an empty option if not found.
  template<typename T>
  std::optional<T> get_parameter(std::vector<std::string> const& argv, fread_t<T> read) {
    for (auto const& a : argv) {
      auto oa = read(a);
      if (oa) { return oa; }
    }
    return {};
  }

  /// Read several parameters with a fread function. Return an empty vector if none found.
  template<typename T>
  std::vector<T> get_parameters(std::vector<std::string> const& argv, fread_t<T> read) {
    std::vector<T> result;
    for (auto const& a : argv) {
      auto oa = read(a);
      if (oa) { result.push_back(oa.value()); }
    }
    return result;
  }

  // --- --- ---
  // --- --- --- fread helpers A: extract data based on a prefix
  // --- --- ---

  /// Extract a string
  inline std::optional<std::string> extract_string(std::string const& arg, std::string const& pfx) {
    if (arg.starts_with(pfx)) {
      return {arg.substr(pfx.size(), arg.size())};
    } else {
      return {};
    }
  }

  /// Extract an integer
  inline std::optional<long long> extract_int(std::string const& arg, std::string const& pfx) {
    if (arg.starts_with(pfx)) {
      std::string str = arg.substr(pfx.size(), arg.size());
      return tempo::reader::as_long(str);
    } else { return {}; }
  }

  /// Extract a "real"
  inline std::optional<double> extract_double(std::string const& arg, std::string const& pfx) {
    if (arg.starts_with(pfx)) {
      std::string str = arg.substr(pfx.size(), arg.size());
      return tempo::reader::as_double(str);
    } else { return {}; }
  }

  // --- --- ---
  // --- --- --- fread helpers B: look for flag, extract data separated with ':'  (e.g. -n:5)
  // --- --- ---

  /// Check for a switch
  inline bool get_switch(std::vector<std::string> const& args, std::string pfx) {
    return get_parameter<bool>(
      args,
      [=](std::string const& a) -> std::optional<bool> { if (a==pfx) { return {true}; } else { return {}; } },
      false
    );
  }

  /// Check for an optional parameter
  template<typename T>
  inline std::optional<T> get_parameter(std::vector<std::string> const& args, std::string pfx, fpfxext_t<T> f) {
    pfx = pfx + ':';
    return get_parameter<T>(args, [=](std::string const& a) { return f(a, pfx); });
  }

  /// Check for a parameter with a default value
  template<typename T>
  inline T get_parameter(std::vector<std::string> const& args, std::string pfx, fpfxext_t<T> f, T def) {
    pfx = pfx + ':';
    return get_parameter<T>(args, [=](std::string const& a) { return f(a, pfx); }, def);
  }

} // End of namespace SCLI