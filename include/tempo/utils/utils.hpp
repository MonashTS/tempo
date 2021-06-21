#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>

#include <map>
#include <set>
#include <vector>
#include <unordered_map>

#include "utils/rand.hpp"
#include "utils/stats.hpp"
#include "utils/unsignedutils.hpp"
#include "utils/timing.hpp"
#include "jsonvalue.hpp"

namespace tempo {

  /// Contains static assertion tools
  namespace stassert {

    /// Helpers extracting the value type of an iterator type and removing the cv qualifiers
    template<typename Iterator>
    using value_type = typename std::remove_cv<typename std::iterator_traits<Iterator>::value_type>::type;

    /// Helper checking the type parameter of an iteratori
    template<typename TargetType, typename Iterator>
    constexpr bool is_iterator_value_type = std::is_same_v<TargetType, value_type<Iterator>>;
  }

  // --- --- --- --- --- ---
  // --- Series constants
  // --- --- --- --- --- ---

  /// Constant to be use when no window is required
  constexpr size_t NO_WINDOW{std::numeric_limits<size_t>::max()};

  /// Constant representing the maximum length allowed for a series.
  /// Account for extra columns/lines that may need to be allocated and representing the window.
  constexpr size_t MAX_SERIES_LENGTH{NO_WINDOW-2};

  // --- --- --- --- --- ---
  // --- Floating point constants
  // --- --- --- --- --- ---

  /// Positive infinity for float types
  template<typename FloatType>
  constexpr FloatType POSITIVE_INFINITY{std::numeric_limits<FloatType>::infinity()};

  /// Negative infinity for float types
  template<typename FloatType>
  constexpr FloatType NEGATIVE_INFINITY{-POSITIVE_INFINITY<FloatType>};

  /// Not A Number
  template<typename FloatType>
  constexpr FloatType DNAN{std::numeric_limits<FloatType>::quiet_NaN()};

  /// Lower Bound inital value, use to deal with numerical instability
  template<typename FloatType>
  FloatType INITLB{-pow(FloatType(10), -(std::numeric_limits<FloatType>::digits10-1))};

  // --- --- --- --- --- ---
  // --- Tooling
  // --- --- --- --- --- ---

  /// Minimum of 3 values using std::min<T>
  template<typename T>
  [[nodiscard]] inline T min(T a, T b, T c) { return std::min<T>(a, std::min<T>(b, c)); }

  /// Maximum of 3 values using std::min<T>
  template<typename T>
  [[nodiscard]] inline T max(T a, T b, T c) { return std::max<T>(a, std::max<T>(b, c)); }

  // --- --- --- --- --- ---
  // --- Should not happen
  // --- --- --- --- --- ---

  /// Throw an exception "should not happen". Used as default case in switches.
  [[noreturn]] void inline should_not_happen() { throw std::logic_error("Should not happen"); }


  // --- --- --- --- --- ---
  // --- Basic set and map "contains" function (need C++20 for those "commodities"...)
  // --- --- --- --- --- ---

  /// std::set "contains" function
  template<typename Key, class Compare = std::less<Key>, class Allocator = std::allocator<Key>>
  [[nodiscard]] inline bool contains(const std::set<Key, Compare, Allocator>& s, const Key& key) {
    return s.find(key)!=s.end();
  }

  /// std::map "contains" function
  template<typename Key, typename T, typename Compare = std::less<Key>, typename Allocator = std::allocator<std::pair<const Key, T> > >
  [[nodiscard]] inline bool contains(const std::map<Key, T, Compare, Allocator>& m, const Key& key) {
    return m.find(key)!=m.end();
  }

  /// std::map "contains" function
  template<class Key, class T, class Hash=std::hash<Key>, class KeyEqual = std::equal_to<Key>, class Allocator = std::allocator<std::pair<const Key, T> > >
  [[nodiscard]] inline bool contains(const std::unordered_map<Key, T, Hash, KeyEqual, Allocator>& m, const Key& key) {
    return m.find(key)!=m.end();
  }


  // --- --- --- --- --- ---
  // --- String helper
  // --- --- --- --- --- ---

  /// Mimic python split
  [[nodiscard]] inline std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> toks;
    std::string tok;
    std::istringstream stream(s);
    while (std::getline(stream, tok, delimiter)) { toks.emplace_back(tok); }
    return toks;
  }



  // --- --- --- --- --- ---
  // --- Initialisation tool
  // --- --- --- --- --- ---

  namespace initBlock_detail {
    struct tag { };

    template<class F>
    decltype(auto) operator+(tag, F&& f) {
      return std::forward<F>(f)();
    }
  }

#define initBlock initBlock_detail::tag{} + [&]() -> decltype(auto)


  // --- --- --- --- --- ---
  // --- JSON helpers
  // --- --- --- --- --- ---

  [[nodiscard]] inline json::JSONValue to_json(timing::duration_t duration) {
    return json::JSONValue({
      {"ns",    duration.count()},
      {"human", timing::as_string(duration)}
    });
  }

  [[nodiscard]] inline std::pair<std::string, json::JSONValue> json_entry_duration(timing::duration_t duration) {
    return {"duration", to_json(duration)};
  }

  [[nodiscard]] inline std::pair<std::string, json::JSONValue> json_entry_accuracy(
    size_t test_size,
    size_t nb_correct
  ) {
    double rate = double(nb_correct)/double(test_size);
    double erate = 1.0-rate;
    return {"accuracy", json::JSONValue({
      {"test_size",  test_size},
      {"correct_nb", nb_correct},
      {"rate",       rate},
      {"error_rate", erate}
    })
    };
  }

} // end of namespace tempo


