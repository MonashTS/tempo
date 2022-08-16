#pragma once

#include <cstddef>
#include <concepts>
#include <limits>

namespace tempo::distance::utils {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Constants
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Constant to be use when no window is required
  constexpr size_t NO_WINDOW{std::numeric_limits<size_t>::max()};

  /// Positive INFinity shorthand: to be used as initial cutoff value, e.g. in 1NN search
  template<typename F>
  constexpr F PINF = std::numeric_limits<F>::infinity();

  /// (Quiet) Not A Number shorthand
  template<typename F>
  constexpr F QNAN = std::numeric_limits<F>::quiet_NaN();


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Concepts
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Subscriptable collection.
  /// Must have the subscript [ ] (random access) operator (e.g. raw pointer or vector)
  template<typename T>
  concept Subscriptable = requires(T const& collection, size_t index) {
    collection[index];
  };

  /// Cost function
  /// Given two series data point, compute a cost
  template<typename Fun, typename F, typename R=F>
  concept CFun = requires(Fun fun, F datum){
    { fun(datum, datum) } -> std::convertible_to<R>;
  };

  /// Indexed Cost function.
  /// Given two series T and S, a cost function f(i,j) computes the cost of aligning T_i with S_j.
  /// Note: T and S aren't part of the signature: the cost function 'f' can, e.g., capture them.
  template<typename Fun, typename R>
  concept ICFun = requires(Fun fun, size_t i, size_t j){
    { fun(i, j) }->std::convertible_to<R>;
  };

  /// Indexed Cost function of one point.
  /// E.g. ERP specific cost function between a point an a "gap value".
  template<typename Fun, typename R>
  concept ICFunOne = requires(Fun fun, size_t i){
    { fun(i) }->std::convertible_to<R>;
  };

} // End of namespace tempo::distance
