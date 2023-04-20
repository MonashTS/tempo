#pragma once

#include <cstdlib>
#include <concepts>

namespace tempo {

  /// Floating point values used to represent data. For now, an alias for std::floating_point
  template<typename F>
  concept Float = std::floating_point<F>;

  /// Must have the subscript [ ] operator and the size() function
  template<typename T>
  concept Subscriptable = requires(const T& c, size_t i) {
    c[i];
  };

  /// Must have the subscript [ ] operator and the length() function
  /// The length function must return the length of the series in the number of points,
  /// not the 'size' of the data which, for a n dimensional series, is size = n * length.
  template<typename T>
  concept TSLike =
  Subscriptable<T> && requires(const T& c, size_t i) {{ c.length() }->std::same_as<size_t>; };

  /// Cost function concept for elastic distances.
  /// Given two series T and S, a cost function f(i,j) computes the cost (a Float) of aligning T_i with S_j.
  /// As T and S are not part of the signature, the cost 'fun' must capture them.
  template<typename Fun>
  concept CFun = requires(Fun fun, size_t i, size_t j){
    { fun(i, j) }->std::convertible_to<F>;
  };

  /// Function creating a CFun based on series
  template<typename T, typename D>
  concept CFunBuilder = requires(T builder, const D& s){
    builder(s, s);
  };

}