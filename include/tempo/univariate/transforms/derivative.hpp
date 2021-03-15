#pragma once

#include <tempo/tseries/transform.hpp>

#include <cstddef>
#include <exception>

namespace tempo::univariate {

  /** Computation of a series derivative according to "Derivative Dynamic Time Warping" by Keogh & Pazzani
   * @tparam FloatType    The floating number type used to represent the series.
   * @param series        Pointer to the series's data
   * @param length        Length of the series
   * @param out           Pointer where to write the derivative. Must be able to store 'length' values.
   * Warning: series and out should not overlap (i.e. not in place derivation)
   */
  template<typename FloatType>
  void derivative(const FloatType* series, size_t length, FloatType* out) {
    if (length>2) {
      for (size_t i{1}; i<length-1; ++i) {
        out[i] = ((series[i]-series[i-1])+((series[i+1]-series[i-1])/2.0))/2.0;
      }
      out[0] = out[1];
      out[length-1] = out[length-2];
    } else {
      std::copy(series, series+length, out);
    }
  }

  template<typename FloatType, typename LabelType>
  struct DerivativeTransformer {
    using TS = TSeries<FloatType, LabelType>;
    using SRC = TransformHandle<std::vector<TS>, FloatType, LabelType>;

    int degree;

    explicit DerivativeTransformer(int degree)
      :degree(degree) { }

    /// Create the transform. Do not add to the src's dataset.
    [[nodiscard]] Transform transform(const SRC& src) {
      if ((src.dataset->get_header()).get_ndim()!=1) {
        throw std::invalid_argument("Dataset is not univariate");
      }
      // --- Transform identification
      auto name = "derivative("+std::to_string(degree)+")";
      auto parent = src.get_transform().get_name_components();
      // --- Compute data
      const std::vector<TS>& src_vec = src.get();
      std::vector<TS> output;
      output.reserve(src_vec.size());
      for (const auto& ts: src_vec) {
        const size_t l = ts.length();
        std::vector<FloatType> d(l);
        if (degree==1) {
          derivative(ts.data(), l, d.data());
        } else {
          // Repeated application: require an extra buffer to hold previous transform.
          std::vector<FloatType> input(ts.data(), ts.data()+l);
          // Do until the penultimate, swapping roles of input and d
          for (int i = 0; i<degree-1; ++i) {
            derivative(input.data(), l, d.data());
            swap(input, d);
          }
          // At the end of the for loop, the last computed derivative is in 'input'.
          // Do the last round derivative, with the result ending up in d
          derivative(input.data(), l, d.data());
        }
        // Build the series using other as source of information (missing, labels, etc...)
        output.template emplace_back(TS(std::move(d), ts));
      }
      // --- Create transform
      Capsule capsule = tempo::make_capsule<std::vector<TS>>(std::move(output));
      const void* ptr = tempo::capsule_ptr<std::vector<TS>>(capsule);
      return Transform(std::move(name), std::move(parent), std::move(capsule), ptr);
    }


    /// Create the transform and add it to the src's dataset. Return the corresponding handle.
    [[nodiscard]] TransformHandle<std::vector<TS>, FloatType, LabelType> transform_and_add(SRC& src){
      auto tr = transform(src);
      return src.dataset->template add_transform<std::vector<TS>>(std::move(tr));
    }


  };


} // End of namespace tempo::univariate
