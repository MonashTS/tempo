#pragma once

#include "splitters.hpp"

#include <tempo/tseries/dataset.hpp>
#include <tempo/tseries/indexSet.hpp>
#include <tempo/univariate/distances/distances.hpp>
#include <tempo/univariate/distances/dtw/dtw.hpp>
#include <tempo/univariate/distances/dtw/cdtw.hpp>
#include <tempo/univariate/distances/dtw/wdtw.hpp>
#include <tempo/univariate/distances/elementwise/elementwise.hpp>

#include <functional>

namespace tempo::univariate::pf {

  template<typename FloatType, typename LabelType>
  struct NN1Splitter : public Splitter<FloatType, LabelType> {
    using DS = Dataset<FloatType, LabelType>;
    using TS = TSeries<FloatType, LabelType>;
    using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
    using DFun = distfun_cutoff_t<FloatType, LabelType>;

    struct LabelIndex {
      LabelType label;
      size_t index;
    };

    const std::vector<TS>* train_set;
    size_t transform_index;
    std::vector<LabelIndex> exemplars;
    DFun distance;

    NN1Splitter(const TH& transform_handle, const ByClassMap<LabelType>& bcm_exemplars, DFun distance)
      :train_set(transform_handle.data), transform_index(transform_handle.index),
       exemplars{}, distance(distance) {
      exemplars.reserve(bcm_exemplars.size());
      for (const auto&[label, vec]: bcm_exemplars) { exemplars.template emplace_back(LabelIndex{label, vec.front()}); }
    }

    std::vector<LabelType> classify_train(const DS& ds, size_t index) {
      const TS& query = ((std::vector<TS>*) ds.get_transform(transform_index).get_data_ptr())->operator[](index);
      double bsf = POSITIVE_INFINITY<double>;
      std::vector<LabelType> labels{};
      for (const auto&[ex_label, ex_index]: exemplars) {
        const TS& candidate = (*train_set)[ex_index];
        auto dist = distance(candidate, query, bsf);
        if (dist<bsf) {
          labels.clear();
          labels.emplace_back(candidate.get_label().value());
          bsf = dist;
        } else if (bsf==dist) { // Manage ties
          const auto& l = candidate.get_label().value();
          if (std::none_of(labels.begin(), labels.end(), [l](const auto& v) { return v==l; })) {
            labels.emplace_back(l);
          }
        }
      }
      return labels;
    }

    std::vector<LabelType> classify_test(const DS& ds, size_t index) { return classify_train(ds, index); }
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- DTW Family
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Generate a NN1-DTW classifier
   * @tparam PRNG Type of the pseudo random number generator
   */
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SG_DTW : public SplitterGenerator<FloatType, LabelType, PRNG> {
    using DS = Dataset<FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;

    Splitter_ptr get_splitter(const DS& ds, const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Get the handler
      const auto& th = ds.get_original_handle();
      // Create the splitter
      auto* nn1splitter = new NN1Splitter(th, exemplars, distfun_cutoff_dtw<FloatType, LabelType>());
      // Embed in a unique ptr
      return std::unique_ptr<Splitter<FloatType, LabelType>>(nn1splitter);
    }
  };

  /** Generate a NN1-CDTW classifier with a random window.
   * Window is chosen randomly in [0, (l+1)/4] (l being the max length if series do not have the same length).
   * @tparam PRNG Type of the pseudo random number generator
   */
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SG_CDTW : public SplitterGenerator<FloatType, LabelType, PRNG> {
    using DS = Dataset<FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;

    Splitter_ptr get_splitter(const DS& ds, const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Compute the size of the window - 0 to
      const size_t top = (ds.get_header().get_maxl()+1)/4;
      const auto w = std::uniform_int_distribution<size_t>(0, top)(prng);
      // Get the handler
      const auto& th = ds.get_original_handle();
      // Create the splitter
      auto* nn1splitter = new NN1Splitter(th, exemplars, distfun_cutoff_cdtw<FloatType, LabelType>(w));
      // Embed in a unique ptr
      return std::unique_ptr<Splitter<FloatType, LabelType>>(nn1splitter);
    }
  };

  /** Create a NN1-WDTW Classifier with a weight ratio randomly taken in [0,1]
   * @tparam PRNG Type of the pseudo random number generator
   */
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SG_WDTW : public SplitterGenerator<FloatType, LabelType, PRNG> {
    using DS = Dataset<FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;

    Splitter_ptr get_splitter(const DS& ds, const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Compute the weight vector
      const FloatType g = std::uniform_real_distribution<FloatType>(0, 1)(prng);
      auto weights = std::make_shared<std::vector<FloatType>>(generate_weights(g, ds.get_header().get_maxl()));
      // Get the handler
      const auto& th = ds.get_original_handle();
      // Create the splitter
      auto* nn1splitter = new NN1Splitter(th, exemplars, distfun_cutoff_wdtw<FloatType, LabelType>(weights));
      // Embed in a unique ptr
      return std::unique_ptr<Splitter<FloatType, LabelType>>(nn1splitter);
    }
  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- Euclidean
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Create a NN1-WDTW Classifier with a weight ratio randomly taken in [0,1]
   * @tparam PRNG Type of the pseudo random number generator
   */
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SG_Eucl : public SplitterGenerator<FloatType, LabelType, PRNG> {
    using DS = Dataset<FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;

    Splitter_ptr get_splitter(const DS& ds, const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Get the handler
      const auto& th = ds.get_original_handle();
      // Create the splitter
      auto* nn1splitter = new NN1Splitter(th, exemplars, distfun_cutoff_elementwise<FloatType, LabelType>());
      // Embed in a unique ptr
      return std::unique_ptr<Splitter<FloatType, LabelType>>(nn1splitter);
    }
  };


} // End of namespace tempo::univariate::pf