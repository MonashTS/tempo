#pragma once

#include "splitters.hpp"

#include <tempo/tseries/dataset.hpp>
#include <tempo/tseries/indexSet.hpp>
#include <tempo/univariate/distances/distances.hpp>
#include <tempo/univariate/distances/dtw/dtw.hpp>
#include <tempo/univariate/distances/dtw/cdtw.hpp>
#include <tempo/univariate/distances/dtw/wdtw.hpp>
#include <tempo/univariate/distances/elementwise/elementwise.hpp>
#include <tempo/univariate/distances/erp/erp.hpp>
#include <tempo/univariate/distances/lcss/lcss.hpp>
#include <tempo/univariate/distances/msm/msm.hpp>
#include <tempo/univariate/distances/msm/wmsm.hpp>
#include <tempo/univariate/distances/dtw/adtw.hpp>
#include <tempo/univariate/distances/twe/twe.hpp>

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
      : train_set(transform_handle.data), transform_index(transform_handle.index), exemplars{}, distance(distance) {
      exemplars.reserve(bcm_exemplars.size());
      for (const auto&[label, vec]: bcm_exemplars) { exemplars.template emplace_back(LabelIndex{label, vec.front()}); }
    }

    /** Classify train
     *  Initialisation with the distance to exemplar of the same class.
     *  Note: in case of ties, order in labels may be different to a simple loop.
     * */
     /*
    std::vector<LabelType> classify_train(const DS& ds, size_t index) {
      const TS& query = ((std::vector<TS>*) ds.get_transform(transform_index).get_data_ptr())->operator[](index);
      const auto& query_label = query.get_label().value();
      double bsf = POSITIVE_INFINITY<double>;
      std::vector<LabelType> labels{};

      { // Init with same label
        size_t idx_ql;
        for (const auto&[exl, exi]: exemplars) {
          if (exl==query_label) {
            const TS& candidate = (*train_set)[exi];
            bsf = distance(candidate, query, bsf);
            labels.template emplace_back(query_label);
            break;
          }
        }
      }

      for (const auto&[ex_label, ex_index]: exemplars) {
        // Skip same label
        if (ex_label==query_label) { continue; }
        // NN1 body
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
    */

    // Classic classify - train
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



      /*
    std::vector<LabelType> classify_test(const DS& ds, size_t index, const LabelType& mflabel) override {

      const TS& query = ((std::vector<TS>*) ds.get_transform(transform_index).get_data_ptr())->operator[](index);

      double bsf = POSITIVE_INFINITY<double>;

      std::vector<LabelType> labels{};

      // --- --- --- Compare with most frequent label
      { // Init with same label
        size_t idx_ql;
        for (const auto&[ex_label, ex_index]: exemplars) {
          if (ex_label==mflabel) {
            const TS& candidate = (*train_set)[ex_index];
            bsf = distance(candidate, query, bsf);
            labels.template emplace_back(mflabel);
            break;
          }
        }
      }

      // --- --- --- Rest

      for (const auto&[ex_label, ex_index]: exemplars) {
        // Skip same label
        if (ex_label==mflabel) { continue; }
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
  };
 */



  // Classic classify - train
  std::vector<LabelType> classify_test(const DS& ds, size_t index, const LabelType& mflabel) {
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
    using TS = TSeries<FloatType, LabelType>;
    using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    using TransfromProvider = std::function<const TH&(PRNG&)>;

    TransfromProvider fun_tp;

    explicit SG_DTW(const TransfromProvider& tp)
      :fun_tp{tp} { }

    Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Get the handler
      const auto& th = fun_tp(prng);
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
    using TS = TSeries<FloatType, LabelType>;
    using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    using TransfromProvider = std::function<const TH&(PRNG&)>;

    TransfromProvider fun_tp;

    explicit SG_CDTW(const TransfromProvider& tp)
      :fun_tp{tp} { }

    Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Compute the size of the window - 0 to max l+1/4
      const size_t top = (ds.get_header().get_maxl()+1)/4;
      const auto w = std::uniform_int_distribution<size_t>(0, top)(prng);
      // Get the handler
      const auto& th = fun_tp(prng);
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
    using TS = TSeries<FloatType, LabelType>;
    using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    using TransfromProvider = std::function<const TH&(PRNG&)>;

    TransfromProvider fun_tp;

    explicit SG_WDTW(const TransfromProvider& tp)
      :fun_tp{tp} { }

    Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Compute the weight vector
      const FloatType g = std::uniform_real_distribution<FloatType>(0, 1)(prng);
      auto weights = std::make_shared<std::vector<FloatType>>(generate_weights(g, ds.get_header().get_maxl()));
      // Get the handler
      const auto& th = fun_tp(prng);
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
    using TS = TSeries<FloatType, LabelType>;
    using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    using TransfromProvider = std::function<const TH&(PRNG&)>;

    TransfromProvider fun_tp;

    explicit SG_Eucl(const TransfromProvider& tp)
      :fun_tp{tp} { }

    Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Get the handler
      const auto& th = fun_tp(prng);
      // Create the splitter
      auto* nn1splitter = new NN1Splitter(th, exemplars, distfun_cutoff_elementwise<FloatType, LabelType>());
      // Embed in a unique ptr
      return std::unique_ptr<Splitter<FloatType, LabelType>>(nn1splitter);
    }
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- ERP
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Create a NN1-ERP classifier with a window in [0, (l+1)/4] and a penalty in [stddev/5, stddev]
   * @tparam PRNG Type of the pseudo random number generator
   */
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SG_ERP : public SplitterGenerator<FloatType, LabelType, PRNG> {
    using DS = Dataset<FloatType, LabelType>;
    using TS = TSeries<FloatType, LabelType>;
    using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    using TransfromProvider = std::function<const TH&(PRNG&)>;

    TransfromProvider fun_tp;

    explicit SG_ERP(const TransfromProvider& tp)
      :fun_tp{tp} { }

    Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Compute the size of the window - 0 to (max l + 1)/4
      const size_t top = (ds.get_header().get_maxl()+1)/4;
      const auto w = std::uniform_int_distribution<size_t>(0, top)(prng);
      // Compute the penalty
      auto stddev_ = stddev(is, ds.get_original_handle());
      const double gValue = std::uniform_real_distribution<double>(0.2*stddev_, stddev_)(prng);
      // Get the handler
      const auto& th = fun_tp(prng);
      // Create the splitter
      auto* nn1splitter = new NN1Splitter(th, exemplars, distfun_cutoff_erp<FloatType, LabelType>(gValue, w));
      // Embed in a unique ptr
      return std::unique_ptr<Splitter<FloatType, LabelType>>(nn1splitter);
    }

  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- LCSS
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Create a NN1-LCSS Classifier with a random window in [0, (l+1)/4] and a random epsilon in [stddev/5, stddev]
   * @tparam PRNG Type of the pseudo random number generator
   */
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SG_LCSS : public SplitterGenerator<FloatType, LabelType, PRNG> {
    using DS = Dataset<FloatType, LabelType>;
    using TS = TSeries<FloatType, LabelType>;
    using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    using TransfromProvider = std::function<const TH&(PRNG&)>;

    TransfromProvider fun_tp;

    explicit SG_LCSS(const TransfromProvider& tp)
      :fun_tp{tp} { }

    Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Compute the size of the window - 0 to max l+1/4
      const size_t top = (ds.get_header().get_maxl()+1)/4;
      const auto w = std::uniform_int_distribution<size_t>(0, top)(prng);
      // Compute the penalty
      auto stddev_ = stddev(is, ds.get_original_handle());
      const double epsilon = std::uniform_real_distribution<double>(0.2*stddev_, stddev_)(prng);
      // Get the handler
      const auto& th = fun_tp(prng);
      // Create the splitter
      auto* nn1splitter = new NN1Splitter(th, exemplars, distfun_cutoff_lcss<FloatType, LabelType>(epsilon, w));
      // Embed in a unique ptr
      return std::unique_ptr<Splitter<FloatType, LabelType>>(nn1splitter);
    }

  };



  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- MSM
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Create a NN1-MSM Classifier with a random cost from a distribution [10^-2, 10^2]
   * @tparam PRNG Type of the pseudo random number generator
   */
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SG_MSM : public SplitterGenerator<FloatType, LabelType, PRNG> {
    using DS = Dataset<FloatType, LabelType>;
    using TS = TSeries<FloatType, LabelType>;
    using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    using TransfromProvider = std::function<const TH&(PRNG&)>;

    TransfromProvider fun_tp;

    explicit SG_MSM(const TransfromProvider& tp)
      :fun_tp{tp} { }

    /** MSM cost parameters */
    static constexpr std::array<FloatType, 100> cost{0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625,
                                                     0.04, 0.04375,
                                                     0.0475, 0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375,
                                                     0.0775, 0.08125,
                                                     0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244,
                                                     0.28, 0.316, 0.352,
                                                     0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712,
                                                     0.748, 0.784,
                                                     0.82, 0.856, 0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8,
                                                     3.16, 3.52, 3.88,
                                                     4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12, 7.48, 7.84,
                                                     8.2, 8.56, 8.92, 9.28,
                                                     9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46,
                                                     49.6, 53.2, 56.8,
                                                     60.4, 64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100};

    Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Get a random cost
      auto distribution = std::uniform_int_distribution<std::size_t>(0, 99);
      const FloatType c = cost[distribution(prng)];
      // Get the handler
      const auto& th = fun_tp(prng);
      // Create the splitter
      auto* nn1splitter = new NN1Splitter(th, exemplars, distfun_cutoff_msm<FloatType, LabelType>(c));
      // Embed in a unique ptr
      return std::unique_ptr<Splitter<FloatType, LabelType>>(nn1splitter);
    }
  };

  /** Create a NN1-WMSM Classifier with a weight ratio randomly taken in [0,1]
   * @tparam PRNG Type of the pseudo random number generator
   */
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SG_WMSM : public SplitterGenerator<FloatType, LabelType, PRNG> {
    using DS = Dataset<FloatType, LabelType>;
    using TS = TSeries<FloatType, LabelType>;
    using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    using TransfromProvider = std::function<const TH&(PRNG&)>;

    TransfromProvider fun_tp;

    explicit SG_WMSM(const TransfromProvider& tp)
      :fun_tp{tp} { }

    /** MSM cost parameters */
    /*
    static constexpr std::array<FloatType, 100> cost{0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625,
                                                     0.04, 0.04375,
                                                     0.0475, 0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375,
                                                     0.0775, 0.08125,
                                                     0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244,
                                                     0.28, 0.316, 0.352,
                                                     0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712,
                                                     0.748, 0.784,
                                                     0.82, 0.856, 0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8,
                                                     3.16, 3.52, 3.88,
                                                     4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12, 7.48, 7.84,
                                                     8.2, 8.56, 8.92, 9.28,
                                                     9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46,
                                                     49.6, 53.2, 56.8,
                                                     60.4, 64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100};
                                                     */

    /** MSM cost parameters */
    static constexpr std::array<FloatType, 50> cost{0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625,
                                                    0.04, 0.04375,
                                                    0.0475, 0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375,
                                                    0.0775, 0.08125,
                                                    0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244,
                                                    0.28, 0.316, 0.352,
                                                    0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712,
                                                    0.748, 0.784,
                                                    0.82, 0.856, 0.892, 0.928, 0.964, 1};

    Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Get a random cost - wmax
      auto distribution = std::uniform_int_distribution<std::size_t>(0, 49);
      const FloatType c = cost[distribution(prng)];
      // Get a max cost based on the stddev
      auto stddev_ = stddev(is, ds.get_original_handle());
      //const double c = std::uniform_real_distribution<double>(0.1*stddev_, 100*stddev_)(prng);
      // Compute the weight vector
      // const FloatType g = std::uniform_real_distribution<FloatType>(0.045, 0.055)(prng);
      auto weights = std::make_shared<std::vector<FloatType>>(
        generate_weights(0.05, ds.get_header().get_maxl(), stddev_));
      // Get the handler
      const auto& th = fun_tp(prng);
      // Create the splitter
      auto* nn1splitter = new NN1Splitter(th, exemplars, distfun_cutoff_wmsm<FloatType, LabelType>(weights));
      // Embed in a unique ptr
      return std::unique_ptr<Splitter<FloatType, LabelType>>(nn1splitter);
    }
  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- ADTW
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  template<typename FloatType, typename LabelType, typename PRNG, auto dist = square_dist<FloatType>>
  struct SG_ADTW : public SplitterGenerator<FloatType, LabelType, PRNG> {
    using DS = Dataset<FloatType, LabelType>;
    using TS = TSeries<FloatType, LabelType>;
    using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    using TransfromProvider = std::function<const TH&(PRNG&)>;

    TransfromProvider fun_tp;

    explicit SG_ADTW(const TransfromProvider& tp)
        :fun_tp{tp} { }

    Splitter_ptr get_splitter(
        const DS& ds, const IndexSet& is,
        const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Get the handler
      const auto& th = fun_tp(prng);
      // Create the penalty array
      // --- Pick a maxdist ratio between [0, 1] - use next after as real distributions pick in [a, b[
      FloatType md_ratio = std::uniform_real_distribution<FloatType>(0.0, std::nextafter(1.0, 2.0))(prng);
      // --- Pick the gindex between [0, 100] - int distributions pick in [a, b]
      size_t gindex = std::uniform_int_distribution<size_t>(0, 100)(prng);
      FloatType g = std::exp(0.001)*(FloatType)gindex-1;
      FloatType maxdist = dist(min_v(is, ds.get_original_handle()), max_v(is, ds.get_original_handle()))*md_ratio;
      // --- Generate the weights
      auto weights = std::make_shared<std::vector<FloatType>>(generate_weights(g, ds.get_header().get_maxl(), maxdist));
      // Create the splitter
      auto* nn1splitter = new NN1Splitter(th, exemplars, distfun_cutoff_adtw<FloatType, LabelType>(weights));
      // Embed in a unique ptr
      return std::unique_ptr<Splitter<FloatType, LabelType>>(nn1splitter);
    }

  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- TWE
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Create a NN1-TWE Classifier with random parameters for lambda (penalties) et nu (stiffness)
   * @tparam PRNG Type of the pseudo random number generator
   */
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SG_TWE : public SplitterGenerator<FloatType, LabelType, PRNG> {
    using DS = Dataset<FloatType, LabelType>;
    using TS = TSeries<FloatType, LabelType>;
    using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    using TransfromProvider = std::function<const TH&(PRNG&)>;

    TransfromProvider fun_tp;

    explicit SG_TWE(const TransfromProvider& tp)
      :fun_tp{tp} { }

    /** TWE nu parameters */
    static constexpr std::array<FloatType, 10> nus{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};

    /** TWE lambda parameters */
    static constexpr std::array<FloatType, 10> lambdas{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444,
                                                       0.055555556, 0.066666667,
                                                       0.077777778, 0.088888889, 0.1};

    Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      // Get a random lambda and nu
      auto distribution = std::uniform_int_distribution<std::size_t>(0, 9);
      const FloatType n = nus[distribution(prng)];
      const FloatType l = lambdas[distribution(prng)];
      // Get the handler
      const auto& th = fun_tp(prng);
      // Create the splitter
      auto* nn1splitter = new NN1Splitter(th, exemplars, distfun_cutoff_twe<FloatType, LabelType>(n, l));
      // Embed in a unique ptr
      return std::unique_ptr<Splitter<FloatType, LabelType>>(nn1splitter);
    }

  };


} // End of namespace tempo::univariate::pf