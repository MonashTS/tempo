#pragma once

#include <tempo/tseries/dataset.hpp>
#include <tempo/tseries/indexSet.hpp>
#include <tempo/utils/utils.hpp>

namespace tempo::univariate::pf {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Splitter & Splitter Generator Interfaces
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Interface: a splitter is a classifier */
  template<typename FloatType, typename LabelType>
  struct Splitter {
    using DS = Dataset<FloatType, LabelType>;

    virtual std::vector<LabelType> classify_train(const DS& dataset, size_t index) = 0;

    virtual std::vector<LabelType> classify_test(const DS& dataset, size_t index) = 0;

    virtual ~Splitter() = default;
  };

  /** Interface: generate a new splitter, given a vector of transforms and a source of randomness.
   *  Note: Method is const as it should be thread-shareable. */
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SplitterGenerator {
    using DS = Dataset<FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;

    virtual Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) = 0;

    virtual ~SplitterGenerator() = default;
  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Splitters chooser
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SplitterChooser:public SplitterGenerator<FloatType, LabelType, PRNG> {

    using DS = Dataset<FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    using SplitterGen_ptr = std::unique_ptr<SplitterGenerator<FloatType, LabelType, PRNG>>;

    std::vector<SplitterGen_ptr> splitter_generators;

    SplitterChooser(std::vector<SplitterGen_ptr>&& splitter_generators)
    : splitter_generators{std::forward<std::vector<SplitterGen_ptr>>(splitter_generators)}
    { }

    Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) override {
      const auto sidx = std::uniform_int_distribution<size_t>(0, splitter_generators.size()-1)(prng);
      return splitter_generators[sidx]->get_splitter(ds, is, exemplars, prng);
    }

  };




} // end of namespace tempo::univariate::pf