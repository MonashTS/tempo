#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>

#include <tempo/classifier/utils.hpp>
#include <tempo/classifier/TSChief/tree.hpp>

namespace tempo::classifier::TSChief::sleaf {

  /// Pure sleaf snode
  struct SplitterLeaf_Pure_SmoothP : public i_SplitterLeaf {

    // --- --- --- Fields
    /// Pure sleaf result is computed at train time
    classifier::Result1 result;

    // --- --- --- Constructor / Destructors
    /// Construction with already built result
    explicit SplitterLeaf_Pure_SmoothP(classifier::Result1&& r) : result(std::move(r)) {}

    // --- --- --- Methods
    /// Simply return a copy of the stored result
    classifier::Result1 predict(TreeState& /* state */, TreeData const& /* data */, size_t /* index */) override {
      return result;
    }

  };

  /// Pure sleaf generator: stop when only one class reaches the node
  struct GenLeaf_PureSmoothP : public i_GenLeaf {

    // --- --- --- types

    DatasetHeader const& get_train_header;


    // --- --- --- Constructors/Destructors

    explicit GenLeaf_PureSmoothP(DatasetHeader const& train_header) :
      get_train_header(train_header) {}

    // --- --- --- Methods

    i_GenLeaf::Result generate(TreeState& /* state */, TreeData const& /* data */ , ByClassMap const& bcm) override {
      // Generate sleaf on pure node:
      // Vector of probabilities at 0 except for the position matching the encoded label
      if (bcm.nb_classes()==1) {
        size_t cardinality = get_train_header.nb_classes();
        EL elabel = *bcm.classes().begin();  // Get the encoded label
        return {
          std::make_unique<SplitterLeaf_Pure_SmoothP>(
            classifier::Result1::make_smooth_probabilities(cardinality, elabel, bcm.size())
          )
        };
      } else { return {}; } // Else, return the empty option
    }

  };

} // End of namespace tempo::classifier::PF2::sleaf
