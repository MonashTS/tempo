#pragma once

#include <map>
#include <memory>
#include <string>

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>

#include <tempo/classifier/TSChief/treedata.hpp>
#include <tempo/classifier/TSChief/treestate.hpp>
#include <tempo/classifier/TSChief/splitter_interface.hpp>

namespace tempo::classifier::TSChief::snode::nn1splitter {


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // NN1 Time Series Distance Splitter

  struct SplitterNN1 : public i_SplitterNode {

    // --- --- --- Types

    /// IndexSet of selected exemplar in the train
    IndexSet train_indexset;

    /// How to map label to index of branches
    std::map<EL, size_t> labels_to_branch_idx;

    /// Distance function
    std::unique_ptr<i_Dist> distance;

    // --- --- --- Constructors/Destructors

    SplitterNN1(
      IndexSet is,
      std::map<EL, size_t> labels_to_branch_idx,
      std::unique_ptr<i_Dist> dist
    ) :
      train_indexset(std::move(is)),
      labels_to_branch_idx(std::move(labels_to_branch_idx)),
      distance(std::move(dist))
    {}

    // --- --- --- Methods

    size_t get_branch_index(TreeState& tstate, TreeData const& tdata, size_t index) override {
      // Distance info access
      std::string tname = distance->get_transformation_name();

      // Data access
      const DTS& train_dataset = at_train(tdata).at(tname);
      const DTS& test_dataset = at_test(tdata).at(tname);
      const TSeries& test_exemplar = test_dataset[index];

      // NN1 test loop
      F bsf = utils::PINF;
      std::set<EL> labels;
      for (size_t candidate_idx : train_indexset) {
        const auto& candidate = train_dataset[candidate_idx];
        F d = distance->eval(candidate, test_exemplar, bsf);
        if (d<bsf) {
          labels.clear();
          labels.insert(train_dataset.label(candidate_idx).value());
          bsf = d;
        } else if (bsf==d) { labels.insert(train_dataset.label(candidate_idx).value()); }
      }
      assert(!labels.empty());

      // Return the branch matching the predicted label
      EL predicted_label;
      std::sample(labels.begin(), labels.end(), &predicted_label, 1, tstate.prng);
      return labels_to_branch_idx.at(predicted_label);

    } // End of function get_branch_index
  };

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
