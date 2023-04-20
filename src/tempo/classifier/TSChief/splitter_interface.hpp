#pragma once

#include <any>
#include <memory>
#include <utility>
#include <vector>

#include "tempo/classifier/utils.hpp"
#include "treedata.hpp"
#include "treestate.hpp"

namespace tempo::classifier::TSChief {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Trained splitter interfaces

  struct i_SplitterLeaf {
    // Virtual destructor
    virtual ~i_SplitterLeaf() = default;

    /// The Leaf Splitter predicts a result using a (mutable) state, the test data,
    /// and an index used to identify the test exemplar within the test data.
    virtual classifier::Result1 predict(TreeState& state, TreeData const& data, size_t index) = 0;
  };

  struct i_SplitterNode {
    // Virtual destructor
    virtual ~i_SplitterNode() = default;

    /// The Node Splitter finds which branch to follow using a (mutable) state, the test data,
    /// and an index used to identify the test exemplar within the test data.
    virtual size_t get_branch_index(TreeState& state, TreeData const& data, size_t index) = 0;
  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Training tree and generating splitters

  struct i_GenLeaf {
    // --- --- --- Types
    using Result = typename std::optional<std::unique_ptr<i_SplitterLeaf>>;

    // --- --- --- Constructor/Destructor
    virtual ~i_GenLeaf() = default;

    // --- --- --- Methods
    /// Given a training state, training data, and a set of index (in a ByClassMap), try to generate a sleaf
    virtual Result generate(TreeState& state, TreeData const& data, ByClassMap const& bcm) = 0;
  };

  struct i_GenNode {

    // --- --- --- Types

    /// Return type of a node generator.
    /// Return a snode and the split of the incoming train data according to that snode.
    /// The size of the branch_splits vector tells us the number of branches.
    /// For each branch, the associated ByClassMap can contain no index, but **cannot** contain no label.
    /// If a branch is required, but no train actually data reaches it,
    /// the BCM must contains at least one label mapping to an empty set of index.
    struct Result {
      std::unique_ptr<i_SplitterNode> splitter;
      std::vector<ByClassMap> branch_splits;
    };

    // --- --- --- Constructor/Destructor

    virtual ~i_GenNode() = default;

    // --- --- --- Methods

    /// Given a training state, training data, and a set of index (in a ByClassMap), try to generate a sleaf
    virtual Result generate(TreeState& state, TreeData const& data, ByClassMap const& bcm) = 0;

  };

} // End of tempo::classifier::PF2