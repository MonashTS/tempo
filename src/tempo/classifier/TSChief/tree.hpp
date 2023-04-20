#pragma once

#include <any>
#include <memory>
#include <utility>
#include <vector>

#include "tempo/classifier/utils.hpp"
#include "treedata.hpp"
#include "treestate.hpp"
#include "splitter_interface.hpp"

namespace tempo::classifier::TSChief {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Result of a trained tree
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  struct TreeNode {
    // --- --- --- Types
    using BRANCH = std::shared_ptr<TreeNode>;

    /// Node Kind type: a TreeNode is either a sleaf or an internal node
    enum Kind { LEAF, NODE };

    /// Payload type when node_kind == LEAF
    struct Leaf {
      std::unique_ptr<i_SplitterLeaf> splitter;
    };

    /// Payload type when node_kind == NODE
    struct Node {
      std::unique_ptr<i_SplitterNode> splitter;
      std::vector<BRANCH> branches;
    };

    // --- --- --- Fields

    Kind node_kind;
    Leaf as_leaf{};
    Node as_node{};

    // --- --- --- Methods

    /// Given a testing state and testing data, do a prediction for the exemplar 'index'
    classifier::Result1 predict(TreeState& state, TreeData const& data, size_t index) const;

    /// Count the number of nodes (number of leaf, number of internal node)
    std::tuple<size_t, size_t> nb_nodes() const;

    /// Get the maximal depth
    size_t depth() const;

    // --- --- --- Static functions
    static std::shared_ptr<TreeNode> make_leaf(std::unique_ptr<i_SplitterLeaf> sleaf);
    static std::shared_ptr<TreeNode> make_node(std::unique_ptr<i_SplitterNode> snode, std::vector<BRANCH>&& branches);
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Training a tree
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  struct TreeTrainer {

    // --- --- --- Fields
    // sleaf and node generators
    std::shared_ptr<i_GenLeaf> leaf_generator;
    std::shared_ptr<i_GenNode> node_generator;

    // --- --- --- Constructors/Destructors

    /// Build a Splitting Tree with a sleaf generator and a node generator
    TreeTrainer(std::shared_ptr<i_GenLeaf> leaf_generator, std::shared_ptr<i_GenNode> node_generator) :
      leaf_generator(std::move(leaf_generator)), node_generator(std::move(node_generator)) {}

    // --- --- --- Methods

    /// Train a splitting tree
    std::shared_ptr<TreeNode> train(TreeState& state, const TreeData& data, ByClassMap const& bcm) const;
  };

} // End of tempo::classifier::PF2