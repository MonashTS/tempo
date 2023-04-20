#include "tree.hpp"

namespace tempo::classifier::TSChief {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Result of a trained tree
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Given a testing state and testing data, do a prediction for the exemplar 'index'
  classifier::Result1 TreeNode::predict(TreeState& state, TreeData const& data, size_t index) const {
    if (node_kind==LEAF) {
      return as_leaf.splitter->predict(state, data, index);
    } else {
      size_t branch_idx = as_node.splitter->get_branch_index(state, data, index);
      const auto& branch = as_node.branches.at(branch_idx);
      return branch->predict(state, data, index);
    }
  }


  std::tuple<size_t, size_t> TreeNode::nb_nodes() const {
    if (node_kind==LEAF) {
      return {1, 0};
    } else {
      size_t nbl=0;
      size_t nbn=1;
      for (const auto& branch : as_node.branches) {
        const auto [nl, nn] = branch->nb_nodes();
        nbl += nl;
        nbn += nn;
      }
      return {nbl, nbn};
    }
  }

  size_t TreeNode::depth() const {
    if (node_kind==LEAF) {
      return 1;
    } else {
      size_t n = as_node.branches[0]->depth();
      for (size_t i = 1; i<as_node.branches.size(); ++i) {
        size_t m = as_node.branches[i]->depth();
        n = std::max<size_t>(n, m);
      }
      return n + 1;
    }
  }

  // --- --- --- Static functions

  std::shared_ptr<TreeNode> TreeNode::make_leaf(std::unique_ptr<i_SplitterLeaf> sleaf) {
    return std::shared_ptr<TreeNode>(
      new TreeNode{.node_kind = LEAF, .as_leaf = Leaf{std::move(sleaf)}}
    );
  }

  std::shared_ptr<TreeNode> TreeNode::make_node(std::unique_ptr<i_SplitterNode> snode,
                                                std::vector<BRANCH>&& branches) {
    return std::shared_ptr<TreeNode>(
      new TreeNode{.node_kind = NODE, .as_node = Node{std::move(snode), std::move(branches)}}
    );
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Training a tree
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  std::shared_ptr<TreeNode> TreeTrainer::train(TreeState& state, const TreeData& data, ByClassMap const& bcm) const {
    // Ensure that we have at least one class reaching this node!
    // Note: there may be no data point associated to the class.
    assert(bcm.nb_classes()>0);

    // Try to generate a sleaf; if successful, make a sleaf node
    typename i_GenLeaf::Result opt_leaf = leaf_generator->generate(state, data, bcm);

    if (opt_leaf) {
      // --- --- --- LEAF
      return TreeNode::make_leaf(std::move(opt_leaf.value()));
    } else {
      // --- --- --- NODE
      // If we could not generate a sleaf, make a node.
      // Recursively build each branches, then build the current node
      i_GenNode::Result rnode = node_generator->generate(state, data, bcm);
      const size_t nb_branches = rnode.branch_splits.size();
      std::vector<TreeNode::BRANCH> branches;
      branches.reserve(nb_branches);

      // Building loop
      for (size_t idx = 0; idx<nb_branches; ++idx) {
        ByClassMap const& branch_bcm = rnode.branch_splits.at(idx);

        // Signal the state that we are going down a new branch
        state.start_branch(idx);

        // Build the branch
        std::shared_ptr<TreeNode> branch = train(state, data, branch_bcm);
        branches.push_back(std::move(branch));

        // Signal the state that we are done with this branch
        state.end_branch(idx);
      }

      // Result
      return TreeNode::make_node(std::move(rnode.splitter), std::move(branches));
    }
  } // End of train method

} // End of tempo::classifier::PF2