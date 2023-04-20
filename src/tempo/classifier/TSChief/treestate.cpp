#include "treestate.hpp"

namespace tempo::classifier::TSChief {

  std::unique_ptr<i_TreeState> TreeState::forest_fork(size_t tree_idx) const {
    // Create the other state and fork substates 1 for 1
    auto fork = std::make_unique<TreeState>(seed, tree_idx);
    for (auto const& substate : states) { fork->states.push_back(substate->forest_fork(tree_idx)); }
    return fork;
  }

  void TreeState::forest_merge_in(std::unique_ptr<i_TreeState>&& other) {
    // Get pointer of the good type
    auto *other_state = dynamic_cast<TreeState *>(other.get());
    if (other_state==nullptr) { tempo::utils::should_not_happen("Dynamic cast to TreeState failed"); }
    // Merge in the substates in a 1 to 1 index matching
    for (size_t i{0}; i<states.size(); ++i) {
      auto&& upt = std::move(other_state->states[i]);
      states[i]->forest_merge_in(std::move(upt));
    }
  }

  void TreeState::start_branch(size_t branch_idx) {
    for (auto& substate : states) { substate->start_branch(branch_idx); }
  }

  void TreeState::end_branch(size_t branch_idx) {
    for (auto& substate : states) { substate->end_branch(branch_idx); }
  }

  std::vector<std::unique_ptr<TreeState>> TreeState::forest_fork_vec(size_t nb_trees) const {
    // Note: override covariant not supported when using smart pointer - use raw pointer cast instead
    //       If it were supported, we could have a std::unique_ptr<TreeState> forest_fork method, avoiding the cast.
    //       This is the case with raw pointer, but not with the "smart" one.
    //       To workaround so much smartness, we release the raw pointer, cast it, and put it back in a new unique_ptr.
    std::vector<std::unique_ptr<TreeState>> local_states;
    for (size_t i = 0; i<nb_trees; ++i) {
      std::unique_ptr<TreeState> uptr;
      uptr.reset((TreeState *)(forest_fork(i).release()));
      local_states.push_back(std::move(uptr));
    }
    return local_states;
  }

  void TreeState::forest_merge_in_vec(std::vector<std::unique_ptr<TreeState>>&& vec) {
    for (auto&& s : vec) { forest_merge_in(std::move(s)); }
  }

} // End of tempo::classifier::PF2