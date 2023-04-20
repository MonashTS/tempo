#pragma once

#include <any>
#include <memory>
#include <utility>
#include <vector>
#include "tempo/classifier/utils.hpp"

namespace tempo::classifier::TSChief {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // State management

  struct i_TreeState {
    // Virtual destructor
    virtual ~i_TreeState() = default;

    // --- --- --- Fork/Merge
    virtual std::unique_ptr<i_TreeState> forest_fork(size_t tree_idx) const = 0;
    virtual void forest_merge_in(std::unique_ptr<i_TreeState>&& other) = 0;

    // --- --- --- Start/End node

    // TODO: think about the callbacks, to be updated (e.g. on leaf)

    /// Method called when a new branch is started - will be called before calling the "train" function for this branch.
    /// Branches are created in a "deep first" fashion.
    virtual void start_branch(size_t branch_idx) = 0;

    /// Method called when a branch is done - will be called after calling the "train" function for this branch.
    virtual void end_branch(size_t branch_idx) = 0;
  };

  struct TreeState;

  template<typename State> requires std::derived_from<State, i_TreeState>
  struct i_GetState {
    virtual ~i_GetState() = default;
    virtual State& at(TreeState& td) = 0;
  };

  /** Maintain a collection of states used in a tree and provide random numbers.
   *  States are represented by unique_ptr<T> where T must subclass i_TreeState.
   *  TreeState itself subclass i_TreeState: all the operation are broadcasted to the states.
   *  Add a state with the 'register_state' method; access the state through the returned object GetState.
   */
  struct TreeState : public i_TreeState {

    // --- --- --- Types
    template<typename State>
    struct GetState : public i_GetState<State> {
      size_t index{};
      explicit GetState(size_t index) : index(index) {}
      State& at(TreeState& ts) override { return *((State *)ts.states[index].get()); }
    };

    // --- --- --- Fields
    std::vector<std::unique_ptr<i_TreeState>> states{};
    size_t seed;
    PRNG prng;
    size_t tree_index;

    // --- --- --- Constructor/Destructor

    /// Build a new tree state
    /// Note: internal seed = seed+tree_index
    explicit TreeState(size_t seed, size_t tree_index) :
      seed(seed), prng(seed + tree_index), tree_index(tree_index) {}

    // --- --- --- Methods

    template<typename State>
    std::shared_ptr<i_GetState<State>> register_state(std::unique_ptr<State>&& uptr) {
      size_t idx = states.size();
      states.push_back(std::move(uptr));
      return std::make_shared<GetState<State>>(idx);
    }

    template<typename State, typename... _Args>
    std::shared_ptr<i_GetState<State>> build_state(_Args&& ... __args) {
      size_t idx = states.size();
      states.push_back(std::make_unique<State>(std::forward<_Args>(__args)...));
      return std::make_shared<GetState<State>>(idx);
    }

    std::unique_ptr<i_TreeState> forest_fork(size_t tree_idx) const override;

    void forest_merge_in(std::unique_ptr<i_TreeState>&& other) override;

    void start_branch(size_t branch_idx) override;

    void end_branch(size_t branch_idx) override;

    /// Self-Fork 'nb_trees' time, putting the forked in a vector
    std::vector<std::unique_ptr<TreeState>> forest_fork_vec(size_t nb_trees) const;

    /// Merge-in a vector of state
    void forest_merge_in_vec(std::vector<std::unique_ptr<TreeState>>&& vec);

  };

} // End of tempo::classifier::PF2