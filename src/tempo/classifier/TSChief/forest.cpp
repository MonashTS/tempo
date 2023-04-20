#include "forest.hpp"

namespace tempo::classifier::TSChief {


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  std::vector<classifier::Result1> Forest::predict(TreeState& state, TreeData const& data, size_t test_index,
                                                   size_t nb_threads) const {
    const size_t nb_trees = forest.size();

    // --- Fork states
    std::vector<std::unique_ptr<TreeState>> local_states = state.forest_fork_vec(nb_trees);

    // --- Multithreaded task
    // Note: each state/result slot is pre-allocated - no shared memory, no need for sync
    std::vector<classifier::Result1> result(nb_trees);
    auto test_task = [&](size_t tree_index) {
      classifier::Result1 r = forest[tree_index]->predict(*local_states[tree_index], data, test_index);
      result[tree_index] = std::move(r);
    };

    tempo::utils::ParTasks p;
    for (size_t i = 0; i<nb_trees; ++i) { p.push_task_args(test_task, i); }
    p.execute(nb_threads);

    // --- Merge states
    state.forest_merge_in_vec(std::move(local_states));

    // --- Return
    return result;
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  std::shared_ptr<Forest> ForestTrainer::train(
    TreeState& state,
    TreeData const& data,
    ByClassMap const& bcm,
    size_t nb_threads,
    std::optional<double> opt_sampling,
    std::ostream *out
  ) const {

    // --- Fork states
    std::vector<std::unique_ptr<TreeState>> local_states = state.forest_fork_vec(nb_trees);

    // --- Multithreaded task
    // Note: each state/result slot is pre-allocated; Mutex still required for output
    std::vector<Forest::TREE> result(nb_trees);
    std::mutex mutex;

    auto test_task = [&](size_t tree_index) {
      if (out!=nullptr) {
        std::lock_guard lock(mutex);
        *out << "Start tree " << tree_index << std::endl;
      }
      //
      auto start = tempo::utils::now();
      ByClassMap const* my_bcm = &bcm;
      ByClassMap local_bcm;
      if(opt_sampling.has_value()){
        local_bcm = bcm.stratified_sampling(opt_sampling.value(), local_states[tree_index]->prng);
        my_bcm = &local_bcm;
      }

      Forest::TREE tree = tree_trainer->train(*local_states[tree_index], data, *my_bcm);
      auto delta = tempo::utils::now() - start;

      //
      if (out!=nullptr) {
        std::lock_guard lock(mutex);
        // --- Printing
        auto& cout = *out;
        auto cf = cout.fill();
        const auto [nbleaf, nbnode] = tree->nb_nodes();
        cout << std::setfill('0');
        cout << std::setw(3) << tree_index + 1 << " / " << nb_trees << "   ";
        cout << std::setw(3) << "Depth = " << tree->depth() << "   ";
        cout << std::setw(3) << "Nb nodes = " << nbnode << "   ";
        cout << std::setw(3) << "Nb leaves = " << nbleaf << "   ";
        cout.fill(cf);
        cout << " timing: " << tempo::utils::as_string(delta) << std::endl;
      }
      //
      result[tree_index] = std::move(tree);
    };

    tempo::utils::ParTasks p;
    for (size_t i = 0; i<nb_trees; ++i) { p.push_task_args(test_task, i); }
    p.execute(nb_threads);

    // --- Merge states
    state.forest_merge_in_vec(std::move(local_states));

    // Build result & return
    return std::make_shared<Forest>(std::move(result), train_header.nb_classes());
  }

} // End of tempo::classifier::PF2
