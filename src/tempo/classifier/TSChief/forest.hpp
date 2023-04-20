#pragma once

#include <memory>
#include <vector>
#include <ostream>

#include "tempo/classifier/utils.hpp"
#include "treedata.hpp"
#include "treestate.hpp"
#include "tree.hpp"

namespace tempo::classifier::TSChief {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Run an ensemble of tree
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Test time splitting forest
  struct Forest {

    // --- --- --- Type/Shorthand
    using TREE = std::shared_ptr<TreeNode>;

    // --- --- --- Fields

    /// Vector of trees forming the forest
    std::vector<TREE> forest;

    /// Number of train class for which this forest has been trained
    size_t trainclass_cardinality{};


    // --- --- --- Constructors/Destructors

    Forest() = default;

    // Copy --- relatively cheap (copy of vector of shared pointer)
    Forest(Forest const& other) = default;
    Forest& operator =(Forest const& other) = default;

    // Move
    Forest(Forest&& other) noexcept = default;
    Forest& operator =(Forest&& other) noexcept = default;

    //
    Forest(std::vector<TREE>&& forest, size_t trainclass_cardinality) :
      forest(std::move(forest)), trainclass_cardinality(trainclass_cardinality) {}

    // --- --- --- Methods

    /** Given a testing state and testing data, do a prediction for one exemplar at 'index'
     *  Returns the prediction per tree - we do so as assembling this prediction can be done in different ways.
     * @param state
     * @param data
     * @param test_index
     * @param nb_threads
     * @return vector of results
     */
    std::vector<classifier::Result1> predict(TreeState& state, TreeData const& data, size_t test_index,
                                             size_t nb_threads) const;
  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Training a Forest
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  struct ForestTrainer {

    // --- --- --- Type/Shorthand

    // --- --- --- Fields

    DatasetHeader const& train_header;

    /// The tree trainer to use
    std::shared_ptr<const TreeTrainer> tree_trainer;

    /// Number of trees to train
    size_t nb_trees;

    // --- --- --- Constructors/Destructors

    ForestTrainer(
      DatasetHeader const& train_header,
      std::shared_ptr<TreeTrainer> tree_trainer,
      size_t nbtrees
    ) :
      train_header(train_header),
      tree_trainer(std::move(tree_trainer)),
      nb_trees(nbtrees) {}


    // --- --- ---- Methods

    /// Training a forest by training each tree individually
    std::shared_ptr<Forest> train(TreeState& state, TreeData const& data, ByClassMap const& bcm,
                                  size_t nb_threads,
                                  std::optional<double> opt_sampling = std::nullopt,
                                  std::ostream* out=nullptr
                                  ) const;

  };

} // End of tempo::classifier::PF2
