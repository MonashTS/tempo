#pragma once

#include "splitters/splitters.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/indexSet.hpp>
#include <tempo/tseries/dataset.hpp>

#include <unordered_map>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <future>

namespace tempo::univariate::pf {

  /** A Proximity Tree */
  template<typename FloatType, typename LabelType>
  class PTree {
    // --- --- --- Types
    /// Dataset type
    using DS = Dataset<FloatType, LabelType>;
    /// Unique pointer type for splitter
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    /// Mapping class->subnode
    using BranchMap_t = std::unordered_map<LabelType, std::unique_ptr<PTree>>;
    /// Type of a split
    using Split = std::unordered_map<LabelType, std::tuple<ByClassMap<LabelType>, std::vector<size_t>>>;
    /// Record ratio of subtree
    using SplitRatios = std::vector<std::tuple<LabelType, double>>;
    /// Internal node (not pure) type: a splitter and a mapping class->subnode. Also record the splitratio.
    using INode_t = std::tuple<Splitter_ptr, BranchMap_t, SplitRatios>;
    /// Leaf node (pure) type: the class.
    using LNode_t = LabelType;

    // --- --- --- Fields

    /// The current node is either a leaf (pure) or an internal node (give access to children)
    std::variant<LNode_t, INode_t> node{};

    // --- --- --- Constructors

    /// Leaf constructor
    explicit PTree(const std::string& cname)
      :node{cname} { }

    /// INode constructor
    PTree(Splitter_ptr splitter, BranchMap_t mapping, SplitRatios&& sr)
      :node{INode_t{std::move(splitter), std::move(mapping), std::move(sr)}} { }


    // --- --- --- Tooling

    /** Compute the weighted gini impurity of a split. */
    [[nodiscard]] static double
    weighted_gini_impurity(const Split& split) {
      double wgini{0};
      double item_number{0};
      for (const auto&[c, bcm_vec]: split) {
        const auto[bcm, _] = bcm_vec;
        double g = gini_impurity(bcm);
        size_t bcm_size = 0;
        for (const auto&[_, v]: bcm) { bcm_size += v.size(); }
        item_number += bcm_size;
        wgini += bcm_size*g;
      }
      return wgini/item_number;
    }

    /** Recursively build a tree */
    template<typename PRNG>
    static std::unique_ptr<PTree> make_tree(const DS& ds, const IndexSet& is, const ByClassMap<LabelType>& bcm,
      size_t nbcandidates, SplitterGenerator<FloatType, LabelType, PRNG>& sg, PRNG& prng
    ) {
      using namespace std;
      assert(bcm.size()>0);

      // --- --- --- CASE 1 - leaf case
      if (bcm.size()==1) { return unique_ptr<PTree>(new PTree(bcm.begin()->first)); }

      // --- --- --- CASE 2 - internal node case
      // Best variables: gini, associated splitter and split (for each branch, the by class map)
      double best_gini = POSITIVE_INFINITY<double>;
      Splitter_ptr best_splitter;
      Split best_split;

      // Generate and evaluate the candidate splitters
      for (auto n = 0; n<nbcandidates; ++n) {
        // Get the set of series exemplars, one per class
        auto exemplars = pick_one_by_class(bcm, prng);
        auto splitter = sg.get_splitter(ds, is, exemplars, prng);
        Split split;
        // Compute the split
        for (const auto& query_idx: is) {
          auto query_label = ds.get_original()[query_idx].get_label().value();
          const vector<LabelType> results = splitter->classify_train(ds, query_idx);
          assert(results.size()>0);
          LabelType predicted_label = rand::pick_one(results, prng);
          auto&[bcm, vec] = split[predicted_label];
          bcm[query_label].push_back(query_idx);
          vec.push_back(query_idx);
        }
        // Compute the weighted gini, save the best split
        double wg = weighted_gini_impurity(split);
        if (wg<best_gini) {
          best_gini = wg;
          best_splitter = move(splitter);
          best_split = move(split);
        }
      }

      // Now, we have our best candidate. Recursively create the sub trees and then create the node itself.
      // Note: iterate using the incoming 'by_class' map and not the computed 'split' as the split may not contains
      // all incoming classes (i.e. never selected by the splitter)
      unordered_map<LabelType, unique_ptr<PTree>> sub_trees;
      SplitRatios split_ratios;
      auto size = (double) is.size();
      for (const auto&[label, _]: bcm) {
        if (contains(best_split, label)) {
          const auto&[bcm, indexes] = best_split[label];
          sub_trees[label] = make_tree(ds, IndexSet(indexes), bcm, nbcandidates, sg, prng);
          split_ratios.push_back({label, indexes.size()/size});
        } else {
          // Label not showing up at all in the split (never selected by the splitter). Create a leaf.
          sub_trees[label] = unique_ptr<PTree>(new PTree(label));
        }
      }

      sort(split_ratios.begin(), split_ratios.end(), [](const auto& a, const auto& b) -> bool {
        return get<1>(a)>get<1>(b);
      });

      return unique_ptr<PTree>(new PTree(std::move(best_splitter), std::move(sub_trees), std::move(split_ratios)));
    }

  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- ---
    // Building a tree
    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- ---

    template<typename PRNG=std::mt19937_64>
    [[nodiscard]] static std::unique_ptr<PTree> make(
      const DS& ds, size_t nbcandidates, SplitterGenerator<FloatType, LabelType, PRNG>& sg, PRNG& prng
    ) {
      IndexSet is(ds);
      auto bcm = get_by_class(ds, is);
      return make_tree<PRNG>(ds, IndexSet(ds), bcm, nbcandidates, sg, prng);
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- ---
    // Obtain a classifier
    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- ---

    /** Classifier interface */
    template<typename PRNG>
    struct Classifier {
    private:
      // Fields
      const PTree& pt;
      PRNG& prng;

      // Main classification function (stateless)
      [[nodiscard]] static std::vector<LabelType> classify_(size_t idx, const DS& ds, const PTree& pt, PRNG& prng) {
        switch (pt.node.index()) {
          case 0: { // Leaf: returns the label
            return {std::get<0>(pt.node)};
          }
          case 1: {
            // Node: find in which branch to go
            const auto&[splitter, subnodes, sr] = std::get<1>(pt.node);
            const auto&[mflabel, _] = sr.front();
            const std::vector<LabelType> res = splitter->classify_test(ds, idx, mflabel);
            assert(res.size()>0);
            std::string cl = rand::pick_one(res, prng);
            const auto& sn = subnodes.at(cl); // Use at (is const, when [ ] is not)
            return classify_(idx, ds, *sn, prng);
          }
          default: throw std::runtime_error("Should not happen");
        }
      }

    public:

      Classifier(const PTree& pt, PRNG& prng)
        :pt(pt), prng(prng) { }

      [[nodiscard]] std::vector<std::string> classify(const DS& ds, size_t index) {
        return classify_(index, ds, pt, prng);
      }
    };// End of Classifier

    template<typename PRNG>
    Classifier<PRNG> get_classifier(PRNG& prng) { return Classifier<PRNG>(*this, prng); }


    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- ---
    // Info on the tree
    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- ---

    /** Recursively compute the depth of the tree */
    [[nodiscard]] size_t depth() const {
      switch (node.index()) {
        case 0: return 1; // Leaf
        case 1: { // Children
          const auto&[_, sub_nodes, __] = std::get<1>(node);
          size_t max = 0;
          for (const auto&[k, v]: sub_nodes) {
            size_t d = v->depth();
            if (d>max) { max = d; }
          }
          return max+1;
        }
        default: throw std::runtime_error("Should not happen");
      }
    }

    /** Recursively compute the number of nodes */
    [[nodiscard]] size_t node_number() const {
      switch (node.index()) {
        case 0: return 1; // Leaf
        case 1: { // Children
          const auto&[_, sub_nodes, __] = std::get<1>(node);
          size_t nb = 1; // include this
          for (const auto&[k, v]: sub_nodes) { nb += v->node_number(); }
          return nb;
        }
        default: throw std::runtime_error("Should not happen");
      }
    }

    /** Number of leaves */
    [[nodiscard]] size_t leaf_number() const {
      switch (node.index()) {
        case 0: return 1; // Leaf
        case 1: { // Children
          const auto&[_, sub_nodes, __] = std::get<1>(node);
          size_t nb = 0; // Exclude this
          for (const auto&[k, v]: sub_nodes) { nb += v->leaf_number(); }
          return nb;
        }
        default: throw std::runtime_error("Should not happen");
      }
    }

  };

  /** A Proximity Forest, made of trees */
  template<typename FloatType, typename LabelType>
  class PForest {
    using DS = Dataset<FloatType, LabelType>;
    using TreeVec = std::vector<std::unique_ptr<PTree<FloatType, LabelType>>>;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- ---
    // Private fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- ---
    TreeVec forest;

    /** Take ownership of a vector of trees */
    explicit PForest(TreeVec&& forest)
      :forest(std::move(forest)) { }

  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- ---
    // Build a forest
    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- ---

    template<typename PRNG, bool doInstrumentation = false>
    [[nodiscard]] static std::unique_ptr<PForest<FloatType, LabelType>> make(
      const DS& ds,
      size_t nbtrees,
      size_t nb_candidates,
      SplitterGenerator<FloatType, LabelType, PRNG>& sg,
      size_t base_seed,
      size_t nb_thread = 1,
      std::ostream* out_ptr = nullptr
    ) {
      using namespace std;
      // Mutex for critical section protecting shared variables
      std::mutex mutex;
      size_t nbdone{0};

      // --- --- --- Function: info printing
      auto print_training_info = [nbtrees, &nbdone](ostream& out, size_t workerid, size_t i, size_t nbtodo,
        const TreeVec& local_forest) {
        auto cf = out.fill();
        out << setfill('0');
        out << setw(3) << nbdone << " / " << nbtrees << "   ";
        out << "Worker " << setw(2) << workerid << ": ";
        out << "Tree " << setw(3) << i;
        out << " / " << setw(3) << nbtodo;
        out.fill(cf);
        out << "    Depth: " << setw(3) << local_forest.back()->depth();
        out << "    NB nodes: " << setw(4) << local_forest.back()->node_number() << endl;
      };

      // --- --- --- Function: Training task for one tree
      auto one_train_task = [&ds, nb_candidates, &sg](PRNG& prng) {
        return PTree<FloatType, LabelType>::template make<PRNG>(ds, nb_candidates, sg, prng);
      };

      // --- --- --- Function: Training task for several trees
      auto multi_train_task = [nbtrees, base_seed, &nbdone, &one_train_task, &mutex, out_ptr, &print_training_info](
        size_t workerid,
        size_t nbtodo) {
        // Create a thread local random number generator, ensuring that each thread gets a different seed.
        static thread_local PRNG prng(base_seed+workerid);
        // Also create a training instrumentation map (will only be used if doInstrumentation=true) for all the trees in this task
        TreeVec local_forest;
        for (size_t i{1}; i<=nbtodo; ++i) { // start at one == nicer printing
          local_forest.push_back(one_train_task(prng));
          if (out_ptr!=nullptr) {
            const lock_guard lock(mutex);
            nbdone++;
            print_training_info(*out_ptr, workerid, i, nbtodo, local_forest);
          }
        }
        return local_forest;
      };

      // --- --- --- Train a forest
      TreeVec forest;
      if (nb_thread==1) {
        // If only one thread, just call the training task
        forest = move(multi_train_task(1, nbtrees));

      } else {
        // Multiple thread: share the work
        size_t trees_per_task = nbtrees/nb_thread;
        size_t extra_trees = nbtrees%nb_thread; // Distributed one by one
        // --- --- --- Launch the tasks.
        // Take the lock as we do some printing (and super fast job may finish quickly!)
        // Collect the (future) results of the task
        vector<future<TreeVec>> tasks;
        for (size_t workerid{1}; workerid<=nb_thread; ++workerid) {
          size_t nbt = trees_per_task;
          // Distribute remaining trees
          if (extra_trees>0) {
            ++nbt;
            extra_trees--;
          }
          if (out_ptr!=nullptr) {
            const lock_guard lock(mutex);
            auto& out = *out_ptr;
            auto cf = out.fill();
            out << "Launch worker " << setw(2) << setfill('0') << workerid << " / " << setw(2) << nb_thread;
            out << " with " << setw(3) << nbt << " tasks." << endl;
            out.fill(cf);
          }
          tasks.push_back(async(launch::async, multi_train_task, workerid, nbt));
        }
        // --- --- --- Collecting the task
        for (auto& f:tasks) {
          auto res = f.get();
          forest.insert(forest.end(), make_move_iterator(res.begin()), make_move_iterator(res.end()));
        }
      }
      return unique_ptr<PForest<FloatType, LabelType>>(new PForest<FloatType, LabelType>(std::move(forest)));
    }



    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- ---
    // Classifier
    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- ---

    template<typename PRNG>
    class Classifier {

      // Fields
      const PForest &pf;
      size_t base_seed;
      size_t nb_threads;

      int majority;

    public:

      Classifier(const PForest &pf, size_t base_seed, size_t nb_threads) :
        pf(pf), base_seed(base_seed), nb_threads(nb_threads) {
        majority = pf.forest.size() / 2 + 1;
      }

      [[nodiscard]] std::vector<std::string> classify(const DS& qset, size_t query_index) {
        std::map<std::string, int> score;
        std::mutex mutex;
        std::atomic<bool> go_on(true);
        int max_score{0};
        std::string max_class;

        // --- --- --- Lambda: classify several trees
        auto multi_classif = [&qset, query_index, &mutex, &score, &max_score, &go_on, this](size_t starti, size_t nb) {
          PRNG prng(this->base_seed + starti);
          auto top = starti + nb;
          for (size_t i{starti}; i < top && go_on.load(); ++i) {
            const auto &tree = this->pf.forest[i];
            auto ctree = tree->get_classifier(prng);
            auto cl = rand::pick_one(ctree.classify(qset, query_index), prng);
            {
              const std::lock_guard lock(mutex);
              score[cl] += 1;
              max_score = std::max(max_score, score[cl]);
              if (max_score >= majority) { go_on.store(false); }
            }
          }
        };

        // --- --- --- Launch the tasks
        std::vector<std::future<void>> tasks;
        const size_t slice = pf.forest.size() / nb_threads;
        size_t extra = pf.forest.size() % nb_threads;
        size_t next_start = 0;
        for (size_t workerid = 1; workerid <= nb_threads; ++workerid) {
          // Distribute remaining trees
          size_t nbt = slice;
          if (extra > 0) {
            ++nbt;
            extra--;
          }
          // Launch
          tasks.push_back(std::async(std::launch::async, multi_classif, next_start, nbt));
          next_start += nbt;
        }

        // --- --- --- Collect the results
        for (auto &f:tasks) { f.get(); }


        // --- --- --- Create a vector of results
        std::vector<std::string> results;
        int bsf = 0;

        for (const auto&[k, v]: score) {
          if (v > bsf) {
            bsf = v;
            results.clear();
            results.push_back(k);
          } else if (v == bsf) {
            results.push_back(k);
          }
        }

        return results;
      }
    };


    /** Method style helper for building a classifier for proximity tree */
    template<typename PRNG=std::mt19937_64>
    [[nodiscard]] inline std::unique_ptr<PForest::Classifier<PRNG>>
    get_classifier(size_t base_seed, size_t nb_threads = 1) {
      return std::unique_ptr<PForest::Classifier<PRNG>>(
        new PForest::Classifier<PRNG>(*this, base_seed, nb_threads)
      );
    }



  };

} // End of namespace tempo::univariate::pf