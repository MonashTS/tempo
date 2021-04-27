#pragma once

#include "splitters/splitters.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/utils/partasks.hpp>
#include <tempo/tseries/indexSet.hpp>
#include <tempo/tseries/dataset.hpp>
#include <tempo/utils/utils/timing.hpp>

#include <unordered_map>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <future>

namespace tempo::univariate::pf {

  /** Unique pointer type for splitter */
  template<typename FloatType, typename LabelType>
  using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;

  /** Type of a split.
   *  A mapping    predicted label (i.e. the branch) ->  (map true label-> series index, series index)
   *  Keeping the map true label -> series index allows for easier computation of the gini impurity.
   */
  template<typename FloatType, typename LabelType>
  using Split = std::unordered_map<LabelType, std::tuple<ByClassMap<LabelType>, std::vector<size_t>>>;

  /** Compute the weighted (ratio of series per branch) gini impurity of a split. */
  template<typename FloatType, typename LabelType>
  [[nodiscard]] static double weighted_gini_impurity(const Split<FloatType, LabelType>& split) {
    double wgini{0};
    double split_size{0};  // Accumulator, total number of series received at this node.
    // For each branch (i.e. assigned class c), get the actual mapping class->series (bcm)
    // and compute the gini impurity of the branch
    for (const auto&[c, bcm_vec]: split) {
      const auto[bcm, vec] = bcm_vec;
      double g = gini_impurity(bcm);
      // Weighted part: the total number of item in this branch is the lenght of the vector of index
      // Accumulate the total number of item in the split (sum the branches),
      // and weight current impurity by size of the branch
      const double bcm_size = vec.size();
      split_size += bcm_size;
      wgini += bcm_size*g;
    }
    // Finish weighted computation by scaling to [0, 1]
    return wgini/split_size;
  }

  /** Make a split, evaluate it, return it with its gini impurity
   * @param ds Dataset with transforms
   * @param is Indexset, indexing into ds, i.e. represents a subset of ds
   * @param bcm By Class mapping of the "is subset", i.e. bcm == get_by_class(ds, is)
   * @param sg Splitter generator
   * @param prng Random number generator
   * @return
   */
  template<typename FloatType, typename LabelType, typename PRNG>
  static std::tuple<Splitter_ptr<FloatType, LabelType>, Split<FloatType, LabelType>, double> mk_split(
    const Dataset<FloatType, LabelType>& ds,
    const IndexSet& is,
    const ByClassMap<LabelType>& bcm,
    SplitterGenerator<FloatType, LabelType, PRNG>& sg,
    PRNG& prng
  ) {
    assert(bcm==get_by_class(ds, is));
    // Get the set of series exemplars, one per class, and generate a splitter
    auto exemplars = pick_one_by_class(bcm, prng);
    auto splitter = sg.get_splitter(ds, is, exemplars, prng);
    Split<FloatType, LabelType> split;
    // For each index in the 'is' subset (including selected exemplars - will eventually form pure leaves)
    for (const auto& query_idx: is) {
      // Predict the branch
      const std::vector<LabelType> results = splitter->classify_train(ds, query_idx);
      assert(!results.empty());
      LabelType predicted_label = rand::pick_one(results, prng);
      // Update the info in the predicted branch (by class map, with the actual class, and the index)
      auto&[branch_bcm, branch_vec] = split[predicted_label];
      auto query_label = ds.get_original()[query_idx].get_label().value();
      branch_bcm[query_label].push_back(query_idx);
      branch_vec.push_back(query_idx);
    }
    // Compute the weighted gini of the splitter
    double wg = weighted_gini_impurity<FloatType, LabelType>(split);
    return {std::move(splitter), std::move(split), wg};
  }

  /** A Proximity Tree */
  template<typename FloatType, typename LabelType>
  class PTree {
    // --- --- --- Private yypes

    /// Dataset type
    using DS = Dataset<FloatType, LabelType>;

    /// Mapping class->subnode
    using BranchMap_t = std::unordered_map<LabelType, std::unique_ptr<PTree>>;

    /// Record ratio of subtree
    using SplitRatios = std::vector<std::tuple<LabelType, double>>;

    /// Internal node (not pure) type: a splitter and a mapping class->subnode. Also record the splitratio.
    using INode_t = std::tuple<Splitter_ptr<FloatType, LabelType>, BranchMap_t, SplitRatios>;

    /// Leaf node (pure) type: the class.
    using LNode_t = LabelType;

    // --- --- --- Private fields

    /// The current node is either a leaf (pure) or an internal node (give access to children)
    std::variant<LNode_t, INode_t> node{};

    // --- --- --- Private constructors

    /// Leaf constructor. The leaf is pure, and will always result in the 'label' class.
    explicit PTree(const LabelType& label)
      :node{label} { }

    /// INode constructor
    PTree(Splitter_ptr<FloatType, LabelType> splitter, BranchMap_t mapping, SplitRatios&& sr)
      :node{INode_t{std::move(splitter), std::move(mapping), std::move(sr)}} { }

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
      Splitter_ptr<FloatType, LabelType> best_splitter;
      Split<FloatType, LabelType> best_split;

      // Generate and evaluate the candidate splitters
      for (auto n = 0; n<nbcandidates; ++n) {
        auto[splitter, split, gini] = mk_split(ds, is, bcm, sg, prng);
        /*
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
         */
        // Compute the weighted gini, save the best split
        double wg = weighted_gini_impurity<FloatType, LabelType>(split);
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
      const DS& ds, const ByClassMap<LabelType>& bcm,
      size_t nbcandidates, SplitterGenerator<FloatType, LabelType, PRNG>& sg, PRNG& prng
    ) {
      IndexSet is(ds);
      return make_tree<PRNG>(ds, IndexSet(ds), bcm, nbcandidates, sg, prng);
    }

    /// Building a tree using an already made root
    template<typename PRNG=std::mt19937_64>
    [[nodiscard]] static std::unique_ptr<PTree> make_from_root(
      const DS& ds, const IndexSet& is, const ByClassMap<LabelType>& bcm,
      const std::tuple<Splitter_ptr<FloatType, LabelType>, Split<FloatType, LabelType>, double>& root_split,
      size_t nbcandidates, SplitterGenerator<FloatType, LabelType, PRNG>& sg, PRNG& prng
    ) {
      using namespace std;
      assert(bcm.size()>0);
      // --- --- --- CASE 1 - leaf case
      if (bcm.size()==1) { return unique_ptr<PTree>(new PTree(bcm.begin()->first)); }
      // --- --- --- CASE 2 - internal node case
      auto[splitter, split, gini] = mk_split(ds, is, bcm, sg, prng);

      unordered_map<LabelType, unique_ptr<PTree>> sub_trees;
      SplitRatios split_ratios;
      auto size = (double) is.size();
      for (const auto&[label, _]: bcm) {
        if (contains(split, label)) {
          const auto&[bcm, indexes] = split[label];
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

      return unique_ptr<PTree>(new PTree(std::move(splitter), std::move(sub_trees), std::move(split_ratios)));


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
          const auto&[_1, sub_nodes, _2] = std::get<1>(node);
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
          const auto&[_1, sub_nodes, _2] = std::get<1>(node);
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
          const auto&[_1, sub_nodes, _2] = std::get<1>(node);
          size_t nb = 0; // Exclude this
          for (const auto&[k, v]: sub_nodes) { nb += v->leaf_number(); }
          return nb;
        }
        default: throw std::runtime_error("Should not happen");
      }
    }
  }; // End of class PTree

  /** A Proximity Forest, made of trees */
  template<typename FloatType, typename LabelType>
  class PForest {
    /// Dataset type
    using DS = Dataset<FloatType, LabelType>;
    /// Type of a PTree
    using PTree_t = PTree<FloatType, LabelType>;
    /// Collection of trees
    using TreeVec = std::vector<std::unique_ptr<PTree_t>>;
    /// Unique pointer type for splitter
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;
    /// Type of a split
    using Split = std::unordered_map<LabelType, std::tuple<ByClassMap<LabelType>, std::vector<size_t>>>;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- ---
    // Private fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- ---
    TreeVec forest;

    /** Take ownership of a vector of trees */
    explicit PForest(TreeVec&& forest) :forest(std::move(forest)) { }

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
      // Mutex for critical section protecting shared variables (output and forest)
      std::mutex mutex;
      TreeVec forest;
      forest.reserve(nbtrees);

      // Shared by the roots:
      auto is = IndexSet(ds);
      auto bcm = get_by_class(ds, is);

      // --- --- --- Task to create one tree
      auto mk_tree = [&ds, &bcm, nb_candidates, &sg, &base_seed, &nbtrees, &mutex, &forest, out_ptr](size_t id){
        // Compute trees
        PRNG prng(base_seed + id);
        auto tree = PTree<FloatType, LabelType>::template make<PRNG>(ds, bcm, nb_candidates, sg, prng);
        // Start lock: protecting the forest and out printing
        std::lock_guard lock(mutex);
        // --- Printing
        auto& out = *out_ptr;
        auto cf = out.fill();
        out << setfill('0');
        out << setw(3) << id+1 << " / " << nbtrees << "   ";
        out.fill(cf);
        out << "    Depth: " << setw(3) << tree->depth();
        out << "    NB nodes: " << setw(4) << tree->node_number();
        out << "    NB leaves: " << setw(4) << tree->leaf_number() << endl;
        // --- Add in the forest
        forest.push_back(std::move(tree));
      };

      // --- --- --- Prepare tasks and execute them
      ParTasks p;
      for(size_t id=0; id<nbtrees; id++){
        p.template push_task(mk_tree, id);
      }
      p.execute(nb_thread);

      // --- --- --- Return the forest
      return unique_ptr<PForest<FloatType, LabelType>>(new PForest<FloatType, LabelType>(std::move(forest)));
    }



    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- ---
    // Classifier
    // --- --- --- --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- --- --- -- --- --- --- --- ---

    template<typename PRNG>
    class Classifier {

      // Fields
      const PForest& pf;
      size_t base_seed;
      size_t nb_threads;

    public:

      Classifier(const PForest& pf, size_t base_seed, size_t nb_threads) :
        pf(pf), base_seed(base_seed), nb_threads(nb_threads) { }

      [[nodiscard]] std::vector<std::string> classify(const DS& qset, size_t query_index) {
        std::map<std::string, int> score; // track the number of vote per label
        std::mutex mutex;

        // --- --- --- Lambda: predict with one tree
        auto predict = [&qset, query_index, &mutex, &score, this](size_t tree_index) {
          // PRNG per tree
          PRNG prng(this->base_seed+tree_index);
          const auto& tree = this->pf.forest[tree_index];
          // Classify
          auto ctree = tree->get_classifier(prng);
          auto cl = rand::pick_one(ctree.classify(qset, query_index), prng);
          // Update score - note score[cl] init to 0 by default on first access
          const std::lock_guard lock(mutex);
          score[cl] += 1;
        };

        // --- --- --- Launch the tasks
        ParTasks p;
        for(size_t i=0; i<this->pf.forest.size(); ++i){
          p.template push_task(predict, i);
        }
        p.execute(nb_threads);


        // --- --- --- Create a vector of results
        std::vector<std::string> results;
        int bsf = 0;

        for (const auto&[k, v]: score) {
          if (v>bsf) {
            bsf = v;
            results.clear();
            results.push_back(k);
          } else if (v==bsf) {
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




//    /** */
//    template<typename PRNG, bool doInstrumentation = false>
//    [[nodiscard]] static std::unique_ptr<PForest<FloatType, LabelType>> make_poolroot(
//      const DS& ds,
//      size_t nbtrees,
//      size_t nb_candidates,
//      SplitterGenerator<FloatType, LabelType, PRNG>& sg,
//      size_t base_seed,
//      size_t nb_thread = 1,
//      std::ostream* out_ptr = nullptr
//    ) {
//      using namespace std;
//      // Mutex for critical section protecting shared variables
//      std::mutex mutex;
//      size_t nbdone{0};
//
//      auto is = IndexSet(ds);
//      auto bcm = get_by_class(ds, is);
//
//      // --- --- --- Create nb trees * nb candidates root splitters, and sort them by gini impurity
//      std::cout << "Starting root pooling..." << std::endl;
//      auto start = tempo::timing::now();
//      // ---
//      using RootSplit = std::tuple<Splitter_ptr, Split, double>;
//      // --- Workload
//      const size_t nbroots = nbtrees*nb_candidates;
//      // --- Per thread worker
//      auto multi_root = [&ds, &is, &bcm, &sg, base_seed](size_t workerid, size_t nb_todo) {
//        // Create a thread local random number generator, ensuring that each thread gets a different seed.
//        static thread_local PRNG prng(base_seed+workerid);
//        vector<RootSplit> result;
//        for (size_t i{0}; i<nb_todo; ++i) { result.template emplace_back(mk_split(ds, is, bcm, sg, prng)); }
//        return result;
//      };
//      // --- Launch thread & collect work
//      vector<RootSplit> roots;
//      if (nb_thread==1) { roots = move(multi_root(1, nbroots)); }
//      else {
//        // Multiple thread: share the work
//        size_t roots_per_task = nbroots/nb_thread;
//        size_t extra_roots = nbroots%nb_thread; // Distributed one by one
//        // --- --- --- Launch the tasks.
//        // Take the lock as we do some printing (and super fast job may finish quickly!)
//        // Collect the (future) results of the task
//        vector<future<vector<RootSplit>>> tasks;
//        for (size_t workerid{1}; workerid<=nb_thread; ++workerid) {
//          // Spread tasks, distributing "extra roots"
//          size_t nbt = roots_per_task;
//          if (extra_roots>0) {
//            ++nbt;
//            extra_roots--;
//          }
//          // Launch
//          tasks.push_back(async(launch::async, multi_root, workerid, nbt));
//        }
//        // --- --- --- Collecting the task
//        for (auto& f:tasks) {
//          auto res = f.get();
//          roots.insert(roots.end(), make_move_iterator(res.begin()), make_move_iterator(res.end()));
//        }
//        // --- --- --- Sort by impurity (small first)
//        sort(roots.begin(), roots.end(), [](const auto& a, const auto& b) -> bool {
//          return std::get<2>(a)<std::get<2>(b);
//        });
//        /*
//        for (const auto&[a, b, c]: roots) { std::cout << " impurity = " << c << std::endl; }
//        std::cout << roots.size() << std::endl;
//         */
//      }
//      auto stop = tempo::timing::now();
//      std::cout << "Root pooling done in" << std::endl;
//      tempo::timing::printDuration(std::cout, stop-start);
//      std::cout << std::endl;
//
//      // -- --- --- Other random number
//      base_seed += nb_thread+7;
//
//      // --- --- --- Create trees using the pool of roots
//      size_t root_index = 0;
//      auto one_train_task = [&root_index, &ds, &is, &bcm, &roots, nb_candidates, &sg, &mutex](PRNG& prng) mutable {
//        mutex.lock();
//        auto root = move(roots[root_index]);
//        ++root_index;
//        mutex.unlock();
//        return PTree<FloatType, LabelType>::template make_from_root<PRNG>(ds, is, bcm, root, nb_candidates, sg, prng);
//      };
//
//
//      // --- --- --- Function: info printing
//      auto print_training_info = [nbtrees, &nbdone](ostream& out, size_t workerid, size_t i, size_t nbtodo,
//        const TreeVec& local_forest) {
//        auto cf = out.fill();
//        out << setfill('0');
//        out << setw(3) << nbdone << " / " << nbtrees << "   ";
//        out << "Worker " << setw(2) << workerid << ": ";
//        out << "Tree " << setw(3) << i;
//        out << " / " << setw(3) << nbtodo;
//        out.fill(cf);
//        out << "    Depth: " << setw(3) << local_forest.back()->depth();
//        out << "    NB nodes: " << setw(4) << local_forest.back()->node_number() << endl;
//      };
//
//
//      // --- --- --- Function: Training task for several trees
//      auto multi_train_task = [nbtrees, base_seed, &nbdone, &one_train_task, &mutex, out_ptr, &print_training_info](
//        size_t workerid,
//        size_t nbtodo) {
//        // Create a thread local random number generator, ensuring that each thread gets a different seed.
//        static thread_local PRNG prng(base_seed+workerid);
//        // Also create a training instrumentation map (will only be used if doInstrumentation=true) for all the trees in this task
//        TreeVec local_forest;
//        for (size_t i{1}; i<=nbtodo; ++i) { // start at one == nicer printing
//          local_forest.push_back(one_train_task(prng));
//          if (out_ptr!=nullptr) {
//            const lock_guard lock(mutex);
//            nbdone++;
//            print_training_info(*out_ptr, workerid, i, nbtodo, local_forest);
//          }
//        }
//        return local_forest;
//      };
//
//      // --- --- --- Train a forest
//      TreeVec forest;
//      if (nb_thread==1) {
//        // If only one thread, just call the training task
//        forest = move(multi_train_task(1, nbtrees));
//
//      } else {
//        // Multiple thread: share the work
//        size_t trees_per_task = nbtrees/nb_thread;
//        size_t extra_trees = nbtrees%nb_thread; // Distributed one by one
//        // --- --- --- Launch the tasks.
//        // Take the lock as we do some printing (and super fast job may finish quickly!)
//        // Collect the (future) results of the task
//        vector<future<TreeVec>> tasks;
//        for (size_t workerid{1}; workerid<=nb_thread; ++workerid) {
//          size_t nbt = trees_per_task;
//          // Distribute remaining trees
//          if (extra_trees>0) {
//            ++nbt;
//            extra_trees--;
//          }
//          if (out_ptr!=nullptr) {
//            const lock_guard lock(mutex);
//            auto& out = *out_ptr;
//            auto cf = out.fill();
//            out << "Launch worker " << setw(2) << setfill('0') << workerid << " / " << setw(2) << nb_thread;
//            out << " with " << setw(3) << nbt << " tasks." << endl;
//            out.fill(cf);
//          }
//          tasks.push_back(async(launch::async, multi_train_task, workerid, nbt));
//        }
//        // --- --- --- Collecting the task
//        for (auto& f:tasks) {
//          auto res = f.get();
//          forest.insert(forest.end(), make_move_iterator(res.begin()), make_move_iterator(res.end()));
//        }
//      }
//      return unique_ptr<PForest<FloatType, LabelType>>(new PForest<FloatType, LabelType>(std::move(forest)));
//    }


  };

} // End of namespace tempo::univariate::pf


// int majority;

// majority = pf.forest.size()/2+1;

// //std::atomic<bool> go_on(true);
// //int max_score{0};

// // --- --- --- Lambda: classify several trees
//auto multi_classif = [&qset, query_index, &mutex, &score, /*&max_score, &go_on,*/ this](size_t starti,
//  size_t nb) {
//  PRNG prng(this->base_seed+starti);
//  auto top = starti+nb;
//  for (size_t i{starti}; i<top /*&& go_on.load()*/; ++i) {
//    const auto& tree = this->pf.forest[i];
//    auto ctree = tree->get_classifier(prng);
//    auto cl = rand::pick_one(ctree.classify(qset, query_index), prng);
//    {
//      const std::lock_guard lock(mutex);
//      score[cl] += 1;
//      //max_score = std::max(max_score, score[cl]);
//      //if (max_score>=majority) { go_on.store(false); }
//    }
//  }
//};
