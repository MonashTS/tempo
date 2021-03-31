#pragma once

#include "splitters/splitters.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/indexSet.hpp>
#include <tempo/tseries/dataset.hpp>

#include <unordered_map>

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
    /// Internal node (not pure) type: a splitter and a mapping class->subnode.
    using INode_t = std::tuple<Splitter_ptr, BranchMap_t>;
    /// Leaf node (pure) type: the class.
    using LNode_t = LabelType;
    /// Type of a split
    using Split = std::unordered_map<LabelType, std::tuple<ByClassMap<LabelType>, std::vector<size_t>>>;

    // --- --- --- Fields

    /// The current node is either a leaf (pure) or an internal node (give access to children)
    std::variant<LNode_t, INode_t> node{};

    // --- --- --- Constructors

    /// Leaf constructor
    explicit PTree(const std::string& cname)
      :node{cname} { }

    /// INode constructor
    PTree(Splitter_ptr splitter, BranchMap_t mapping)
      :node{INode_t{std::move(splitter), std::move(mapping)}} { }


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
        auto splitter = sg.get_splitter(is, ds, exemplars, prng);
        Split split;
        // Compute the split
        for (const auto& query_idx: is) {
          auto query_label = ds.get_original()[query_idx].get_label().value();
          const vector<LabelType> results = splitter->classify_train(ds, query_idx);
          assert(results.size()>0);
          LabelType predicted_label = rand::pick_one(results, prng);
          auto& [bcm, vec] = split[predicted_label];
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
      for (const auto&[label, _]: bcm) {
        if (contains(best_split, label)) {
          const auto&[bcm, indexes] = best_split[label];
          sub_trees[label] = make_tree(ds, IndexSet(indexes), bcm, nbcandidates, sg, prng);
        } else {
          // Label not showing up at all in the split (never selected by the splitter). Create a leaf.
          sub_trees[label] = unique_ptr<PTree>(new PTree(label));
        }
      }

      return unique_ptr<PTree>(new PTree(std::move(best_splitter), std::move(sub_trees)));
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
            const auto&[splitter, subnodes] = std::get<1>(pt.node);
            const std::vector<LabelType> res = splitter->classify_test(ds, idx);
            assert(res.size()>0);
            std::string cl = tempo::rand::pick_one(res, prng);
            const auto& sn = subnodes.at(cl); // Use at (is const, when [ ] is not)
            return classify_(idx, ds, *sn, prng);
          }
          default: throw std::runtime_error("Should not happen");
        }
      }

    public:

      Classifier(const PTree& pt, PRNG& prng) :pt(pt), prng(prng) { }

      [[nodiscard]] std::vector<std::string> classify(const DS& ds, size_t index) {
        return classify_(index, ds, pt, prng);
      }
    };// End of Classifier

    template<typename PRNG>
    Classifier<PRNG> get_classifier(PRNG& prng) { return Classifier<PRNG>(*this, prng); }


  };


} // End of namespace tempo::univariate::pf