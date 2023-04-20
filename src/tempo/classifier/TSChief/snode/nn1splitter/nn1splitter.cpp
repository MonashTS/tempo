#include <algorithm>
#include <set>
#include <vector>

#include "nn1splitter.hpp"
#include "nn1splitter.private.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  /// Generate a snode based on the distance generator specifed at build time
  i_GenNode::Result GenSplitterNN1::generate(TreeState& state, TreeData const& data, ByClassMap const& bcm) {
    using MDTS = std::map<std::string, DTS>;

    // --- --- --- Generate a distance
    auto distance = distance_generator->generate(state, data, bcm);
    std::string transform_name = distance->get_transformation_name();

    // --- --- --- Access State
    // Get/Compute the index set matching 'bcm'
    const IndexSet& all_indexset = get_train_state->at(state).get_index_set(bcm);

    // --- --- --- Data access
    const DTS& train_dataset = at<MDTS>(data, "train_mdts").at(transform_name);

    // --- --- --- Splitter training algorithm
    // Pick on exemplar per class using the pseudo random number generator from the state
    ByClassMap train_bcm = bcm.pick_one_by_class(state.prng);
    IndexSet train_idxset = train_bcm.to_IndexSet();

    // Build return:
    //  Number of branches == number of classes
    //  We maintain mapping from labels to branch index
    //  We build the "by branch BCM" vector resulting from this snode
    const std::map<EL, size_t>& label_to_branchIdx = bcm.labels_to_index();
    std::vector<ByClassMap::BCMvec_t> result_bcm_vec(bcm.nb_classes());

    // For each incoming series (including selected train exemplars - will eventually form pure leaves)
    // Do 1NN classification, managing ties
    for (auto query_idx : all_indexset) {
      const auto& query = train_dataset[query_idx];
      EL query_label = train_dataset.label(query_idx).value();

      // 1NN variables, use a set to manage ties
      // Start with same class: better chance to have a tight cutoff
      std::set<EL> labels = {query_label};
      const size_t first_idx = train_bcm[query_label][0];
      F bsf = distance->eval(train_dataset[first_idx], query, utils::PINF);

      // Continue with other classes
      for (size_t candidate_idx : train_idxset) {
        if (candidate_idx!=first_idx) {
          const auto& candidate = train_dataset[candidate_idx];
          auto dist = distance->eval(candidate, query, bsf);
          if (dist<bsf) {
            labels.clear();
            labels.insert(train_dataset.label(candidate_idx).value());
            bsf = dist;
          } else if (bsf==dist) { labels.insert(train_dataset.label(candidate_idx).value()); }
        }
      }

      // Break ties and choose the branch according to the predicted label
      tempo::EL predicted_label;
      std::sample(labels.begin(), labels.end(), &predicted_label, 1, state.prng);
      size_t predicted_index = label_to_branchIdx.at(predicted_label);
      // The predicted label gives us the branch, but the BCM at the branch must contain the real label
      result_bcm_vec[predicted_index][query_label].push_back(query_idx);
    }

    // Convert the vector of ByClassMap::BCMvec_t in a vector of ByClassMap.
    // IMPORTANT: ensure that no empty BCM is generated
    // If we get an empty map, we have to add the  mapping (label for this index -> empty vector)
    // This ensures that no empty BCM is ever created. This is also why we iterate over the label: so we have them!
    std::vector<ByClassMap> v_bcm;
    for (EL label : bcm.classes()) {
      size_t idx = label_to_branchIdx.at(label);
      if (result_bcm_vec[idx].empty()) { result_bcm_vec[idx][label] = {}; }
      v_bcm.emplace_back(std::move(result_bcm_vec[idx]));
    }

    return i_GenNode::Result{
      .splitter = std::make_unique<SplitterNN1>(train_idxset, label_to_branchIdx, std::move(distance)),
      .branch_splits = std::move(v_bcm)
    };
  } // End of generate function

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
