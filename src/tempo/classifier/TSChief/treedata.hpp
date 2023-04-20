#pragma once

#include <tempo/classifier/utils.hpp>
#include <tempo/dataset/dts.hpp>

#include <any>
#include <memory>
#include <utility>
#include <vector>

namespace tempo::classifier::TSChief {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Data management

  struct TreeData {
    std::map<std::string, std::shared_ptr<void>> storage;

    template<typename Data>
    void register_data(std::shared_ptr<Data> sptr, std::string const& key) {
      storage[key] = std::move(sptr);
    }
  };

  template<typename Data>
  Data const& at(TreeData const& td, const std::string& key){
    return *std::static_pointer_cast<Data>(td.storage.at(key)).get();
  }

  // --- --- --- Helpers for named transforms

  using MDTS = std::map<std::string, tempo::DTS>;

  inline void register_train(TreeData& td, std::shared_ptr<MDTS> sptr){
    td.register_data<MDTS>(std::move(sptr), "train_mdts");
  }

  inline void register_test(TreeData& td, std::shared_ptr<MDTS> sptr){
    td.register_data<MDTS>(std::move(sptr), "test_mdts");
  }

  inline MDTS const& at_train(TreeData const& td){ return at<MDTS>(td, "train_mdts"); }

  inline MDTS const& at_test(TreeData const& td){ return at<MDTS>(td, "test_mdts"); }

} // End of tempo::classifier::PF2