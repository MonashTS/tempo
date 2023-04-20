#pragma once

#include "nn1dist_interface.hpp"

#include <string>

namespace tempo::classifier::TSChief::snode::nn1splitter {

  /// Base implementation for i_Dist taking care of the transformation name.
  struct BaseDist : public i_Dist {

    // --- --- --- Destructor/Constructor

    explicit BaseDist(std::string str) : transformation_name(std::move(str)) {}

    ~BaseDist() override = default;

    // --- --- --- Method

    /// Store the name of the transform
    std::string transformation_name;

    /// Name of the transformation to draw the data from
    std::string get_transformation_name() override { return transformation_name; }
  };

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
