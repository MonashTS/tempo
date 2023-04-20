#pragma once

#include <string>
#include <functional>

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>

#include <tempo/classifier/TSChief/treedata.hpp>
#include <tempo/classifier/TSChief/treestate.hpp>

namespace tempo::classifier::TSChief::snode::nn1splitter {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // NN1 Splitter: Interface for distance between two time series

  /// Interface for the distance component
  struct i_Dist {

    // --- --- --- Destructor/Constructor
    virtual ~i_Dist() = default;

    // --- --- --- Method
    /// Distance function computing a similarity score between two series, the lower the more similar.
    /// 'bsf' ('Best so far') allows early abandoning and pruning in a NN1 classifier (upper bound on the whole process)
    virtual F eval(TSeries const& t1, TSeries const& t2, F bsf) = 0;

    /// Name of the transformation to draw the data from
    virtual std::string get_transformation_name() = 0;

    virtual std::string get_distance_name() = 0;
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Time series distance generator interface

  /// Interface for the generation of an i_Dist
  struct i_GenDist {

    virtual ~i_GenDist() = default;

    virtual std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& data, const ByClassMap& bcm) = 0;

  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Helper for parameter generation (usually used as parameters for i_GenDist implementing-class constructors)

  /// Generate a transform name suitable for elastic distances
  using TransformGetter = std::function<std::string(TreeState& state)>;

  /// Generate an cfe e used in some elastic distances' cost function cost(a,b)=|a-b|^e
  using ExponentGetter = std::function<F(TreeState& state)>;

  /// Generic getter
  template<typename T>
  using T_GetterState = std::function<T(TreeState& state)>;

  /// Generate a warping window
  using WindowGetter = std::function<size_t(TreeState& state, TreeData const& data)>;

  /// Some distances (like ADTW, ERP and LCSS) generate a random value based on the dataset
  /// (requires 'data' and the dataset name, and 'bcm' for the local subset)
  using StatGetter = std::function<F(TreeState& state,
                                     TreeData const& data,
                                     ByClassMap const& bcm,
                                     std::string const& tn)>;

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
