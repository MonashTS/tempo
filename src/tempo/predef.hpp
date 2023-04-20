#pragma once

// STD C LIB
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>

// STD C++ LIB
#include <algorithm>
#include <any>
#include <chrono>
#include <concepts>
#include <filesystem>
#include <functional>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

// MLPACK which also gives us Armadillo
//#include <mlpack/core.hpp>

// JSONCPP
#include <nlohmann/json.hpp>

namespace tempo {

  using LabelType = std::string;
  using L = LabelType;

  using EncodedLabelType = size_t;
  using EL = EncodedLabelType;

  using FloatType = double;
  using F = FloatType;

  using PRNG = std::mt19937_64;

} // End of namespace tempo