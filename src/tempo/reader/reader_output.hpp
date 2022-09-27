#pragma once

#include <concepts>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <set>

namespace tempo::reader {

  /// Encoded label type
  using EL = size_t;

  /// Label type
  using L = std::string;

  /** Label Encoder
   *  Allow to manipulate the 'EL' (an integral type) instead of strings.
   *  This allows to represent labels with simple vector - particularly useful for python/numpy bindings
   */
  class LabelEncoder {

    /// Set of labels for the dataset, with index encoding
    std::map<L, EL> _label_to_index;

    /// Reverse mapping index top label - Note: EL must be an integral type for indexing purposes
    std::vector<L> _index_to_label;

    /// Helper function: create the mapping structures with the content from the set _labels.
    /// Also use as an update function if the collection of labels has been *extended*.
    template<typename Collection>
    void update(Collection const& clabels) {
      std::set<std::string> ulabels(std::begin(clabels), std::end(clabels));
      size_t idx = _index_to_label.size(); // Next index == size of the vector
      for (L const& k : ulabels) {
        if (!_label_to_index.contains(k)) {
          _label_to_index[k] = idx;
          _index_to_label.push_back(k);
          ++idx;
        }
      }
    }

  public:

    LabelEncoder() = default;

    LabelEncoder(LabelEncoder const& other) = default;

    LabelEncoder& operator =(LabelEncoder const& other) = default;

    LabelEncoder(LabelEncoder&& other) = default;

    LabelEncoder& operator =(LabelEncoder&& other) = default;

    /// Create a new encoder from an iterable collection of labels
    template<typename Collection>
    explicit LabelEncoder(Collection const& labels) { update<Collection>(labels); }

    /// Copy other into this, then add unknown labels from the iterable collection 'labels'
    template<typename Collection>
    LabelEncoder(LabelEncoder other, Collection const& labels) : LabelEncoder(std::move(other)) {
      update<Collection>(labels);
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Labels to indexes encoding (reverse from index_to_label)
    inline std::map<L, EL> const& label_to_index() const { return _label_to_index; }

    /// Indexes to labels encoding (reverse from label_to_index)
    inline std::vector<L> const& index_to_label() const { return _index_to_label; }

    /// Encode a label
    inline EL encode(L const& l) const { return _label_to_index.at(l); }

    /// Decode a label
    inline L decode(EL el) const { return _index_to_label[el]; }
  };

  template<std::floating_point F>
  struct ResultData {

    /// Storing the data of a series as a vector - gives contiguous block of memory
    using SData = std::vector<F>;

    /// The collection of series from the dataset; a series is represented by its index.
    std::vector<SData> series;

    /// Optionally, if series have labels, we have a vector of encoded label (matching series by index)
    /// See the LabelEncoder 'encoder'
    std::optional<std::vector<std::string>> labels;

    /// Label Encoder
    LabelEncoder encoder;

    /// Indexes of the series containing NaN
    std::set<size_t> series_with_nan;

    /// Smallest length read
    size_t length_min{};

    /// Largest length read
    size_t length_max{};

    /// Number of dimensions
    size_t nb_dimensions{};

    // --- --- ---

    // Do not copy me!
    ResultData(ResultData const& other) = delete;
    ResultData& operator =(ResultData const& other) = delete;

    // Move me instead!
    ResultData(ResultData&&) noexcept = default;
    ResultData& operator =(ResultData&&) noexcept = default;

    ResultData() = default;

  };

  /// Result type for the readers; any error shall be written in the variant's string
  template<std::floating_point F>
  using Result = std::variant<std::string, ResultData<F>>;

} // End of namespace tempo