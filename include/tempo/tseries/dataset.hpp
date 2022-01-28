#pragma once

#include "tseries.hpp"
#include "transform.hpp"
#include <tempo/utils/capsule.hpp>
#include <tempo/utils/jsonvalue.hpp>

#include <stdexcept>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace tempo {

  /** Dataset header. Contains basic info about the dataset. */
  template<typename LabelType_>
  struct DatasetHeader {
    using LabelType = LabelType_;
    static_assert(std::is_copy_constructible_v<LabelType>);
  private:
    // About the dataset
    std::string dataset_identifier{};
    std::size_t dataset_length{};

    // About its series
    std::size_t series_nbdim{};
    std::size_t series_length_min{};
    std::size_t series_length_max{};
    bool series_have_missing_value{};
    std::set<LabelType> labels{};

  public:

    DatasetHeader() = default;

    DatasetHeader(std::string datasetIdentifier, size_t datasetLength, size_t seriesNbdim, size_t seriesLengthMin,
      size_t seriesLengthMax, bool seriesHaveMissingValue, const std::set<LabelType>& labels)
      :dataset_identifier(std::move(datasetIdentifier)), dataset_length(datasetLength), series_nbdim(seriesNbdim),
       series_length_min(seriesLengthMin), series_length_max(seriesLengthMax),
       series_have_missing_value(seriesHaveMissingValue), labels(labels) { }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    [[nodiscard]] inline std::string get_identifier() const { return dataset_identifier; }

    [[nodiscard]] inline std::size_t get_size() const { return dataset_length; }

    [[nodiscard]] inline std::size_t get_ndim() const { return series_nbdim; }

    [[nodiscard]] inline std::size_t get_minl() const { return series_length_min; }

    [[nodiscard]] inline std::size_t get_maxl() const { return series_length_max; }

    [[nodiscard]] inline bool has_missing_values() const { return series_have_missing_value; }

    [[nodiscard]] inline const std::set<LabelType>& get_labels() const { return labels; }

    /// Get header as a JSON object
    [[nodiscard]] json::JSONValue to_json() const {
      using json::JSONValue;
      return JSONValue({
        {"identifier",    JSONValue(get_identifier())},
        {"nb_series",     JSONValue((double) get_size())},
        {"nb_dimensions", JSONValue((double) get_ndim())},
        {"min_length",    JSONValue((double) get_minl())},
        {"max_length",    JSONValue((double) get_maxl())},
        {"has_missing",   JSONValue(has_missing_values())},
        {"labels",        JSONValue(get_labels().begin(), get_labels().end())}
      });
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    template<typename FloatType>
    static DatasetHeader from(const std::vector<TSeries<FloatType, LabelType>>& series, std::string&& url) {
      DatasetHeader result;
      if (series.empty()) { return result; }
      else {
        result.dataset_identifier = std::forward<std::string>(url);
        result.dataset_length = series.size();
        const auto& s0 = series.front();
        // Initialization based on s0:
        result.series_nbdim = s0.nb_dimensions();
        result.series_length_min = s0.length();
        result.series_length_max = s0.length();
        result.series_have_missing_value = s0.has_missing_values();
        if (s0.get_label()) { result.labels.insert(s0.get_label().value()); }
        // Rest of the series
        for (const auto& s:series) {
          if (s.nb_dimensions()!=result.series_nbdim) {
            throw std::domain_error("Series from a dataset must have the same number of dimensions");
          }
          result.series_length_min = std::min<FloatType>(result.series_length_min, s.length());
          result.series_length_max = std::max<FloatType>(result.series_length_max, s.length());
          result.series_have_missing_value = result.series_have_missing_value || s.has_missing_values();
        }
        //
        return result;
      }
    }
  };

  /** A dataset containing transforms.
   *  By convention, the first transform (index 0) is the original collection of time series,
   *  and we provide an easy access to it through the "original" function.
   *
   *  Note that the original data are stored as a vector of time series.
   *  We can rely on this, i.e. a given series is identified by its index in this vector.
   */
  template<typename FloatType_, typename LabelType_>
  struct Dataset : std::enable_shared_from_this<Dataset<FloatType_, LabelType_>> {
    using FloatType = FloatType_;
    using LabelType = LabelType_;
    static_assert(std::is_floating_point_v<FloatType>);
    static_assert(std::is_copy_constructible_v<LabelType>);
    using TS = TSeries<FloatType, LabelType>;
    using DH = DatasetHeader<LabelType>;
    using Self = Dataset<FloatType, LabelType>;
    template<typename T>
    using TH = TransformHandle<T, FloatType, LabelType>;

  private:
    DH header{};
    std::vector<Transform> transforms{};
    TH<std::vector<TS>> original_handle;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructors
    Dataset() = default;

    Dataset(std::vector<TS>&& series, DatasetHeader<LabelType> h)
      :header(h) {
      original_handle = add_transform("original", {}, std::move(series));
    }

    Dataset(std::vector<TS>&& series, std::string&& identifier) {
      original_handle = add_transform("original", {}, std::move(series));
      header = DH::from(get_original(), std::move(identifier));
    }

  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // --- Delete copy and move constructor: handler are referring to the dataset by pointer, don't move it!
    Dataset(const Self& other) = delete;

    Dataset(Self&& other) = delete;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Factories

    /// Create a new dataset from a collection of series (taking ownership) and a header
    [[nodiscard]] static std::shared_ptr<Self> make(std::vector<TS>&& series, DatasetHeader<LabelType> h) {
      auto ptr = new Dataset(std::move(series), std::move(h));
      return std::shared_ptr<Self>(ptr);
    }

    /// Create a new dataset from a collection of series (taking ownership) and an identifier. Computes the header.
    [[nodiscard]] static std::shared_ptr<Self> make(std::vector<TS>&& series, std::string&& identifier) {
      auto ptr = new Dataset(std::move(series), std::move(identifier));
      return std::shared_ptr<Self>(ptr);
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Add a transform in the dataset **/
    template<typename T>
    TH<T> add_transform(Transform&& tr) {
      transforms.template emplace_back(std::move(tr));
      return TH<T>(this, transforms.size()-1);
    }

    /** Add a new transform in the dataset.
     * @tparam T Type of the data making the transform
     * @param name  Name of the transform
     * @param parents_name Name of the parent of the transform
     * @param transform_data Actual data of the transform
     * @return A handle on the transform
     */
    template<typename T>
    TH<T> add_transform(std::string name, std::vector<std::string> parents_name, T&& transform_data) {
      // Create data
      Capsule c = make_capsule<T>(std::forward<T>(transform_data));
      void* ptr = (T*) capsule_ptr<T>(c);
      // Create the transform
      transforms.emplace_back(std::move(name), std::move(parents_name), std::move(c), ptr);
      // Create the handle
      return TH<T>(this, transforms.size()-1);
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    [[nodiscard]] const DH& get_header() const { return header; }

    [[nodiscard]] const std::vector<Transform>& get_transforms() const { return transforms; }

    [[nodiscard]] const Transform& get_transform(size_t idx) const { return transforms[idx]; }

    [[nodiscard]] const std::vector<TS>& get_original() const { return original_handle.get(); }

    [[nodiscard]] const TH<std::vector<TS>>& get_original_handle() const { return original_handle; }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Shortcuts
    [[nodiscard]] size_t size() const { return header.get_size(); }

    [[nodiscard]] const TS& get(size_t index) const { return get_original()[index]; }

    [[nodiscard]] const TS& operator[](size_t index) const { return get(index); }

  };


} // End of namespace tempo