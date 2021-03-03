#pragma once

#include "tseries.hpp"
#include "transform.hpp"
#include <tempo/utils/capsule.hpp>

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
        std::string dataset_url{};
        std::size_t dataset_length{};

        // About its series
        std::size_t series_nbdim{};
        std::size_t series_length_min{};
        std::size_t series_length_max{};
        bool series_have_missing_value{};
        std::set<LabelType> labels{};

    public:

        DatasetHeader() = default;

        DatasetHeader(std::string datasetUrl, size_t datasetLength, size_t seriesNbdim, size_t seriesLengthMin,
                      size_t seriesLengthMax, bool seriesHaveMissingValue, const std::set<LabelType> &labels)
                : dataset_url(std::move(datasetUrl)), dataset_length(datasetLength), series_nbdim(seriesNbdim),
                  series_length_min(seriesLengthMin), series_length_max(seriesLengthMax),
                  series_have_missing_value(seriesHaveMissingValue), labels(labels) {}

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        [[nodiscard]] inline std::string get_dataset_url() const {return dataset_url;}
        [[nodiscard]] inline std::size_t get_dataset_length() const {return dataset_length;}

        [[nodiscard]] inline std::size_t get_nbdimensions() const {return series_nbdim;}
        [[nodiscard]] inline std::size_t get_length_min()  const {return series_length_min;}
        [[nodiscard]] inline std::size_t get_length_max()  const {return series_length_max;}
        [[nodiscard]] inline bool has_missing_values() const {return series_have_missing_value;}
        [[nodiscard]] inline const std::set<LabelType>& get_labels() const {return labels;}

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        template<typename FloatType>
        static DatasetHeader from(const std::vector<TSeries<FloatType, LabelType>> &series, std::string &&url) {
            DatasetHeader result;
            if (series.empty()) { return result; }
            else {
                result.dataset_url = std::forward<std::string>(url);
                result.dataset_length = series.size();
                const auto &s0 = series.front();
                // Initialization based on s0:
                result.series_nbdim = s0.nb_dimensions();
                result.series_length_min = s0.length();
                result.series_length_max = s0.length();
                result.series_have_missing_value = s0.has_missing_values();
                if (s0.get_label()) { result.labels.insert(s0.get_label().value()); }
                // Rest of the series
                for (const auto &s:series) {
                    if (s.nb_dimensions() != result.series_nbdim) {
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
    struct Dataset {
        using FloatType = FloatType_;
        using LabelType = LabelType_;
        static_assert(std::is_floating_point_v<FloatType>);
        static_assert(std::is_copy_constructible_v<LabelType>);
        using TS = TSeries<FloatType, LabelType>;
        using DH = DatasetHeader<LabelType>;
        template<typename T>
        using TH = TransformHandle<T, FloatType, LabelType>;

    private:
        DH header{};
        std::vector<Transform> transforms{};
        TH<std::vector<TS>> original_handle;

    public:

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Constructor
        Dataset() = default;

        Dataset(std::vector<TS> &&series, DatasetHeader<LabelType> h) : header(h) {
            original_handle = add_transform("original", {}, std::move(series));
        }

        Dataset(std::vector<TS> &&series, std::string &&url) {
            original_handle = add_transform("original", {}, std::move(series));
            header = DH::from(get_original(), std::move(url));
        }


        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        /** Add a new transform in the dataset.
         * @tparam T Type of the data making the transform
         * @param name  Name of the transform
         * @param parents_name Name of the parent of the transform
         * @param transform_data Actual data of the transform
         * @return A handle on the transform
         */
        template<typename T>
        TH<T> add_transform(std::string name, std::vector<std::string> parents_name, T &&transform_data) {
            // Create data
            Capsule c = make_capsule<T>(std::forward<T>(transform_data));
            void *ptr = (void *) capsule_ptr<T>(c);
            // Create the transform
            transforms.template emplace_back(std::move(name), std::move(parents_name), std::move(c), ptr);
            // Create the handle
            return TH<T>(this, transforms.size() - 1);
        }

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        [[nodiscard]] inline const DH& get_header() const {return header;}

        [[nodiscard]] inline const std::vector<Transform>& get_transforms() const {return transforms;}
        [[nodiscard]] inline const Transform& get_transform(size_t idx) const {return transforms[idx];}

        [[nodiscard]] inline const std::vector<TS>& get_original() const { return original_handle.get(); }
        [[nodiscard]] inline const TH<std::vector<TS>>& get_original_handle() const {return original_handle;}

    };


} // End of namespace tempo