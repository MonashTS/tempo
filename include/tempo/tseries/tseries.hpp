#pragma once

#include <tempo/utils/capsule.hpp>

#include <optional>
#include <stdexcept>
#include <vector>

namespace tempo {

    /** Representation of a time series.
     * Can own or not its data. In any case, copying is cheap as the actual data are not duplicated.
     * @tparam FloatType_ Type of values
     * @tparam LabelType_ Type of labels
     */
    template<typename FloatType_, typename LabelType_>
    class TSeries {
    public:
        using FloatType = FloatType_;
        using LabelType = LabelType_;
        static_assert(std::is_floating_point_v<FloatType>);
        static_assert(std::is_copy_constructible_v<LabelType>);
    private: // Order of field matters!

        size_t nb_dimensions_{};
        size_t length_{};
        bool has_missing_values_{};
        std::optional<LabelType> opt_label_{};

        /// Keeps the data alive when owning.
        /// Accessing the data through the capsule is too slow (required a runtime cast with type check).
        /// Use the 'public const FloatType* data' pointer instead.
        Capsule c_{}; // Remark: keep before *data - required for correct construction order.
        const FloatType *data_{nullptr};    /// Direct access to the raw data

    public:

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Constructor
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        /// Default constructor: create an empty univariate series
        TSeries() = default;

        /// Create a new TSeries owning its data
        TSeries(std::vector<FloatType> &&vec, size_t nb_dimensions, bool has_missing, std::optional<LabelType> label) :
                nb_dimensions_{nb_dimensions},
                has_missing_values_{has_missing},
                opt_label_{std::move(label)} {
            if (nb_dimensions < 1) {
                throw std::domain_error("nb_dimensions must be >= 1");
            }

            if (vec.size() % nb_dimensions != 0) {
                throw std::domain_error("Vector size is not a multiple of nb_dimensions");
            }
            length_ = vec.size() / nb_dimensions;
            data_ = vec.data();
            c_ = make_capsule<std::vector<FloatType>>(std::move(vec));
        }


        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        [[nodiscard]] inline const FloatType *data() const { return data_; }

        [[nodiscard]] inline size_t nb_dimensions() const { return nb_dimensions_; }

        [[nodiscard]] inline size_t length() const { return length_; }

        [[nodiscard]] inline bool has_missing_values() const { return has_missing_values_; }

        [[nodiscard]] inline std::optional<LabelType> get_label() const { return opt_label_; }

        /// Access to a give dimension
        [[nodiscard]] inline const FloatType *get_dim(size_t dim) const { return &(data_[length_ * dim]); }

        /// Access in (dim, idx) -- or (li, co) -- style
        [[nodiscard]] inline FloatType operator()(size_t dim, size_t idx) const { return get_dim(dim)[idx]; }
    };


} // End of namespace tempo