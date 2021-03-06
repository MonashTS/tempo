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
            c_ = make_capsule<std::vector<FloatType>>(std::move(vec));
            const auto* ptr = capsule_ptr<std::vector<FloatType>>(c_);
            data_ = ptr->data();
        }

        /// Create a new TSeries owning its data, copying its information from "other".
        TSeries(std::vector<FloatType> &&vec, const TSeries<FloatType, LabelType>& other)
        :TSeries(std::move(vec), other.nb_dimensions_, other.has_missing_values_, other.opt_label_){}


      /** Constructor with a raw pointer.
       *  The new instance relies on the raw pointer and does not directly manage the memory.
       *  However, by providing a "capsule", TSeries can maintain a reference on the actual memory owner,
       *  preventing collection while alive.
       * @param data_ptr         Pointer to the data
       * @param length           Length of the dimension (not the total size of the buffer!)
       * @param nb_dimensions    Number of dimension. The size of the buffer should be nb_dimensions * length.
       * @param has_missing      Is there any missing data?
       * @param label            The label, optional.
       * @param capsule          Allow to maintain a reference on the actual storage
       */
      TSeries(const FloatType *data_ptr, size_t length, size_t nb_dimensions, bool has_missing,
        std::optional<LabelType> label, Capsule capsule) :
        nb_dimensions_(nb_dimensions),
        length_(length),
        has_missing_values_(has_missing),
        opt_label_(std::move(label)),
        c_(std::move(capsule)),
        data_(data_ptr){

        if ((length == 0) ^ (data_ptr == nullptr)) {
          throw std::domain_error("A length of 0 requires a null pointer, and vice versa");
        }

        if (nb_dimensions < 1) {
          throw std::domain_error("nb_dimensions must be >= 1");
        }
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