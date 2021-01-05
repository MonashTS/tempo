#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "../utils/utils.hpp"
#include "../utils/uncopyable.hpp"

namespace tempo {

    /** Helper for times series data. Prevent us from implicitly copying the series.
     *  Can own or not the underlying data.
     *  Once created, a series cannot be modified.
      * @tparam FloatType   Type of the values of the series. Must be a floating point type with NAN support.
      * @tparam LabelType   Type of the label. Must be copy-constructible.
     */
    template<typename FloatType, typename LabelType>
    class TSeries: private Uncopyable {
        static_assert(std::is_floating_point_v<FloatType>);
        static_assert(std::is_copy_constructible_v<LabelType>);
    protected:

        /// When owning: backend storage
        std::vector<FloatType> data_v_{};

        /// Pointer on the backend
        const FloatType* data_{nullptr};

        /// Length of the series (if multivariate, same length for each "dimension")
        size_t length_{0};

        /// Number of "dimensions" (series are always 2D, so this is the number of "tracks")
        size_t nbdim_{1};

        /// Record if the series has any missing data, represented by NaN
        bool has_missing_{false};

        /// Record if the series was given a label
        std::optional<LabelType> label_{};

    public:

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Constructors & factories & destructor
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        /// Default constructor: create an empty univariate series
        TSeries():data_(data_v_.data()){}

        /** Constructor taking ownership of a vector<FloatType>.
         * The length is computed based on the vector's length and the number of dimensions.
         * In other words, data.size() must be a multiple of nb_dimensions.
         * @param data              Vector of FloatType, moved into the new TSeries instance
         * @param nb_dimensions     1 for univariate, more than 1 for multivariate.
         * @param has_missing       Set the flag. Not checked against the data (so you better be right).
         * @param label             Optional class label
         */
        TSeries(std::vector<FloatType> &&data, size_t nb_dimensions, bool has_missing, std::optional<LabelType> label)
                : data_v_(std::move(data)), nbdim_(nb_dimensions), has_missing_(has_missing),
                  label_(std::move(label)) {

            if(nb_dimensions<1){
                throw std::domain_error("nb_dimensions must be >= 1");
            }

            if(data_v_.size()%nb_dimensions != 0){
                throw std::domain_error("Vector size is not a multiple of nb_dimensions");
            }

            length_ = data_v_.size() / nbdim_;
            data_v_.shrink_to_fit();
            data_ = data_v_.data();
        }

        /** Constructor taking ownership of a vector<FloatType>.
         * Same as above, but check data for missing data.
         * The length is computed based on the vector's length and the number of dimensions.
         * In other words, data.size() must be a multiple of nb_dimensions.
         * @param data              Vector of FloatType, moved into the new TSeries instance
         * @param nb_dimensions     1 for univariate, more than 1 for multivariate.
         * @param label             Optional class label
         */
        TSeries(std::vector<FloatType> &&data, size_t nb_dimensions, const std::optional<LabelType>& label):
                TSeries(std::move(data), nb_dimensions, false, label) {
            has_missing_ = std::any_of(data_, data_+nbdim_*length_, std::isnan);
        }

        /** Constructor taking ownership of a vector<FloatType>, copying information from another series.
         * Convenient when transforming series.
         * @param data
         * @param info_source
         */
        TSeries(std::vector<FloatType> &&data, const TSeries<FloatType, LabelType>& info_source) :
                TSeries(std::move(data), info_source.nbdim_, info_source.has_missing_, info_source.label_){ }

        /** Constructor with a raw pointer.
         *  The new instance does not own the data and will not free the memory when destroyed. */
        TSeries(const FloatType *data_ptr, size_t length, size_t nb_dimensions, bool has_missing, std::optional<LabelType> label)
                : data_(data_ptr), length_(length), nbdim_(nb_dimensions), has_missing_(has_missing),
                  label_(std::move(label)) {

            if(length == 0 ^ data_ptr != nullptr) {
                throw std::domain_error("A length of 0 requires a null pointer, and vice versa");
            }

            if(nb_dimensions<1){
                throw std::domain_error("nb_dimensions must be >= 1");
            }

            if(data_v_.size()%nb_dimensions != 0){
                throw std::domain_error("Vector size is not a multiple of nb_dimensions");
            }
        }

        /** Factory function creating a non-owning series from another one.
         *  Be sure thath the backing instance lives longer than the new one!
         */
        static TSeries<FloatType, LabelType> from(const TSeries<FloatType, LabelType>& backing){
            return std::move(TSeries<FloatType, LabelType>(
                    backing.data_, backing.length_, backing.nbdim_, backing.has_missing_, backing.label_
            ));
        }

        /** Default destructor, automatically free the backend when owning. */
        ~TSeries() = default;

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Movement
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        /// Move-constructor
        TSeries(TSeries &&other) noexcept {
            // Use the move-assignment operator
            *this = std::move(other);
        }

        /// Move-assignment
        TSeries &operator=(TSeries &&other) noexcept {
            if (this != &other) {
                if (other.is_owning()) {
                    data_v_ = std::move(other.data_v_);
                    data_ = data_v_.data();
                } else {
                    data_ = other.data_;
                }
                nbdim_ = other.nbdim_;
                length_ = other.length_;
                has_missing_ = other.has_missing_;
                label_ = std::move(other.label_);
            }
            return *this;
        }

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Methods
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        [[nodiscard]] inline size_t length() const { return length_; }

        [[nodiscard]] inline size_t nb_dimensions() const { return nbdim_; }

        [[nodiscard]] inline bool has_missing() const { return has_missing_; }

        [[nodiscard]] inline const std::optional<LabelType> &label() const { return label_; }

        [[nodiscard]] inline const double *data() const { return data_; }

        /// Access to the start of a dimension (first dimension is 0)
        [[nodiscard]] inline const double *operator[](size_t dim) const { return data_ + (nbdim_ * dim); }

        /// Access a value using a pair of coordinate (Dimension,index)
        [[nodiscard]] inline double operator()(size_t dim, size_t idx) const {
            return *(data_ + (nbdim_ * dim + idx));
        }

        /// return true if the series is owning its data
        [[nodiscard]] inline bool is_owning() const { return data_v_.data() == data_; }


        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Comparison
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        [[nodiscard]] friend inline bool operator==(const TSeries &lhs, const TSeries &rhs) {
            bool res1 = lhs.nb_dimensions() == rhs.nb_dimensions()
                        && lhs.length() == rhs.length()
                        && lhs.has_missing() == rhs.has_missing()
                        && lhs.label() == rhs.label();
            if (res1) {
                const auto *ld = lhs.data();
                const auto *rd = rhs.data();
                if (ld == rd) {
                    return true; // Same pointer, so all good
                } else { // Else, compare item one by one
                    bool same = true;
                    size_t index = 0;
                    while (same && index < lhs.length()) {
                        same = ld[index] == rd[index];
                        ++index;
                    }
                    return same;
                }
            } else {
                return false;
            }
        }


        [[nodiscard]] friend inline bool operator!=(const TSeries &lhs, const TSeries &rhs) { return !operator==(lhs, rhs); }
    };


} // End of namespace tempo
