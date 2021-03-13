#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/utils/capsule.hpp>

#include <any>

namespace tempo {

    /** Helper for times series data.
     *  Can own or not the underlying data.
     *  When owning, data are behind a shared pointer, allowing for cheap copies.
     *  Once created, a series cannot be modified.
      * @tparam FloatType   Type of the values of the series. Must be a floating point type with NAN support.
      * @tparam LabelType   Type of the label. Must be copy-constructible.
     */
    template<typename FloatType, typename LabelType>
    class TSeries {
        static_assert(std::is_floating_point_v<FloatType>);
        static_assert(std::is_copy_constructible_v<LabelType>);
    protected:

        /// Pointer on the data
        const FloatType *data_{nullptr};

        /// Length of the series (if multivariate, same length for each "dimension")
        size_t length_{0};

        /// Number of "dimensions" (series are always 2D, so this is the number of "tracks")
        size_t nbdim_{1};

        /// Record if the series has any missing data, represented by NaN
        bool has_missing_{false};

        /// Record if the series was given a label
        std::optional<LabelType> label_{};

        /// When owning (backend storage) or holding reference
        std::shared_ptr<std::any> capsule_{};

    public:

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Constructors & factories & destructor
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        /// Default constructor: create an empty univariate series
        TSeries() = default;

        /** Constructor taking ownership of a vector<FloatType>.
         * The length is computed based on the vector's length and the number of dimensions.
         * In other words, data.size() must be a multiple of nb_dimensions.
         * @param data              Vector of FloatType, moved into the new TSeries instance
         * @param nb_dimensions     1 for univariate, more than 1 for multivariate.
         * @param has_missing       Set the flag. Not checked against the data (so you better be right).
         * @param label             Optional class label
         */
        TSeries(std::vector<FloatType> &&data, size_t nb_dimensions, bool has_missing, std::optional<LabelType> label) :
                data_(data.data()),
                nbdim_(nb_dimensions),
                has_missing_(has_missing),
                label_(std::move(label)) {

            if (nb_dimensions < 1) {
                throw std::domain_error("nb_dimensions must be >= 1");
            }

            if (data.size() % nb_dimensions != 0) {
                throw std::domain_error("Vector size is not a multiple of nb_dimensions");
            }

            length_ = data.size() / nbdim_;
            capsule_ = make_capsule<std::vector<FloatType>>(std::move(data));
            auto* ptr = capsule_ptr<std::vector<FloatType>>(capsule_);
            data_ = ptr->data();
        }

        /** Constructor taking ownership of a vector<FloatType>.
         * Same as above, but check data for missing data.
         * The length is computed based on the vector's length and the number of dimensions.
         * In other words, data.size() must be a multiple of nb_dimensions.
         * @param data              Vector of FloatType, moved into the new TSeries instance
         * @param nb_dimensions     1 for univariate, more than 1 for multivariate.
         * @param label             Optional class label
         */
        TSeries(std::vector<FloatType> &&data, size_t nb_dimensions, const std::optional<LabelType> &label) :
                TSeries(std::move(data), nb_dimensions, false, label) {
            has_missing_ = std::any_of(data_, data_ + nbdim_ * length_, std::isnan);
        }

        /** Constructor taking ownership of a vector<FloatType>, copying information from another series.
         * Convenient when transforming series.
         * @param data
         * @param info_source
         */
        TSeries(std::vector<FloatType> &&data, const TSeries<FloatType, LabelType> &info_source) :
                TSeries(std::move(data), info_source.nbdim_, info_source.has_missing_, info_source.label_) {}

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
                std::optional<LabelType> label, std::shared_ptr<std::any> capsule) :
                data_(data_ptr),
                length_(length),
                nbdim_(nb_dimensions),
                has_missing_(has_missing),
                label_(std::move(label)),
                capsule_(std::move(capsule)) {

            if ((length == 0) ^ (data_ptr == nullptr)) {
                throw std::domain_error("A length of 0 requires a null pointer, and vice versa");
            }

            if (nb_dimensions < 1) {
                throw std::domain_error("nb_dimensions must be >= 1");
            }
        }

        /** Factory function creating a non-owning series from another one.
         *  Be sure thath the backing instance lives longer than the new one!
         */
        static TSeries<FloatType, LabelType> from(const TSeries<FloatType, LabelType> &backing) {
            return TSeries<FloatType, LabelType>(
                    backing.data_, backing.length_, backing.nbdim_, backing.has_missing_, backing.label_
            );
        }

        /** Default destructor, automatically free the backend when owning. */
        ~TSeries() = default;

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Methods
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        [[nodiscard]] inline size_t length() const { return length_; }

        [[nodiscard]] inline size_t nb_dimensions() const { return nbdim_; }

        [[nodiscard]] inline size_t size() const { return length_ * nbdim_; }

        [[nodiscard]] inline bool has_missing() const { return has_missing_; }

        [[nodiscard]] inline const std::optional<LabelType> &label() const { return label_; }

        [[nodiscard]] inline const FloatType *data() const { return data_; }

        /// Access a value using a pair of coordinate (Dimension,index)
        [[nodiscard]] inline FloatType operator()(size_t dim, size_t idx) const {
            return *(data_ + (length_ * dim + idx));
        }

        /// Return true if the series is owning its data
        [[nodiscard]] inline bool is_owning() const { return capsule_->has_value(); }

        /** Attempt to extends the series with a new dimension (always added as the last one).
         *  - Take ownership of the vector
         *  - The series must own its data
         *  - The added dimension must be of the same length as the series
         *  Throw a std::logic_error if the condition are not met.
         *  Note: previous reference do data() must be considered invalid
         */
         void push_dimension(std::vector<FloatType>&& new_dim, bool has_missing) {
             if(new_dim.size() != length_){
                 throw std::logic_error("Pushing new dimension with length " + new_dim.size() + " != " + length_);
             }
             if(capsule_->has_value()){
                 std::vector<FloatType>& v{};
                 try {
                        v = std::any_cast<std::vector<FloatType>>(*capsule_);
                 } catch (...) { throw std::logic_error("The series does not own its data"); }
                 v.insert(v.end(), std::make_move_iterator(new_dim.begin()), std::make_move_iterator(new_dim.end()));
                 data_ = v.data(); // Get the pointer, as we may have reallocated
                 nbdim_++;
                 has_missing_ = has_missing_ || has_missing;
             } else {
                 throw std::logic_error("The series does not own its data");
             }
         }


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
                    return std::equal(ld, ld+lhs.length(), rd);
                }
            } else {
                return false;
            }
        }

        [[nodiscard]] friend inline bool operator!=(const TSeries &lhs, const TSeries &rhs) {
            return !operator==(lhs, rhs);
        }
    };

} // End of namespace tempo
