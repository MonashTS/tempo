#pragma once

#include "../utils/capsule.hpp"

#include <optional>

namespace tempo {

    /** Representation of a time series.
     * Can own or not its data. In any case, copying is cheap as the actual data are not duplicated.
     * @tparam FloatType_ Type of values
     * @tparam LabelType_ Type of labels
     */
    template <typename FloatType_, typename LabelType_>
    class TSeries {
    public:
        using FloatType = FloatType_;
        using LabelType = LabelType_;
        static_assert(std::is_floating_point_v<FloatType>);
        static_assert(std::is_copy_constructible_v<LabelType>);
    private:
        /// Keeps the data alive.
        /// Accessing the data through the capsule is too slow (required a runtime cast with type check).
        /// Use the 'public const FloatType* data' pointer instead.
        Capsule c{};

    public:

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Constructor
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        const FloatType* data{nullptr};    /// Direct access to the raw data
        const size_t nbdim{};
        const size_t length{};
        const bool has_missing_value{};
        const std::optional<LabelType> mb_label{};

        [[nodiscard]] inline FloatType operator(size_t dim, size_t idx){ return data[length*dim + idx]; }

    };

} // End of namespace tempo