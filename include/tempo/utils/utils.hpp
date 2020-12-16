#pragma once

#include <sys/types.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace tempo {


    // --- --- --- --- --- ---
    // --- Series constants
    // --- --- --- --- --- ---

    /// Constant to be use when no window is required
    constexpr size_t NO_WINDOW{std::numeric_limits<size_t>::max()};

    /// Constant representing the maximum length allowed for a series.
    /// Account for extra columns/lines that may need to be allocated and representing the window.
    constexpr size_t MAX_SERIES_LENGTH{NO_WINDOW - 2};

    // --- --- --- --- --- ---
    // --- Floating point constants
    // --- --- --- --- --- ---

    /// Positive infinity for float types
    template<typename FloatType>
    constexpr FloatType POSITIVE_INFINITY{std::numeric_limits<FloatType>::infinity()};

    /// Negative infinity for float types
    template<typename FloatType>
    constexpr FloatType NEGATIVE_INFINITY{-POSITIVE_INFINITY<FloatType>};

    /// Not A Number
    template<typename FloatType>
    constexpr FloatType DNAN{std::numeric_limits<FloatType>::quiet_NaN()};


    // --- --- --- --- --- ---
    // --- Tooling
    // --- --- --- --- --- ---

    /// Minimum of 3 values using std::min<T>
    template<typename T>
    [[nodiscard]] inline T min(T a, T b, T c) { return std::min<T>(a, std::min<T>(b, c)); }

    /// Maximum of 3 values using std::min<T>
    template<typename T>
    [[nodiscard]] inline T max(T a, T b, T c) { return std::max<T>(a, std::max<T>(b, c)); }


    // --- --- --- --- --- ---
    // --- Unsigned arithmetic
    // --- --- --- --- --- ---

    /** Unsigned arithmetic:
     * Given an 'index' and a 'window', get the start index corresponding to std::max(0, index-window) */
    [[nodiscard]] inline size_t cap_start_index_to_window(size_t index, size_t window) {
        if (index > window) { return index - window; } else { return 0; }
    }

    /** Unsigned arithmetic:
     * Given an 'index', a 'window' and an 'end', get the stop index corresponding to std::min(end, index+window+1).
     * The expression index+window+1 is illegal for any index>0 as window could be MAX-1
     * */
    [[nodiscard]] inline size_t
    cap_stop_index_to_window_or_end(size_t index, size_t window, size_t end) {
        // end-window is valid when window<end
        if (window < end && index + 1 < end - window) { return index + window + 1; } else { return end; }
    }

    /** Absolute value for any comparable and subtractive type, without overflowing risk for unsigned types. */
    template<typename T>
    [[nodiscard]] inline T absdiff(T a, T b) { return (a > b) ? a - b : b - a; }

    /** From unsigned to signed for integral types*/
    template<typename UIType>
    [[nodiscard]] inline typename std::make_signed_t<UIType> to_signed(UIType ui) {
        static_assert(std::is_unsigned_v<UIType>, "Template parameter must be an unsigned type");
        using SIType = std::make_signed_t<UIType>;
        if (ui > (UIType) (std::numeric_limits<SIType>::max())) {
            throw std::overflow_error("Cannot store unsigned type in signed type.");
        }
        return (SIType) ui;
    }

    // --- --- --- --- --- ---
    // --- Random tools
    // --- --- --- --- --- ---

    /** Generate a vector of a given size with random real values in the half-closed interval [min, max[.
     *  Use a provided random number generator. */
    template<typename T=double, typename PRNG>
    [[nodiscard]] std::vector<T> generate_random_real_vector(PRNG &prng, size_t size, T min, T max) {
        std::uniform_real_distribution<T> udist{min, max};
        auto generator = [&udist, &prng]() { return udist(prng); };
        std::vector<T> v(size);
        std::generate(v.begin(), v.end(), generator);
        return v;
    }

    /** Generate a vector of a given size with random integer values in the closed interval [min, max].
     *  Use a provided random number generator. */
    template<typename T=int, typename PRNG>
    [[nodiscard]] std::vector<T> generate_random_integer_vector(PRNG &prng, size_t size, T min, T max) {
        std::uniform_int_distribution<T> udist{min, max};
        auto generator = [&udist, &prng]() { return udist(prng); };
        std::vector<T> v(size);
        std::generate(v.begin(), v.end(), generator);
        return v;
    }

    // --- --- --- --- --- ---
    // --- Should not happen
    // --- --- --- --- --- ---

    /// Throw an exception "should not happen". Used as default case in switchs.
    [[noreturn]] void inline should_not_happen(){ throw std::logic_error("Should not happen"); }

    // --- --- --- --- --- ---
    // --- Initialisation tool
    // --- --- --- --- --- ---

    namespace initBlock_detail {
        struct tag { };

        template <class F>
        decltype(auto) operator + (tag, F &&f) {
            return std::forward<F>(f)();
        }
    }

#define initBlock initBlock_detail::tag{} + [&]() -> decltype(auto)

} // end of namespace tempo

