#pragma once

#include <cstddef>
#include <limits>

namespace tempo {

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

    /** Absolute value for any comparable and subtractive type, without overflowing risk for unsigned types.
     *  Also work for signed type. */
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

} // End of namespace tempo
