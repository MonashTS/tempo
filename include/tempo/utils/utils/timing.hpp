#pragma once

#include <chrono>

namespace tempo::timing {

    using myclock_t = std::chrono::steady_clock;
    using duration_t = myclock_t::duration;
    using time_point_t = myclock_t::time_point;

    /** Create a time point for "now" */
    inline time_point_t now() { return myclock_t::now(); }

    /** Print a duration in a human readable form (from nanoseconds to hours) in an output stream. */
    inline void printDuration(std::ostream &out, const duration_t &elapsed) {
        namespace c = std::chrono;
        auto execution_time_ns = c::duration_cast<c::nanoseconds>(elapsed).count();
        auto execution_time_us = c::duration_cast<c::microseconds>(elapsed).count();
        auto execution_time_ms = c::duration_cast<c::milliseconds>(elapsed).count();
        auto execution_time_sec = c::duration_cast<c::seconds>(elapsed).count();
        auto execution_time_min = c::duration_cast<c::minutes>(elapsed).count();
        auto execution_time_hour = c::duration_cast<c::hours>(elapsed).count();

        bool first = true;

        if (execution_time_hour > 0) {
            first = false; // no need to test, if above condition is true, this is the first
            out << execution_time_hour << "h";
        }
        if (execution_time_min > 0) {
            if (first) { first = false; } else { out << " "; }
            out << execution_time_min % 60 << "m";
        }
        if (execution_time_sec > 0) {
            if (first) { first = false; } else { out << " "; }
            out << "" << execution_time_sec % 60 << "s";
        }
        if (execution_time_ms > 0) {
            if (first) { first = false; } else { out << " "; }
            out << "" << execution_time_ms % long(1E+3) << "ms";
        }
        if (execution_time_us > 0) {
            if (first) { first = false; } else { out << " "; }
            out << "" << execution_time_us % long(1E+3) << "us";
        }
        if (execution_time_ns >= 0) {
            if (first) { first = false; } else { out << " "; }
            out << "" << execution_time_ns % long(1E+3) << "ns";
        }
    }


    /** Shortcut for the above function, converting two time points into a duration. */
    inline void printExecutionTime(std::ostream &out, time_point_t start_time, time_point_t end_time) {
        const auto elapsed = end_time - start_time;
        printDuration(out, elapsed);
    }

    /** Shortcut to print in a string */
    [[nodiscard]] inline std::string as_string(time_point_t start_time, time_point_t end_time) {
        std::stringstream ss;
        printExecutionTime(ss, start_time, end_time);
        return ss.str();
    }

} // End of namespace tempo::timing
