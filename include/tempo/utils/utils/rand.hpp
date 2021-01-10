#pragma once

#include <algorithm>
#include <stdexcept>
#include <random>
#include <vector>

namespace tempo::rand {

    /** Pick a random item from a vector.
     * Note: optional references did not make it in C++17, so just throw if the vector v is empty*/
    template<typename T, typename PRNG>
    [[nodiscard]] auto pick_one(const std::vector<T> &v, PRNG &prng) {
        if (v.size() == 1) {
            return v.back();
        } else if (v.size() > 1) {
            auto distribution = std::uniform_int_distribution<int>(0, v.size() - 1);
            return v[distribution(prng)];
        } else {
            throw std::invalid_argument("Picking from an empty vector");
        }
    }

    /** Generate a vector of a given size with random real values in the half-closed interval [min, max[.
     *  Use a provided random number generator. */
    template<typename T=double, typename PRNG>
    [[nodiscard]] std::vector<T> generate_random_real_vector(PRNG &prng, size_t size, T min, T max) {
        static_assert(std::is_floating_point_v<T>);
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
        static_assert(std::is_integral_v<T>);
        std::uniform_int_distribution<T> udist{min, max};
        auto generator = [&udist, &prng]() { return udist(prng); };
        std::vector<T> v(size);
        std::generate(v.begin(), v.end(), generator);
        return v;
    }

} // End of namespace tempo