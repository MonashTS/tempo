#pragma once

namespace tempo::utils {

    /** Private inherit from this class to create an uncopyable (but still movable) class */
    class Uncopyable {
    protected:
        // Protect the constructor and the destructor (class not usable by itself)
        Uncopyable() = default;

        ~Uncopyable() = default;

        // Still movable
        Uncopyable(Uncopyable &&) = default;

        Uncopyable &operator=(Uncopyable &&) = default;

    public:
        // Delete copy and copy-assignment operator
        Uncopyable(const Uncopyable &other) = delete;

        Uncopyable &operator=(const Uncopyable &other) = delete;
    };
}