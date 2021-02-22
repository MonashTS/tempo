#pragma once

namespace tempo {

    using Capsule = std::shared_ptr<std::any>;

    /// Capsule builder helper
    template<typename CapsuleT, typename... Args>
    [[nodiscard]] inline Capsule make_capsule(Args &&... args) {
        return std::make_shared<std::any>(std::make_any<CapsuleT>(args...));
    }

    /// Capsule pointer accessor
    template<typename CapsuleT>
    [[nodiscard]] inline CapsuleT *capsule_ptr(const std::shared_ptr <std::any> &ptr) {
        return std::any_cast<CapsuleT>(ptr.get());
    }

}