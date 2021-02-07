#pragma once

#include "tseries.hpp"

#include <any>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace tempo {

    /** A transform in a TS pack contains:
     *  - name:string       The name of the transform
     *  - capsule:any       The result of the transform, in a capsule. Keep the result alive.
     *  - transform:void*   Raw pointer access to the transform
     */
    struct TSPackTR {
        std::string name;
        std::any capsule;
        void* transform;
    };

    // Declaration of TSPack for use in TSPackTransformer
    template <typename FloatType, typename LabelType>
    struct TSPack;

    /** Result type for a transformation function.
     * Fails with a message or returns a TSPackTR
     */
    using TSPackResult = std::variant<std::string, TSPackTR>;

    /** A TSPackTransformer contains the same fields as a TSPackTR,
     * except for the transform, as it instead contains the transformation function.
     * Note that the transform takes a TSPack as input, not a TSeries.
     * Hence, it can rely on other transforms computed previously.
     * Fields will be copied into a TSPackTR.
     * Consider putting your args behind a shred pointer if they are heavy or not copyable.
     */
    template <typename FloatType, typename LabelType>
    struct TSPackTransformer {
        std::string name;
        std::string extra_json;
        std::function<TSPackResult(const TSPack<FloatType, LabelType>&)> transfun{};
    };

    /** A TSPack contains a TSeries ("raw") and a map of transforms (string, any).
     *  A transform is identified by its name and can be anything.
     */
    template<typename FloatType, typename LabelType>
    struct TSPack {
        using TS = TSeries<FloatType, LabelType>;
        using Transformer = TSPackTransformer<FloatType, LabelType>;
        using Self = TSPack<FloatType, LabelType>;

        /// Original raw series, source of the transforms
        std::shared_ptr<TS> raw_capsule;
        const TS& raw;

        /// Collection of transforms information
        std::vector<TSPackTR> transform_infos{};

        /// Collection of transforms pointer for direct access
        std::vector<void*> transforms{};

        /// Create a new pack from a shared series. Register "raw" in the map of transforms.
        explicit TSPack(std::shared_ptr<TS> raw):
                raw_capsule(std::move(raw)), raw(*raw_capsule)
        {
            transforms.push_back(raw_capsule.get());
            transform_infos.emplace_back(
                    TSPackTR {
                            .name = "raw",
                            .capsule = std::any(raw_capsule),
                            .transform = raw_capsule.get()
                    }
            );
        }

        /// Extend the pack with a new transformation. Can fail.
        /// Overwrite if the name is already defined
        std::variant<std::string, size_t> apply(const Transformer& transformer){
            TSPackResult result = transformer.transfun(*this);
            switch(result.index()){
                case 0: { // Error case
                    return {std::get<0>(result)};
                }
                case 1: { // Success case
                    size_t idx = transform_infos.size();
                    transform_infos.template emplace_back(std::move(std::get<1>(result)));
                    transforms.push_back(transform_infos.back().transform);
                    return {idx};
                }
                default: should_not_happen();
            }
            return {};
        }

        /// Linear lookup of a transform's index by name
        [[nodiscard]] std::optional<size_t> lookup(const std::string& name){
            for(size_t i=0; i<transform_infos.size(); ++i){
                if(transform_infos[i].name == name){ return {i}; }
            }
            return {};
        }


        /** Helper wrapping a vector of series into a vector of TSPack.
         * @param series Input vector - take ownership
         * @return A vector of TSPack
         */
        [[nodiscard]] static inline std::vector<Self> wrap(std::vector<TS>&& series) {
            std::vector<Self> storedata;
            std::transform(
                    std::move_iterator(series.begin()),
                    std::move_iterator(series.end()),
                    std::back_inserter(storedata),
                    [](TS&& ts){ return Self(std::make_shared<TS>(std::move(ts))); }
            );
            return storedata;
        }

        /// Label of the underlying series
        [[nodiscard]] inline const std::optional<LabelType> &label() const { return raw.label_; }

        /// Shorthand for transforms that are supposed to be TSeries
        [[nodiscard]] inline const TS& at(size_t idx) const {
            return *(static_cast<TS*>(transforms[idx]));
        }


    };


}
