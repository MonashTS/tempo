#pragma once

#include "../../tseries/tspack.hpp"

#include <cstddef>
#include <exception>

namespace tempo::univariate {

    /** Computation of a series derivative according to "Derivative Dynamic Time Warping" by Keogh & Pazzani
     * @tparam FloatType    The floating number type used to represent the series.
     * @param series        Pointer to the series's data
     * @param length        Length of the series
     * @param out           Pointer where to write the derivative. Must be able to store 'length' values.
     */
    template<typename FloatType>
    void derivative(const FloatType *series, size_t length, FloatType *out) {
        if (length > 2) {
            for (size_t i{1}; i < length - 1; ++i) {
                out[i] = ((series[i] - series[i - 1]) + ((series[i + 1] - series[i - 1]) / 2.0)) / 2.0;
            }
            out[0] = out[1];
            out[length - 1] = out[length - 2];
        } else {
            std::copy(series, series + length, out);
        }
    }


    template<typename FloatType, typename LabelType>
    struct DerivativeTransformer {
        using TS = TSeries<FloatType, LabelType>;
        using TSP = TSPack<FloatType, LabelType>;
        using TSPTr = TSPackTransformer<FloatType, LabelType>;
        using ElemType = TS;
        using AnyType = std::shared_ptr<ElemType>;

        [[nodiscard]] static TSPTr get(size_t source_index, const std::string& name){
            return TSPTr {
                    .name = name,
                    .extra_json = R"({"derivative name": ")"+name+"\"}",
                    .transfun = [source_index, name](const TSPack<FloatType, LabelType>& tsp){
                        const auto& s = *(static_cast<TS*>(tsp.transforms[source_index]));
                        std::vector<FloatType> d(s.size());
                        derivative(s.data(), s.size(), d.data());
                        AnyType ant = std::make_shared<ElemType>(std::move(d));
                        ElemType* ptr = ant.get();
                        std::any any = std::any(ant);
                        return TSPackResult{
                                TSPackTR {
                                        .name = name,
                                        .capsule = any,
                                        .transform = ptr
                                }
                        };
                    }
            };
        }

        [[nodiscard]] static TSPTr get_nth(size_t source_index, const std::string& name, int nth) {
            if(nth<1){
                throw std::invalid_argument("Derivative transform: called with invalid derivative rank");
            }
            return TSPTr{
                    .name = name,
                    .extra_json = R"({"derivative name": ")"+name+"\"}",
                    .transfun = [nth, source_index, name](const TSPack<FloatType, LabelType> &tsp) {
                        const auto &s = *(static_cast<TS *>(tsp.transforms[source_index]));
                        std::vector<FloatType> d(s.size());
                        if(nth==1){
                            derivative(s.data(), s.size(), d.data());
                        } else {
                            // Repeated application: require an extra buffer to hold previous transform.
                            std::vector<FloatType> input(s.data(), s.data()+s.size());
                            for (int i = 0; i<nth; ++i) {
                                derivative(input.data(), input.size(), d.data());
                                swap(input, d);
                            }
                        }
                        AnyType ant = std::make_shared<ElemType>(std::move(d), tsp.at(source_index));
                        ElemType *ptr = ant.get();
                        std::any any = std::any(ant);
                        return TSPackResult{
                                TSPackTR{
                                        .name = name,
                                        .capsule = any,
                                        .transform = ptr
                                }
                        };
                    }
            };
        }


        [[nodiscard]] inline static const ElemType& cast(void* ptr){
            return *(static_cast<ElemType*>(ptr));
        }
    };

} // End of namespace tempo::univariate
