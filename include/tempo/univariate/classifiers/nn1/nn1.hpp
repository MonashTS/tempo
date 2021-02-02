#pragma once

#include <vector>
#include <string>
#include <algorithm>


#include "../../distances/distances.hpp"


namespace tempo::univariate {


    /** NN1 function on TSeries. Given a database and a query, returns a vector of labels.
     *  The vector will hold more than one label if ties are found, and will be empty if the database is empty.
     * @tparam FloatType        The floating number type used to represent the series.
     * @tparam LabelType        Type of the labels
     * @tparam InputIterator    Type of the InputIterator representing the database
     * @param distance          Distance without cut-off - used for the first pair (query, database[0])
     * @param distance_co       Distance with cut-off - used for the other pair (query, database[i>0])
     * @param begin             Input Iterator pointing on the start of the database
     * @param end               Input Iterator pointing on the end of the database
     * @param query             Query whose label is to be determined
     * @return                  Vector of labels, containing more than 1 if ties occur, and 0 if the database is empty.
     */
    template<typename FloatType, typename LabelType, typename InputIterator>
    [[nodiscard]] std::vector<LabelType> nn1(
            const distfun_cutoff_t<FloatType, LabelType>& distance_co,
            InputIterator begin, InputIterator end,
            const TSeries<FloatType, LabelType>& query){
        // Static check the value type parameter of the iterator type
        static_assert(
                stassert::is_iterator_value_type<TSeries<FloatType, LabelType>, InputIterator>,
                "Iterator must contain TSeries<FloatType, LabelType>");
        // Check if the database isn't empty, else immediately return an empty vector
        if(begin!=end){
            // Use the distance without cut-off to compute the first pair.
            auto bsf = POSITIVE_INFINITY<FloatType>;
            std::vector<LabelType> labels{};
            // Keep going, exhaust the database
            while(begin!=end){
                // We can now use the distance with a cutoff
                const TSeries<FloatType, LabelType>& candidate = *begin;
                auto result = distance_co(candidate, query, bsf);
                if(result<bsf){
                    labels.clear();
                    labels.emplace_back(candidate.label().value());
                    bsf=result;
                } else if (bsf == result){ // Manage ties
                    const auto& l = candidate.label().value();
                    if( std::none_of(labels.begin(), labels.end(), [l](const auto& v){return v==l;}) ){
                        labels.emplace_back(l);
                    }
                }
                ++begin;
            }
            return labels;
        } else {
            return {};
        }
    }

}