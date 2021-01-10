#pragma once

namespace tempo::stats {

    /** Given a collection of cardinalities, compute the Gini Impurity.
     * @tparam ForwardIterator Must be a forward iterator (used twice)
     * @param begin First item of the collection
     * @param end Marked the end of the collection
     * @return 0<=gi<1 where 0 means total purity (all item in one class).
     */
    template<typename ForwardIterator>
    double gini_impurity(ForwardIterator begin, ForwardIterator end){
        // Ensure that we never encounter a "floating point near 0" issue.
        if(std::distance(begin, end)==1){return 0;}
        // Gini impurity computation
        double total_size{0};
        for(auto it=begin; it!=end; ++it){total_size += *it;}
        double sum{0};
        for(auto it=begin; it!=end; ++it){
            double p = (*it) / total_size;
            sum += p * p;
        }
        return 1 - sum;
    }


} // End of namespace tempo