#pragma once

#include "../utils/utils.hpp"
#include "tspack.hpp"

#include <memory>
#include <sstream>
#include <variant>
#include <vector>

namespace tempo {


    template<typename LabelType>
    struct DatasetInfo {
        size_t nb_dimensions{0};
        size_t min_length{0};
        size_t max_length{0};
        size_t size{0};
        bool has_missing{false};
        std::set<LabelType> labels{};

        /** Factory method: build a dataset info from a collection of TSeries<FloatType, LabelType> represented by iterators.
         * Can throw if the collection contains times series with inconsistent number of dimensions.
         * @tparam FloatType        Type of time series's values
         * @tparam ForwardIterator  Type of the iterator
         * @param begin             Start of the collection
         * @param end               End of the collection
         */
        template<typename FloatType, typename ForwardIterator>
        static DatasetInfo<LabelType> make_from(ForwardIterator begin, ForwardIterator end){
            using TSP = TSPack<FloatType, LabelType>;
            DatasetInfo<LabelType> result;
            if(begin!=end){
                // Init with the first series
                {
                    const TSP& s = *begin;
                    result.nb_dimensions = s.raw.nb_dimensions();
                    result.min_length = s.raw.length();
                    result.max_length = s.raw.length();
                    result.has_missing = s.raw.has_missing();
                    result.size++;
                    if (s.raw.label()) { result.labels.insert(s.raw.label().value()); }
                    ++begin;
                }
                // Main loop
                for (auto it = begin; it != end; ++it) {
                    const TSP& s = *it;
                    if(result.nb_dimensions != s.raw.nb_dimensions()){
                        throw std::logic_error("Cannot mix series with different number of dimensions");
                    }
                    const auto l = s.raw.length();
                    result.min_length = std::min(result.min_length, l);
                    result.max_length = std::max(result.max_length, l);
                    result.has_missing = result.has_missing || s.raw.has_missing();
                    result.size++;
                    if(s.raw.label()) { result.labels.insert(s.raw.label().value()); }
                }
            }
            return result;
        }

        /** Create a JSON representation */
        inline std::string to_json() const {
            std::stringstream ss;
            ss << "{";
            ss << "\"Dim\": " <<  nb_dimensions << ", ";
            ss << "\"lmin\": " << min_length << ", ";
            ss << "\"lmax\": " << max_length << ", ";
            ss << "\"size\": " << size << ", ";
            ss << "\"has_missing\": " << has_missing << ", ";
            ss << "\"nb_labels\": " <<   labels.size() << ", ";
            ss << "\"labels\": [";
            if (!labels.empty()) {
                const auto end = labels.cend();
                auto it = labels.cbegin();
                ss << '"' << *it << '"'; // set not empty: no need to check it!=end
                for (++it; it != end; ++it) { ss << ", \"" << *it; }
            }
            ss << "] }";
            return ss.str();
        }
    };


    /** Store: backend of a dataset and sub-dataset.
     *  The store is made of a vector of TSPack i.e a series and its transforms.
     *  @tparam FloatType    Type of time series values
     *  @tparam LabelType    Type of labels
     */
    template<typename FloatType, typename LabelType>
    using DSStore = std::vector<TSPack<FloatType, LabelType>>;


    /** A dataset of time series
     *  @tparam FloatType    Type of time series values
     *  @tparam LabelType    Type of labels
     */
    template<typename FloatType, typename LabelType>
    struct Dataset {
        using DI = DatasetInfo<LabelType>;
        using TSP = TSPack<FloatType, LabelType>;
        using Store = DSStore<FloatType, LabelType>;
        using Self = Dataset<FloatType, LabelType>;

        /// Subset by range
        struct  Range{size_t start; size_t end; };

        /// Subset by selection of indexes (indexing into store_)
        using   Subset = std::shared_ptr<std::vector<size_t>>;

        /// Disjoint union: range xor indexing
        using   SubKind = std::variant<Range, Subset>;

    private:

        /// Backend store, made of series packs.
        std::shared_ptr<Store> store;

        /// Info about the backed storage
        std::shared_ptr<const DI> store_info_;

        /// Info about this particular subset
        std::optional<DI> my_info {};

        /** Subset of the store: either by range or by subset
         *  Note: Always used: no subset == range [0, store_->size()[*/
        SubKind subset;

    public:

        /// New empty dataset, with an empty store
        Dataset() = default;

        /** New dataset based on a vector of TSPacks. Take ownership of the vector.
         * Requires information about the store
         */
        Dataset(std::vector<TSP>&& series, DatasetInfo<LabelType> info):
                store(std::make_shared<std::vector<TSP>>(std::move(series))),
                store_info_(std::make_shared<const DI>(info)),
                my_info({*store_info_}),
                subset(Range{0, store->size()})
        { }

        /** New dataset based on a vector of TSPacks. Take ownership of the vector.
         * Compute the DatasetInfo related to the series
         */
        explicit Dataset(std::vector<TSP>&& series):
                store(std::make_shared<std::vector<TSP>>(std::move(series))),
                subset(Range{0, store->size()})
        {
            auto di = DI::template make_from<FloatType>(store->cbegin(), store->cend());
            store_info_ = std::make_shared<const DI>(di);
            my_info = {*store_info_};
        }


        /** New dataset based on a vector of TSPacks with shared ownership.
         * Requires information about the store
         */
        Dataset(std::shared_ptr<std::vector<TSP>> series, DatasetInfo<LabelType> info):
                store(std::move(series)),
                store_info_(std::make_shared<const DI>(info)),
                my_info({*store_info_}),
                subset(Range{0, store->size()})
        { }

        /** New dataset based on a vector of TSPacks with shared ownership.
         *  Compute the DatasetInfo related to the series
         */
        explicit Dataset(std::shared_ptr<std::vector<TSP>> series):
                store(std::move(series)),
                subset(Range{0, store->size()})
        {
            auto di = DI::template make_from<FloatType>(store->cbegin(), store->cend());
            store_info_ = std::make_shared<const DI>(di);
            my_info = {*store_info_};
        }


        /** New dataset made out of a subrange [0<=start <= end<=other.size()[ of another one
         * @param other     Other dataset
         * @param start     Start of range
         * @param end       End of range
         */
        Dataset(const Self& other, size_t start, size_t end):
                store(other.store), store_info_(other.store_info_) {
            const auto os = other.size();
            // Test requested range
            if (start > end) {
                throw std::out_of_range("Creating a range dataset with invalid indexes (start>end)");
            }
            if (end>os){
                throw std::out_of_range("Requested range is out of range");
            }
            // Check how other encodes its subset
            switch (other.subset.index()) {
                case 0: { // Subset is a range over store_. Take a range of that
                    auto [other_start, other_end] = std::get<0>(other.subset);
                    size_t s = other_start+start;
                    size_t e = other_start+end;
                    subset = {Range{s,e}};
                    break;
                }
                case 1: { // Subset is a selection of indexes (directly in store_). Take a range.
                    auto other_it_begin = std::get<1>(other.subset)->cbegin();
                    auto it_start = other_it_begin+start;
                    auto it_end = other_it_begin+end;
                    subset = {std::make_shared<std::vector<size_t>>(it_start, it_end)};
                    break;
                }
                default:
                    should_not_happen();
            }
        }

        /** Create a new dataset from another dataset and a selection of indexes
         * @param other             Other dataset
         * @param indexes_in_other  Vector of indexes i, indexing in other 0 <= i < other.size()
         */
        Dataset(const Self& other, const std::vector<size_t>& indexes_in_other):
                store(other.store), store_info_(other.store_info_) {
            // Test requested subset
            if (indexes_in_other.size()> other.size()){
                throw std::out_of_range("Requested subset contains more elements than 'other'");
            }
            // Check how other encodes its subset
            switch (other.subset.index()) {
                case 0: { // Subset is a range over store_. Take a subset of that, generating a new subset
                    auto [other_start, other_end] = std::get<0>(other.subset);
                    std::vector<size_t> nv;
                    for(auto i:indexes_in_other){
                        auto nidx = other_start + i;
                        if(nidx>=other_end){ throw std::out_of_range("Requested subset is out of range"); }
                        nv.push_back(nidx);
                    }
                    subset = {std::make_shared<std::vector<size_t>>(std::move(nv))};
                    break;
                }
                case 1: { // Subset is a selection of indexes (directly in store_). Take a subset.
                    const std::vector<size_t>& ov = *(std::get<1>(other.subset));
                    std::vector<size_t> nv;
                    nv.reserve(ov.size());
                    for(auto i: indexes_in_other){ nv.push_back(ov.at(i)); }
                    subset = {std::make_shared<std::vector<size_t>>(std::move(nv))};
                    break;
                }
                default:
                    should_not_happen();
            }
        }

        ~Dataset() = default;

        // Copy
        Dataset(const Dataset &other) = default;

        Dataset &operator=(const Dataset &other) = default;

        // Move
        Dataset(Dataset &&)  noexcept = default;

        Dataset &operator=(Dataset &&)  noexcept = default;


        // --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- -- --- --- --- --- -- --- --- --- --- -- --- ---
        // Methods
        // --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- -- --- --- --- --- -- --- --- --- --- -- --- ---

        /** Apply a transformation to the dataset. Always apply to the full store. */
        inline std::variant<std::string, size_t> apply(const TSPackTransformer<FloatType, LabelType>& tr){
            auto& s = *store;
            if(!s.empty()) {
                size_t idx = s.front().transforms.size();
                for(auto& p:s){
                    auto res = p.apply(tr);
                    switch(res.index()){
                        case 0: { return res; } // Transmit error
                        case 1: { if(idx != std::get<1>(res)){ return {"Store is out of sync"}; } }
                    }
                } // End of application loop
                return {idx};
            } else {
                return {"Store is empty"};
            }
        }

        /** All TSPack in a dataset are supposed to be "in sync".
         *  Linearly lookup an index using the first TSPack from the store.
         */
         inline std::optional<size_t> lookup(const std::string& name){
             if(store->size()!=0){
                 return store->at(0).lookup(name);
             }
             return {};
         }


        /** Get the index of the item in the underlying store, index
         * @param index  [0, size[
         * @return [0, store size[
         */
        [[nodiscard]] inline ssize_t get_store_index(size_t index) const {
            // Check how other encodes its subset
            switch (subset.index()) {
                case 0: { // Subset is a range over store_. Convert in the range [0, end[
                    auto[start, _] = std::get<0>(subset);
                    return start + index;
                }
                case 1: { // Subset is a selection of indexes (directly in store_). The size marks the end.
                    const std::vector<size_t> &s = *std::get<1>(subset);
                    return s[index];
                }
                default:
                    should_not_happen();
            }
        }

        /** Mutable access an item by its index in the dataset
         * @param index  [0, size [
         * @return The time series pack at 'index' in this dataset (not in the store)
         */
        [[nodiscard]] inline TSP& get(size_t index) {
            return (*store)[get_store_index(index)];
        }

        /// Mutable bracket operator aliasing 'get'
        [[nodiscard]] inline TSP& operator[](size_t index) { return get(index); }

        /** Const access an item by its index in the dataset
         * @param index  [0, size [
         * @return The time series pack at 'index' in this dataset (not in the store)
         */
        [[nodiscard]] inline const TSP& get(size_t index) const {
            return (*store)[get_store_index(index)];
        }

        /// Const Bracket operator aliasing 'get'
        [[nodiscard]] inline const TSP& operator[](size_t index) const { return get(index); }

        /// Size of the dataset
        [[nodiscard]] inline size_t size() const {
            switch (subset.index()) {
                case 0: { // Subset is a range
                    auto [start, end] = std::get<0>(subset);
                    return end-start;
                }
                case 1: { // Subset is a selection of indexes
                    auto s = std::get<1>(subset);
                    return s->size();
                }
                default:
                    should_not_happen();
            }
        }

        /// IS the dataset empty?
        [[nodiscard]] inline bool empty() const {return store->empty();}

        /// Access information of the underlying store (about all the series, not only the one in this dataset)
        [[nodiscard]] inline const DI& store_info() const { return *store_info_; }

        /// Compute information about the series in this dataset. O(n). Results are cached.
        [[nodiscard]] inline const DI& info() const {
            if(!my_info){
                my_info = DI::template make_from<FloatType>(begin(), end());
            }
            return my_info;
        }

        // --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- -- --- --- --- --- -- --- --- --- --- -- --- ---
        // Iterator
        // --- --- --- --- --- --- --- --- --- --- -- --- --- --- --- -- --- --- --- --- -- --- --- --- --- -- --- ---

        /// Random access iterator
        class It {
        public:
            // --- --- --- Types
            using iterator_category = std::random_access_iterator_tag;
            using difference_type = ssize_t;
            using value_type = TSP;
            using pointer = const TSP*;
            using reference = const TSP&;

        private:
            // --- --- --- Fields
            ssize_t it_index;
            const Self* dataset;

        public:
            // --- --- --- Constructor
            It(ssize_t it, const Self* ds):it_index(it), dataset(ds){}

            // --- --- --- Iterator methods

            /// Prefix increment
            inline It &operator++() { ++it_index; return *this; }

            /// Prefix decrement
            inline It &operator--() { --it_index; return *this; }

            /// Postfix increment
            // (int) is just a placeholder to indicate that this is the postfix operation
            inline It operator++(int) { It i = *this; ++it_index; return i; }

            /// Prefix decrement
            inline It operator--(int) { It i = *this; --it_index; return i; }

            /// Equality testing, assuming that the backing collection and the range are the same
            inline bool operator==(const It &other) const { return (it_index == other.it_index); }

            /// Difference testing, based on equality
            inline bool operator!=(const It &other) const { return !(*this == other); }

            // Comparison: assume that the backing store and the range are the same
            inline bool operator<(const It &other) const { return it_index < other.it_index; }

            inline bool operator>(const It &other) const { return other < *this; }

            inline bool operator<=(const It &other) const { return !(*this > other); }

            inline bool operator>=(const It &other) const { return !(*this < other); }

            // Arithmetic operations
            Dataset &operator+=(const difference_type &add) {
                this->it_index += add;
                return *this;
            }

            Dataset operator+(const difference_type &add) const {
                Dataset copy(*this);
                return copy += add;
            }

            Dataset &operator-=(const difference_type &add) {
                this->it_index -= add;
                return *this;
            }

            Dataset operator-(const difference_type &add) const {
                Dataset copy(*this);
                return copy -= add;
            }

            /// Subtraction between iterators
            difference_type operator-(const Dataset &other) const { return it_index - other.it_index; }

            /// Dereference
            reference operator*() const { return dataset->get(it_index); }

        }; // End of inner class It

        /// Get an iterator on the beginning of the dataset
        [[nodiscard]] It begin() const { It a = {0, this}; return a; }

        /// Get an iterator pas the end of the dataset
        [[nodiscard]] It end() const {
            ssize_t it_index;
            switch (subset.index()) {
                case 0: { // Subset is a range over store_. Convert in the range [0, end[
                    auto[start, end] = std::get<0>(subset);
                    it_index = end - start;
                    break;
                }
                case 1: { // Subset is a selection of indexes (directly in store_). The size marks the end.
                    it_index = std::get<1>(subset)->size();
                    break;
                }
                default:
                    should_not_happen();
            } // End of switch
            return It(it_index, this);
        }
    };

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Function over datasets
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Type of mappings Class -> indexes in a dataset */
    template<typename LabelType>
    using ByClassMap = std::map<LabelType, std::vector<size_t>>;

    /** Create a mapping Class -> indexes for a dataset */
    template<typename FloatType, typename LabelType>
    [[nodiscard]] ByClassMap<LabelType> getByClassMap(const Dataset<FloatType, LabelType>& dataset){
        ByClassMap<LabelType> result;
        for(size_t idx=0; idx<dataset.size(); ++idx){
            const auto& s = dataset[idx].raw;
            if(s.label()) {
                auto l = s.label().value();
                result[l].push_back(idx); // Default construction of the vector in the map on first access.
            }
        }
        return result;
    }

    /** Giving a ByClassMap, compute the gini impurity */
    template<typename LabelType>
    [[nodiscard]] double gini_impurity(const ByClassMap<LabelType>& bcm){
        std::vector<size_t> cardinalities;
        std::transform(bcm.cbegin(), bcm.cend(), std::back_inserter(cardinalities), [](const auto& kv){return kv.second.size();} );
        return tempo::stats::gini_impurity(cardinalities.begin(), cardinalities.end());
    }





} // End of namespace tempo
