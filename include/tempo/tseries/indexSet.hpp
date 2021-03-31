#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/dataset.hpp>

#include <variant>
#include <vector>
#include <unordered_map>

namespace tempo {

  /** Type gathering indexes by class */
  template<typename LabelType>
  using ByClassMap = std::unordered_map<LabelType, std::vector<size_t>>;

  /** Manage a set of indexes.
   * Used to create sub-dataset.
   * Index in dataset are assumed to be "stable",
   * i.e. a series's index from the original transform also is its identifier.
   **/
  struct IndexSet {
  private:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Types
    using Self = IndexSet;

    /// Subset by range
    struct Range { size_t start; size_t end; };
    static_assert(std::is_default_constructible_v<Range>);

    /// Subset by selection of indexes
    using Subset = std::shared_ptr<std::vector<size_t>>;

    /// Disjoint union: range xor indexing
    using SubKind = std::variant<Range, Subset>;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields

    size_t low{0};
    size_t high{0};

    /// Subset of the store: either by range or by subset. No subset == range [0, original.size()[
    SubKind subset{Range{low, high}};

  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructors

    /// Create a default empty range
    IndexSet() = default;

    /// New set of index, from low (included) to high (excluded).
    IndexSet(size_t low, size_t high)
      :low(low), high(high), subset{Range{low, high}} { }


    /// Set of index based on a collection. Collection must be ordered!
    explicit IndexSet(std::vector<size_t> collection) {
      Subset s = std::make_shared<std::vector<size_t>>(std::move(collection));
      if (!s->empty()) {
        low = s->front();
        high = s->back();
        subset = {s};
      }
    }

    /// Create an IndexSet over a Dataset
    template<typename FloatType, typename LabelType>
    explicit IndexSet(const Dataset<FloatType, LabelType>& ds)
      :IndexSet(0, ds.size()) { }


    /// Create an IndexSet over a ByClassMap
    template<typename LabelType>
    explicit IndexSet(const ByClassMap<LabelType>& bcm){
      using std::begin, std::end;
      std::shared_ptr<std::vector<size_t>> idx = std::make_shared<std::vector<size_t>>();
      for(const auto& [c,v]: bcm){  (*idx).insert(end(*idx), begin(v), end(v)); }
      std::sort(idx->begin(), idx->end());
      if (!idx->empty()) {
        low = idx->front();
        high = idx->back();
        subset = {std::move(idx)};
      }
    }


    /** New dataset made out of a subrange [0<=start <= end<=other.size()[ of another one
     * @param other     Other dataset
     * @param start     Start of range
     * @param end       End of range
     */
    IndexSet(const Self& other, size_t start, size_t end) {
      const auto os = other.size();
      // Test requested range
      if (end>os) {
        throw std::out_of_range("Requested end if out of range ("+std::to_string(end)+">"+std::to_string(os)+")");
      }
      // Check how other encodes its subset
      switch (other.subset.index()) {
        case 0: { // Subset is a range over store_. Take a range of that
          auto[other_start, other_end] = std::get<0>(other.subset);
          if (other_start>=other_end || start>=end) {
            low = 0;
            high = 0;
          } else {
            low = other_start+start;
            high = other_start+end;
          }
          subset = {Range{low, high}};
          break;
        }
        case 1: { // Subset is a selection of indexes (directly in store_). Take a range.
          auto other_it_begin = std::get<1>(other.subset)->cbegin();
          auto it_start = other_it_begin+start;
          auto it_end = other_it_begin+end;
          auto s = std::make_shared<std::vector<size_t>>(it_start, it_end);
          if (!s->empty()) {
            low = s->front();
            high = s->back();
            subset = {s};
          } else {
            low = 0;
            high = 0;
          }
          break;
        }
        default:should_not_happen();
      }
    }

    /** Create a new dataset from another dataset and a selection of indexes
     * @param other             Other dataset
     * @param indexes_in_other  Vector of indexes i, indexing in other 0 <= i < other.size()
     */
    IndexSet(const Self& other, const std::vector<size_t>& indexes_in_other) {
      // Test requested subset
      if (indexes_in_other.size()>other.size()) {
        throw std::out_of_range("Requested subset contains more elements than 'other'");
      }
      // Check how other encodes its subset
      switch (other.subset.index()) {
        case 0: { // Take a subset of a range
          auto[other_start, other_end] = std::get<0>(other.subset);
          if (other_start>=other_end || indexes_in_other.empty()) {
            low = 0;
            high = 0;
            subset = std::make_shared<std::vector<size_t>>();
          } else {
            std::vector<size_t> nv;
            for (auto i:indexes_in_other) {
              auto nidx = other_start+i;
              if (nidx>=other_end) { throw std::out_of_range("Requested subset is out of range"); }
              nv.push_back(nidx);
            }
            if (nv.empty()) {
              low = 0;
              high = 0;
            } else {
              low = nv.front();
              high = nv.back();
            }
            subset = {std::make_shared<std::vector<size_t>>(std::move(nv))};
          }
          break;
        }
        case 1: { // Subset of a subset
          const std::vector<size_t>& ov = *(std::get<1>(other.subset));
          std::vector<size_t> nv;
          nv.reserve(ov.size());
          for (auto i: indexes_in_other) { nv.push_back(ov.at(i)); }
          if (nv.empty()) {
            low = 0;
            high = 0;
          } else {
            low = nv.front();
            high = nv.back();
          }
          subset = {std::make_shared<std::vector<size_t>>(std::move(nv))};
          break;
        }
        default:should_not_happen();
      }
    }

    // Default destructor is fine
    ~IndexSet() = default;

    // Default Copy ok
    IndexSet(const Self& other) = default;

    IndexSet& operator=(const Self& other) = default;

    // Default Move ok
    IndexSet(Self&&) noexcept = default;

    IndexSet& operator=(Self&&) noexcept = default;


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Accesses

    /** index into the indexSet, returning a "real index" usable in a dataset.
     * @param index  [0, size[
     * @return [low, high[
     */
    [[nodiscard]] inline size_t get(size_t index) const {
      // Check how other encodes its subset
      switch (subset.index()) {
        case 0: { // Subset is a range. Convert in the range [low, high[
          auto[low, high] = std::get<0>(subset);
          assert(low+index<high);
          return low+index;
        }
        case 1: { // Subset is a selection: index into it (size of selection marks the end).
          const std::vector<size_t>& s = *std::get<1>(subset);
          return s[index];
        }
        default:should_not_happen();
      }
    }

    /// Const Bracket operator aliasing 'get'
    [[nodiscard]] inline size_t operator[](size_t index) const { return get(index); }

    /// Size of the set
    [[nodiscard]] inline size_t size() const {
      switch (subset.index()) {
        case 0: { // Subset is a range
          auto[start, end] = std::get<0>(subset);
          if (start>end) { return 0; }
          else { return end-start; }
        }
        case 1: { // Subset is a selection of indexes
          auto s = std::get<1>(subset);
          return s->size();
        }
        default:should_not_happen();
      }
    }

    /// Is the dataset empty?
    [[nodiscard]] inline bool empty() const { return size()==0; }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Iterator

    /// Provide an iterator over the indexes contained in an IndexSet
    struct It {
      // --- --- --- Types
      using iterator_category = std::random_access_iterator_tag;
      using difference_type = ssize_t;
      using value_type = std::size_t;
      using pointer = const value_type*;
      using reference = const value_type&;

    private:
      // --- --- --- Fields
      ssize_t it_index;
      const Self* idxset;

    public:
      // --- --- --- Constructor
      It(ssize_t it, const Self* idxset)
        :it_index(it), idxset(idxset) { }

      // --- --- --- Iterator methods

      /// Prefix increment
      inline It& operator++() {
        ++it_index;
        return *this;
      }

      /// Prefix decrement
      inline It& operator--() {
        --it_index;
        return *this;
      }

      /// Postfix increment
      // (int) is just a placeholder to indicate that this is the postfix operation
      inline It operator++(int) {
        It i = *this;
        ++it_index;
        return i;
      }

      /// Prefix decrement
      inline It operator--(int) {
        It i = *this;
        --it_index;
        return i;
      }

      /// Equality testing
      inline bool operator==(const It& other) const {
        return (idxset==other.idxset) && (it_index==other.it_index);
      }

      /// Difference testing, based on equality
      inline bool operator!=(const It& other) const { return !(*this==other); }

      // Comparison: assume that the backing store and the range are the same
      inline bool operator<(const It& other) const {
        assert(idxset==other.idxset);
        return it_index<other.it_index;
      }

      inline bool operator>(const It& other) const { return other<*this; }

      inline bool operator<=(const It& other) const { return !(*this>other); }

      inline bool operator>=(const It& other) const { return !(*this<other); }

      // Arithmetic operations
      inline It& operator+=(const difference_type& add) {
        this->it_index += add;
        return *this;
      }

      inline It operator+(const difference_type& add) const {
        It copy(*this);
        return copy += add;
      }

      inline It& operator-=(const difference_type& add) {
        this->it_index -= add;
        return *this;
      }

      inline It operator-(const difference_type& add) const {
        It copy(*this);
        return copy -= add;
      }

      /// Subtraction between iterators
      inline difference_type operator-(const It& other) const {
        assert(idxset==other.idxset);
        return it_index-other.it_index;
      }

      /// Dereference
      inline value_type operator*() const { return idxset->get(it_index); }

    }; // End of inner class It

    /// Get an iterator on the beginning of the indexSet
    [[nodiscard]] inline It begin() const { return It{0, this}; }

    /// Get an iterator pas the end of the dataset
    [[nodiscard]] inline It end() const {
      ssize_t it_index;
      switch (subset.index()) {
        case 0: { // Subset is a range. Convert in the range [0, end[
          auto[start, end] = std::get<0>(subset);
          if (start>end) { it_index = 0; }
          else { it_index = end-start; }
          break;
        }
        case 1: { // Subset is a selection of indexes. The size marks the end.
          it_index = std::get<1>(subset)->size();
          break;
        }
        default:should_not_happen();
      } // End of switch
      return It(it_index, this);
    }
  };




  /** Obtain a "by class" structure given an IndexSet and a Dataset.
   *  Throw an exception if series don't have a class label.*/
  template<typename FloatType, typename LabelType>
  [[nodiscard]] ByClassMap<LabelType> get_by_class(const Dataset<FloatType, LabelType>& ds, const IndexSet& is) {
    ByClassMap<LabelType> result;
    auto it = is.begin();
    auto end = is.end();
    while (it!=end) {
      const auto idx = *it;
      const auto& label = ds[idx].get_label().value();
      result[label].push_back(idx); // Note: default construction of the vector of first access
      ++it;
    }
    return std::move(result);
  }

  /** Given a by class mapping, pick an exemplar by class */
  template<typename PRNG, typename LabelType>
  ByClassMap<LabelType> pick_one_by_class(const ByClassMap<LabelType>& bcm, PRNG& prng) {
    ByClassMap<LabelType> result;
    for (const auto&[c, v]: bcm) {
      assert(v.size()>0);
      std::uniform_int_distribution<size_t> dist(0, v.size()-1);
      result[c].push_back(v[dist(prng)]);
    }
    return result;
  }

  /** Gini impurity of a "by class" map */
  template<typename LabelType>
  [[nodiscard]] inline double gini_impurity(const ByClassMap<LabelType>& by_classes) {
    assert(!by_classes.empty());
    // Ensure that we never encounter a "floating point near 0" issue.
    if (by_classes.size()==1) { return 0; }
    else {
      double total_size = 0;
      for (const auto&[k, v]: by_classes) { total_size += v.size(); }
      double sum{0};
      for (const auto&[cl, val]: by_classes) { double p = val.size()/total_size; sum += p*p; }
      return 1-sum;
    }
  }

  /** Compute the stddev of a selection of a dataset, over a transform that is serie-like */
  template<typename FloatType, typename LabelType>
  FloatType stddev(
    const IndexSet& is,
    const TransformHandle<std::vector<TSeries<FloatType, LabelType>>, FloatType, LabelType>& t) {
    stats::StddevWelford s;
    for (const auto& ts: t.get()) { for (size_t i{0}; i<ts.length(); ++i) { s.update(ts(0, i)); }}
    return s.get_stddev_p();
  }


} // End of namespace tempo