#pragma once

#include <tempo/utils/capsule.hpp>

#include <string>
#include <typeinfo>
#include <vector>


/** Transforms.
 * Transforms form the basis of the dataset - even more than the time series themselves!
 * A transform is an arbitrary piece of data D of type T.
 * When inserting D in a dataset, we obtain in return a "handle" on the transform allowing us to recover
 * the type T of the underlying data.
 * Hence transforms allow to store anything in a dataset, and the handles allow to access these data safely.
 *
 */

namespace tempo {

  /// Declaration of the datasets
  template<typename FloatType, typename LabelType>
  struct Dataset;

  /** Base transform - provide type erasure */
  struct Transform {
  private:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Identification
    std::string name{};
    std::string full_name{};
    std::vector<std::string> parents{};

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Data
    Capsule capsule{};
    const void* data_ptr{};   // Direct access to the capsule's data

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Helpers
    /// Create the full name of a transform, given its name and its parents' name
    [[nodiscard]] static inline std::string mk_full_name(const std::string& n, const std::vector<std::string>& pn) {
      std::string full_name{};
      if (n.empty()) {
        throw std::domain_error("Transform name cannot be empty");
      } else if (pn.size()==1) {
        full_name = pn.front();
      } else if (pn.size()>1) {
        full_name = "{"+pn.front();
        for (auto it = pn.begin()+1; it<pn.end(); ++it) { full_name += " & "+*it; }
      }
      return full_name+"/"+n;
    }

  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructors
    Transform() = default;

    /** Construct a new transformation
     * @param n     Name of the transform
     * @param pn    Parents' name
     * @param c     Capsule containing the data
     * @param p     Raw pointer to the data
     */
    Transform(std::string&& n, std::vector<std::string>&& pn, Capsule&& c, const void* p) :
      name(std::move(n)), parents(std::move(pn)), capsule(std::move(c)), data_ptr(p) {
      full_name = mk_full_name(name, parents);
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    [[nodiscard]] inline const std::string& get_name() const { return name; }

    [[nodiscard]] inline const std::string& get_full_name() const { return full_name; }

    [[nodiscard]] inline const std::vector<std::string>& get_parents() const { return parents; }

    [[nodiscard]] inline std::vector<std::string> get_name_components() const {
      auto p = parents;
      p.emplace_back(name);
      return p;
    }

    [[nodiscard]] inline const Capsule& get_capsule() const { return capsule; }

    [[nodiscard]] inline const void* get_data_ptr() const { return data_ptr; }

  };

  /** A typed handle on a transform. */
  template<typename T, typename FloatType, typename LabelType>
  struct TransformHandle {
    using DS = Dataset<FloatType, LabelType>;
    using Self = TransformHandle<T, FloatType, LabelType>;
    const DS* dataset{};
    size_t index{};

    TransformHandle() = default;

    /// Create a handler over a dataset and transform index inside this dataset
    TransformHandle(const DS* ds, size_t idx)
      :
      dataset(ds), index(idx), data((T*) ds->get_transform(idx).get_data_ptr()) {
    }

    [[nodiscard]] const Transform& get_transform() const { return dataset->get_transform(index); }

    [[nodiscard]] const T& get() const { return *data; }

  private:
    const T* data{nullptr};
  };

} // End of namespace tempo