#pragma once

#include <tempo/utils/utils.hpp>

#include <armadillo>

namespace tempo {

  namespace { // Unnamed namespace: visibility local to the file (when in header, do not declare variable here!)
    namespace lu = tempo::utils;
  }

  class TSeries {
    /// Missing data in the time series? We use a floating point type, so should be represented by "nan"
    bool _missing{false};

    /// Optional label
    std::optional<std::string> _olabel{};

    /// Capsule: used when we own the data
    lu::Capsule _capsule;

    /// Raw Data pointer: the matrix must be built on top of that. Must be Column Major.
    F const *_rawdata{nullptr};

    /// Representation of the matrix - by default, 1 line (univariate), 0 cols (empty)
    arma::Mat<F> _matrix{1, 0};

    // --- Statistics
    arma::Col<F> _min;        /// Min value per dimension
    arma::Col<F> _max;        /// Max value per dimension
    arma::Col<F> _mean;       /// Mean value per dimension
    arma::Col<F> _median;     /// Median value per dimension
    arma::Col<F> _stddev;     /// Standard deviation per dimension

    /// Private "moving-in" constructor
    TSeries(
      // Column major data
      lu::Capsule&& c,
      F const *p,
      arma::Mat<F>&& m,
      // Other
      std::optional<std::string> olabel,
      bool has_missing
    ) :
      _missing(has_missing),
      _olabel(std::move(olabel)),
      _capsule(std::move(c)),
      _rawdata(p),
      _matrix(std::move(m)) {

      // Pointer fixup
      if(_rawdata ==nullptr){ _rawdata = _matrix.memptr(); }

      // Statistics (matrix, 1==along the row)
      // Doing the statistics along the rows restul in a column vector, with statistics per dimension.
      _min = arma::min(_matrix, 1);
      _max = arma::max(_matrix, 1);
      _mean = arma::mean(_matrix, 1);
      _median = arma::median(_matrix, 1);
      // Note:  Second argument is norm_type = 0: performs normalisation using N-1 (signal sampled in the "population")
      //        Third argument means "along the row"
      _stddev = arma::stddev(_matrix, 0, 1);
      //
    }

  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Construction
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Default constructor: create an univariate empty time series */
    TSeries() = default;

    /** Default move constructor */
    TSeries(TSeries&&) noexcept = default;

    /** Disable copy (we have to move instead; prevent unwanted data duplication) */
    TSeries(const TSeries&) = delete;

    /// Build a new univariate series from an arma::Row<F>
    static TSeries mk_from(arma::Row<F>&& v, std::optional<std::string> olabel, std::optional<bool> omissing) {
      // Build matrix from incoming row vector
      arma::Mat<F> matrix(std::move(v));
      // Check missing data (NAN)
      bool has_missing;
      if (omissing.has_value()) { has_missing = omissing.value(); }
      else { has_missing = matrix.has_nan(); }
      //
      return TSeries({}, nullptr, std::move(matrix), olabel, has_missing);
    }

    /// Build a new univariate series copying its info from an existing TSeries,
    /// and getting its data from an arma::Row<F>
    /// Usefull when transforming series into other series (e.g. normalisation)
    /// Does not check if the info from other actually match v.
    static TSeries mk_from(TSeries const& other, arma::Row<F>&& v) {
      return mk_from(std::move(v), other.label(), {other.missing()});
    }

    /// Build a new series from a row major vector
    static TSeries mk_from_rowmajor(std::vector<F>&& v,
                                    size_t nbvar,
                                    std::optional<std::string> olabel,
                                    std::optional<bool> omissing) {
      using namespace std;

      // --- Checking
      size_t vsize = v.size();
      if (nbvar<1) { throw domain_error("Number of variable can't be < 1"); }
      if (vsize%nbvar!=0) { throw domain_error("Vector size is not a multiple of 'nbvar'"); }

      // --- lico
      const size_t nb_lines = nbvar;
      const size_t nb_cols = vsize/nbvar;

      // --- Take ownership of the incoming vector
      auto capsule = lu::make_capsule<vector<F>>(std::move(v));

      // --- Build Armadillo matrix
      // Armadillo works with column major data, but we are given a row major one.
      // Build the data, and proceed with an "in place" transposition
      // This transposition **will** change the underlying vector, which is fine.
      F *rawptr = lu::get_capsule_ptr<vector<F>>(capsule)->data();
      // Invert line/column here: we get the "right" matrix after transposition
      arma::Mat<F> matrix(rawptr, nb_cols, nb_lines,
                          false,   // copy_aux_mem = false: use the auxiliary memory (i.e. no copying)
                          true     // struct = true: matrix bounds to the auxiliary memory for its lifetime; can't be resized
      );

      // Transpose
      inplace_trans(matrix);
      assert(nb_lines==matrix.n_rows);
      assert(nb_cols==matrix.n_cols);

      // Check missing data (NAN)
      bool has_missing;
      if (omissing.has_value()) { has_missing = omissing.value(); }
      else { has_missing = matrix.has_nan(); }

      // Build the TSeries
      return TSeries(std::move(capsule), rawptr, std::move(matrix), olabel, has_missing);
    }

    /// Copy data from other, except for the actual data. Allow to easily do transforms. No checking done.
    static TSeries mk_from_rowmajor(TSeries const& other, std::vector<F>&& v) {
      return mk_from_rowmajor(std::move(v), other.nb_dimensions(), other.label(), {other.missing()});
    }



    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Basic access
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Access number of variable
    size_t nb_dimensions() const { return _matrix.n_rows; }

    /// Check if a series is univariate
    bool is_univariate() const { return nb_dimensions() == 1;}

    /// Access the length
    size_t length() const { return _matrix.n_cols; }

    /// Access the size of the data == ndim*length
    size_t size() const { return _matrix.n_elem; }

    /// Check if has missing values
    bool missing() const { return _missing; }

    /// Get the label (perform a copy)
    std::optional<std::string> label() const { return _olabel; }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Data access
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Column major access over all the points AND dimensions
    F operator [](size_t idx) const { return _matrix(idx); }

    /// Column major access to the raw pointer
    const F *data() const { return _rawdata; }

    /// Matrix access (li, co)
    const arma::Mat<F>& matrix() const { return _matrix; }

    /// As row vector, only for univariate
    arma::Row<F> rowvec() const {
      if (!is_univariate()) { throw std::logic_error("rowvec can only be used with univariate series"); }
      return arma::Row<F>(matrix());
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Statistic access
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Minimum value per dimension
    const arma::Col<F>& min() const { return _min; };

    /// Maximum value per dimension
    const arma::Col<F>& max() const { return _max; };

    /// Mean value per dimension
    const arma::Col<F>& mean() const {
      assert(_mean.size()>0);
      return _mean;
    };

    /// Median value per dimension
    const arma::Col<F>& median() const { return _median; };

    /// Standard deviation per dimension
    const arma::Col<F>& stddev() const { return _stddev; };


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Series transformation
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Univariate Map low level transformation.
     * @param fun univariate function to apply to the series. Works on the totality of the raw data.
     *            See other map function for multivariate transforms
     * @return a new transformed series
     */
    TSeries map(typename std::function<void(F const *in, size_t total_size, F *out)> fun) const {
      assert(is_univariate());
      arma::Row<F> arow(size());
      fun(data(), size(), arow.memptr());
      return mk_from(std::move(arow), label(), missing());
    }

    auto static mapfun(std::function<void(F const *in, size_t total_size, F *out)> fun) {
      return [fun](TSeries const& in) -> TSeries { return in.map(fun); };
    }

    /** Multivariate Map low level transformation - remember that data is stored in column major order
     * @param fun multivariate function to apply to the series. Works on the totality of the raw data.
     * @return a new transformed series
     */
     /*
    TSeries map(typename std::function<void(F const *in, size_t nbdim, size_t length, F *out)> fun) const {
      throw std::runtime_error("not implemented");
    }
    */

  }; // End of class TSeries

} // End of namespace tempo