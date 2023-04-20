#pragma once

#include <tempo/dataset/dataset.hpp>

#include <tempo/predef.hpp>

#include <armadillo>

namespace tempo::classifier {

  /// Classifier result for one test exemplar
  struct Result1 {

    /// A row vector representing the classification result of one test exemplar.
    /// In the row vector, the ith column represent the probability of the ith class,
    /// where the ith class is determined by a label encoder.
    arma::rowvec probabilities;

    /// Weight associated to the probabilities
    double weight{};

    Result1() = default;

    Result1(Result1 const&) = default;

    Result1& operator =(Result1 const&) = default;

    Result1(Result1&&) = default;

    Result1& operator =(Result1&&) = default;

    inline Result1(arma::rowvec&& proba, double weight) : probabilities(std::move(proba)), weight(weight) {}

    inline explicit Result1(size_t nbclasses) : probabilities(nbclasses, arma::fill::zeros), weight(0) {}

    /// Create a Result for 'cardinality' classes, with the index (coming from a label encoder) of 'proba_at_one'
    /// being 1.0, and all the other at 0. The weight is copied as is.
    static inline Result1 make_probabilities_one(size_t cardinality, EL proba_at_one, double weight) {
      arma::rowvec p(cardinality, arma::fill::zeros);
      p[proba_at_one] = 1.0;
      return Result1(std::move(p), weight);
    }

    /// Create a Result for 'cardinality' classes, with smooth cardinality.
    /// Count 'one' for each classes, add 'weight' (must be a count) to 'top_proba'
    static inline Result1 make_smooth_probabilities(size_t cardinality, EL top_proba, double weight) {
      arma::rowvec p(cardinality, arma::fill::ones);
      p[top_proba] += weight;
      double total = (double)cardinality+weight;
      p = p/total;
      return Result1(std::move(p), total);
    }

    /// Obtain the classes with the max probability
    inline std::tuple<std::vector<EL>, double> most_probable_classes(){
      double maxv = probabilities.max();
      std::vector<size_t> maxp;
      for (size_t c{0}; c<probabilities.n_cols; ++c) { if (probabilities[c]==maxv) { maxp.push_back(c); }}
      return {maxp, maxv};
    }

  };

  /// Classifier result for several test exemplars
  struct ResultN {

    /// A matrix representing the classification result of several test exemplar.
    /// A row represents the classification result of one test exemplar.
    /// In the row vector, the ith column represent the probability of the ith class,
    /// where the ith class is determined by a label encoder.
    arma::mat probabilities{};

    /// Column vector of length nb_test_exemplars, indicating the "weights" (can be used to indicate confidence)
    /// associated to each row from the probabilities matrix
    arma::colvec weight{};

    ResultN() = default;

    ResultN(ResultN const&) = default;

    ResultN& operator =(ResultN const&) = default;

    ResultN(ResultN&&) = default;

    ResultN& operator =(ResultN&&) = default;

    inline void append(Result1 const& res1) {
      size_t n_rows = probabilities.n_rows;
      probabilities.insert_rows(n_rows, res1.probabilities);
      weight.insert_rows(n_rows, res1.weight);
    }

    inline size_t nb_correct_01loss(DatasetHeader const& test_header, IndexSet const& test_iset, PRNG& prng) {
      size_t nb_correct = 0;

      for (size_t r{0}; r<probabilities.n_rows; ++r) {
        // Find max probabilities, break ties randomly
        auto row = probabilities.row(r);
        double maxv = row.max();
        std::vector<size_t> maxp;
        for (size_t c{0}; c<probabilities.n_cols; ++c) { if (row[c]==maxv) { maxp.push_back(c); }}
        // Predicted and true encoded label
        EL predicted_elabel = utils::pick_one(maxp, prng);
        EL true_elabel = test_header.label(test_iset[r]).value();
        // Compare
        if (predicted_elabel==true_elabel) { nb_correct++; }
      }

      return nb_correct;
    }

  };

} // end of namespace tempo::classifier