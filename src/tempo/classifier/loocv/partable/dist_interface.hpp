#pragma once

#include <tempo/dataset/dts.hpp>

#include <functional>

namespace tempo::classifier::nn1loocv {

  /// Given a query, a candidate, and a parameter index, compute the distance between the query and the candidate.
  /// Can be early abandoned with 'bsf' (best so far)
  using distParam_ft = std::function<F(size_t train_idx1, size_t train_idx2, size_t param_idx, F bsf)>;

  /// Same as above, but must produce an Upper Bound (UB) of the distance.
  /// For elastic distances using a cost matrix, this is usually done by taking the diagonal of the cost matrix,
  /// i.e. a form of direct alignment (eventually completed along the last line/column for disparate lengths)
  using distUB_ft = std::function<F(size_t train_idx1, size_t train_idx2, size_t param_idx)>;

  /// Test function - must capture the best parameterization found by LOOCV
  using distTest_ft = std::function<F(size_t tst_idx, size_t train_idx, F bsf)>;

  /// result structure for LOOCV train and test accuracy
  struct result_LOOCVDist {
    size_t size;
    size_t nb_correct;
    double accuracy;
    tempo::utils::duration_t time;

    nlohmann::json to_json() {
      nlohmann::json j;
      j["size"] = size;
      j["nb_correct"] = nb_correct;
      j["accuracy"] = accuracy;
      j["time_ns"] = time.count();
      j["time_human"] = tempo::utils::as_string(time);
      return j;
    }
  };

  struct i_LOOCVDist {

    /// Number of parameter to evaluate
    size_t nb_params;

    /// Train result - updated by partable
    result_LOOCVDist result_train;

    /// Test result - updated by partable
    result_LOOCVDist result_test;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor / Destructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Build this with the number of parameters to evaluate
    explicit i_LOOCVDist(size_t nbp) : nb_params(nbp) {}

    virtual ~i_LOOCVDist() = default;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Interface
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Given a query, a candidate, and a parameter index, compute the distance between the query and the candidate.
    /// Can be early abandoned with 'bsf' (best so far)
    /// Operate on train indexes.
    virtual F distance_param(size_t train_idx1, size_t train_idx2, size_t param_idx, F bsf) = 0;

    /// Same as above, but must produce an Upper Bound (UB) of the distance.
    /// For elastic distances using a cost matrix, this is usually done by taking the diagonal of the cost matrix,
    /// i.e. a form of direct alignment (eventually completed along the last line/column for disparate lengths)
    /// Operate on train indexes.
    virtual F distance_UB(size_t train_idx1, size_t train_idx2, size_t param_idx) = 0;

    /// The LOOCV process will produce a collection of best parameter ids (more than one due to ties),
    /// and the number of correctly classified instance.
    /// Also record the time taken.
    /// This function is supposed to generate a unique parameter (e.g. minimum, median, etc...) based on the
    /// collection of best parameters.
    /// Note: this->result_train is now valid
    virtual void set_loocv_result(std::vector<size_t> bestp) = 0;

    /// Test function - must capture the best parameterization as computed by set_loocv_result'
    /// Operate and the test index and the train index. Test index is first!
    virtual F distance_test(size_t test_idx, size_t train_idx, F bsf) = 0;

  };

}
