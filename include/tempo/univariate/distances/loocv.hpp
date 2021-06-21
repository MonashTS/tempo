#pragma once

#include <tempo/tseries/dataset.hpp>
#include <tempo/utils/progressmonitor.hpp>
#include <tempo/utils/utils/timing.hpp>
#include <tempo/utils/partasks.hpp>

#include <functional>
#include <vector>
#include <mutex>
#include <tuple>

namespace tempo::univariate {


  /// LOOCV Task.
  /// A function learning on a (captured) train dataset, excluding  leftout_index, with param.
  /// Then, classify leftout_index.
  /// @param leftout_index    in [0, train_size[, given a LOOCV task on a train dataset of size train_size
  /// @return true if the classification of leftout succeeded, else false.
  template<typename ParamType>
  using LOOCVTask = std::function<bool(
    size_t leftout_index,
    const ParamType& param
  )>;

  /// LOOCV process.
  /// @return a tuple of (vector with best paramter index, best number of correctly classified instance)
  template<typename ParamType>
  std::tuple<std::vector<size_t>, size_t> do_loocv(
    const std::vector<ParamType>& params,      // Vector of parameter to test. The return indexes are into this vector
    size_t train_size,                         // Size of the train dataset. 'leftout' index is in [0, train_size[
    const LOOCVTask<ParamType>& loocv_task,    // The LOOCV task
    size_t nbthreads=1,
    std::ostream* out= nullptr
  ){
    std::mutex mutex;
    namespace tt = tempo::timing;

    // --- Constants
    const size_t params_size = params.size();
    // --- Best results
    std::vector<size_t> best_params;
    size_t best_nbcorrect{0};
    // --- For each parameter, generate a task per left out index
    for (size_t param_index{0}; param_index<params_size; ++param_index) {
      size_t nbcorrect{0};
      size_t nbdone{0};
      tempo::ParTasks p;
      tempo::ProgressMonitor pm{train_size};
      // --- Wrap the provided LOOCVTask in a task tracking the number of correct classification + progress monitor
      auto task = [&params, &mutex, &loocv_task, &pm, &nbcorrect, &nbdone, &out](size_t idparam, size_t leftout) {
        // Call the LOOCV task
        if (loocv_task(leftout, params[idparam])) {
          std::lock_guard g(mutex);
          nbcorrect++;
        }
        // Update progress
        if(out!=nullptr){
          std::lock_guard lg(mutex);
          nbdone++;
          pm.print_progress(*out, nbdone);
        }
      };
      // --- Initial output
      if(out !=nullptr){
        *out << "pidx " << param_index << ": ";
      }
      // --- Generate the tasks
      for (size_t leftout_index{0}; leftout_index<train_size; ++leftout_index) {
        p.push_task(task, param_index, leftout_index);
      }
      // --- Execute the tasks
      auto start = tt::now();
      p.execute(nbthreads);
      auto duration = tt::now()-start;
      // --- Check best param
      if (nbcorrect>best_nbcorrect) {
        best_nbcorrect = nbcorrect;
        best_params.clear();
        best_params.push_back(param_index);
      } else if (nbcorrect==best_nbcorrect) { best_params.push_back(param_index); }

      // --- Final output
      if(out!=nullptr) {
        *out << std::endl << "     nb corrects = " << nbcorrect << "/" << train_size << " accuracy = "
             << (double) (nbcorrect)/train_size << "  (" << tt::as_string(duration) << ")" << std::endl;
      }
    }
    return {std::move(best_params), best_nbcorrect};
  }

} // End of namespace tempo::univariate