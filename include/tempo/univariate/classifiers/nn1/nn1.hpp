#pragma once

#include <tempo/univariate/distances/distances.hpp>
#include <tempo/utils/utils.hpp>
#include <tempo/utils/progressmonitor.hpp>
#include <tempo/tseries/dataset.hpp>
#include <tempo/utils/partasks.hpp>

#include <vector>
#include <string>
#include <algorithm>
#include <mutex>

namespace tempo::univariate {

  /** NN1 function on TSeries. Given a database and a query, returns a vector of labels.
   *  The vector will hold more than one get_label if ties are found, and will be empty if the database is empty.
   * @tparam FloatType        The floating number type used to represent the series.
   * @tparam LabelType        Type of the labels
   * @tparam InputIterator    Type of the InputIterator representing the database
   * @param distance          Distance without cut-off - used for the first pair (query, database[0])
   * @param distance_co       Distance with cut-off - used for the other pair (query, database[i>0])
   * @param begin             Input Iterator pointing on the start of the database
   * @param end               Input Iterator pointing on the end of the database
   * @param query             Query whose get_label is to be determined
   * @return                  Vector of labels, containing more than 1 if ties occur, and 0 if the database is empty.
   */
  template<typename FloatType, typename LabelType, typename Funtype, typename InputIterator>
  [[nodiscard]] std::vector<LabelType> nn1(
    const Funtype& distance_co,
    InputIterator begin, InputIterator end,
    const typename InputIterator::value_type& query) {
    /*
    // Static check the value type parameter of the iterator type
    static_assert(
            stassert::is_iterator_value_type<TSPack<FloatType, LabelType>, InputIterator>,
            "Iterator does not contain TSPack<FloatType, LabelType>");
            */
    // Check if the database isn't empty, else immediately return an empty vector
    if (begin!=end) {
      // Use the distance without cut-off to compute the first pair.
      auto bsf = POSITIVE_INFINITY<FloatType>;
      std::vector<LabelType> labels{};
      // Keep going, exhaust the database
      while (begin!=end) {
        // We can now use the distance with a cutoff
        const auto& candidate = *begin;
        auto result = distance_co(candidate, query, bsf);
        if (result<bsf) {
          labels.clear();
          labels.emplace_back(candidate.get_label().value());
          bsf = result;
        } else if (bsf==result) { // Manage ties
          const auto& l = candidate.get_label().value();
          if (std::none_of(labels.begin(), labels.end(), [l](const auto& v) { return v==l; })) {
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

  template<typename FloatType>
  using nn1dist_t = std::function<FloatType(size_t idxTrain, size_t idxTest, FloatType bsf)>;

  template<typename FloatType, typename LabelType>
  size_t nn1(
    const Dataset<FloatType, LabelType>& train,
    const Dataset<FloatType, LabelType>& test,
    nn1dist_t<FloatType> distance,
    size_t nbthread = 1,
    std::ostream* out = nullptr
  ) {
    std::mutex mutex;
    size_t nb_correct{0};
    size_t nb_done{0};
    tempo::ProgressMonitor pm(train.size()*test.size());

    // --- --- --- NN1 task
    auto nn1task = [out, &mutex, &pm, &nb_done, &train, &test, &distance](size_t idxTest) -> LabelType {
      double bsf = tempo::POSITIVE_INFINITY<FloatType>;
      size_t bcandidate = train.size()+1;
      // --- --- --- Try first:
      std::set<size_t> try_first{};
      for(const auto& [key, val]: train.get_by_class()){
        const size_t top = std::min<size_t>(2, val.size());
        for(size_t idxOfIdx{0}; idxOfIdx<top; idxOfIdx++){
          const size_t idxTrain = val[idxOfIdx];
          try_first.insert(idxTrain);
          double res = distance(idxTrain, idxTest, bsf);
          if (res<bsf) {
            bsf = res;
            bcandidate = idxTrain;
          }
        }
      }

      // --- --- --- NN1 loop
      for (size_t idxTrain{0}; idxTrain<train.size(); idxTrain++) {
        //if(tempo::contains(try_first, idxTrain)){continue;}
        double res = distance(idxTrain, idxTest, bsf);
        if (res<bsf) {
          bsf = res;
          bcandidate = idxTrain;
        }
        // --- --- ---
        if (out!=nullptr) {
          std::lock_guard lg(mutex);
          nb_done++;
          pm.print_progress(*out, nb_done);
        }
      }
      // --- --- ---
      return train.get_original()[bcandidate].get_label().value();
    };

    // --- --- --- Task generator
    auto nn1task_gen = [&mutex, &test, &nn1task, &nb_correct, idxTest = 0]() mutable {
      if (idxTest<test.size()) {
        tempo::ParTasks::task_t task = [&test, idxTest, &nn1task, &mutex, &nb_correct]() {
          LabelType l = nn1task(idxTest);
          if (l==test.get_original()[idxTest].get_label().value()) {
            std::lock_guard lg(mutex);
            nb_correct++;
          }
        };
        idxTest++;
        return std::optional<tempo::ParTasks::task_t>(task);
      } else {
        return std::optional<tempo::ParTasks::task_t>();
      }
    };

    // --- --- ---
    tempo::ParTasks p;
    p.execute(nbthread, nn1task_gen);
    return nb_correct;
  }

}