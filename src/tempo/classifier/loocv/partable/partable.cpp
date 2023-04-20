#include "partable.hpp"

#include <tempo/utils/utils.hpp>

namespace tempo::classifier::nn1loocv {

  std::tuple<std::vector<size_t>, size_t> partable(
    distParam_ft distance,
    distUB_ft distanceUB,
    size_t nbtrain,
    DatasetHeader const& train_header,
    size_t nbparams,
    size_t nbthreads,
    std::ostream *out
  ) {
    const size_t NBLINE = nbtrain;
    const size_t NBCOL = nbparams;
    const size_t NBCELL = NBLINE*NBCOL;

    // NNTable: one line per series, |params| column.
    // At each column, register the closest NN ID and the associated distance
    std::vector<NNC> NNTable(NBCELL);
    for (size_t i = 0; i<NBCELL; ++i) {
      NNTable[i].NNindex = NBLINE;
      NNTable[i].NNdistance = tempo::utils::PINF;
    }

    // Ordered set of added series, sorted by their (approximated) descending NN distance in NNTable[T, last]
    // i.e. the first series has a large distance to its NN, and the last one has a small distance to its NN.
    std::vector<size_t> intable;
    intable.reserve(NBLINE);

    auto update = [&](size_t table_idx, size_t nnindex, double nndist) {
      auto& nn = NNTable[table_idx];
      std::lock_guard lock(nn.mutex);
      if (nndist<nn.NNdistance) {
        nn.NNindex = nnindex;
        nn.NNdistance = nndist;
      }
    };

    // --- --- --- Add the first pair
    {
      // Start with the upper bound
      {
        const size_t LASTP = nbparams - 1;
        F UB = distanceUB(0, 1, LASTP);
        F d_last = distance(0, 1, LASTP, UB);
        update(0*NBCOL + NBCOL - 1, 1, d_last);
        update(1*NBCOL + NBCOL - 1, 0, d_last);
      }
      // Complete the table cutting with the UB (we are not going to EA anything here)
      for (int pidx = (int)NBCOL - 2; pidx>=0; --pidx) {
        F UB = NNTable[0*NBCOL + pidx + 1].NNdistance;
        F d = distance(0, 1, pidx, UB);
        update(0*NBCOL + pidx, 1, d);
        update(1*NBCOL + pidx, 0, d);
      }
      intable.push_back(0);
      intable.push_back(1);
    }

    tempo::utils::ParTasks ptask;

    // --- --- --- Add the other series
    for (size_t S = 2; S<NBLINE; ++S) {
      auto start = tempo::utils::now();

      // --- Complete with other series already in the table
      // This loop generates a set of task (one per Ti)
      for (size_t Ti : intable) {
        // --- Define the tasks
        auto task = [&, Ti]() {
          // Max bound: if above this, S and T cannot be each other NN
          const F dmax = std::max(NNTable[S*NBCOL + NBCOL - 1].NNdistance, NNTable[Ti*NBCOL + NBCOL - 1].NNdistance);
          // Start the process with the first parameters, and no lower bound
          size_t Pi = 0;
          F LB = 0;
          do {
            const F d_S = NNTable[S*NBCOL + Pi].NNdistance;
            const F d_Ti = NNTable[Ti*NBCOL + Pi].NNdistance;
            const F d_nn = std::max(d_S, d_Ti);
            if (LB<d_nn) {
              const F cutoff = std::min(dmax, distanceUB(S, Ti, Pi));
              const F di = distance(S, Ti, Pi, cutoff);
              update(S*NBCOL + Pi, Ti, di);
              update(Ti*NBCOL + Pi, S, di);
              if (di==tempo::utils::PINF) { Pi = NBCOL; }
              else { LB = di; }
            }
            Pi++;
          } while (Pi<NBCOL);
        };
        // --- Add the tasks
        ptask.push_task(std::move(task));
      }

      // --- Execute the tasks in parallel
      ptask.execute(nbthreads);

      // --- Put S in intable, maintaining approximate descending order on NNTable[S*NBCOL+NBCOL-1]
      intable.push_back(S);
      for (size_t Tidx = intable.size() - 1; Tidx>=1; --Tidx) {
        const size_t Ti = intable[Tidx];
        const size_t Tiprev = intable[Tidx - 1];
        if (NNTable[Ti*NBCOL + NBCOL - 1].NNdistance>NNTable[Tiprev*NBCOL + NBCOL - 1].NNdistance) {
          std::swap(intable[Ti], intable[Tiprev]);
        }
      }

      tempo::utils::duration_t duration = tempo::utils::now() - start;
      if (out!=nullptr) {
        std::ostream& o = *out;
        o << S << "/" << NBLINE << " " << tempo::utils::as_string(duration) << std::endl;
      }

    }// End of Table filling


    // --- --- --- NNTable is full: find the param with the fewest error
    std::vector<size_t> result;
    size_t bestError = std::numeric_limits<size_t>::max();
    {
      for (size_t pidx = 0; pidx<NBCOL; ++pidx) {
        size_t nError = 0;
        for (size_t Ti = 0; Ti<NBLINE; Ti++) {
          if (train_header.label(Ti).value()!=train_header.label(NNTable[Ti*NBCOL + pidx].NNindex).value()) {
            nError++;
          }
        }
        //
        if (nError<bestError) {
          result.clear();
          result.push_back(pidx);
          bestError = nError;
        } else if (nError==bestError) { result.push_back(pidx); }
      }
    }

    return {result, NBLINE - bestError};
  }

  void partable(
    i_LOOCVDist& instance,
    size_t nbtrain, // May be smaller than train_header.size() for a subset
    DatasetHeader const& train_header,
    size_t nbtest,  // May be smaller than test_header.size() for a subset
    DatasetHeader const& test_header,
    PRNG& prng,
    size_t nbthreads,
    std::ostream *out
  ) {

    // --- --- --- Get our functions from the instance
    distParam_ft distance = [&instance](size_t train_idx1, size_t train_idx2, size_t param_idx, F bsf) {
      return instance.distance_param(train_idx1, train_idx2, param_idx, bsf);
    };

    distUB_ft distanceUB = [&instance](size_t train_idx1, size_t train_idx2, size_t param_idx) {
      return instance.distance_UB(train_idx1, train_idx2, param_idx);
    };

    distTest_ft distanceTest = [&instance](size_t train_idx1, size_t train_idx2, F bsf) {
      return instance.distance_test(train_idx1, train_idx2, bsf);
    };

    // --- --- --- LOOCV process
    {
      auto start = tempo::utils::now();
      auto [loocv_params, loocv_nbcorrect] =
        partable(distance, distanceUB, nbtrain, train_header, instance.nb_params, nbthreads, out);
      tempo::utils::duration_t loocv_time = tempo::utils::now() - start;

      // Write result
      result_LOOCVDist result_train{};
      result_train.size = nbtrain;
      result_train.nb_correct = loocv_nbcorrect;
      result_train.accuracy = (double)loocv_nbcorrect/(double)nbtrain;
      result_train.time = loocv_time;
      instance.result_train = result_train;
      instance.set_loocv_result(loocv_params);
    }

    // --- --- --- Test parameterization
    {
      const size_t train_size = train_header.size();
      const size_t test_size = test_header.size();
      size_t test_nb_correct = 0;

      // --- Progress reporting
      tempo::utils::ProgressMonitor pm(test_size);    // How many to do, both train and test accuracy
      size_t nb_done = 0;                             // How many done up to "now"

      // --- Multithreading control
      tempo::utils::ParTasks ptasks;
      std::mutex mutex;

      // --- NN1 test task with tie management
      auto nn1_test_task = [&](size_t test_idx) mutable {
        double bsf = tempo::utils::PINF;
        std::set<tempo::EL> labels{};       // manage ties
        for (size_t train_idx{0}; train_idx<train_size; train_idx++) {
          double d = distanceTest(test_idx, train_idx, bsf);
          if (d<bsf) { // Best: clear labels and insert new
            labels.clear();
            labels.insert(train_header.label(train_idx).value());
            bsf = d;
          } else if (d==bsf) { // Same: add label in
            labels.insert(train_header.label(train_idx).value());
          }
        }
        // --- Update accuracy
        {
          std::lock_guard lock(mutex);
          tempo::EL result = -1;
          std::sample(labels.begin(), labels.end(), &result, 1, prng);
          assert(result<train_header.nb_classes());
          if (result==test_header.label(test_idx).value()) { ++test_nb_correct; }
          nb_done++;
          pm.print_progress(out, nb_done);
        }
      };

      // --- Create the tasks per tree. Note that we clone the state.
      tempo::utils::ParTasks p;
      auto test_start = tempo::utils::now();
      p.execute(nbthreads, nn1_test_task, 0, test_size, 1);
      tempo::utils::duration_t test_time = tempo::utils::now() - test_start;

      // --- Write result
      result_LOOCVDist result_test{};
      result_test.size = nbtest;
      result_test.nb_correct = test_nb_correct;
      result_test.accuracy = (double)test_nb_correct/(double)nbtest;
      result_test.time = test_time;
      instance.result_test = result_test;
    }

  } // End of function partable


}
