#include <iostream>
#include <unordered_map>

#include <tempo/utils/utils.hpp>
#include <tempo/utils/partasks.hpp>
#include <tempo/utils/utils/timing.hpp>
#include <tempo/utils/jsonvalue.hpp>
#include <tempo/univariate/distances/sed/sed.hpp>
#include <tempo/reader/ts/ts.hpp>
#include <tempo/tseries/dataset.hpp>
#include <filesystem>
#include <fstream>

// --- --- --- Namespaces
using namespace std;
namespace fs = std::filesystem;
namespace tt = tempo::timing;
namespace tu = tempo::univariate;
using namespace tempo::json;

// --- --- --- Types
using FloatType = double;
using LabelType = string;
using TS = tempo::TSeries<FloatType, LabelType>;
using DS = tempo::Dataset<FloatType, LabelType>;
using TH = tempo::TransformHandle<vector<TS>, FloatType, LabelType>;
using distfun_t = function<FloatType(size_t query_idx, size_t candidate_idx, FloatType bsf)>;

variant<string, std::shared_ptr<tempo::Dataset<double, string>>> read_data(ostream& log, fs::path& dataset_path) {
  log << "Loading " << dataset_path << "... ";
  ifstream istream(dataset_path);
  auto start = tempo::timing::now();
  auto res = tempo::reader::TSReader::read(istream);
  auto stop = tempo::timing::now();
  if (res.index()==0) { return {get<0>(res)}; }
  auto tsdata = std::move(get<1>(res));
  cout << "Done in ";
  tempo::timing::printDuration(cout, stop-start);
  cout << endl;
  return {tempo::reader::make_dataset(std::move(tsdata), dataset_path.filename().replace_extension(""))};
}

int main(int argc, char** argv) {
  std::vector<std::string> argList(argv, argv+argc);
  if (argc<3) {
    cout << "<path to ucr> <dataset name> required" << endl;
    exit(1);
  }
  fs::path path_ucr(argList[1]);
  string dataset_name(argList[2]);

  size_t nbp = 1;
  if (argc>=4) { nbp = stoi(argList[3]); }

  // --- Dataset path
  fs::path path_train = path_ucr/dataset_name/(dataset_name+"_TRAIN.ts");
  fs::path path_test = path_ucr/dataset_name/(dataset_name+"_TEST.ts");

  // --- Load the datasets
  shared_ptr<DS> train;
  shared_ptr<DS> test;
  {
    auto res_train = read_data(cout, path_train);
    if (res_train.index()==0) {
      std::cout << "Could not read train set" << endl;
      exit(1);
    }
    train = std::move(get<1>(std::move(res_train)));
    cout << to_string(train->get_header().to_json()) << endl;

    auto res_test = read_data(cout, path_test);
    if (res_test.index()==0) {
      std::cout << "Could not read test set" << endl;
      exit(1);
    }
    test = std::move(get<1>(std::move(res_test)));
    cout << to_string(test->get_header().to_json()) << endl;
  }
  size_t maxl = std::max(train->get_header().get_maxl(), test->get_header().get_maxl());
  size_t minl = std::min(train->get_header().get_minl(), test->get_header().get_minl());
  if (maxl!=minl) {
    std::cout << "Skipping series with unequal lengths" << endl;
    exit(1);
  }

  const auto train_source = train->get_original_handle();
  const auto test_source = test->get_original_handle();

  const size_t tsize = train->size();
  const size_t tlvoo = tsize-1;

  // Parameters
  vector<double> params;
  for (int i = 0; i<100; ++i) { params.push_back(0.01*(double) (i+1)); }
  vector<size_t> best_param;
  size_t best_nbcorrect = 0;
  size_t next_param = 0;

  // Manage tasks
  std::mutex mutex;
  auto loocv_task = [&mutex, &params, &best_param, &best_nbcorrect, &tsize, &tlvoo, &train_source](
    size_t pindex) mutable {
    auto diagw = params[pindex];
    size_t nbcorrect = 0;
    // 'i' is the item left out - being classified
    for (size_t i = 0; i<tsize; ++i) {
      auto query = (*train_source.data)[i];
      double bsf = tempo::POSITIVE_INFINITY<double>;
      size_t bid = i;
      // NN1 loop on the other train
      for (size_t j = 0; j<tsize; ++j) {
        // Skip self
        if (i==j) { continue; }
        auto candidate = (*train_source.data)[j];
        double dist = tu::sed(query, candidate, (double*) nullptr, diagw, bsf);
        if (dist<bsf) {
          bsf = dist;
          bid = j;
        }
      }
      // Check result
      auto nn = (*train_source.data)[bid];
      if (query.get_label().value()==nn.get_label().value()) { nbcorrect++; }
    }
    // --- Check best param
    if (nbcorrect>best_nbcorrect) {
      lock_guard g(mutex);
      best_nbcorrect = nbcorrect;
      best_param.clear();
      best_param.push_back(pindex);
    } else if (nbcorrect==best_nbcorrect) {
      lock_guard g(mutex);
      best_param.push_back(pindex);
    }
    lock_guard g(mutex);
    cout << "Param " << pindex << " = " << diagw << ": nb corrects = "
         << nbcorrect << "/" << tlvoo << " accuracy = "
         << (double) (nbcorrect)/tlvoo << endl;
  };


  // --- --- --- LVOO loop in parallal
  if (nbp!=1) { cout << "Using " << nbp << " threads" << endl; }
  tempo::ParTasks p;
  for (size_t pindex = 0; pindex<params.size(); ++pindex) {
    p.push_task(loocv_task, pindex);
  }
  p.execute(nbp);




  // Report result
  cout << "Best parameter(s): " << best_nbcorrect << "/" << tlvoo << " = " << (double) best_nbcorrect/tlvoo << endl;
  for (auto pi: best_param) {
    cout << "  " << params[pi] << endl;
  }

  {
    // Do NN1 with a parameter
    // --- NN1 classification
    size_t nb_ea{0};
    size_t nb_done{0};
    size_t total = test->size()*train->size();
    size_t centh = total/100;
    size_t tenth = centh*10;
    size_t nbtenth = 0;

    // --- --- --- NN1 loop
    double nb_correct{0};
    tt::duration_t duration{0};
    auto start = tt::now();
    for (const auto& q: *test_source.data) {
      double bsf = tempo::POSITIVE_INFINITY<double>;
      const TS* bcandidates = nullptr;
      for (const auto& c: *train_source.data) {
        double res = tu::sed(q, c, (double*) (nullptr), params[best_param.front()], bsf);
        if (res==tempo::POSITIVE_INFINITY<double>) {
          nb_ea++;
        } else if (res<bsf) { // update BSF
          bsf = res;
          bcandidates = &c;
        }
        nb_done++;
        if (nb_done%centh==0) {
          if (nb_done%tenth==0) {
            nbtenth++;
            std::cout << nbtenth*10 << "% ";
            std::flush(std::cout);
          } else {
            std::cout << ".";
            std::flush(std::cout);
          }
        }
      }
      if (bcandidates!=nullptr && bcandidates->get_label().value()==q.get_label().value()) {
        nb_correct++;
      }
    }
    cout << endl;
    cout << "NN1 test result: " << nb_correct << "/" << test->size() << " = " << (double) nb_correct/test->size()
         << endl;
  }


  // Do NN1 with a parameter
  // --- NN1 classification
  size_t nb_ea{0};
  size_t nb_done{0};
  size_t total = test->size()*train->size();
  size_t centh = total/100;
  size_t tenth = centh*10;
  size_t nbtenth = 0;

  // --- --- --- NN1 loop
  double nb_correct{0};
  tt::duration_t duration{0};
  auto start = tt::now();
  for (const auto& q: *test_source.data) {
    double bsf = tempo::POSITIVE_INFINITY<double>;
    const TS* bcandidates = nullptr;
    for (const auto& c: *train_source.data) {
      double res = tu::sed(q, c, (double*) (nullptr), params[best_param.back()], bsf);
      if (res==tempo::POSITIVE_INFINITY<double>) {
        nb_ea++;
      } else if (res<bsf) { // update BSF
        bsf = res;
        bcandidates = &c;
      }
      nb_done++;
      if (nb_done%centh==0) {
        if (nb_done%tenth==0) {
          nbtenth++;
          std::cout << nbtenth*10 << "% ";
          std::flush(std::cout);
        } else {
          std::cout << ".";
          std::flush(std::cout);
        }
      }
    }
    if (bcandidates!=nullptr && bcandidates->get_label().value()==q.get_label().value()) {
      nb_correct++;
    }
  }
  cout << endl;
  cout << "NN1 test result: " << nb_correct << "/" << test->size() << " = " << (double) nb_correct/test->size() << endl;
}
