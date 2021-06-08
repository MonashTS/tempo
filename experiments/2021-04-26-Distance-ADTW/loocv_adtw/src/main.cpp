#include <iostream>

#include <tempo/utils/utils.hpp>
#include <tempo/utils/partasks.hpp>
#include <tempo/utils/progressmonitor.hpp>
#include <tempo/utils/utils/timing.hpp>
#include <tempo/utils/jsonvalue.hpp>
#include <tempo/univariate/distances/dtw/adtw.hpp>
#include <tempo/reader/ts/ts.hpp>
#include <tempo/tseries/dataset.hpp>
#include <tempo/tseries/indexSet.hpp>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <tempo/univariate/transforms/derivative.hpp>

// --- --- --- Namespaces
using namespace std;
namespace fs = std::filesystem;
namespace tt = tempo::timing;
namespace tu = tempo::univariate;
using namespace tempo::json;

// --- --- --- Types
using FloatType = double;
using LabelType = string;
using PRNG = std::mt19937_64;
using TS = tempo::TSeries<FloatType, LabelType>;
using DS = tempo::Dataset<FloatType, LabelType>;
using TH = tempo::TransformHandle<vector<TS>, FloatType, LabelType>;
using distfun_t = function<FloatType(size_t query_idx, size_t candidate_idx, FloatType bsf)>;

FloatType pickvalue(const TH& th, PRNG& prng) {
  const auto& s = tempo::rand::pick_one(*th.data, prng);
  auto distribution = std::uniform_int_distribution<int>(0, s.length()-1);
  return s(0, distribution(prng));
}

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
  const size_t SAMPLE_SIZE = 100000;

  std::vector<std::string> argList(argv, argv+argc);
  if (argc<6) {
    cout << "<path to ucr> <dataset name> <derivative|original> <nbthreads> <output> required" << endl;
    exit(1);
  }
  fs::path path_ucr(argList[1]);
  string dataset_name(argList[2]);
  string transform(argList[3]);
  size_t nbthreads = stoi(argList[4]);
  string outpath = argList[5];

  std::random_device rd;
  PRNG prng(rd());

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

  // --- --- --- --- Original
  const auto train_source_o = train->get_original_handle();
  const auto test_source_o = test->get_original_handle();
  const auto is_train = tempo::IndexSet(*train);
  auto train_source = train_source_o;
  auto test_source = test_source_o;

  // --- --- --- --- Derivative
  if (transform=="derivative") {
    tu::DerivativeTransformer<FloatType, LabelType> deriver1(1);
    const auto train_source_d1 = deriver1.transform_and_add(train->get_original_handle());
    const auto test_source_d1 = deriver1.transform_and_add(test->get_original_handle());
    train_source = train_source_d1;
    test_source = test_source_d1;
  } else if (transform!="original") {
    cout << "<path to ucr> <dataset name> <derivative|original> <fixed|weighted> <nbthreads> <output> required" << endl;
    cout << "transform found: " << transform << endl;
    exit(1);
  }

  const size_t train_size = train->size();

  double sampled_mean_dist = 0;
  tempo::stats::StddevWelford welford;
  {
    auto start = tt::now();
    // --- --- --- Distance sampling
    for (int i = 0; i<SAMPLE_SIZE; ++i) {
      // Pick two random values from the train set (random series, random point) and compute the distance
      FloatType a = pickvalue(train_source, prng);
      FloatType b = pickvalue(train_source, prng);
      welford.update(tempo::univariate::square_dist(a, b));
    }
    sampled_mean_dist = welford.get_mean();
  }


  // --- --- --- Create parameters
  vector<tuple<double, double>> params;
  for (int i = -100; i<=100; ++i) {
    double r = (double) i/100;
    params.emplace_back(make_tuple(r, r*sampled_mean_dist));
  }

  // --- --- --- LOOCV task per left out
  std::mutex mutex;
  // --- --- --- Task to NN1 one series
  auto loocv_task_i = [&mutex, &params, &train_size, &train_source](
    size_t param_index, size_t leftout_index, size_t* nbcorrect,
    tempo::ProgressMonitor& pm,
    size_t* nb_done) {
    const double weight = get<1>(params[param_index]);
    auto query = (*train_source.data)[leftout_index];
    double bsf = tempo::POSITIVE_INFINITY<double>;
    size_t bid = leftout_index;
    // NN1 loop on the other train
    for (size_t j = 0; j<train_size; ++j) {
      // Skip self
      if (j==leftout_index) { continue; }
      const auto candidate = (*train_source.data)[j];
      double dist = tu::adtw(query, candidate, weight, bsf);
      if (dist<bsf) {
        bsf = dist;
        bid = j;
      }
    }
    // Check result
    auto nn = (*train_source.data)[bid];
    if (query.get_label().value()==nn.get_label().value()) {
      lock_guard g(mutex);
      (*nbcorrect)++;
    }
    {
      std::lock_guard lg(mutex);
      (*nb_done)++;
      pm.print_progress(std::cout, *nb_done);
    }
  };

  // --- --- --- LOOCV
  cout << dataset_name << " Using " << nbthreads << " threads" << endl;
  vector<size_t> best_param;
  size_t best_nbcorrect = 0;
  tempo::ParTasks p;
  auto loocv_start = tt::now();
  for (size_t pindex = 0; pindex<params.size(); ++pindex) {
    size_t nbcorrect = 0;
    size_t nbdone = 0;
    tempo::ProgressMonitor pm(train_size);
    for (size_t i = 0; i<train_size; ++i) {
      p.push_task(loocv_task_i, pindex, i, &nbcorrect, pm, &nbdone);
    }
    cout << endl << dataset_name << " Param " << pindex << " with g = " << get<0>(params[pindex]);
    cout << " with penalty = " << get<1>(params[pindex]) << ":" << endl;
    auto start = tt::now();
    p.execute(nbthreads);
    auto duration = tt::now()-start;
    // --- Check best param
    if (nbcorrect>best_nbcorrect) {
      best_nbcorrect = nbcorrect;
      best_param.clear();
      best_param.push_back(pindex);
    } else if (nbcorrect==best_nbcorrect) {
      best_param.push_back(pindex);
    }
    cout << endl << "nb corrects = " << nbcorrect << "/" << train_size << " accuracy = "
         << (double) (nbcorrect)/train_size << "  (" << tt::as_string(duration) << ")" << endl;
  }
  auto loocv_duration = tt::now()-loocv_start;



  // Report result
  cout << dataset_name << " Best parameter(s): " << best_nbcorrect << "/" << train_size << " = "
       << (double) best_nbcorrect/train_size << "  (" << tt::as_string(loocv_duration) << ")" << endl;
  // Sort required: multithread does not guarantee order on insertion in the vector.
  std::sort(best_param.begin(), best_param.end());
  for (auto pi: best_param) { cout << "  " << get<0>(params[pi]) << endl; }

  double bestg;

  {
    auto size = best_param.size();
    if (size%2==0) {
      bestg = (get<0>(params[best_param[size/2-1]])+get<0>(params[best_param[size/2]]))/2;

    } else {
      bestg = get<0>(params[best_param[size/2]]);
    }
    cout << dataset_name << " Pick median: g=" << bestg << endl;
  }


  // Do NN1 with a parameter
  // --- NN1 classification

  const double weight = bestg*sampled_mean_dist;

  // --- --- --- NN1 loop
  size_t nb_correct{0};
  size_t nb_done{0};
  tempo::ProgressMonitor pm(train->size()*test->size());
  // --- --- --- Task generator
  auto tgen = [&mutex, &pm, weight, &train_source, it = test_source.data->begin(), end = test_source.data->end(), &nb_done, &nb_correct]() mutable {
    if (it!=end) {
      tempo::ParTasks::task_t t = [&mutex, &pm, &nb_correct, &nb_done, it, &train_source, weight]() {
        double bsf = tempo::POSITIVE_INFINITY<double>;
        const TS* bcandidates = nullptr;
        for (const auto& c: *train_source.data) {
          double res = tu::adtw(*it, c, weight, bsf);
          if (res<bsf) { // update BSF
            bsf = res;
            bcandidates = &c;
          }
          {
            std::lock_guard lg(mutex);
            nb_done++;
            pm.print_progress(std::cout, nb_done);
          }
        } // End looping over candidates
        if (bcandidates!=nullptr && bcandidates->get_label().value()==(*it).get_label().value()) {
          std::lock_guard lg(mutex);
          nb_correct++;
        }
      };

      ++it;
      return std::optional<tempo::ParTasks::task_t>(t);

    } else {
      return std::optional<tempo::ParTasks::task_t>();
    }
  };

  auto testnn1_start = tt::now();
  p.execute(nbthreads, tgen);
  auto testnn1_duration = tt::now()-testnn1_start;

  double accuracy = ((double) nb_correct)/test->size();
  cout << endl;
  cout << dataset_name << " NN1 test result: " << nb_correct << "/" << test->size() << " = " << accuracy
       << "  (" << tt::as_string(testnn1_duration) << ")" << endl;

  string distname{"adtw"};
  if (transform=="derivative") { distname = distname+"-d1"; }

  JSONValue json_sample({
      {"mean",   welford.get_mean()},
      {"stddev", welford.get_stddev_s()},
      {"size",   SAMPLE_SIZE},
    }
  );

  JSONValue json_results(JSONValue::JSONObject({
    // Dataset info
    tempo::json_entry_dataset(dataset_name, *train, *test),
    // Result info
    tempo::json_entry_accuracy(test->size(), nb_correct),
    // Distance info
    {"distance", JSONValue::JSONObject({
      {"name",           distname},
      {"penalty_factor", bestg},
      {"sample",         json_sample},
      {"penalty",        bestg*sampled_mean_dist}
    })},
    // Timings
    {"threads", nbthreads},
    {"duration_loocv", tempo::to_json(loocv_duration)},
    {"duration_nn1test", tempo::to_json(testnn1_duration)}
  }));

  std::cout << dataset_name << " output to " << outpath << endl;
  std::ofstream outfile(outpath);
  print(json_results, outfile);
  print(json_results, cout);
  cout << std::endl;
}
