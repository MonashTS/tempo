#include <tempo/utils/utils.hpp>
#include <tempo/utils/progressmonitor.hpp>
#include <tempo/utils/utils/timing.hpp>
#include <tempo/utils/jsonvalue.hpp>
#include <tempo/univariate/distances/dtw/adtw.hpp>
#include <tempo/univariate/distances/dtw/dtw.hpp>
#include <tempo/univariate/distances/elementwise/elementwise.hpp>
#include <tempo/reader/ts/ts.hpp>
#include <tempo/tseries/dataset.hpp>
#include <tempo/tseries/indexSet.hpp>
#include <tempo/univariate/transforms/derivative.hpp>
#include <tempo/univariate/distances/loocv.hpp>
#include <tempo/univariate/classifiers/nn1/nn1.hpp>

#include <filesystem>
#include <fstream>
#include <tuple>
#include <iostream>

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

  std::vector<std::string> argList(argv, argv+argc);
  if (argc<9) {
    cout << "<path to ucr> <dataset name> <points|sqed|dtw> <sampling_number> <derivative|original> <pstart:pend:pstep:pdiv> <nbthreads> <output> required" << endl;
    exit(1);
  }
  fs::path path_ucr(argList[1]);
  string dataset_name(argList[2]);
  string sampling_method(argList[3]);
  size_t sampling_number = std::stoi(argList[4]);
  string transform(argList[5]);
  std::string pargs_str = argList[6];
  vector<std::string> pargs = tempo::split(pargs_str, ':');
  int pstart = stoi(pargs[0]);
  int pend = stoi(pargs[1]);
  int pstep = stoi(pargs[2]);
  int pdiv = stoi(pargs[3]);
  size_t nbthreads = stoi(argList[7]);
  string outpath = argList[8];

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
    cout << "<path to ucr> <dataset name> <points|sqed|dtw> <sampling_number> <derivative|original> <pstart:pend:pstep:pdiv> <nbthreads> <output> required" << endl;
    cout << "transform found: " << transform << endl;
    exit(1);
  }

  const size_t train_size = train->size();



  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Train with LOOCV
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- --- Create parameters

  // --- Sample mean
  const size_t SAMPLE_SIZE = sampling_number;

  double sampled_mean_dist = 0;
  tempo::stats::StddevWelford welford;
  {
    auto start = tt::now();
    // --- --- --- Distance sampling
    for (int i = 0; i<SAMPLE_SIZE; ++i) {
      // Pick two random values from the train set (random series, random point) and compute the distance
      if(sampling_method=="points") {
        FloatType a = pickvalue(train_source, prng);
        FloatType b = pickvalue(train_source, prng);
        welford.update(tempo::univariate::square_dist(a, b));
      } else if(sampling_method=="sqed"){
        const auto& q = tempo::rand::pick_one(*train_source.data, prng);
        const auto& s = tempo::rand::pick_one(*train_source.data, prng);
        welford.update(tempo::univariate::elementwise(q, s) / std::min<double>(q.length(), s.length()));
      } else if (sampling_method=="dtw"){
        const auto& q = tempo::rand::pick_one(*train_source.data, prng);
        const auto& s = tempo::rand::pick_one(*train_source.data, prng);
        welford.update(tempo::univariate::dtw(q, s) /std::min<double>(q.length(), s.length()));
      } else {
        cout << "<path to ucr> <dataset name> <points|sqed|dtw> <sampling_number> <derivative|original> <pstart:pend:pstep:pdiv> <nbthreads> <output> required" << endl;
        cout << "sampling method found: " << sampling_method << endl;
        exit(1);
      }
    }
    sampled_mean_dist = welford.get_mean();
  }

  JSONValue json_sample({
      {"method", sampling_method},
      {"mean",   welford.get_mean()},
      {"stddev", welford.get_stddev_s()},
      {"size",   SAMPLE_SIZE},
    }
  );
  print(json_sample, cout);
  cout << endl;


  // --- Generate parameters
  using PType = tuple<double, double>;
  vector<PType> params;
  for (int i = pstart; i<=pend; i=i+pstep) {
    double r = (double) i/ (double)(pdiv);
    params.emplace_back(make_tuple(r, r*sampled_mean_dist));
  }






  // --- --- --- LOOCV

  // --- Create the LOOCV task per left out item
  tempo::univariate::LOOCVTask<PType> task =
    [&train_source](size_t leftout, const PType& param) -> bool {
      const double penalty = get<1>(param);
      const auto& vectrain = *train_source.data;
      const auto train_size = vectrain.size();
      const auto& query = vectrain[leftout];
      double bsf = tempo::POSITIVE_INFINITY<double>;  // Best so far
      size_t bid = leftout;                           // Best ID of the bsf
      // NN1 loop on train, excluding leftout
      for (size_t i = 0; i<train_size; ++i) {
        if (i==leftout) { continue; }   // Skip leftout
        const auto candidate = vectrain[i];
        const double dist = tu::adtw(query, candidate, penalty, bsf);
        if (dist<bsf) {
          bsf = dist;
          bid = i;
        }
      }
      // Check result
      const auto nn = vectrain[bid];
      return query.get_label().value()==nn.get_label().value();
    };

  // --- Do the LOOCV process
  auto loocv_start = tt::now();
  auto[best_param, best_nbcorrect] = tempo::univariate::do_loocv<PType>(params, train_size, task, nbthreads, &cout);
  auto loocv_duration = tt::now()-loocv_start;

  // --- --- --- "Best parameter": Pick the median
  // Sort required: multi-thread does not guarantee order on insertion in the vector.
  double bestg;
  {
    // Report all the best
    cout << dataset_name << " Best parameter(s): " << best_nbcorrect << "/" << train_size << " = "
         << (double) best_nbcorrect/train_size << "  (" << tt::as_string(loocv_duration) << ")" << endl;
    std::sort(best_param.begin(), best_param.end());
    for (auto pi: best_param) { cout << "  " << get<0>(params[pi]) << endl; }
    {
      auto size = best_param.size();
      if (size%2==0) { bestg = (get<0>(params[best_param[size/2-1]])+get<0>(params[best_param[size/2]]))/2; }
      else { bestg = get<0>(params[best_param[size/2]]); }
    }
    cout << dataset_name << " Pick median: g=" << bestg << endl;
  }




  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Test
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- --- NN1-ADTW classification
  // --- Penalty
  // double not_sampled_mean = 2.01298588537;
  const double penalty = sampled_mean_dist*bestg;

  // --- Distance
  tempo::univariate::nn1dist_t<FloatType> distance = [penalty, &train_source, &test_source](
    size_t train_idx, size_t test_idx, FloatType bsf) -> FloatType {
    const auto& t = (*train_source.data)[train_idx];
    const auto& q = (*test_source.data)[test_idx];
    return tu::adtw<FloatType, LabelType>(t, q, penalty, bsf);
  };

  // --- Classification
  auto testnn1_start = tt::now();
  size_t nb_correct = tempo::univariate::nn1<FloatType, LabelType>(*train, *test, distance, nbthreads, &std::cout);
  auto testnn1_duration = tt::now()-testnn1_start;

  // --- Generate results
  double accuracy = ((double) nb_correct)/test->size();
  cout << endl;
  cout << dataset_name << " NN1 test result: " << nb_correct << "/" << test->size() << " = " << accuracy
       << "  (" << tt::as_string(testnn1_duration) << ")" << endl;

  string distname{"adtw"};
  if (transform=="derivative") { distname = distname+"-d1"; }

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
      {"penalty",        penalty}
    })},
    // Timings
    {"threads", nbthreads},
    //{"duration_loocv", tempo::to_json(loocv_duration)},
    {"duration_nn1test", tempo::to_json(testnn1_duration)},
    {"pargs", pargs_str}
  }));

  std::cout << dataset_name << " output to " << outpath << endl;
  std::ofstream outfile(outpath);
  print(json_results, outfile);
  print(json_results, cout);
  cout << std::endl;
}
