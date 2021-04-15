#include "any.hpp"
#include "../../utils/parsarg.hpp"

#include <tempo/tseries/tseries.hpp>
#include <tempo/tseries/dataset.hpp>
#include <tempo/reader/ts/ts.hpp>

#include <tempo/utils/utils/timing.hpp>

#include <tempo/univariate/classifiers/proximity_forest/pf.hpp>
#include <tempo/univariate/classifiers/proximity_forest/splitters/distances_splitters.hpp>
#include <tempo/univariate/transforms/derivative.hpp>

#include <tempo/utils/jsonvalue.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace tempo;
using FloatType = double;
using LabelType = std::string;
using PRNG = std::mt19937_64;
using TS = TSeries<FloatType, LabelType>;
using DS = Dataset<FloatType, LabelType>;
namespace tu = tempo::univariate;

std::variant<std::string, std::shared_ptr<DS>> read_data(std::ostream& log, fs::path& dataset_path) {
  log << "Loading " << dataset_path << "... ";
  std::ifstream istream(dataset_path);
  auto start = tempo::timing::now();
  auto res = tempo::reader::TSReader::read(istream);
  auto stop = tempo::timing::now();
  if (res.index()==0) { return {std::get<0>(res)}; }
  auto tsdata = std::move(std::get<1>(res));
  std::cout << "Done in ";
  tempo::timing::printDuration(std::cout, stop-start);
  std::cout << std::endl;
  return {tempo::reader::make_dataset(std::move(tsdata), dataset_path.filename().replace_extension(""))};
}

int main(int argc, char** argv) {
  using namespace std;

  // --- Manage command line argument (defined in "any")
  CMDArgs config = read_args(argc, argv);

  // --- --- --- Manage random seed and Pseudo Random Number Generator
  size_t base_seed = config.random_seed;
  if (base_seed==0) {
    std::random_device rd;
    base_seed = rd();
    std::cout << "base seed = " << base_seed << std::endl;
  }
  PRNG prng(base_seed);

  // --- Dataset path
  fs::path path_train;
  fs::path path_test;
  {
    switch (config.ucr_traintest_path.index()) {
      case 0: {
        auto[path, name] = get<0>(config.ucr_traintest_path);
        if (path.empty()) { print_error_exit(argv[0], "No dataset specified", 1); }
        path_train = path/name/(name+"_TRAIN.ts");
        path_test = path/name/(name+"_TEST.ts");
        break;
      }
      case 1: {
        auto[train, test] = get<0>(config.ucr_traintest_path);
        path_train = train;
        path_test = test;
        break;
      }
      default: tempo::should_not_happen();
    }
  }

  // --- --- --- Load dataset
  std::shared_ptr<DS> train;
  std::shared_ptr<DS> test;
  {
    auto res_train = read_data(std::cout, path_train);
    if (res_train.index()==0) {
      std::cerr << std::get<0>(res_train) << std::endl;
      exit(2);
    }
    train = std::move(std::get<1>(std::move(res_train)));
    std::cout << to_string(train->get_header().to_json()) << std::endl;

    auto res_test = read_data(std::cout, path_test);
    if (res_test.index()==0) {
      std::cerr << std::get<0>(res_train) << std::endl;
      exit(2);
    }
    test = std::move(std::get<1>(std::move(res_test)));
    std::cout << to_string(test->get_header().to_json()) << std::endl;
  }

  // --- --- --- Compute some info on the dataset
  /*
  std::cout << std::endl;
  std::cout << "Train" << std::endl;
  double train_gini = gini_impurity(train_bcm);
  std::cout << "Size = " << train_is.size() << " Train nb class = " << train_bcm.size() << " - gini impurity = "
            << train_gini << std::endl;
  for (const auto&[label, vec]: train_bcm) { std::cout << "label='" << label << "' size= " << vec.size() << "  "; }
  std::cout << std::endl;
  auto train_stddev = stddev(train_is, train_source);
  std::cout << "stddev = " << train_stddev << std::endl;
  FloatType train_max=NEGATIVE_INFINITY<FloatType>;
  FloatType train_min=POSITIVE_INFINITY<FloatType>;
  for(const auto& ts: *train_source.data){
    for(size_t idx=0; idx < ts.length(); ++idx){
      train_min = std::min<FloatType>(ts(0, idx), train_min);
      train_max = std::max<FloatType>(ts(0, idx), train_max);
    }
  }
  std::cout << "min v = " << train_min << std::endl;
  std::cout << "max v = " << train_max << std::endl;
  std::cout << "delta = " << train_max - train_min << std::endl;

  std::cout << std::endl;
  std::cout << "Test" << std::endl;
  double test_gini = gini_impurity(test_bcm);
  std::cout << "Size = " << test_is.size() << " Test nb class = " << test_bcm.size() << " - gini impurity = "
            << test_gini << std::endl;
  for (const auto&[label, vec]: test_bcm) { std::cout << "label='" << label << "' size= " << vec.size() << "  "; }
  std::cout << std::endl;
  auto test_stddev = stddev(test_is, test_source);
  std::cout << "stddev = " << test_stddev << std::endl;
   */

  // --- --- --- Transforms
  // --- --- --- --- Original
  auto train_source = train->get_original_handle();
  IndexSet train_is(*train);
  auto train_bcm = get_by_class(*train, train_is);

  auto test_source = test->get_original_handle();
  IndexSet test_is(*test);
  auto test_bcm = get_by_class(*test, test_is);

  // --- --- --- --- Derivative
  tu::DerivativeTransformer<FloatType, LabelType> deriver1(1);
  const auto train_source_d1 = deriver1.transform_and_add(train_source);
  const auto test_source_d1 = deriver1.transform_and_add(test_source);

  // --- --- --- --- Transform provider
  using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
  using TransfromProvider = std::function<const TH&(PRNG&)>;

  TransfromProvider transform_provider = [&train_source](PRNG& prng) -> const TH& { return train_source; };

  TransfromProvider transform_provider_d1 = [&train_source_d1](PRNG& prng) -> const TH& {return train_source_d1; };



  // --- --- --- Splitters

  namespace pf = tempo::univariate::pf;
  using splitgen_ptr = std::unique_ptr<pf::SplitterGenerator<FloatType, LabelType, PRNG>>;
  std::vector<splitgen_ptr> gens;
  gens.emplace_back(splitgen_ptr(new pf::SG_DTW<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_CDTW<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_WDTW<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_DTW<FloatType, LabelType, PRNG>(transform_provider_d1)));
  gens.emplace_back(splitgen_ptr(new pf::SG_CDTW<FloatType, LabelType, PRNG>(transform_provider_d1)));
  gens.emplace_back(splitgen_ptr(new pf::SG_WDTW<FloatType, LabelType, PRNG>(transform_provider_d1)));
  gens.emplace_back(splitgen_ptr(new pf::SG_Eucl<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_ERP<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_LCSS<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_MSM<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_TWE<FloatType, LabelType, PRNG>(transform_provider)));

  pf::SplitterChooser<FloatType, LabelType, PRNG> sg(std::move(gens));

  // --- --- --- Test a Forest
  std::cout << "Starting training..." << std::endl;
  auto start = tempo::timing::now();
  //auto pforest = tempo::univariate::pf::PForest<FloatType, LabelType>::make(*train, 100, 5, sg, 700, 4, &std::cout);
  auto pforest = tempo::univariate::pf::PForest<FloatType, LabelType>::make_poolroot(*train, 100, 5, sg, 700, 7, &std::cout);
  auto stop = tempo::timing::now();
  std::cout << "Training done in" << std::endl;
  auto train_time_ns = stop-start;
  tempo::timing::printDuration(std::cout, train_time_ns);
  std::cout << std::endl;
  std::cout << "Starting testing..." << std::endl;
  start = tempo::timing::now();
  auto classifier = pforest->get_classifier(900, 7);
  size_t nbcorrect{0};
  for (const auto& idx:test_is) {
    auto res = tempo::rand::pick_one(classifier->classify(*test, idx), prng);
    if (res==test->get_original()[idx].get_label().value()) { nbcorrect++; }
  }
  stop = tempo::timing::now();
  std::cout << "Testing done in" << std::endl;
  auto test_time_ns = stop-start;
  tempo::timing::printDuration(std::cout, test_time_ns);
  double accuracy = double(nbcorrect)/test_is.size();
  std::cout << std::endl;
  std::cout << "Correct:  " << nbcorrect << "/" << test_is.size() << std::endl;
  std::cout << "Accuracy: " <<  accuracy*100 << "%" << std::endl;
  std::cout << "Error:    " << 100.0-(accuracy*100.0) << "%" << std::endl;

  using json::JSONValue;
  auto jsv = JSONValue({
    {"task", "Tempo.pf2018"},
    {"train_set", train->get_header().to_json() },
    {"test_set", test->get_header().to_json() },
    {"train_time_ns", train_time_ns.count()},
    {"train_time", tempo::timing::as_string(train_time_ns)},
    {"test_time_ns", test_time_ns.count()},
    {"test_time", tempo::timing::as_string(test_time_ns)},
    {"Correct", nbcorrect},
    {"Accuracy", accuracy},
    {"Error", 1.0-accuracy}
  });

  std::cout << to_string(jsv) << std::endl;
}