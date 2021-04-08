#include "any.hpp"
#include "../../utils/parsarg.hpp"

#include <tempo/tseries/tseries.hpp>
#include <tempo/tseries/dataset.hpp>
#include <tempo/reader/ts/ts.hpp>

#include <tempo/utils/utils/timing.hpp>

#include <tempo/univariate/classifiers/proximity_forest/pf.hpp>
#include <tempo/univariate/classifiers/proximity_forest/splitters/distances_splitters.hpp>
#include <tempo/univariate/transforms/derivative.hpp>


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
  size_t base_seed=config.random_seed;
  if(base_seed == 0){
    std::random_device rd;
    base_seed = rd();
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

  // --- --- --- Transforms
  // --- --- --- --- Original
  auto train_source = train->get_original_handle();
  IndexSet train_is(*train);
  auto train_bcm = get_by_class(*train, train_is);

  auto test_source = test->get_original_handle();
  IndexSet test_is(*test);
  auto test_bcm = get_by_class(*test, test_is);

  // --- --- --- --- Derivative 1
  tu::DerivativeTransformer<FloatType, LabelType> deriver1(1);
  const auto train_source_d1 = deriver1.transform_and_add(train_source);
  const auto test_source_d1 = deriver1.transform_and_add(test_source);

  // --- --- --- --- Derivative 2
  tu::DerivativeTransformer<FloatType, LabelType> deriver2(2);
  const auto train_source_d2 = deriver2.transform_and_add(train_source);
  const auto test_source_d2 = deriver2.transform_and_add(test_source);

  // --- --- --- --- Derivative 3
  tu::DerivativeTransformer<FloatType, LabelType> deriver3(3);
  const auto train_source_d3 = deriver3.transform_and_add(train_source);
  const auto test_source_d3 = deriver3.transform_and_add(test_source);

  // --- --- --- --- Transform provider
  using TH = TransformHandle<std::vector<TS>, FloatType, LabelType>;
  using TransfromProvider = std::function<const TH&(PRNG&)>;

  TransfromProvider transform_provider =
    [&train_source, &train_source_d1, &train_source_d2, &train_source_d3](PRNG& prng) -> const TH& {
      std::discrete_distribution<size_t> transform_weights{4, 2, 1, 1};
      size_t index = transform_weights(prng);
      if (index==0) {
        return train_source;
      } else if (index==1) {
        return train_source_d1;
      } else if (index==2) {
        return train_source_d2;
      } else if (index==3) {
        return train_source_d3;
      }
      tempo::should_not_happen();
    };



  // --- --- --- Compute some info on the dataset
  std::cout << "Train" << std::endl;
  double train_gini = gini_impurity(train_bcm);
  std::cout << "Size = " << train_is.size() << " Train nb class = " << train_bcm.size() << " - gini impurity = "
            << train_gini << std::endl;
  for (const auto&[label, vec]: train_bcm) { std::cout << "label='" << label << "' size= " << vec.size() << "  "; }
  std::cout << std::endl << std::endl;

  std::cout << "Test" << std::endl;
  double test_gini = gini_impurity(test_bcm);
  std::cout << "Size = " << test_is.size() << " Test nb class = " << test_bcm.size() << " - gini impurity = "
            << test_gini << std::endl;
  for (const auto&[label, vec]: test_bcm) { std::cout << "label='" << label << "' size= " << vec.size() << "  "; }
  std::cout << std::endl;

  // --- --- --- Test splitter generation
  std::cout << "Exemplars" << std::endl;
  auto exemplars = pick_one_by_class(train_bcm, prng);
  for (const auto&[label, vec]: exemplars) {
    std::cout << "l=" << label << " size= " << vec.size() << " idx = " << vec.front() << std::endl;
  }
  IndexSet exemplars_is(exemplars);
  std::cout << "Size = " << exemplars_is.size() << std::endl;


  namespace pf = tempo::univariate::pf;
  using splitgen_ptr = std::unique_ptr<pf::SplitterGenerator<FloatType, LabelType, PRNG>>;
  std::vector<splitgen_ptr> gens;
  gens.emplace_back(splitgen_ptr(new pf::SG_DTW<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_CDTW<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_WDTW<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_Eucl<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_ERP<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_LCSS<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_MSM<FloatType, LabelType, PRNG>(transform_provider)));
  gens.emplace_back(splitgen_ptr(new pf::SG_TWE<FloatType, LabelType, PRNG>(transform_provider)));

  pf::SplitterChooser<FloatType, LabelType, PRNG> sg(std::move(gens));

  // --- --- --- Test a tree
  /*
  std::cout << "Starting training..." << std::endl;
  auto start = tempo::timing::now();
  auto ptree = tu::pf::PTree<FloatType, LabelType>::template make<PRNG>(*train, train_bcm, 5, sg, prng);
  auto stop = tempo::timing::now();
  std::cout << "Training done in" << std::endl;
  tempo::timing::printDuration(std::cout, stop-start);
  std::cout << std::endl;
  std::cout << "  Tree depth: " << ptree->depth() << std::endl;
  std::cout << "  Nb node:    " << ptree->node_number() << std::endl;
  std::cout << "  Nb leaf:    " << ptree->leaf_number() << std::endl;

  std::cout << "Starting testing..." << std::endl;
  start = tempo::timing::now();
  auto classifier = ptree->get_classifier(prng);
  size_t nbcorrect{0};
  for (const auto& idx:test_is) {
    auto res = tempo::rand::pick_one(classifier.classify(*test, idx), prng);
    if (res==test->get_original()[idx].get_label().value()) { nbcorrect++; }
  }
  stop = tempo::timing::now();
  std::cout << "Testing done in" << std::endl;
  tempo::timing::printDuration(std::cout, stop-start);
  std::cout << std::endl;
  std::cout << "Correct:  " << nbcorrect << "/" << test_is.size() << std::endl;
  std::cout << "Accuracy: " << double(nbcorrect)/test_is.size()*100.0 << "%" << std::endl;
   */

// --- --- --- Test a Forest
std::cout << "Starting training..." << std::endl;
auto start = tempo::timing::now();
//auto pforest = tempo::univariate::pf::PForest<FloatType, LabelType>::make(*train, 100, 5, sg, 700, 7, &std::cout);
auto pforest = tempo::univariate::pf::PForest<FloatType, LabelType>::make_poolroot(*train, 100, 5, sg, 700, 7, &std::cout);
auto stop = tempo::timing::now();
std::cout << "Training done in" << std::endl;
tempo::timing::printDuration(std::cout, stop-start);
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
tempo::timing::printDuration(std::cout, stop-start);
std::cout << std::endl;
std::cout << "Correct:  " << nbcorrect << "/" << test_is.size() << std::endl;
std::cout << "Accuracy: " << double(nbcorrect)/test_is.size()*100.0 << "%" << std::endl;
std::cout << "Error:    " << 100.0-(double(nbcorrect)/test_is.size()*100.0) << "%" << std::endl;
}