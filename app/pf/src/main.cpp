#include "../../utils/parsarg.hpp"

#include <tempo/univariate/classifiers/proximity_forest/pf.hpp>
#include <tempo/univariate/classifiers/proximity_forest/splitters/distances_splitters.hpp>
#include <tempo/reader/ts/ts.hpp>

#include <tempo/utils/utils/timing.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace tempo;
using FloatType = double;
using LabelType = std::string;
using PRNG = std::mt19937_64;
using TS = TSeries<FloatType, LabelType>;
using DS = Dataset<FloatType, LabelType>;

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

  PRNG prng(500);

  // --- --- --- Fields
  fs::path path_train;
  fs::path path_test;

  // --- --- --- Simple arg parser
  std::vector<std::string> arguments(argv+1, argv+argc);
  {
    for (size_t i = 0; i<argc-1; ++i) {
      if (arguments[i]=="-train") {
        ++i;
        path_train = fs::path(arguments.at(i));
      }
      if (arguments[i]=="-test") {
        ++i;
        path_test = fs::path(arguments.at(i));
      }
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
  std::cout << "Train" << std::endl;
  IndexSet train_is(*train);
  auto train_bcm = get_by_class(*train, train_is);
  double train_gini = gini_impurity(train_bcm);
  std::cout << "Size = " << train_is.size() << " Train nb class = " << train_bcm.size() << " - gini impurity = "
            << train_gini << std::endl;
  for (const auto&[label, vec]: train_bcm) { std::cout << "label='" << label << "' size= " << vec.size() << "  "; }
  std::cout << std::endl << std::endl;

  std::cout << "Test" << std::endl;
  IndexSet test_is(*test);
  auto test_bcm = get_by_class(*test, test_is);
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
  gens.emplace_back(splitgen_ptr(new pf::SG_DTW<FloatType, LabelType, PRNG>()));
  gens.emplace_back(splitgen_ptr(new pf::SG_CDTW<FloatType, LabelType, PRNG>()));
  gens.emplace_back(splitgen_ptr(new pf::SG_WDTW<FloatType, LabelType, PRNG>()));
  gens.emplace_back(splitgen_ptr(new pf::SG_Eucl<FloatType, LabelType, PRNG>()));
  gens.emplace_back(splitgen_ptr(new pf::SG_ERP<FloatType, LabelType, PRNG>()));
  gens.emplace_back(splitgen_ptr(new pf::SG_LCSS<FloatType, LabelType, PRNG>()));
  gens.emplace_back(splitgen_ptr(new pf::SG_MSM<FloatType, LabelType, PRNG>()));
  gens.emplace_back(splitgen_ptr(new pf::SG_TWE<FloatType, LabelType, PRNG>()));

  pf::SplitterChooser<FloatType, LabelType, PRNG> sg(std::move(gens));

  /*
  // --- --- --- Test a tree
  std::cout << "Starting training..." << std::endl;
  auto start = tempo::timing::now();
  auto ptree = tempo::univariate::pf::PTree<FloatType, LabelType>::make(*train, 5, sg, prng);
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
}