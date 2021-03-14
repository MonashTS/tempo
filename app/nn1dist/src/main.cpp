#include <iostream>
#include <unordered_map>

#include <tempo/utils/utils.hpp>
#include <tempo/utils/jsonvalue.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/envelopes.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/lb_keogh.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/lb_enhanced.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/lb_webb.hpp>
#include <tempo/univariate/distances/dtw/dtw.hpp>
#include <tempo/univariate/distances/dtw/cdtw.hpp>
#include <tempo/univariate/distances/dtw/wdtw.hpp>
#include <tempo/univariate/distances/elementwise/elementwise.hpp>
#include <tempo/univariate/distances/erp/erp.hpp>
#include <tempo/univariate/distances/lcss/lcss.hpp>
#include <tempo/univariate/distances/msm/msm.hpp>
#include <tempo/univariate/distances/twe/twe.hpp>
#include <tempo/univariate/transforms/derivative.hpp>

#include "any.hpp"

// --- --- --- Namespaces
using namespace std;
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

/** Helper to compute the window distance */
template<typename P>
inline size_t get_w(const P& param, size_t maxl) { return param.wint ? param.wratio : maxl*param.wratio; }

/** Wrap DTW/CDTW distance behind lower bounds */
distfun_t dtw_lb(distfun_t&& df, DTWLB lb, TH& test, TH& train, size_t w) {
  switch (lb.kind) {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // None
    case DTWLB_Kind::NONE: { return std::move(df); }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Keogh
    case DTWLB_Kind::KEOGH: {
      tu::KeoghEnvelopesTransformer<FloatType, LabelType> keogh_envelopes_transformer;
      auto train_env = keogh_envelopes_transformer.transform_and_add(train, w);

      switch (lb.lb_param.keogh.kind) {

        case LB_KEOGH_Kind::BASE: {
          return [df, train_env, test](size_t q, size_t c, FloatType bsf) -> FloatType {
            const TS& tsq = test.get()[q];
            const auto& candidate_env = train_env.get()[c];
            auto res = tu::lb_Keogh(tsq.data(), tsq.length(), candidate_env.up.data(), candidate_env.lo.data(), bsf);
            if (res<=bsf) { return df(q, c, bsf); } else { return tempo::POSITIVE_INFINITY<double>; }
          };
        }

        case LB_KEOGH_Kind::CASCADE2: {
          auto test_env = keogh_envelopes_transformer.transform_and_add(test, w);
          return [df, test_env, train_env, test, train](size_t q, size_t c, FloatType bsf) -> FloatType {
            double res;
            const TS& tsq = test.get()[q];
            const auto& candidate_env = train_env.get()[c];
            res = tu::lb_Keogh(tsq.data(), tsq.length(), candidate_env.up.data(), candidate_env.lo.data(), bsf);
            if (res<=bsf) {
              const TS& tsc = train.get()[c];
              const auto& query_env = test_env.get()[q];
              res = tu::lb_Keogh(tsc.data(), tsc.length(), query_env.up.data(), query_env.lo.data(), bsf);
              if (res<=bsf) { return df(q, c, bsf); } else { return tempo::POSITIVE_INFINITY<double>; }
            } else { return tempo::POSITIVE_INFINITY<double>; }
          };
        }

        case LB_KEOGH_Kind::JOINED2: {
          auto test_env = keogh_envelopes_transformer.transform_and_add(test, w);
          return [df, test_env, train_env, test, train](size_t q, size_t c, FloatType bsf) -> FloatType {
            const TS& tsq = test.get()[q];
            const auto& query_env = test_env.get()[q];
            const TS& tsc = train.get()[c];
            const auto& candidate_env = train_env.get()[c];
            double res = tu::lb_Keogh2j(
              tsq.data(), tsq.length(), query_env.up.data(), query_env.lo.data(),
              tsc.data(), tsc.length(), candidate_env.up.data(), candidate_env.lo.data(),
              bsf);
            if (res<=bsf) { return df(q, c, bsf); } else { return tempo::POSITIVE_INFINITY<double>; }
          };
        }
      } // End switch lb keogh kind
      tempo::should_not_happen(); // Unreachable
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Enhanced
    case DTWLB_Kind::ENHANCED: {
      tu::KeoghEnvelopesTransformer<FloatType, LabelType> keogh_envelopes_transformer;
      auto train_env = keogh_envelopes_transformer.transform_and_add(train, w);
      size_t v = lb.lb_param.enhanced.v;
      switch (lb.lb_param.enhanced.kind) {

        case LB_ENHANCED_Kind::BASE: {
          return [df, test, train, train_env, v, w](size_t q, size_t c, FloatType bsf) -> FloatType {
            const TS& tsq = test.get()[q];
            const TS& tsc = train.get()[c];
            const auto& candidate_env = train_env.get()[c];
            auto res = tu::lb_Enhanced(
              tsq.data(), tsq.length(),
              tsc.data(), tsc.length(), candidate_env.up.data(), candidate_env.lo.data(), v, w, bsf);
            if (res<=bsf) { return df(q, c, bsf); } else { return tempo::POSITIVE_INFINITY<double>; }
          };
        }

        case LB_ENHANCED_Kind::JOINED2: {
          auto test_env = keogh_envelopes_transformer.transform_and_add(test, w);
          return [df, test, test_env, train, train_env, v, w](size_t q, size_t c, FloatType bsf) -> FloatType {
            const TS& tsq = test.get()[q];
            const auto& query_env = test_env.get()[q];
            const TS& tsc = train.get()[c];
            const auto& candidate_env = train_env.get()[c];
            auto res = tu::lb_Enhanced2j(
              tsq.data(), tsq.length(), query_env.up.data(), query_env.lo.data(),
              tsc.data(), tsc.length(), candidate_env.up.data(), candidate_env.lo.data(), v, w, bsf);
            if (res<=bsf) { return df(q, c, bsf); } else { return tempo::POSITIVE_INFINITY<double>; }
          };
        }

      } // End switch lb enhanced kind
      tempo::should_not_happen(); // Unreachable
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Webb
    case DTWLB_Kind::WEBB: {
      tu::WebbEnvelopesTransformer<FloatType, LabelType> webb_envelopes_transformer{};
      auto test_env = webb_envelopes_transformer.transform_and_add(test, w);
      auto train_env = webb_envelopes_transformer.transform_and_add(train, w);
      return [df, test, test_env, train, train_env, w](size_t q, size_t c, FloatType bsf) -> FloatType {
        const TS& tsq = test.get()[q];
        const auto& query_env = test_env.get()[q];
        const TS& tsc = train.get()[c];
        const auto& candidate_env = train_env.get()[c];
        auto res = tu::lb_Webb(
          tsq.data(), tsq.length(),
          query_env.up.data(), query_env.lo.data(), query_env.lo_up.data(), query_env.up_lo.data(),
          tsc.data(), tsc.length(),
          candidate_env.up.data(), candidate_env.lo.data(), candidate_env.lo_up.data(), candidate_env.up_lo.data(),
          w, bsf);
        if (res<=bsf) { return df(q, c, bsf); } else { return tempo::POSITIVE_INFINITY<double>; }
      };
    }

    default: tempo::should_not_happen();
  }

}


/** Determine the distance measure */
distfun_t mk_distance(const CMDArgs& config, TH& test_source, TH& train_source){
  size_t maxl = std::max(train_source.dataset->get_header().get_maxl(), test_source.dataset->get_header().get_maxl());

  switch (config.distance) {

    case DISTANCE::DTW: {
      // DTW
      distfun_t distance =
        [&test_source, &train_source](size_t q, size_t c, FloatType bsf) -> FloatType {
        const TS& tsq = test_source.get()[q];
        const TS& tsc = train_source.get()[c];
        return tu::dtw(tsq, tsc, bsf);
      };
      // Embeds under lower bound
      return dtw_lb(std::move(distance), config.distargs.dtw.lb, test_source, train_source, maxl);
    }

    case DISTANCE::CDTW: {
      auto param = config.distargs.cdtw;
      size_t w = get_w(param, maxl);
      distfun_t distance =
        [&test_source, &train_source, w](size_t q, size_t c, FloatType bsf) -> FloatType {
          const TS& tsq = test_source.get()[q];
          const TS& tsc = train_source.get()[c];
          return tu::cdtw(tsq, tsc, w, bsf);
        };
      // Embeds under lower bound
      return dtw_lb(std::move(distance), config.distargs.dtw.lb, test_source, train_source, maxl);
    }

    case DISTANCE::WDTW: {
      auto param = config.distargs.wdtw;
      auto weights = std::make_shared<vector<FloatType>>(tu::generate_weights(param.weight_factor, maxl));
      return [&test_source, &train_source, weights](size_t q, size_t c, FloatType bsf) -> FloatType {
        const TS& tsq = test_source.get()[q];
        const TS& tsc = train_source.get()[c];
        return tu::wdtw(tsq, tsc, *weights, bsf);
      };
    }

    case DISTANCE::ERP: {
      auto param = config.distargs.erp;
      size_t w = get_w(param, maxl);
      return [&test_source, &train_source, w, gv = param.gv](size_t q, size_t c, FloatType bsf) -> FloatType {
        const TS& tsq = test_source.get()[q];
        const TS& tsc = train_source.get()[c];
        return tu::erp(tsq, tsc, gv, w, bsf);
      };
    }

    case DISTANCE::LCSS: {
      auto param = config.distargs.lcss;
      size_t w = get_w(param, maxl);
      return [&test_source, &train_source, w, e = param.epsilon](size_t q, size_t c, FloatType bsf) -> FloatType {
        const TS& tsq = test_source.get()[q];
        const TS& tsc = train_source.get()[c];
        return tu::lcss(tsq, tsc, e, w, bsf);
      };
    }

    case DISTANCE::MSM: {
      auto param = config.distargs.msm;
      return [&test_source, &train_source, cost = param.cost](size_t q, size_t c, FloatType bsf) -> FloatType {
        const TS& tsq = test_source.get()[q];
        const TS& tsc = train_source.get()[c];
        return tu::msm(tsq, tsc, cost, bsf);
      };
    }

    case DISTANCE::SQED: {
      return [&test_source, &train_source](size_t q, size_t c, FloatType bsf) -> FloatType {
        const TS& tsq = test_source.get()[q];
        const TS& tsc = train_source.get()[c];
        return tu::elementwise(tsq, tsc, bsf);
      };
    }

    case DISTANCE::TWE: {
      auto param = config.distargs.twe;
      return [&test_source, &train_source, nu = param.nu, la = param.lambda](size_t q, size_t c,
        FloatType bsf) -> FloatType {
        const TS& tsq = test_source.get()[q];
        const TS& tsc = train_source.get()[c];
        return tu::twe(tsq, tsc, nu, la, bsf);
      };
    }

    default: tempo::should_not_happen();
  }

}



int main(int argc, char** argv) {
  // --- Manage command line argument (defined in "any")
  CMDArgs config = read_args(argc, argv);

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

  // --- Parameter recall
  {
    cout << "Configuration: distance: " << dist_to_JSON(config) << endl;
    cout << "Configuration: train path: " << path_train << endl;
    cout << "Configuration: test path: " << path_test << endl;
  }

  // --- Load the datasets
  shared_ptr<DS> train;
  shared_ptr<DS> test;
  {
    auto res_train = read_data(cout, path_train);
    if (res_train.index()==0) { print_error_exit(argv[0], get<0>(res_train), 2); }
    train = std::move(get<1>(std::move(res_train)));
    cout << to_string(train->get_header().to_json()) << endl;

    auto res_test = read_data(cout, path_test);
    if (res_test.index()==0) { print_error_exit(argv[0], get<0>(res_test), 2); }
    test = std::move(get<1>(std::move(res_test)));
    cout << to_string(test->get_header().to_json()) << endl;
  }
  size_t maxl = std::max(train->get_header().get_maxl(), test->get_header().get_maxl());
  size_t minl = std::min(train->get_header().get_minl(), test->get_header().get_minl());
  if (maxl!=minl) { print_error_exit(argv[0], "Series have disparate lengths.", 2); }

  // --- Deal with transforms
  auto train_source = train->get_original_handle();
  auto test_source = test->get_original_handle();
  switch (config.transforms) {
    case TRANSFORM::NONE: { break; }
    case TRANSFORM::DERIVATIVE: {
      tu::DerivativeTransformer<FloatType, LabelType> deriver(config.transargs.derivative.rank);
      train_source = deriver.transform_and_add(train_source);
      test_source = deriver.transform_and_add(test_source);
      break;
    }
  }


  // --- Get the distance
  distfun_t distance = mk_distance(config, test_source, train_source);


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
  for (size_t q{0}; q<test->size(); ++q) {
    double bsf = tempo::POSITIVE_INFINITY<double>;
    const TS* bcandidates = nullptr;
    for (size_t c{0}; c<train->size(); ++c) {
      double res = distance(q, c, bsf);
      if (res==tempo::POSITIVE_INFINITY<double>) {
        nb_ea++;
      } else if (res<bsf) { // update BSF
        bsf = res;
        bcandidates = &(train->get(c));
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
    if (bcandidates!=nullptr && bcandidates->get_label().value()==test->get(q).get_label().value()) {
      nb_correct++;
    }
  }
  auto stop = tt::now();
  std::cout << endl;
  duration += (stop-start);
  double acc = nb_correct/(test->size());

  std::cout << "nb ea = " << nb_ea << std::endl;

  std::cout << "Classification done" << std::endl;
  stringstream ss;
  ss << "{" << endl;
  ss << R"(  "type":"NN1",)" << endl;
  ss << R"(  "nb_correct":)" << nb_correct << "," << endl;
  ss << R"(  "accuracy":)" << acc << "," << endl;
  ss << R"(  "distance":)" << dist_to_JSON(config) << ',' << endl;
  ss << R"(  "timing_ns":)" << duration.count() << "," << endl;
  ss << R"(  "timing":")";
  tt::printDuration(ss, duration);
  ss << "\"" << endl;
  ss << "}" << endl;
  string str = ss.str();
  std::cout << str;
  if (config.outpath) {
    auto p = config.outpath.value();
    std::ofstream out(p);
    out << str;
  }

}
