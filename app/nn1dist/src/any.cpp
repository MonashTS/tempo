#include "any.hpp"
#include "parsarg.hpp"
#include <tempo/utils/utils.hpp>

using namespace std;
using tempo::reader::as_double;
using tempo::reader::as_int;

string to_string(DISTANCE dist)
{
  switch (dist) {
    case DISTANCE::DTW:return "dtw";
    case DISTANCE::CDTW:return "cdtw";
    case DISTANCE::WDTW:return "wdtw";
    case DISTANCE::ERP:return "erp";
    case DISTANCE::LCSS:return "lcss";
    case DISTANCE::MSM:return "msm";
    case DISTANCE::SQED:return "sqed";
    case DISTANCE::TWE:return "twe";
    default:tempo::should_not_happen();
  }
}

string to_string(TRANSFORM tr)
{
  switch (tr) {
    case TRANSFORM::NONE: return "none";
    case TRANSFORM::DERIVATIVE: return "derivative";
    default: tempo::should_not_happen();
  }
}

string to_string(LB_KEOGH_Kind k)
{
  switch (k) {
    case LB_KEOGH_Kind::BASE: return "lb-keogh";
    case LB_KEOGH_Kind::CASCADE2: return "lb-keogh2";
    case LB_KEOGH_Kind::JOINED2: return "lb-keogh2j";
    default: tempo::should_not_happen();
  }
}

string to_string(LB_ENHANCED_Kind k)
{
  switch (k) {
    case LB_ENHANCED_Kind::BASE: return "lb-enhanced";
    case LB_ENHANCED_Kind::JOINED2: return "lb-enhanced2j";
    default: tempo::should_not_happen();
  }
}

string to_string(DTWLB lb)
{
  switch (lb.kind) {
    case DTWLB_Kind::NONE: return "\"none\"";
    case DTWLB_Kind::KEOGH: return '"'+to_string(lb.lb_param.keogh.kind)+'"';
    case DTWLB_Kind::ENHANCED:
      return '"'+to_string(lb.lb_param.enhanced.kind)+'"'+R"(, "v" : )"+to_string(lb.lb_param.enhanced.v);
    case DTWLB_Kind::WEBB: return "\"lb-webb\"";
    default: tempo::should_not_happen();
  }
}

string to_string_JSON(bool wint, double wratio)
{
  stringstream ss;
  if (wint) { ss << "\"wint\": " << to_string((int) wratio); }
  else { ss << "\"wratio\": " << to_string(wratio); }
  return ss.str();
}

string dist_to_JSON(const CMDArgs& args)
{
  stringstream ss;
  ss << R"({"name": ")" << to_string(args.distance) << "\"";
  switch (args.distance) {
    case DISTANCE::DTW: {
      ss << ", \"lb\": " << to_string(args.distargs.dtw.lb);
      break;
    }
    case DISTANCE::CDTW: {
      ss << ", \"lb\": " << to_string(args.distargs.cdtw.lb);
      ss << ", " << to_string_JSON(args.distargs.cdtw.wint, args.distargs.cdtw.wratio);
      break;
    }
    case DISTANCE::WDTW: {
      ss << ", \"weight_factor\": " << args.distargs.wdtw.weight_factor;
      break;
    }
    case DISTANCE::ERP: {
      ss << ", \"gvalue\": " << args.distargs.erp.gv;
      ss << ", " << to_string_JSON(args.distargs.erp.wint, args.distargs.erp.wratio);
      break;
    }
    case DISTANCE::LCSS: {
      ss << ", \"epsilon\": " << args.distargs.lcss.epsilon;
      ss << ", " << to_string_JSON(args.distargs.lcss.wint, args.distargs.lcss.wratio);
      break;
    }
    case DISTANCE::MSM: {
      ss << ", \"cost\": " << args.distargs.msm.cost;
      break;
    }
    case DISTANCE::SQED: {
      break;
    }
    case DISTANCE::TWE: {
      ss << ", \"nu\": " << args.distargs.twe.nu;
      ss << ", \"lambda\": " << args.distargs.twe.lambda;
      break;
    }
    default:tempo::should_not_happen();
  }
  ss << "}";

  return ss.str();
}

void print_usage(const string& execname, ostream& out)
{
  out << "Elastic Distance NN1 classification - Monash University, Melbourne, Australia, 2021" << endl;
  out << "    Command:" << endl;
  out << "        " << execname
      << "< -ucr path name | -tt trainpath testpath> <distance> [-tr transform] [-out outfile]";
  out << R"(
    Dataset:
        -ucr <path to dir> <name>   Path to a directory containing datasets in ts format.
                                    A dataset is a directory named 'name' containing the 'name_TRAIN.ts' and 'name_TEST.ts' files.
        -tt <train> <test>          Specify train and test files (in the ts format).

    Distance: -dist <distance name and args>:
      dtw [LB]              DTW distance, optional lower bound
      cdtw <wr> [LB]        CDTW distance with a window ratio 0<=wr<=1, optional lower bound
        LB:
          lb-none           Do not use any lower bound (default)
          lb-keogh          LB-Keogh between the query and the envelopes of the candidate
          lb-keogh2         LB-Keogh both way, cascading.
          lb-keogh2j        LB-Keogh both way, joined.
          lb-enhanced <v>   LB-Enhanced with tightness parameter (5 recommended)
          lb-enhanced2j <v> LB-Enhanced, two ways joined, with tightness parameter (5 recommended)
          lb-webb           LB-WEBB
      wdtw <g>              WDTW distance with a weight factor 0<=g
      sqed                  Squared Euclidean distance
      erp <gv> <wr>         ERP distance with a gap value 0<=gv and window ratio 0<=wr<=1
      lcss <e> <wr>         LCSS distance with an epsilon 0<=e and window ratio 0<=wr<=1
      msm <cost>            MSM distance with a Split and Merge cost 0<=cost
      twe <nu> <lambda>     TWE distance with a stiffness 0<=nu and a constant delete penalty 0<=lambda
    Notes:
      Windows are computed using the window ratio and the longest series in the dataset.
      CDTW, ERP and LCSS window can also be given with an absolute value using ``int n'' where n>0 for <wr>

    Apply a transform: -tr <transform>:
        none                No transform (default)
        derivative <n>       Nth derivative, n must be an integer n>=1

    Create an output file: '-out <outfile>'
    )";
  out << "Examples:" << endl;
  out << "  " << execname << " -dist cdtw 0.2 lb-keogh2 -ucr /path/to/Univariate_ts Crop" << endl;
  out << "  " << execname << " -dist erp 0.4 int 5 lb-keogh2 -ucr /path/to/Univariate_ts Crop" << endl;
  out << "  " << execname << " -dist dtw lb-keogh2 -ucr ~/Univariate_ts/ Crop -tr derivative 1" << endl;
  out << endl;
}

CMDArgs read_args(int argc, char** argv)
{
  // No argument: print the usage
  if (argc==1) {
    print_usage(argv[0], cout);
    exit(0);
  }

  // Type alias
  using namespace PArg;
  using PS = Parser<CMDArgs>;
  using PA = PAlt<CMDArgs>;

  // Fields
  auto stack = newArgStack(argc, argv);
  CMDArgs args;
  auto as = std::make_unique<ArgState_<CMDArgs>>(ArgState_<CMDArgs>{std::move(stack), std::move(args)});

  // --- --- --- Warping window

  /// Access to the "window is integer" parameter
  auto refWInt = [](CMDArgs& a) -> bool& {
    switch (a.distance) {
      case DISTANCE::CDTW: return a.distargs.cdtw.wint;
      case DISTANCE::ERP: return a.distargs.erp.wint;
      case DISTANCE::LCSS: return a.distargs.erp.wint;
      default:tempo::should_not_happen();
    }
  };

  /// Access to the window value
  auto refWVal = [](CMDArgs& a) -> double& {
    switch (a.distance) {
      case DISTANCE::CDTW: return a.distargs.cdtw.wratio;
      case DISTANCE::ERP: return a.distargs.erp.wratio;
      case DISTANCE::LCSS: return a.distargs.erp.wratio;
      default:tempo::should_not_happen();
    }
  };

  /// Read the window
  auto pReadW = PS{
    .name = "Warping window parameter",
    .post = not_recognize<CMDArgs>("<['int'] wr>"),
    .repeat = no_repeat<CMDArgs>(),
    .alternatives = {
      PA{
        .name = "<wr>",
        .head = read_value_check<CMDArgs, double>(number(),
          [refWInt, refWVal](CMDArgs& a, const int& v) -> optional<string> {
            if (!(0<=v && v<=1)) { return {"Windows ratio must be between 0 and 1"}; }
            refWInt(a) = false;
            refWVal(a) = v;
            return {};
          })
      }, PA{
        .name = "<int wr>",
        .head = read_value<CMDArgs>(flag("int")),
        .tuples = {
          PS{
            .name = "integer window",
            .post = not_recognize<CMDArgs>("integer window"),
            .repeat = no_repeat<CMDArgs>(),
            .alternatives = {
              PA{
                .name = "<wr>",
                .head = read_value_check<CMDArgs, double>(number(),
                  [refWInt, refWVal](CMDArgs& a, const int& v) {
                    refWInt(a) = true;
                    refWVal(a) = v;
                    return std::optional<std::string>{};
                  })
              }
            }
          }
        }
      }
    }
  };

  // --- --- --- DTW Lower bound

  /// Access to the DTWLB depending on an already set a.distance
  auto refDTWLB = [](CMDArgs& a) -> DTWLB& {
    if (a.distance==DISTANCE::DTW) { return a.distargs.dtw.lb; } else { return a.distargs.cdtw.lb; }
  };

  /// Parser for lb enhanced band
  auto pLBEnhancedV = mandatory<CMDArgs>("number of bands", "<n>",
    read_value_check<CMDArgs, int>(integer(), [refDTWLB](CMDArgs& a, const int& v) -> optional<string> {
      if (v<0) { return {"number of bands must be >=0"}; }
      refDTWLB(a).lb_param.enhanced.v = v;
      return {};
    }));

  /// Parser for the lower bounds of DTW and CDTW
  auto pLB = PS{
    .name = "[lower bound]",
    .post = optional_parser<CMDArgs>([refDTWLB](CMDArgs& a) { refDTWLB(a).kind = DTWLB_Kind::NONE; }),
    .repeat = no_repeat<CMDArgs>(),
    .alternatives = {
      pa_switch<CMDArgs>("lb-keogh", [refDTWLB](CMDArgs& a) {
        refDTWLB(a).kind = DTWLB_Kind::KEOGH;
        refDTWLB(a).lb_param.keogh.kind = LB_KEOGH_Kind::BASE;
      }),
      pa_switch<CMDArgs>("lb-keogh2", [refDTWLB](CMDArgs& a) {
        refDTWLB(a).kind = DTWLB_Kind::KEOGH;
        refDTWLB(a).lb_param.keogh.kind = LB_KEOGH_Kind::CASCADE2;
      }),
      pa_switch<CMDArgs>("lb-keogh2j", [refDTWLB](CMDArgs& a) {
        refDTWLB(a).kind = DTWLB_Kind::KEOGH;
        refDTWLB(a).lb_param.keogh.kind = LB_KEOGH_Kind::JOINED2;
      }),
      pa_switch<CMDArgs>("lb-webb", [refDTWLB](CMDArgs& a) {
        refDTWLB(a).kind = DTWLB_Kind::WEBB;
      }),
      pa_switch<CMDArgs>("lb-enhanced", [refDTWLB](CMDArgs& a) {
        refDTWLB(a).kind = DTWLB_Kind::ENHANCED;
        refDTWLB(a).lb_param.enhanced.kind = LB_ENHANCED_Kind::BASE;
      }) && pLBEnhancedV,
      pa_switch<CMDArgs>("lb-enhanced2j", [refDTWLB](CMDArgs& a) {
        refDTWLB(a).kind = DTWLB_Kind::ENHANCED;
        refDTWLB(a).lb_param.enhanced.kind = LB_ENHANCED_Kind::JOINED2;
      }) && pLBEnhancedV,
    }
  };

  // --- --- --- Main argument parser

  auto p = PS{
    .name = std::string(argv[0]),
    .post = not_recognize<CMDArgs>(),
    .repeat = if_token<CMDArgs>(),
    .alternatives = {
      // --- --- --- Distances
      PA{
        .name = "-dist",
        .head = read_value<CMDArgs>(flag("-dist")),
        .tuples = {
          PS{
            .name = "distance name",
            .post = not_recognize<CMDArgs>("distance name"),
            .repeat = no_repeat<CMDArgs>(),
            .alternatives {
              // --- DTW
              pa_switch<CMDArgs>("dtw", [](CMDArgs& a) { a.distance = DISTANCE::DTW; }) && pLB,
              // --- CDTW
              pa_switch<CMDArgs>("cdtw", [](CMDArgs& a) { a.distance = DISTANCE::CDTW; }) && pReadW && pLB,
              // --- WDTW
              pa_switch<CMDArgs>("wdtw", [](CMDArgs& a) { a.distance = DISTANCE::WDTW; })
                && mandatory("weight factor", "<g>", read_value<CMDArgs, double>(number(),
                  [](CMDArgs& a, double v) { a.distargs.wdtw.weight_factor = v; })),
              // --- SQED
              pa_switch<CMDArgs>("sqed", [](CMDArgs& a) { a.distance = DISTANCE::SQED; }),
              // --- ERP
              pa_switch<CMDArgs>("erp", [](CMDArgs& a) { a.distance = DISTANCE::ERP; })
                && mandatory<CMDArgs>("gap value", "<gv>", read_value<CMDArgs, double>(number(),
                  [](CMDArgs& a, double v) { a.distargs.erp.gv = v; }))
                && pReadW,
              // --- LCSS
              pa_switch<CMDArgs>("lcss", [](CMDArgs& a) { a.distance = DISTANCE::LCSS; })
                && mandatory<CMDArgs>("epsilon", "<e>", read_value<CMDArgs, double>(number(),
                  [](CMDArgs& a, double v) { a.distargs.lcss.epsilon = v; }))
                && pReadW,
              // --- MSM
              pa_switch<CMDArgs>("msm", [](CMDArgs& a) { a.distance = DISTANCE::MSM; })
                && mandatory<CMDArgs>("cost", "<cost>",
                  read_value<CMDArgs, double>(number(), [](CMDArgs& a, double v) { a.distargs.msm.cost = v; })),
              // --- TWE
              pa_switch<CMDArgs>("twe", [](CMDArgs& a) { a.distance = DISTANCE::TWE; })
                && mandatory<CMDArgs>("nu", "<nu>",
                  read_value<CMDArgs, double>(number(), [](CMDArgs& a, double v) { a.distargs.twe.nu = v; }))
                && mandatory<CMDArgs>("lambda", "<lambda>",
                  read_value<CMDArgs, double>(number(), [](CMDArgs& a, double v) { a.distargs.twe.lambda = v; }))
            }
          }
        }
      },
      // --- --- --- Transform
      pa_switch<CMDArgs>("-tr")
        && mandatory<CMDArgs>("transform name")
          | (pa_switch<CMDArgs>("derivative")
            && mandatory<CMDArgs>("derivative degree", "<n>",
              read_value_check<CMDArgs, int>(integer(),
                [](CMDArgs& a, int v) {
                  if (v<1) { return optional<string>{"derivative degree must be >= 1"}; }
                  a.transforms = TRANSFORM::DERIVATIVE;
                  a.transargs.derivative.rank = v;
                  return optional<string>{};
                }
              )
            )
          ),
      // --- --- --- Datasets -urc
      pa_switch<CMDArgs>("-ucr")
        && mandatory<CMDArgs>("path to UCR folder", "<path>", read_value_check<CMDArgs, string>(read_token(),
          [](CMDArgs& a, const string& v) -> optional<string> {
            try {
              fs::path ucrpath(v);
              ucrpath = fs::canonical(ucrpath);
              if (!(fs::exists(ucrpath) && fs::is_directory(ucrpath))) { return {"cannot find UCR folder '"+v+"'"}; }
              a.ucr_traintest_path = std::tuple<fs::path, string>(ucrpath, "");
              return {};
            }
            catch (...) { return {"cannot find UCR folder '"+v+"'"}; }
          }
        ))
        && mandatory<CMDArgs>("dataset name", "<name>", read_value_check<CMDArgs, string>(read_token(),
          [](CMDArgs& a, const string& v) -> optional<string> {
            try {
              fs::path p = std::get<0>(std::get<0>(a.ucr_traintest_path));
              auto fp = fs::canonical(p/v);
              if (!(fs::exists(fp) && fs::is_directory(fp))) { return {"cannot find the dataset '"+v+"'"}; }
              a.ucr_traintest_path = std::tuple<fs::path, string>(p, v);
              return {};
            }
            catch (...) { return {"cannot find the dataset '"+v+"'"}; }
          })
        ),
      // --- --- --- Datasets -tt
      pa_switch<CMDArgs>("-tt")
        && mandatory<CMDArgs>("path to the train and test set", "<path> <path>",
          [](ArgState<CMDArgs>&& as) -> Result<CMDArgs> {
            fs::path ptrain;
            fs::path ptest;
            Tok train = pop(as->stack);
            if (train) {
              try {
                ptrain = fs::canonical(fs::path(train.value()));
                if (!(fs::exists(ptrain) && fs::is_regular_file(ptrain))) {
                  return {.state = std::move(as), .status = ERROR, .emsg = "Cannot find train set"};
                }
              }
              catch (...) { return {.state = std::move(as), .status = ERROR, .emsg = "Cannot find train set"}; }
            } else { return Result<CMDArgs>{.state = std::move(as), .status = REJECTED, .emsg = ""}; }
            Tok test = pop(as->stack);
            if (test) {
              try {
                ptest = fs::canonical(fs::path(train.value()));
                if (!(fs::exists(ptest) && fs::is_regular_file(ptest))) {
                  return {.state = std::move(as), .status = ERROR, .emsg = "Cannot find test set"};
                }
              }
              catch (...) { return {.state = std::move(as), .status = ERROR, .emsg = "Cannot find test set"}; }
            } else { return Result<CMDArgs>{.state = std::move(as), .status = REJECTED, .emsg = ""}; }
            as->args.ucr_traintest_path = {std::tuple{ptrain, ptest}};
            return {.state = std::move(as), .status = ACCEPTED, .emsg = ""};
          })
      // --- --- --- Out
      , pa_switch<CMDArgs>("-out") && mandatory<CMDArgs>("output file", "<path>",
        read_value_check<CMDArgs, string>(read_token(), [](CMDArgs& a, const string& v) -> optional<string> {
          fs::path path(v);
          try {
            if (path.has_parent_path()) { fs::create_directories(path.parent_path()); }
            ofstream of(path);
            of.close();
            a.outpath = {path};
          }
          catch (std::exception& e) { return {e.what()}; }
          return {};
        })
      )
    }
  };

  // --- --- ---
  auto res = parse<CMDArgs>(p, std::move(as));
  if (is(ERROR, res)) {
    std::cout << "ERROR: " << res.emsg << std::endl;
    print_usage(argv[0], cout);
    exit(1);
  } else { return res.state->args; }
}

variant<string, std::shared_ptr<tempo::Dataset<double, string>>> read_data(ostream& log, fs::path& dataset_path)
{
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