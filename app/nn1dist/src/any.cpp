#include "any.hpp"
#include <tempo/utils/utils.hpp>

using namespace std;
using tempo::reader::as_double;
using tempo::reader::as_int;

string to_string(DISTANCE dist) {
    switch (dist) {
        case DISTANCE::DTW:
            return "dtw";
        case DISTANCE::CDTW:
            return "cdtw";
        case DISTANCE::WDTW:
            return "wdtw";
        case DISTANCE::ERP:
            return "erp";
        case DISTANCE::LCSS:
            return "lcss";
        case DISTANCE::MSM:
            return "msm";
        case DISTANCE::SQED:
            return "sqed";
        case DISTANCE::TWE:
            return "twe";
        default:
            tempo::should_not_happen();
    }
}

string to_string(TRANSFORM tr) {
    switch (tr) {
        case TRANSFORM::DERIVATIVE:
            return "derivative";
        default:
            tempo::should_not_happen();
    }
}

string to_string(DTWLB lb) {
    switch (lb) {
        case DTWLB::NONE:
            return "none";
        case DTWLB::KEOGH:
            return "keogh";
        case DTWLB::KEOGH2:
            return "keogh2";
        case DTWLB::WEBB:
            return "webb";
        default:
            tempo::should_not_happen();
    }
}

string to_string_JSON(bool wint, double wratio){
    stringstream ss;
    if(wint){
       ss << "\"wint\": " << to_string((int)wratio);
    } else {
        ss << "\"wratio\": " << to_string(wratio);
    }
    return ss.str();
}

string dist_to_JSON(const CMDArgs &args) {
    stringstream ss;
    ss << "\"name: \"" << to_string(args.distance);
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
        default:
            tempo::should_not_happen();
    }

    return ss.str();
}


void print_usage(const string &execname, ostream &out) {
    out << "Elastic Distance NN1 classification - Monash University, Melbourne, Australia, 2021" << endl;
    out << "Command:" << endl;
    out << "  " << execname << "< -ucr path name | -tt trainpath testpath> <distance> [-out outfile]" << endl;
    out << "Dataset:" << endl;
    out << "  -ucr <path to dir> <name>" << endl;
    out << "                        Path to a directory containing datasets in ts format." << endl;
    out << "                        A dataset is itself a directory containing a _TRAIN.ts and a _TEST.ts files."
        << endl;
    out << "                        The name is the name of the datasets is also the name of its directory." << endl;
    out << "  -tt <train> <test>" << endl;
    out << "                        Specify train and test (in the ts format) " << endl;
    out << "Distance: -dist <distance name and args>:" << endl;
    out << "  dtw [LB]              DTW distance, optional lower bound" << endl;
    out << "  cdtw <wr> [LB]        CDTW distance with a window ratio 0<=wr<=1, optional lower bound" << endl;
    out << "    LB:" << endl;
    out << "      lb-none           Do not use any lower bound (default)" << endl;
    out << "      lb-keogh          LB-Keogh between the query and the envelopes of the candidate" << endl;
    out << "      lb-keogh2         LB-Keogh both way" << endl;
    out << "      lb-webb           LB-WEBB" << endl;
    out << "  wdtw <g>              WDTW distance with a weight factor 0<=g" << endl;
    out << "  sqed                  Squared Euclidean distance" << endl;
    out << "  erp <gv> <wr>         ERP distance with a gap value 0<=gv and window ratio 0<=wr<=1" << endl;
    out << "  lcss <e> <wr>         LCSS distance with an epsilon 0<=e and window ratio 0<=wr<=1" << endl;
    out << "  msm <cost>            MSM distance with a Split and Merge cost 0<=cost" << endl;
    out << "  twe <nu> <lambda>     TWE distance with a stiffness 0<=nu and a constant delete penalty 0<=lambda"
        << endl;
    out << "Notes:" << endl;
    out << "  Windows are computed using the window ratio and the longest series in the dataset." << endl;
    out << "  CDTW, ERP and LCSS window can also be given with an absolute value using ``int n'' where n>0 for <wr>"
        << endl;
    out << "Create an output file: '-out <outfile>'" << endl;
    out << "Examples:" << endl;
    out << "  " << execname << " -dist cdtw 0.2 lb-keogh2 -ucr /path/to/Univariate_ts Crop" << endl;
    out << "  " << execname << " -dist erp 0.4 int 5 lb-keogh2 -ucr /path/to/Univariate_ts Crop" << endl;
    out << endl;
}




CMDArgs read_args(int argc, char **argv) {
    // No argument: print the usage
    if (argc == 1) {
        print_usage(argv[0], cout);
        exit(0);
    }

    // Else, read the args. Declare local var.
    optional<string> error;
    int i{1};
    string arg;
    CMDArgs config;

    // Lambda: arg reading
    auto next_arg = [argv, &i]() {
        string res{argv[i]};
        ++i;
        return res;
    };

    // Argument parsing loop
    while (i < argc) {
        arg = next_arg();

        if (arg == "-ucr") { // UCR dataset
            if (i + 1 < argc) {
                fs::path ucrpath(next_arg());
                ucrpath = fs::canonical(ucrpath);
                string name = next_arg();
                auto fullpath = ucrpath / name;
                if (fs::exists(fullpath) && fs::is_directory(fullpath)) {
                    config.ucr_traintest_path = std::tuple(ucrpath, name);
                } else { error = {"Cannot find dataset directory " + string(fullpath)}; }
            } else { error = {"Directory path and dataset name expected after '-ucr'"}; }
        } else if (arg == "-tt") { // Train/Test dataset
            if (i + 1 < argc) {
                fs::path train = fs::canonical(fs::path(next_arg()));
                fs::path test = fs::canonical(fs::path(next_arg()));
                if (fs::exists(train)) {
                    if (fs::exists(test)) {
                        config.ucr_traintest_path = std::tuple(train, test);
                    } else { error = {"Cannot find test " + string(test)}; }
                } else { error = {"Cannot find train " + string(train)}; }
            } else { error = {"Directory path and dataset name expected after '-ucr'"}; }
        } else if (arg == "-out") { // Specify output file
            if (i < argc) {
                arg = next_arg();
                fs::path path(arg);
                try {
                    if (path.has_parent_path()) { fs::create_directories(path.parent_path()); }
                    ofstream of(path);
                    of.close();
                    config.outpath = {path};
                } catch (std::exception &e) {
                    error = {e.what()};
                }
            } else { error = {"Directory path expected after '-out'"}; }
        } else if (arg == "-dist") { // Specify the distance to run
            if (i < argc) {
                auto distname = next_arg();
                std::transform(distname.begin(), distname.end(), distname.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                // --- --- ---
                if (distname == "dtw") {
                    config.distance = DISTANCE::DTW;
                    config.distargs.dtw.lb = DTWLB::NONE;
                    /* Maybe a lower bound */
                    if (i < argc) {
                        arg = next_arg();
                        if (arg == "lb-none") { }
                        else if (arg == "lb-keogh") { config.distargs.dtw.lb = DTWLB::KEOGH; }
                        else if (arg == "lb-keogh2") { config.distargs.dtw.lb = DTWLB::KEOGH2; }
                        else if (arg == "lb-webb") { config.distargs.dtw.lb = DTWLB::WEBB; }
                        else { --i; /* revert */}
                    }
                } // --- --- ---
                else if (distname == "cdtw") {
                    config.distance = DISTANCE::CDTW;
                    config.distargs.cdtw.lb = DTWLB::NONE;
                    if (i < argc) {
                        arg = next_arg();
                        if (arg == "int") {
                            arg = next_arg();
                            auto wint = as_int(arg);
                            if (wint && wint.value() >= 0) {
                                config.distargs.cdtw.wint = true;
                                config.distargs.cdtw.wratio = wint.value();
                            } else { error = {"cdtw with int window expects a value >=0"}; }
                        } else {
                            auto wr = as_double(arg);
                            if (wr && 0 <= wr.value() && wr.value() <= 1) {
                                config.distargs.cdtw.wint = false;
                                config.distargs.cdtw.wratio = wr.value();
                            } else { error = {"cdtw expects a windows value 0<=w<=1"}; }
                        }
                    } else { error = {"cdtw expects a windows value 0<=w<=1"}; }
                    /* Maybe a lower bound */
                    if (i < argc) {
                        arg = next_arg();
                        if (arg == "lb-none") { }
                        else if (arg == "lb-keogh") { config.distargs.cdtw.lb = DTWLB::KEOGH; }
                        else if (arg == "lb-keogh2") { config.distargs.cdtw.lb = DTWLB::KEOGH2; }
                        else if (arg == "lb-webb") { config.distargs.cdtw.lb = DTWLB::WEBB; }
                        else { --i; /* revert */}
                    }
                } // --- --- ---
                else if (distname == "wdtw") {
                    config.distance = DISTANCE::WDTW;
                    if (i < argc) {
                        arg = next_arg();
                        auto g = as_double(arg);
                        if (g && 0 <= g.value()) { config.distargs.wdtw.weight_factor = g.value(); }
                        else { error = {"wdtw expects a weight factor 0<=g"}; }
                    } else { error = {"wdtw expects a weight factor 0<=g"}; }
                } // --- --- --
                else if (distname == "sqed") {
                    config.distance = DISTANCE::SQED;
                    /* no parameter */
                } // --- --- ---
                else if (distname == "erp") {
                    config.distance = DISTANCE::ERP;
                    if (i < argc) {
                        arg = next_arg();
                        auto gv = as_double(arg);
                        if (gv && 0 <= gv.value()) { config.distargs.erp.gv = gv.value(); }
                        else { error = {"erp expects a gap value 0<=gv"}; }
                        if (arg == "int") {
                            arg = next_arg();
                            auto wint = as_int(arg);
                            if (wint && wint.value() >= 0) {
                                config.distargs.erp.wint = true;
                                config.distargs.erp.wratio = wint.value();
                            } else { error = {"erp with int window expects a value >=0"}; }
                        } else {
                            auto wr = as_double(arg);
                            if (wr && 0 <= wr.value() && wr.value() <= 1) {
                                config.distargs.erp.wint = false;
                                config.distargs.erp.wratio = wr.value();
                            } else { error = {"erp expects a windows value 0<=w<=1"}; }
                        }
                    } else { error = {"erp expects a gap value 0<=gv followed by a window ratio 0<=w<=1"}; }
                } // --- --- ---
                else if (distname == "lcss") {
                    config.distance = DISTANCE::LCSS;
                    if (i + 1 < argc) {
                        arg = next_arg();
                        auto epsilon = as_double(arg);
                        if (epsilon && 0 <= epsilon.value()) { config.distargs.lcss.epsilon = epsilon.value(); }
                        else { error = {"lcss expects an epsilon value"}; }
                        arg = next_arg();
                        if (arg == "int") {
                            arg = next_arg();
                            auto wint = as_int(arg);
                            if (wint && wint.value() >= 0) {
                                config.distargs.lcss.wint = true;
                                config.distargs.lcss.wratio = wint.value();
                            } else { error = {"erp with int window expects a value >=0"}; }
                        } else {
                            auto wr = as_double(arg);
                            if (wr && 0 <= wr.value() && wr.value() <= 1) {
                                config.distargs.erp.wint = false;
                                config.distargs.erp.wratio = wr.value();
                            } else { error = {"erp expects a windows value 0<=w<=1"}; }
                        }
                    } else { error = {"lcss expects an epsilon value followed by a window ratio 0<=w<=1"}; }
                } // --- --- ---
                else if (distname == "msm") {
                    config.distance = DISTANCE::MSM;
                    if (i < argc) {
                        arg = next_arg();
                        auto cost = as_double(arg);
                        if (cost && 0 <= cost.value()) {
                            config.distargs.msm.cost = cost.value();
                        } else { error = {"msm expects a cost 0<=c"}; }
                    } else { error = {"msm expects a cost 0<=c"}; }
                } // --- --- ---
                else if (distname == "twe") {
                    config.distance = DISTANCE::TWE;
                    if (i + 1 < argc) {
                        arg = next_arg();
                        auto nu = as_double(arg);
                        arg = next_arg();
                        auto lambda = as_double(arg);
                        if (nu && 0 <= nu.value() && lambda && 0 <= lambda.value()) {
                            config.distargs.twe.nu = nu.value();
                            config.distargs.twe.lambda = lambda.value();
                        }
                    } else {
                        error = {"twe expects a stiffness parameter 0<=nu followed by a cost parameter 0<=lambda"};
                    }
                } else { error = {"twe expects a stiffness parameter 0<=nu followed by a cost parameter 0<=lambda"}; }
            } else { error = {"Distance name expected after '-dist'"}; }
        } // end of -dist
        // --- --- --- Unkwnon args
        else { error = {"Unkwnon arg: " + arg}; }

        // --- --- --- Error checking
        if (error.has_value()) { print_error_exit(argv[0], error.value(), 1); }

    } // end of while loop over args

    return config;
}



variant<string, tempo::Dataset<double, string>> read_data(ostream &log, fs::path& dataset_path){
    log << "Loading " << dataset_path << "... ";
    ifstream istream(dataset_path);
    auto start = tempo::timing::now();
    auto res = tempo::reader::TSReader::read(istream);
    auto stop = tempo::timing::now();
    if(res.index()==0){return {get<0>(res)}; }
    auto tsdata = std::move(get<1>(res));
    cout << "Done in "; tempo::timing::printDuration(cout, stop-start); cout << endl;
    return {tempo::reader::make_dataset(std::move(tsdata))};
}