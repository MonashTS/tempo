#include <iostream>
#include <unordered_map>

#include <tempo/utils/utils.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/envelopes.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/lb_keogh.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/lb_enhanced.hpp>
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

// --- --- --- Types
using FloatType = double;
using LabelType = string;
using TS = tempo::TSeries<FloatType, LabelType>;
using TSP = tempo::TSPack<FloatType, LabelType>;
using DS = tempo::Dataset<FloatType, LabelType>;
using distfun = tu::distpackfun_cutoff_t<FloatType, LabelType>;
using MBFun = variant<string, distfun>;

/** Helper to compute the window distance */
template<typename P>
inline size_t get_w(const P &param, size_t maxl) {
    return param.wint ? param.wratio : maxl * param.wratio;
}

/** Envelope computation for lb keogh */
tuple<vector<FloatType>, vector<FloatType>> compute_envelopes(const TS &series, size_t w) {
    const auto l = series.length();
    vector<FloatType> up(l);
    vector<FloatType> lo(l);
    tu::get_keogh_envelopes(series.data(), l, up.data(), lo.data(), w);
    return {up, lo};
}

size_t GLB_KEOGH1{0};
size_t GLB_KEOGH2{0};
size_t GLB_ENHANCED{0};

/// Compute envelopes of a dataset, returns an error or the index of the transform
std::variant<std::string, size_t> compute_envelope(DS &ds, size_t w, size_t source_index,
                                                   const std::string &LBName, const std::string &DSName
) {
    using KET = tu::KeoghEnvelopesTransformer<FloatType, LabelType>;
    auto start = tt::now();
    const string &source_name = ds[0].transform_infos[source_index].name;
    auto env_transformer = KET::get(w, source_index, source_name); // Get the transformer
    auto res = ds.apply(env_transformer);             // Apply the transformation, may fail
    auto stop = tt::now();
    if (res.index() == 0) { return res; }
    auto duration = stop - start;
    std::cout << LBName << ": pre-computation of " << DSName << " envelopes in ";
    tt::printDuration(std::cout, duration);
    std::cout << " (transform index " << std::get<1>(res) << ")";
    std::cout << std::endl;
    return res;
}


MBFun lbDTW(distfun &&df, DTWLB lb, DS &train, DS &test, size_t w, size_t source_index) {
    using KET = tu::KeoghEnvelopesTransformer<FloatType, LabelType>;
    // Pre check
    if (train.empty()) { return {"Empty train dataset"}; }
    auto maxl = max(train.store_info().max_length, test.store_info().max_length);
    auto minl = min(train.store_info().min_length, test.store_info().min_length);
    if (minl != maxl) { return {"Lower bounds require same-length series"}; }
    // Do bound by bound
    switch (lb.kind) {
        case DTWLB_Kind::NONE: { return {df}; }
        case DTWLB_Kind::KEOGH: {
            const auto kind = lb.lb_param.keogh.kind;
            size_t env_idx_train;
            { // --- --- --- Envelopes TRAIN
                auto res = compute_envelope(train, w, source_index, "lb-keogh2", "train");
                if (res.index() == 0) { return {std::get<0>(res)}; }
                env_idx_train = std::get<1>(res);
            }
            switch (kind) {
                case LB_KEOGH_Kind::BASE: {
                    return { // Remember: returns a variant, hence the { } for construction
                            [w, source_index, env_idx_train, d = std::move(df)](const TSP &q, const TSP &c,
                                                                                FloatType cutoff) {
                                const auto&[u, l] = KET::cast(c.transforms[env_idx_train]);
                                const auto &qs = q.tseries_at(source_index);
                                double v = tu::lb_Keogh(qs.data(), qs.size(), u.data(), l.data(), u.size(), w, cutoff);
                                if (v < cutoff) { return d(q, c, cutoff); }
                                else {
                                    GLB_KEOGH1++;
                                    return tempo::POSITIVE_INFINITY<double>;
                                }
                            }
                    };
                }
                case LB_KEOGH_Kind::CASCADE2: {
                    size_t env_idx_test;
                    { // --- --- --- Envelopes TEST
                        auto res = compute_envelope(test, w, source_index, "lb-keogh2", "test");
                        if (res.index() == 0) { return {std::get<0>(res)}; }
                        env_idx_test = std::get<1>(res);
                    }
                    // --- --- --- Embed distance behind 2 rounds of lb keogh
                    return {
                            [w, source_index, env_idx_train, env_idx_test, d = std::move(df)]
                                    (const TSP &q, const TSP &c, FloatType cutoff) {
                                // Query with candidate envelopes
                                const auto &qs = q.tseries_at(source_index);
                                const auto&[cup, clo] = KET::cast(c.transforms[env_idx_train]);
                                double v = tu::lb_Keogh(qs.data(), qs.size(), cup.data(), clo.data(), cup.size(), w,
                                                        cutoff);
                                if (v < cutoff) {
                                    // Candidate with query envelopes
                                    const auto &cs = c.tseries_at(source_index);
                                    const auto&[qup, qlo] = KET::cast(q.transforms[env_idx_test]);
                                    v = tu::lb_Keogh(cs.data(), cs.size(), qup.data(), qlo.data(), qlo.size(), w,
                                                     cutoff);
                                    if (v < cutoff) { return d(q, c, cutoff); }
                                    else {
                                        GLB_KEOGH2++;
                                        return tempo::POSITIVE_INFINITY<double>;
                                    }
                                } else {
                                    GLB_KEOGH1++;
                                    return tempo::POSITIVE_INFINITY<double>;
                                }
                            }
                    };
                }
                case LB_KEOGH_Kind::JOINED2: {
                    size_t env_idx_test;
                    { // --- --- --- Envelopes TEST
                        auto res = compute_envelope(test, w, source_index, "lb-keogh2j", "test");
                        if (res.index() == 0) { return {std::get<0>(res)}; }
                        env_idx_test = std::get<1>(res);
                    }
                    // --- --- --- Embed distance behind joined lb keogh
                    return {
                            [w, source_index, env_idx_train, env_idx_test, d = std::move(df)]
                                    (const TSP &q, const TSP &c, FloatType cutoff) {
                                const auto &qs = q.tseries_at(source_index);
                                const auto&[qup, qlo] = KET::cast(q.transforms[env_idx_test]);
                                const auto &cs = c.tseries_at(source_index);
                                const auto&[cup, clo] = KET::cast(c.transforms[env_idx_train]);
                                double v = tu::lb_Keogh2(qs.data(), qs.size(), qup.data(), qlo.data(),
                                                         cs.data(), cs.size(), cup.data(), clo.data(),
                                                         cutoff);
                                if (v < cutoff) { return d(q, c, cutoff); }
                                else {
                                    GLB_KEOGH1++;
                                    return tempo::POSITIVE_INFINITY<double>;
                                }
                            }
                    };
                }
                default: tempo::should_not_happen();
            }
        }
        case DTWLB_Kind::ENHANCED : {
            const auto vt = lb.lb_param.enhanced.v;
            const auto kind = lb.lb_param.enhanced.kind;
            size_t env_idx_train;
            { // --- --- --- Envelopes TRAIN
                auto res = compute_envelope(train, w, source_index, "lb-keogh2", "train");
                if (res.index() == 0) { return {std::get<0>(res)}; }
                env_idx_train = std::get<1>(res);
            }
            switch (kind) {
                case LB_ENHANCED_Kind::BASE: {
                    return {
                            [vt, w, source_index, env_idx_train, d = std::move(df)]
                                    (const TSP &q, const TSP &c, FloatType cutoff) {
                                const auto &qs = q.tseries_at(source_index);
                                const auto &cs = c.tseries_at(source_index);
                                const auto&[cu, cl] = KET::cast(c.transforms[env_idx_train]);
                                double v = tu::lb_Enhanced(qs.data(), qs.size(), cs.data(), cs.size(), cu.data(),
                                                           cl.data(), vt, w, cutoff);
                                if (v < cutoff) { return d(q, c, cutoff); }
                                else {
                                    GLB_ENHANCED++;
                                    return tempo::POSITIVE_INFINITY<double>;
                                }
                            }
                    };
                }
                case LB_ENHANCED_Kind::JOINED2: {
                    size_t env_idx_test;
                    { // --- --- --- Envelopes TEST
                        auto res = compute_envelope(test, w, source_index, "lb-keogh2", "test");
                        if (res.index() == 0) { return {std::get<0>(res)}; }
                        env_idx_test = std::get<1>(res);
                    }

                    return { // Remember: returns a variant, hence the { } for construction
                            [vt, w, source_index, env_idx_train, env_idx_test, d = std::move(df)]
                                    (const TSP &q, const TSP &c, FloatType cutoff) {
                                const auto &qs = q.tseries_at(source_index);
                                const auto&[qu, ql] = KET::cast(q.transforms[env_idx_test]);
                                const auto &cs = c.tseries_at(source_index);
                                const auto&[cu, cl] = KET::cast(c.transforms[env_idx_train]);
                                double v = tu::lb_Enhanced2(qs.data(), qs.size(), qu.data(), ql.data(), cs.data(),
                                                            cs.size(), cu.data(), cl.data(), vt, w, cutoff);
                                if (v < cutoff) { return d(q, c, cutoff); }
                                else {
                                    GLB_ENHANCED++;
                                    return tempo::POSITIVE_INFINITY<double>;
                                }
                            }
                    };
                }
                default: tempo::should_not_happen();
            }
        }

        case DTWLB_Kind::WEBB: { return {"Sorry, lb-webb not implemented yet"}; }
        default: tempo::should_not_happen();
    }
}

/** Create a distance function given the command line argument and the min/max size of the dataset.
 *  Can precompute info (like the envelope), directly updating the datasets */
MBFun mk_distfun(const CMDArgs &conf, DS &train, DS &test, size_t source_index) {
    auto maxl = max(train.store_info().max_length, test.store_info().max_length);
    auto minl = min(train.store_info().min_length, test.store_info().min_length);

    switch (conf.distance) {
        case DISTANCE::DTW: {
            auto param = conf.distargs.dtw;
            distfun df = tu::wrap(tu::distfun_cutoff_dtw<FloatType, LabelType>(), source_index);
            return lbDTW(std::move(df), param.lb, train, test, maxl, source_index);
        }
        case DISTANCE::CDTW: {
            auto param = conf.distargs.cdtw;
            size_t w = get_w(param, maxl);
            distfun df = tu::wrap(tu::distfun_cutoff_cdtw<FloatType, LabelType>(w), source_index);
            return lbDTW(std::move(df), param.lb, train, test, w, source_index);
        }
        case DISTANCE::WDTW: {
            auto param = conf.distargs.wdtw;
            auto weights = std::make_shared<vector<FloatType>>(tu::generate_weights(param.weight_factor, maxl));
            return tu::wrap(tu::distfun_cutoff_wdtw<FloatType, LabelType>(weights), source_index);
        }
        case DISTANCE::ERP: {
            auto param = conf.distargs.erp;
            size_t w = get_w(param, maxl);
            return tu::wrap(tu::distfun_cutoff_erp<FloatType, LabelType>(param.gv, w), source_index);
        }
        case DISTANCE::LCSS: {
            auto param = conf.distargs.lcss;
            size_t w = get_w(param, maxl);
            return tu::wrap(tu::distfun_cutoff_lcss<FloatType, LabelType>(param.epsilon, w), source_index);
        }
        case DISTANCE::MSM: {
            auto param = conf.distargs.msm;
            return tu::wrap(tu::distfun_cutoff_msm<FloatType, LabelType>(param.cost), source_index);
        }
        case DISTANCE::SQED: {
            return tu::wrap(tu::distfun_cutoff_elementwise<FloatType, LabelType>(), source_index);
        }
        case DISTANCE::TWE: {
            auto param = conf.distargs.twe;
            return tu::wrap(tu::distfun_cutoff_twe<FloatType, LabelType>(param.nu, param.lambda), source_index);
        }
        default: tempo::should_not_happen();
    }
    return {"Should not happen"};
}

/** Manage the transforms.
 *  Can precompute info (like the derivative), directly updtaing the datasets */
MBFun mk_transform(const CMDArgs &conf, DS &train, DS &test) {
    switch (conf.transforms) {
        case TRANSFORM::NONE: { return mk_distfun(conf, train, test, 0); }
        case TRANSFORM::DERIVATIVE: {
            // Get the transformer
            using DTR = tu::DerivativeTransformer<FloatType, LabelType>;
            int dnumber = conf.transargs.derivative.rank;
            std::string dname = "d" + std::to_string(dnumber);
            auto transformer = DTR::get_nth(0, dname, dnumber);

            auto start = tt::now();
            auto res = train.apply(transformer);
            if (res.index() == 0) { return {std::get<0>(res)}; }
            auto duration = tt::now() - start;
            std::cout << "derivative: pre-computation of TRAIN " << dname << " in ";
            tt::printDuration(std::cout, duration);
            std::cout << " (transform index " << std::get<1>(res) << ")";
            std::cout << std::endl;

            start = tt::now();
            res = test.apply(transformer);
            if (res.index() == 0) { return {std::get<0>(res)}; }
            duration = tt::now() - start;
            std::cout << "derivative: pre-computation of TEST " << dname << " in ";
            tt::printDuration(std::cout, duration);
            std::cout << " (transform index " << std::get<1>(res) << ")";
            std::cout << std::endl;

            return mk_distfun(conf, train, test, std::get<1>(res));
        }
        default: tempo::should_not_happen();
    }
}

[[nodiscard]] inline std::string percent(double v, double total) {
    return to_string(100 * v / total) + "%";
}

/** NN1, in case of ties, first found win
 * Return a tuple (nb correct, accuracy, duration)
 * where accuracy = nb correct/test size*/
variant<string, tuple<size_t, double, tt::duration_t>> do_NN1(const CMDArgs &conf, DS &train, DS &test) {
    // --- --- --- Get the transform/distance function/lower bound
    MBFun mbfun = mk_transform(conf, train, test);
    if (mbfun.index() == 0) { return {get<0>(mbfun)}; }
    distfun dfun = get<1>(mbfun);

    size_t nb_ea{0};
    size_t nb_done{0};
    size_t total = test.size() * train.size();
    size_t centh = total / 100;
    size_t tenth = centh * 10;
    size_t nbtenth = 0;

    // --- --- --- NN1 loop
    double nb_correct{0};
    tt::duration_t duration{0};
    auto start = tt::now();
    for (auto &query : test) {
        double bsf = tempo::POSITIVE_INFINITY<double>;
        const TSP *bcandidates = nullptr;
        for (auto &candidate : train) {
            double res = dfun(query, candidate, bsf);
            if (res == tempo::POSITIVE_INFINITY<double>) {
                nb_ea++;
            } else if (res < bsf) { // update BSF
                bsf = res;
                bcandidates = &candidate;
            }
            nb_done++;
            if (nb_done % centh == 0) {
                if (nb_done % tenth == 0) {
                    nbtenth++;
                    std::cout << nbtenth * 10 << "% ";
                    std::flush(std::cout);
                } else {
                    std::cout << ".";
                    std::flush(std::cout);
                }
            }
        }
        if (bcandidates != nullptr && bcandidates->raw.label().value() == query.raw.label().value()) {
            nb_correct++;
        }
    }
    std::cout << endl;
    auto stop = tt::now();
    duration += (stop - start);
    DTWLB lb{.kind = DTWLB_Kind::NONE};
    if (conf.distance == DISTANCE::CDTW) { lb = conf.distargs.cdtw.lb; }
    else if (conf.distance == DISTANCE::DTW) { lb = conf.distargs.dtw.lb; }
    switch (lb.kind) {
        case DTWLB_Kind::NONE: {
            std::cout << "nn1: total number of early abandon: "
                      << nb_ea << "/" << total << " (" << percent(nb_ea, total) << ")" << std::endl;
            break;
        }
        case DTWLB_Kind::KEOGH: {
            switch (lb.lb_param.keogh.kind) {
                case LB_KEOGH_Kind::BASE: {
                    size_t nn1 = nb_ea - GLB_KEOGH1;
                    std::cout << "lb-keogh: number of early abandon: "
                              << GLB_KEOGH1 << "/" << total << " (" << percent(GLB_KEOGH1, total) << ")" << std::endl;
                    std::cout << "nn1: number of early abandon: "
                              << nn1 << "/" << total << " (" << percent(nn1, total) << ")" << std::endl;
                    std::cout << "nn1: total number of early abandon: "
                              << nb_ea << "/" << total << " (" << percent(nb_ea, total) << ")" << std::endl;
                    break;
                }
                case LB_KEOGH_Kind::CASCADE2: {
                    size_t nn1 = nb_ea - GLB_KEOGH1 - GLB_KEOGH2;
                    std::cout << "lb-keogh2: number of early abandon keogh1: "
                              << GLB_KEOGH1 << "/" << total << " (" << percent(GLB_KEOGH1, total) << ")" << std::endl;
                    std::cout << "lb-keogh2: number of early abandon keogh2: "
                              << GLB_KEOGH2 << "/" << total << " (" << percent(GLB_KEOGH2, total) << ")" << std::endl;
                    std::cout << "nn1: number of early abandon: "
                              << nn1 << "/" << total << " (" << percent(nn1, total) << ")" << std::endl;
                    std::cout << "nn1: total number of early abandon: "
                              << nb_ea << "/" << total << " (" << percent(nb_ea, total) << ")" << std::endl;
                    break;
                }
                case LB_KEOGH_Kind::JOINED2: {
                    size_t nn1 = nb_ea - GLB_KEOGH1;
                    std::cout << "lb-keogh2j: number of early abandon keogh: "
                              << GLB_KEOGH1 << "/" << total << " (" << percent(GLB_KEOGH1, total) << ")" << std::endl;
                    std::cout << "nn1: number of early abandon: "
                              << nn1 << "/" << total << " (" << percent(nn1, total) << ")" << std::endl;
                    std::cout << "nn1: total number of early abandon: "
                              << nb_ea << "/" << total << " (" << percent(nb_ea, total) << ")" << std::endl;

                    break;
                }
            }
            break;
        }
        case DTWLB_Kind::ENHANCED: {
            switch (lb.lb_param.enhanced.kind) {
                case LB_ENHANCED_Kind::BASE: {
                    size_t nn1 = nb_ea - GLB_ENHANCED;
                    std::cout << "lb-enhanced: number of early abandon: "
                              << GLB_ENHANCED << "/" << total << " (" << percent(GLB_ENHANCED, total) << ")"
                              << std::endl;
                    std::cout << "nn1: number of early abandon: "
                              << nn1 << "/" << total << " (" << percent(nn1, total) << ")" << std::endl;
                    std::cout << "nn1: total number of early abandon: "
                              << nb_ea << "/" << total << " (" << percent(nb_ea, total) << ")" << std::endl;
                    break;
                }
                case LB_ENHANCED_Kind::JOINED2: {
                    size_t nn1 = nb_ea - GLB_ENHANCED;
                    std::cout << "lb-enhanced2j: number of early abandon: "
                              << GLB_ENHANCED << "/" << total << " (" << percent(GLB_ENHANCED, total) << ")"
                              << std::endl;
                    std::cout << "nn1: number of early abandon: "
                              << nn1 << "/" << total << " (" << percent(nn1, total) << ")" << std::endl;
                    std::cout << "nn1: total number of early abandon: "
                              << nb_ea << "/" << total << " (" << percent(nb_ea, total) << ")" << std::endl;
                    break;
                }
            }
            break;
        }
        case DTWLB_Kind::WEBB: {
            break;
        }
        default: tempo::should_not_happen();
    }

    return {tuple<size_t, double, tt::duration_t>{nb_correct, nb_correct / (test.size()), duration}};
}


int main(int argc, char **argv) {
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // --- Read args
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    CMDArgs config = read_args(argc, argv);

    // --- Dataset path
    fs::path path_train;
    fs::path path_test;
    {
        switch (config.ucr_traintest_path.index()) {
            case 0: {
                auto[path, name] = get<0>(config.ucr_traintest_path);
                path_train = path / name / (name + "_TRAIN.ts");
                path_test = path / name / (name + "_TEST.ts");
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
    tempo::Dataset<double, string> train;
    tempo::Dataset<double, string> test;
    {
        auto res_train = read_data(cout, path_train);
        if (res_train.index() == 0) { print_error_exit(argv[0], get<0>(res_train), 2); }
        train = std::move(get<1>(std::move(res_train)));
        cout << train.store_info().to_json() << endl;

        auto res_test = read_data(cout, path_test);
        if (res_test.index() == 0) { print_error_exit(argv[0], get<0>(res_test), 2); }
        test = std::move(get<1>(std::move(res_test)));
        cout << test.store_info().to_json() << endl;
    }

    // --- Classification
    auto res = do_NN1(config, train, test);

    // --- Analys results
    switch (res.index()) {
        case 0: {
            std::cout << "Error: " << std::get<0>(res) << std::endl;
            return 2;
        }
        case 1: {
            std::cout << "Classification done" << std::endl;
            auto[nbcorrect, acc, duration] = get<1>(res);
            stringstream ss;
            ss << "{" << endl;
            ss << R"(  "type":"NN1",)" << endl;
            ss << R"(  "nb_correct":)" << nbcorrect << "," << endl;
            ss << R"(  "accuracy":)" << acc << "," << endl;
            ss << R"(  "distance":)" << dist_to_JSON(config) << ',' << endl;
            ss << R"(  "timing_ns":)" << duration.count() << "," << endl;
            ss << R"(  "timing":")";
            tt::printDuration(ss, duration);
            ss << "\"" << endl;
            ss << "}" << endl;
            string str = ss.str();
            std::cout << str;
        }
    }
}



