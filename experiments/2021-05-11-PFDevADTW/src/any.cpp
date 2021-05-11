#include "any.hpp"
#include "../../../app/utils/parsarg.hpp"
#include "../../../app/utils/parsarg_predefs.hpp"

#include <tempo/utils/utils.hpp>

namespace fs = std::filesystem;
using namespace std;
using tempo::reader::as_double;
using tempo::reader::as_int;




void print_usage(const string& execname, ostream& out) {
  out << "Proximity Forest DEV ADTW - Monash University, Melbourne, Australia, 2021" << endl;
  out << "Proximity Forest 2018  +  ADTW and Derivative ADTW." << endl;
  out << "    Command:" << endl;
  out << "        " << execname
      << "< -ucr path name | -tt trainpath testpath> [-c nb candidates] [-t nb trees] [-out outfile]";
  out << R"(
    Dataset:
        -ucr <path to dir> <name>   Path to a directory containing datasets in ts format.
                                    A dataset is a directory named 'name' containing the
                                    'name_TRAIN.ts' and 'name_TEST.ts' files.
        -tt <train> <test>          Specify train and test files (in the ts format).

    Parameters:
      -c <n>                Number of candidates to evaluate per node (default=5)
      -t <n>                Number of trees in the forest (default=100)
      -s <n>                Random seed. Use 0 to generate a random seed (default=0)

    Create an output file: '-out <outfile>'
    )";
  out << endl;
}

CMDArgs read_args(int argc, char** argv) {
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

  // Read -c
  auto read_c = pa_switch<CMDArgs>("-c")
    && mandatory<CMDArgs>("Number of candidates", "<n>",
      read_value_check<CMDArgs, int>(integer(),
        [](CMDArgs& a, int v) {
          if (v<1) { return optional<string>{"Number of candidates must be >= 1"}; }
          a.nb_candidates = v;
          return optional<string>{};
        }
      )
    );

  // Read -t
  auto read_t = pa_switch<CMDArgs>("-t")
    && mandatory<CMDArgs>("Number of trees", "<n>",
      read_value_check<CMDArgs, int>(integer(),
        [](CMDArgs& a, int v) {
          if (v<1) { return optional<string>{"Number of trees must be >= 1"}; }
          a.nb_trees = v;
          return optional<string>{};
        }
      )
    );

  // Read -s
  auto read_s = pa_switch<CMDArgs>("-s")
    && mandatory<CMDArgs>("Random seed", "<n>",
      read_value_check<CMDArgs, size_t>(read_size_t(),
        [](CMDArgs& a, size_t v) {
          a.random_seed = v;
          return optional<string>{};
        }
      )
    );

  // Read -p
  auto read_p = pa_switch<CMDArgs>("-p")
    && mandatory<CMDArgs>("Number of threads", "<n>",
      read_value_check<CMDArgs, int>(integer(),
        [](CMDArgs& a, int v) {
          if (v<1) { return optional<string>{"Number of thread must be >= 0"}; }
          a.nb_thread = v;
          return optional<string>{};
        }
      )
    );

  // Build the parser
  auto parser = PS{
    .name = std::string(argv[0]),
    .post = not_recognize<CMDArgs>(),
    .repeat = if_token<CMDArgs>(),
    .alternatives = {
      read_c,
      read_t,
      read_s,
      read_p,
      PArg::get_ucr<CMDArgs>(),
      PArg::get_tt<CMDArgs>(),
      PArg::get_out<CMDArgs>()
    }
  };

  // --- --- ---
  auto res = parse<CMDArgs>(parser, std::move(as));
  if (is(ERROR, res)) {
    std::cout << "ERROR: " << res.emsg << std::endl;
    print_usage(argv[0], cout);
    exit(1);
  } else { return res.state->args; }

}
