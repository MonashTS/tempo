#include "parsarg.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <fstream>

namespace PArg {


  /// Make a Parser Alternative reading a ucr dataset in ts format
  template<typename CMDArgs>
  PAlt<CMDArgs> get_ucr() {
    using namespace std;
    namespace fs = filesystem;
    return
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
        );
  }


  /// Make a Parser Alternative reading a train set and a test set, each file in s format
  template<typename CMDArgs>
  PAlt<CMDArgs> get_tt() {
    using namespace std;
    namespace fs = filesystem;
    return pa_switch<CMDArgs>("-tt")
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
        });
  }


  /// Make a Parser Alternative reading the path to an output file
  template<typename CMDArgs>
  PAlt<CMDArgs> get_out() {
    using namespace std;
    namespace fs = filesystem;
    return pa_switch<CMDArgs>("-out") && mandatory<CMDArgs>("output file", "<path>",
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
    );
  }




} // End of namespace PArg