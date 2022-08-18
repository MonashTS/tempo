#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <set>

#include "../reader_output.hpp"

namespace tempo::reader {

  enum CSVLabel { NONE, FIRST, LAST };

  struct CSVReaderParam {
    CSVLabel label_position;
    bool has_header;
    char field_sep;
    std::set<char> comment_skip;

    explicit CSVReaderParam(CSVLabel lp, bool has_header = false, char fsep = ',', std::set<char> cskip = {'#'}) :
      label_position(lp), has_header(has_header), field_sep(fsep), comment_skip(std::move(cskip)) {}
  };

  template<std::floating_point F>
  Result<F> read_csv(std::istream& input, LabelEncoder const& other, CSVReaderParam params);

} // End of namespace tempo::reader

