#include "csv.hpp"
#include <rapidcsv.h>

namespace tempo::reader::univariate {

  namespace {
    template<std::floating_point F>
    inline F convert(bool& flag, std::string const& str) {
      try { return (F)std::stod(str); }
      catch (std::exception const& e) {
        flag = true;
        return std::numeric_limits<F>::quiet_NaN();
      }
    }
  } // End of namespace <ANONYMOUS>

  template<std::floating_point F>
  tempo::reader::Result<F> read_csv(
    std::istream& input,
    LabelEncoder const& other,
    CSVReaderParam params
  ) {

    using SData = typename ResultData<F>::SData;

    // Configure header
    rapidcsv::LabelParams label_params;
    if (params.has_header) {
      label_params.mColumnNameIdx = 0;
    } else {
      label_params.mColumnNameIdx = -1;
    }

    // Configure Separator
    rapidcsv::SeparatorParams sep_params;
    sep_params.mHasCR = false;
    sep_params.mTrim = true;
    sep_params.mSeparator = params.field_sep;
    sep_params.mAutoQuote = true;

    // Converter
    rapidcsv::ConverterParams conv_params;

    // LineReader
    rapidcsv::LineReaderParams lreader_params;
    lreader_params.mSkipEmptyLines = true;
    if (!params.comment_skip.empty()) {
      lreader_params.mSkipCommentLines = true;
      lreader_params.mCommentPrefix = std::move(params.comment_skip);
    } else {
      lreader_params.mSkipCommentLines = false;
    }

    // Prepare reader
    rapidcsv::Document doc(input, label_params, sep_params, conv_params, lreader_params);

    // Data per row
    size_t nbrow = doc.GetRowCount();
    std::vector<SData> rows;
    rows.reserve(nbrow);
    std::vector<std::string> labels;
    labels.reserve(nbrow);

    size_t length_min = std::numeric_limits<size_t>::max();
    size_t length_max = 0;

    // Series with missing data
    std::set<size_t> series_with_nan;

    // Read the lines
    for (size_t i = 0; i<nbrow; ++i) {

      // Read the line
      std::vector<std::string> row = doc.GetRow<std::string>(i);

      // Get info
      size_t row_length = row.size();
      if (params.label_position!=NONE) { row_length--; } // Minus 1 for the label
      length_min = std::min(length_min, row_length);
      length_max = std::max(length_max, row_length);

      // Extract label and data
      std::vector<double> data(row.size() - 1);
      std::optional<std::string> label = std::nullopt;
      bool flag = false;

      switch (params.label_position) {
      case NONE: {
        std::transform(
          row.begin(), row.end(), data.begin(), [&](std::string const& str) { return convert<F>(flag, str); }
        );
        break;
      }
      case FIRST: {
        label = row.front();
        std::transform(
          row.begin() + 1, row.end(), data.begin(), [&](std::string const& str) { return convert<F>(flag, str); }
        );
        break;
      }
      case LAST: {
        label = row.back();
        std::transform(
          row.begin(), row.end() - 1, data.begin(), [&](std::string const& str) { return convert<F>(flag, str); }
        );
        break;
      }
      }

      // Update our structs
      if (label) { labels.push_back(label.value()); }
      if (flag) { series_with_nan.insert(i); }
      rows.push_back(std::move(data));
    }

    // Build the result
    ResultData<F> r;
    r.series = std::move(rows);
    r.series_with_nan = std::move(series_with_nan);
    r.length_min = length_min;
    r.length_max = length_max;
    r.nb_dimensions = 1;

    if (params.label_position!=CSVLabel::NONE) {
      r.labels = std::move(labels);
      r.encoder = LabelEncoder(other, r.labels.value());
    } else {
      r.encoder = other;
    }

    return {std::move(r)};
  }

} // End of namespace tempo::reader::univariate
