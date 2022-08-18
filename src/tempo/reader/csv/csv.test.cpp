#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <filesystem>
#include <cmath>

#include "csv.hpp"

TEST_CASE("CSV reader") {

  using F = double;

  std::string filename = "src/tempo/reader/csv/test.csv";
  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("Cannot access file " + filename + "\nCWD = " + std::filesystem::current_path().string());
  }

  std::ifstream input(filename, std::ios::binary);

  tempo::reader::CSVReaderParam params(
    tempo::reader::CSVLabel::FIRST,
    false,
    ',',
    {'%', '@'}
  );

  tempo::reader::Result<F> vresult = tempo::reader::read_csv<F>(input, tempo::reader::LabelEncoder(), params);
  tempo::reader::ResultData<F> const& result = std::get<1>(vresult);

  // Check that we read the correct number lines
  REQUIRE(result.series.size()==5);

  // Check the label encoder
  tempo::reader::LabelEncoder const& e = result.encoder;
  REQUIRE(e.index_to_label().size()==2);
  REQUIRE(e.label_to_index().size()==2);
  REQUIRE(e.encode(e.decode(0))==0);
  REQUIRE(e.encode(e.decode(1))==1);

  // Check the length
  REQUIRE(result.length_min==5);
  REQUIRE(result.length_max==10);

  // Check Missing data
  REQUIRE(result.series_with_nan.contains(1));

  // Check infinity
  auto const& s = result.series.back();
  REQUIRE((std::isinf(s[0])&&s[0]>0));
  REQUIRE((std::isinf(s[1])&&s[1]>0));
  REQUIRE((std::isinf(s[2])&&s[2]<0));
  REQUIRE((std::isinf(s[3])&&s[3]<0));

}
