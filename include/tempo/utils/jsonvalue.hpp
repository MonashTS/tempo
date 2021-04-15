#pragma once

#include <any>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <sstream>
#include <iomanip>

namespace tempo::json {

  template<typename E>
  constexpr auto to_underlying(E e) noexcept { return static_cast<std::underlying_type_t<E>>(e); }

  struct JSONValue {
    using JSONObject = std::map<std::string, JSONValue>;
    using JSONArray = std::vector<JSONValue>;

    enum class Index {
      object,
      array,
      number,
      str,
      boolean,
      null
    };

    union Datum {
      std::map<std::string, JSONValue>* object;
      std::vector<JSONValue>* array;
      double number;
      std::string* str;
      bool boolean;
    };

    Index index{Index::null};
    Datum datum{nullptr};

    //
    // --- --- ---- Constructors
    //

    JSONValue()
      :index{Index::null} { }

    explicit JSONValue(bool b)
      :index{Index::boolean}, datum{.boolean = b} { }

    explicit JSONValue(const char* s)
      :index{Index::str}, datum{.str = new std::string(s)} { }

    explicit JSONValue(const std::string& s)
      :index{Index::str}, datum{.str = new std::string(s)} { }

    explicit JSONValue(double n)
      :index{Index::number}, datum{.number = n} { }

    explicit JSONValue(JSONArray&& vec)
      :index{Index::array}, datum{.array = new JSONArray(std::forward<JSONArray>(vec))} { }

    explicit JSONValue(JSONObject&& obj)
      :index{Index::object}, datum{.object = new JSONObject(std::forward<JSONObject>(obj))} { }

    // Collection into vector
    template<typename T>
    JSONValue(const T& begin, const T& end)
      :index{Index::array} {
      datum.array = new JSONArray();
      auto b = begin;
      while (b!=end) {
        datum.array->push_back(JSONValue(*b));
        ++b;
      }
    }


    //
    // --- --- --- Copy Constructor
    //

    JSONValue(const JSONValue& other)
      :index{other.index} {
      if (index==Index::object) {
        datum.object = new std::map(*other.datum.object);
      } else if (index==Index::array) {
        datum.array = new std::vector(*other.datum.array);
      } else if (index==Index::str) {
        datum.str = new std::string(*other.datum.str);
      } else {
        datum = other.datum;
      }
    }

    //
    // --- --- --- Move constructor
    //

    JSONValue(JSONValue&& other) noexcept
      :index{other.index}, datum{other.datum} {
      other.index = Index::null;
    }

    //
    // --- --- --- Destructors
    //

    inline void cleanup() {
      switch (index) {
        case Index::object: {
          delete datum.object;
          break;
        }
        case Index::array: {
          delete datum.array;
          break;
        }
        case Index::number: { break; }
        case Index::str: {
          delete datum.str;
          break;
        }
        case Index::boolean: { break; }
        case Index::null: { break; }
      }
    }

    ~JSONValue() {
      cleanup();
    }

    //
    // --- --- --- Swap
    //

    void swap(JSONValue& other) {
      using std::swap;
      swap(index, other.index);
      swap(datum, other.datum);
    }

    //
    // --- --- --- Assignments
    //

    JSONValue& operator=(const JSONValue& other) {
      JSONValue{other}.swap(*this);
      return *this;
    }

    JSONValue& operator=(JSONValue&& other) noexcept {
      if (&other!=this) {
        cleanup();
        index = other.index;
        datum = other.datum;
        other.index = Index::null;
      }
      return *this;
    }
  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Free functions
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  inline std::string js_string(const std::string& str) { return '"'+str+'"'; }

  inline void print(const JSONValue& value, std::ostream& os, int indent_width = 2, int indent_level = 0) {
    switch (value.index) {
      case JSONValue::Index::object: {
        os << '{' << std::endl;
        const auto old_indent = std::string(indent_level*indent_width, ' ');
        indent_level += 1;
        const auto indent = std::string(indent_level*indent_width, ' ');
        auto it = value.datum.object->begin();
        const auto end = value.datum.object->end();
        while (it!=end) {
          const auto&[k, v] = *it;
          os << indent << js_string(k) << " : ";
          print(v, os, indent_width, indent_level);
          ++it;
          if (it!=end) { os << "," << std::endl; } else { os << std::endl; }
        }
        os << old_indent << '}';
        break;
      }

      case JSONValue::Index::array: {
        os << '[';
        auto it = value.datum.array->begin();
        const auto end = value.datum.array->end();
        while (it!=end) {
          const auto& v = *it;
          print(v, os, indent_width, indent_level);
          ++it;
          if (it!=end) { os << ", "; }
        }
        os << ']';
        break;
      }

      case JSONValue::Index::number: {
        std::ios_base::fmtflags f( os.flags() );
        os << std::setprecision(12);
        os << value.datum.number;
        os.flags( f );
        break;
      }

      case JSONValue::Index::str: {
        os << js_string(*value.datum.str);
        break;
      }

      case JSONValue::Index::boolean: {
        if (value.datum.str) { os << "true"; } else { os << "false"; }
        break;
      }

      case JSONValue::Index::null: {
        os << "null";
        break;
      }
    }
  }

  inline std::string to_string(const JSONValue& value, int indent_width = 2) {
    using std::stringstream;
    stringstream ss;
    print(value, ss, indent_width);
    return ss.str();
  }

} // end of namespace tempo::utils
