#pragma once


#include "../../../src/tempo/reader/readingtools.hpp"

#include <memory>
#include <stdexcept>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <set>
#include <functional>


namespace tempo::utils {

    struct ArgStream {
        std::vector<std::string> argv;
        int argc;
        int current_arg;

        [[nodiscard]] inline bool has_next() const{return current_arg<argc; }

        [[nodiscard]] inline std::optional<std::string> read(){
            if(current_arg<argc){
                const auto& s = argv[current_arg];
                ++current_arg;
                return {s};
            } else {
                return {};
            }
        }

        ArgStream(int c_argc, char** c_argv):
            argv(c_argv + 1, c_argv + c_argc), argc(c_argc-1), current_arg(0)
        { }
    };

    using ArgPResult = std::variant<std::string, bool>;

    [[nodiscard]] inline bool isError(const ArgPResult & r){ return r.index() == 0;}
    [[nodiscard]] inline bool accepted(const ArgPResult & r){
        if(isError(r)){throw std::logic_error("Cannot check accepted status on error");}
        return std::get<1>(r);
    }

    using ArgAction = std::function<ArgPResult(const std::optional<std::string>&, ArgStream*)>;

    static ArgAction reject = [](const std::optional<std::string>& token, ArgStream* stream) {
        return ArgPResult {false};
    };

    struct ArgSum;
    struct ArgAlt;

    struct ArgSum {
        std::string                             name;
        std::string                             description;
        ArgAction                               default_action;
        std::vector<std::shared_ptr<ArgAlt>>    alternatives;
    };

    struct ArgAlt {
        std::string                             name;
        std::string                             description;
        ArgAction                               parse_action;
        std::vector<std::shared_ptr<ArgSum>>    product;
    };


    [[nodiscard]] static ArgPResult parse(const std::shared_ptr<ArgSum>& sum, ArgStream* stream){
        bool did_accept = false;
        auto token = stream->read();
        for(const auto& alt: sum->alternatives){
            auto r = alt->parse_action(token, stream);
            if(isError(r)){
                return {sum->name + " - " + alt->name+": " + std::get<0>(r)};
            } else{
                did_accept = accepted(r);
                if(did_accept){
                    for(const auto& fact: alt->product){
                        auto r2 = parse(fact, stream);
                        if(isError(r2)){ return {sum->name + " - " + alt->name+": " + std::get<0>(r)}; }
                        else if(!accepted(r2)){
                            return {sum->name + " - " + alt->name+": missing argument"};
                        }
                    } // End Product loop
                    break;
                }
            }
        } // End Sum loop
        if(did_accept){
            return {did_accept};
        } else {
            return sum->default_action(token, stream);
        }
    }

    template<typename T>
    using Validator = std::function<std::optional<std::string>(const T&)>;

    template <typename T>
    static Validator<T> defvaldt = [](const T& t){ return std::optional<std::string>{}; };

    template<typename T>
    struct BaseReader {
        std::function<std::optional<T>(const std::string&)> reader;
        std::string emsg;
    };

    template<typename T>
    static ArgAction mkAction(const BaseReader<T>& br, std::optional<T>& target, Validator<T> valdt = defvaldt<T>){
        using namespace std;
        return [&](const optional<string>& token, ArgStream* stream) mutable -> ArgPResult {
            if(token){
                auto result = br.reader(token.value());
                if(result){
                    T v = result.value();
                    auto mb_error = valdt(v);
                    if(mb_error){
                        return ArgPResult {br.emsg + ", " + mb_error.value()};
                    } else {
                        target = {v};
                        return ArgPResult {true};
                    }
                }
            }
            return {false};
        };
    }

    const static BaseReader<int> int_reader{.reader = tempo::reader::as_int, .emsg = "integer expected"};
    const static BaseReader<double> double_reader{.reader = tempo::reader::as_double, .emsg = "number expected"};
    const static BaseReader<bool> bool_reader{.reader = tempo::reader::as_bool, .emsg = "'true' or 'false' expected"};


} // End of namespace tempo