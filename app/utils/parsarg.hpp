#pragma once

#include "../src/tempo/reader/readingtools.hpp"

#include <functional>
#include <optional>
#include <stdexcept>
#include <stack>
#include <string>
#include <variant>
#include <vector>
#include <memory>

namespace PArg {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Token management
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Arguments are stored in a stack
    using ArgStack = std::stack<std::string>;

    /// Type of a token given by the argument stream
    using Tok = std::optional<std::string>;

    /// Create an argument stack based on the usual argc, argv parameters. Does not save argv[0] (the program name)
    [[nodiscard]] inline ArgStack newArgStack(int argc, char **argv) {
        ArgStack stack;
        for (int i = argc - 1; i > 0; --i) {
            stack.push(argv[i]);
        }
        return stack;
    }

    /// Safe access for the argument stack
    [[nodiscard]] inline Tok top(const ArgStack &stack) {
        if (stack.empty()) {
            return Tok{};
        } else {
            return Tok{stack.top()};
        }
    }

    /// Pop the top of the stack, returns it. Safe operation.
    inline Tok pop(ArgStack &stack) {
        if (stack.empty()) {
            return Tok{};
        } else {
            Tok ret = Tok{stack.top()};
            stack.pop();
            return ret;
        }
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // State
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Internal state, made of the tokens stack and the type of the argument (updated by the parser).
    /// Must be managed linearly through a std::unique_ptr. Use the alias ArgState.
    template<typename T>
    struct ArgState_ {
        ArgStack stack;
        T args;
    };

    /// Linear management of ArgState_
    template<typename T>
    using ArgState = std::unique_ptr<ArgState_<T>>;



    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Result type of parser and parsing functions
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Parser status
    enum Status { // Status of the parser
        ACCEPTED, // Successfully matched
        REJECTED, // Did not match, but this is not an error
        ERROR     // Error
    };

    /// Return value of a parser
    template<typename T>
    struct Result {
        ArgState<T> state;
        Status status{REJECTED};
        std::string emsg;
    };

    /// Test the status of a result
    template<typename T>
    [[nodiscard]] inline bool is(Status s, Result<T> &r) { return r.status == s; }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Base types
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    template<typename T>
    struct PAlt;

    /// Type of function called at the end of a Parser
    template<typename T>
    using fnPost = std::function<Result<T>(Result<T> &&)>;

    /// Type of functions controlling if and how a parser loop (evaluate its alternative more than once).
    /// Called AFTER "post"
    template<typename T>
    using fnRepeat = std::function<bool(Result<T> &)>;

    /// Argument parser
    template<typename T>
    struct Parser {
        std::string name;
        fnPost<T> post;
        fnRepeat<T> repeat;
        std::vector<PAlt<T>> alternatives;
    };

    /// Type of functions testing alternatives' head. Must consume the tokens on success.
    template<typename T>
    using fnTryHead = std::function<Result<T>(ArgState<T> &&as)>;

    /// An alternative. Can return "Rejected" without failing.
    template<typename T>
    struct PAlt {
        std::string name;
        fnTryHead<T> head;
        std::vector<Parser<T>> tuples;
    };

    /// Append a parser at the end of an alternative tuple
    template<typename T>
    [[nodiscard]] inline PAlt<T> operator && (PAlt<T>&& alt, const Parser<T>& factor){
        alt.tuples.push_back(factor);
        return std::forward<PAlt<T>>(alt);
    }

    /// Append an alternative to a parser
  template<typename T>
  [[nodiscard]] inline Parser<T> operator | (Parser<T>&& sum, const PAlt<T>& alt){
    sum.alternatives.push_back(alt);
    return std::forward<Parser<T>>(sum);
  }



  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Parsing loop
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    template<typename T>
    Result<T> parse(const Parser<T> &p, ArgState<T> &&as) {
        Result<T> r{.state = std::forward<ArgState<T>>(as)};
        bool do_repeat;
        do {
            // --- --- --- Try alternatives
            for (const auto &alt: p.alternatives) {
                r = alt.head(std::move(r.state));
                switch (r.status) {
                    // --- --- --- Alternative found. Check tuple.
                    case Status::ACCEPTED: {
                        // All the factors must match, in the order they are defined.
                        for (const auto &fact: alt.tuples) {
                            r = parse(fact, std::move(r.state));
                            switch (r.status) {
                                case Status::ACCEPTED: {
                                    continue;
                                }
                                    // --- --- --- Not accepted becomes an error!
                                case Status::REJECTED: {
                                    r.emsg = alt.name + ": argument required.";
                                    r.status = Status::ERROR;
                                    return r;
                                }
                                    // --- --- --- Error: prepend our name and transmit
                                case Status::ERROR: {
                                    r.emsg = alt.name + " " + r.emsg;
                                    return r;
                                }
                            }
                        }
                        // All tuple done - alternative is accepted
                        goto done;
                    }
                        // --- --- --- Alternative rejected. Keep looping.
                    case Status::REJECTED: {
                        continue;
                    }
                        // --- --- --- Error: prepend our name and transmit
                    case Status::ERROR: {
                        r.emsg = alt.name + ": " + r.emsg;
                        return r;
                    }
                }
            }
            done:
            r = p.post(std::move(r));
            if (is(REJECTED, r)) {
                throw std::logic_error("Parser should not be in the 'REJECTED' state at the end of the loop");
            } else if (is(ERROR, r)) {
                r.emsg = p.name + ": " + r.emsg;
                return r;
            }
            // Do we repeat ourselves?
            do_repeat = p.repeat(r);
        } while (do_repeat);
        return r;
    }


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Try Head Helpers
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Base reader
    template<typename V>
    using fnBaseReader = std::function<std::optional<V>(const std::string &tok)>;

    // --- --- ---

    /// Updater function - may fail with an error. Put your validation here!
    template<typename T, typename V>
    using fnUpdaterChecker = std::function<std::optional<std::string>(T &args, const V& value)>;

    /// Read a value and update the arguments
    template<typename T, typename V>
    [[nodiscard]] fnTryHead<T> read_value_check(fnBaseReader<V> br, fnUpdaterChecker<T, V> update) {
        return [br, update](ArgState<T> &&as) -> Result<T> {
            Tok toptok = top(as->stack);
            // ---
            if (toptok) {
                auto mbv = br(toptok.value());
                if (mbv) {
                    pop(as->stack);
                    auto res = update(as->args, mbv.value());
                    if (res) {
                        return Result<T>{.state = std::move(as), .status = ERROR, .emsg = res.value()};
                    } else {
                        return Result<T>{.state = std::move(as), .status = ACCEPTED, .emsg = ""};
                    }
                }
            }
            // ---
            return Result<T>{.state = std::move(as), .status = REJECTED, .emsg = ""};
        };
    }

    // --- --- ---

    /// Updater function without validation
    template<typename T, typename V>
    using fnUpdater = std::function<void(T &args, const V& value)>;

    /// Read a value and update the arguments
    template<typename T, typename V>
    [[nodiscard]] fnTryHead<T> inline read_value(fnBaseReader<V> br, fnUpdater<T,V> su) {
        fnUpdaterChecker<T, V> fu = [su](T &t, const V &v) -> std::optional<std::string> {
            su(t, v);
            return {};
        };
        return read_value_check(br, fu);
    }

    // --- --- ---

    /// Simple Updater function ignoring the actual value of the token
    template<typename T>
    using fnSimpleUpdater = std::function<void(T &args)>;

    /// Read a value and update the arguments
    template<typename T, typename V>
    [[nodiscard]] fnTryHead<T> inline read_value(fnBaseReader<V> br, fnSimpleUpdater<T> su = [](T &) {}) {
        fnUpdaterChecker<T, V> fu = [su](T &t, const V &) -> std::optional<std::string> {
            su(t);
            return {};
        };
        return read_value_check(br, fu);
    }

    // --- --- --- Collection of base reader

    /// Accept the next token
    [[nodiscard]] inline fnBaseReader<std::string> read_token() {
        return [](const std::string &tok) -> std::optional<std::string> { return {tok}; };
    }

    /// Read a string - accept several possibilities (e.g. {"-v", "--verbose"} )
    [[nodiscard]] inline fnBaseReader<std::string> flag(const std::vector<std::string> &vec) {
        return [vec](const std::string &tok) -> std::optional<std::string> {
            for (const auto &s: vec) {
                if (tok == s) {
                    return {tok};
                }
            }
            return {};
        };
    }

    /// Read a string - accept one possibility
    [[nodiscard]] inline fnBaseReader<std::string> flag(const std::string &s) {
        return flag(std::vector<std::string>{s});
    }

    /// Read an integer. Do the validation in the 'update' function.
    [[nodiscard]] inline fnBaseReader<int> integer() {
        return [](const std::string &s) { return tempo::reader::as_int(s); };
    }

    /// Read a size_t. Do the validation in the 'update' function.
    [[nodiscard]] inline fnBaseReader<size_t> read_size_t() {
      return [](const std::string &s) { return tempo::reader::as_size_t(s); };
    }

  /// Read a floating point number. Do the validation in the 'update' function.
    [[nodiscard]] inline fnBaseReader<double> number() {
        return [](const std::string &s) { return tempo::reader::as_double(s); };
    }






    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Repeat Helpers
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Never repeat
    template<typename T>
    [[nodiscard]] inline fnRepeat<T> no_repeat() { return [](Result<T> &) { return false; }; }

    /// Repeat while we have unread arguments
    template<typename T>
    [[nodiscard]] inline fnRepeat<T> if_token() { return [](const Result<T> &r) { return !r.state->stack.empty(); }; }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Post Helpers
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// No post. Identity function.
    template<typename T>
    [[nodiscard]] inline fnPost<T> no_post() { return [](Result<T> &&r) { return std::forward<Result<T>>(r); }; }

    /// Type of functions giving a default value for optional parser
    template<typename T>
    using fnDefault = std::function<void(T &)>;

    /// Convert the parser into an optional parser. If the result is rejected, changed into accepted.
    /// Can be used to set a default value
    template<typename T>
    [[nodiscard]] inline fnPost<T> optional_parser(fnDefault<T> set_default) {
        return [set_default](Result<T> &&r) {
            if (is(REJECTED, r)) {
                r.status = ACCEPTED;
                set_default(r.state->args);
            }
            return std::forward<Result<T>>(r);
        };
    }

    /// "Catch all": If the result is "Rejected", push the argument in a vector and switch status on "Accepted"
    template<typename T>
    [[nodiscard]] inline fnPost<T> catch_all(std::vector<std::string> &vec) {
        return [&vec](Result<T> &&r) {
            if (r.status == REJECTED) {
                auto tok = pop(r.state->stack);
                if (tok) {
                    vec.push_back(tok.value());
                    r.status = ACCEPTED;
                    return std::forward<Result<T>>(r);
                } else {
                    throw std::logic_error("Catch all should not receive an empty token");
                }
            }
            return std::forward<Result<T>>(r);
        };
    }

    /// "Not recognized": Fail if the result is "Rejected", either with 'missing token' or 'top token'
    template<typename T>
    [[nodiscard]] inline fnPost<T> not_recognize(const std::string &what = "") {
        return [what](Result<T> &&r) -> Result<T> {
            if (is(REJECTED, r)) {
                r.status = ERROR;
                auto toptok = top(r.state->stack);
                auto str = what.empty() ? std::string("argument") : what;
                if (toptok) {
                    r.emsg = "unrecognized " + str + " '" + toptok.value() + "'";
                } else {
                    r.emsg = str + " expected";
                }
            }
            return std::forward<Result<T>>(r);
        };
    }



    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Parser helpers
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Create a new parser expecting a mandatory arguments - to be specified!
    template<typename T>
    [[nodiscard]] inline Parser<T> mandatory(const std::string &name) {
      return Parser<T>{
        .name = "<"+name+">", .post = not_recognize<T>(name), .repeat = no_repeat<T>(),
        .alternatives = {}
      };
    }

    /// Create a new parser with one unique mandatory argument
    template<typename T>
    [[nodiscard]] inline Parser<T> mandatory(const std::string &name, const std::string &cmd_name, fnTryHead<T> head) {
          return Parser<T>{
                  .name = "<"+name+">", .post = not_recognize<T>(name), .repeat = no_repeat<T>(),
                  .alternatives = {PAlt<T>{
                          .name = cmd_name,
                          .head = head,
                          .tuples = {}
                  } }
          };
      }

    /// Create a new switch parser
    template<typename T>
    [[nodiscard]] inline PAlt<T> pa_switch(const std::string& s, fnSimpleUpdater<T> updater = [](T &) {}){
        return PAlt<T> {
            .name = s,
            .head = read_value<T>(flag(s), updater)
        };
    }

} // end of namespace PArg