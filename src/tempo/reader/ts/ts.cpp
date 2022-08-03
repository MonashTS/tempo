// Standard
#include <variant>
#include <functional>

// Tempo
#include <tempo/reader/ts/ts.hpp>
#include <tempo/reader/readingtools.hpp>

namespace tempo::reader {

    using namespace std::string_literals;

    /** Main reading loop
     * Initialize the state-function in the 'read_header' state.
     * Then, loop (i.e. call the state-function) until the file is read or an error occured.
     */
    std::variant <std::string, TSData> TSReader::read() {
        // Init the state-function on 'read_header'
        state = &TSReader::read_header;
        Result r = {};
        std::variant <std::string, TSData> result{"TSReader: read error"};
        while (state != nullptr) {
            r = std::invoke(state, this);
            if (r) { // Error case when
                result.emplace<0>(*r);
                return result;
            }
        }
        result.emplace<1>(std::move(data));
        return result;
    }

    /** Read header function.
     *  Ignore comments and blanks, switch the state to 'read_directive' when reading the start of a directive '@'
     */
    TSReader::Result TSReader::read_header() {
        buffer.clear();
        int c{EOF};
        while ((c = input.get()) != EOF) {
            // Ignore comments: any number of char, until end of line.
            if (c == '#') { skip_line(input); }
            else if (std::isspace(c)) { /*Ignore whitespaces: continue */ }
            else if (c == '@') { // Else, it should be a '@'. Update the state. and return
                state = &TSReader::read_directive;
                return {};
            } else { // Else it is an error
                return {"Error while reading the header."};
            }
        }
        // The only way to stop the while loop is buy reading EOF, which means we did not read @data
        return {"Error while reading the header: reached the end of file. Missing '@data'?"};
    }

    /** Identify a directive.
     *  Called while in the header after reading a '@'. */
    TSReader::Result TSReader::read_directive() {
        // Consume the next word (non whites char), converting to lower case.
        buffer.clear();
        int c = read_word_to_lower(input, buffer);
        if (c == EOF) { return {"Error while reading the header: reached the end of file (after \"" + buffer + "\"."}; }
        // Whatever we read, skip whites
        skip_white(input);

        // Test what we read. The buffer is in lower cases.
        auto it = directive_map.find(buffer);
        if (it != directive_map.end()) {
            // Clear the buffer here (most cases need to read something)
            buffer.clear();
            switch (it->second) {

                // Read the name of the problem
                case DirectiveCode::dir_problem_name: {
                    if (read_word(input, buffer) == EOF) {
                        return {"Error while reading directive @problemname: reached EOF."};
                    }
                    data.problem_name = {buffer};
                    state = (&TSReader::read_header);
                    return {};
                }

                    // Are we using the timestamps?
                case DirectiveCode::dir_timestamp: {
                    if (read_word_to_lower(input, buffer) == EOF) {
                        return {"Error while reading directive @timestamps: reached EOF."};
                    }
                    auto ob = as_bool(buffer);
                    if (ob.has_value()) { data.timestamps = {ob.value()}; }
                    else { return {"Error while reading directive @timestamps ('true' or 'false' expected)"}; }
                    state = (&TSReader::read_header);
                    return {};
                }

                    // Do we have missing values?
                case DirectiveCode::dir_missing: {
                    if (read_word_to_lower(input, buffer) == EOF) {
                        return {"Error while reading directive @missing: reached EOF."};
                    }
                    auto ob = as_bool(buffer);
                    if (ob.has_value()) { data.missing = {ob.value()}; }
                    else { return {"Error while reading directive @missing ('true' or 'false' expected)"}; }
                    state = (&TSReader::read_header);
                    return {};
                }

                    // Is this an univariate time series?
                case DirectiveCode::dir_univariate: {
                    if (read_word_to_lower(input, buffer) == EOF) {
                        return {"Error while reading directive @univariate: reached EOF."};
                    }
                    auto ob = as_bool(buffer);
                    if (ob.has_value()) { data.univariate = {ob.value()}; }
                    else { return {"Error while reading directive @univariate ('true' or 'false' expected)"}; }
                    state = (&TSReader::read_header);
                    return {};
                }

                    // Do we have series of equal length_?
                case DirectiveCode::dir_equal_length: {
                    if (read_word_to_lower(input, buffer) == EOF) {
                        return {"Error while reading directive @equallength: reached EOF."};
                    }
                    auto ob = as_bool(buffer);
                    if (ob.has_value()) {
                        if (!ob.value() && data.serieslength.has_value()) {
                            return {"Error while reading the directive @equallength: cannot be false if a length is specified"};
                        }
                        data.equallength = {ob.value()};
                    } else { return {"Error while reading directive @equallength ('true' or 'false' expected)"}; }
                    state = (&TSReader::read_header);
                    return {};
                }

                    // What is the length_ of the series? Implies equal length.
                case DirectiveCode::dir_series_length: {
                    if (read_word_to_lower(input, buffer) == EOF) {
                        return {"Error while reading directive @serieslength: reached EOF."};
                    }
                    auto oi = as_int(buffer);
                    if (oi.has_value() && oi.value() > 0) {
                        if (data.equallength.has_value() && !data.equallength.value()) {
                            return {"Error while reading directive @serieslength: cannot be set with @equallength set to false"};
                        }
                        data.serieslength = {oi.value()};
                    } else {
                        return {"Error while reading directive @serieslength (positive integer number expected)"};
                    }
                    state = (&TSReader::read_header);
                    return {};
                }

                    // Do we have class label?
                case DirectiveCode::dir_class_label: {
                    if (read_word_to_lower(input, buffer) == EOF) {
                        return {"Error while reading directive @classlabel: reached EOF."};
                    }
                    auto ob = as_bool(buffer);
                    if (ob.has_value()) {
                        if (ob.value()) {
                            // Read the classes
                            state = (&TSReader::read_classes);
                            return {};
                        } else {
                            // Skip until the end of line.
                            skip_line(input);
                            state = (&TSReader::read_header);
                            return {};
                        }
                    } else {
                        return {"Error whole reading directive @classlabel ('true' or 'false' expected)"};
                    }
                }

                    // Data directive
                case DirectiveCode::dir_data: {
                    skip_line(input);
                    state = (&TSReader::read_data);
                    return {};
                }

                    // Default case: should not happen
                default:
                    return {"Error while reading the header: internal"};
            }
        } else {
            return {"Error while reading the header: unrecognized directive \"" + buffer + "\"."};
        }
    }

    /** Read a list of classes on a line.
     *  Called while reading a '@classlabel' directive
     */
    TSReader::Result TSReader::read_classes() {
        int c{};
        while ((c = input.get()) != EOF) {
            // Skip whitespaces, but not '\n'
            if (is_white(c)) { continue; }
            else if (c == '\n') { // New line: end of the directive, back in the header.
                if (!data.has_labels()) { return {"Error while reading directive @classlabel: labels expected"}; }
                state = (&TSReader::read_header);
                return {};
            } else { // Else, read a word
                buffer.clear();
                buffer.push_back((char)c); // Keep the first letter of the word!
                read_word(input, buffer);
                data.labels.insert(buffer);
            }
        }
        // We read EOF
        return {"Error while reading directive @classlabel: reached EOF."};
    }


    /** Read the data section.
     *  Called after reading the '@data' directive.
     *  Start by fully reading the first line in order to determine the number of dimensions and the expected length of the series.
     *  Then use 'read_data_' to read in a streaming fashion.
     */
    TSReader::Result TSReader::read_data() {
        // --- --- --- Check the dimensions
        // We use the fist line of data to count the number of dimensions by counting the number of separator ':'
        // * There is one more dimensions than separators
        // * Labels, if present, occupy the last "dimension"
        buffer.clear();
        std::getline(input, buffer);
        auto has_labels = data.has_labels();
        data.nb_dimensions = std::count(buffer.begin(), buffer.end(), ':') + (has_labels ? 0 : 1);
        if (data.nb_dimensions == 0) {
            return {"Initialisation: Error reading the data: no dimension could be read"};
        } else if (data.univariate.has_value() && data.univariate.value() && data.nb_dimensions != 1) {
            return {"Initialisation: Error reading the data: the dataset is not univariate."};
        }

        // --- --- --- Check the length
        // Get the length_ of the first series (looking at its first dimension)
        auto first_band = buffer.substr(0, buffer.find(':'));
        length1st = std::count(first_band.begin(), first_band.end(), ',') + 1;
        // Ensure we have the good length_ if it was specified
        if (data.has_equallength() && data.serieslength.has_value() &&
            data.serieslength.value() != length1st) {
            return {"Initialisation: Error reading the data: non matching length_ "s +
                    std::to_string(data.serieslength.value()) + " vs " + std::to_string(length1st)};
        }

        // --- --- --- Read the first line based on the buffer
        auto in = std::istringstream(buffer);
        auto res = read_data_(in);
        if (res.has_value()) { return res; }

        // --- --- --- Rest of the data
        res = read_data_(input);
        if (res.has_value()) { return res; }

        // --- --- --- End of the parsing process
        state = nullptr;
        return {};
    }

    /** Actual data-reading function, cCalled by 'read_data'. */
    TSReader::Result TSReader::read_data_(std::istream &in) {
        // Shorthands
        auto &dataset = data.series;
        auto &missing = data.series_with_missing_values;
        // Constants
        const bool has_labels = data.has_labels();
        const bool has_equallength = data.has_equallength();
        const size_t ndim = data.nb_dimensions;
        const size_t expected_length = length1st; // Only used if has_equallength == true

        // --- Inside data loop
        bool loop_data = true;
        do {
            // --- Inside series loop

            // Data for a time series
            std::vector<double> series;
            size_t length{0};
            size_t cur_dim{0};
            std::string label{};
            bool has_missing{false};

            // Data for the loop
            bool loop_series = true;
            do {
                // --- Class or inside band loop
                // If we have label and we read all the dimensions, read the label
                if (has_labels && cur_dim == ndim) {
                    buffer.clear();
                    int c = read_until_delim(in, buffer);
                    trim(buffer);
                    // Ok if end of line/end of file
                    if (c == '\n' || c == EOF) {
                        // Construct the series
                        dataset.push_back(TSData::TS(std::move(series), ndim, has_missing, {buffer}));
                        // Update min/max length
                        data.shortest_length = std::min(data.shortest_length, length);
                        data.longest_length = std::max(data.longest_length, length);
                        // Stop the series loop, and the data loop if EOF
                        loop_series = false;
                        loop_data = c != EOF; // stop if EOF
                    } // Else, error
                    else { return {"Error while reading the data: the class should be the last item on the row."}; }
                } else {
                    // Else, loop inside the dimension
                    bool loop_dimension = true;
                    do {
                        buffer.clear();
                        skip_white(in);
                        int c = read_until_delim(in, buffer);
                        rtrim(buffer);

                        // --- --- --- If we haven't read anything...
                        if (buffer.empty()) {
                            // EOF: exit
                            if (in.eof()) { return {}; }
                                // Not EOF: error
                            else { return {"Error reading @data"}; }
                        }

                        // --- --- --- Else, What have we read?
                        // Missing value? If so, also record the index of the series
                        if (buffer == "?") {
                            series.push_back(std::numeric_limits<double>::quiet_NaN());
                            size_t missing_index = dataset.size();
                            if (missing.empty() || missing.back() != missing_index) {
                                missing.push_back(missing_index);
                            }
                        } else {
                            auto od = as_double(buffer);
                            if (od.has_value()) { series.push_back(od.value()); }
                            else {
                                return {"Error reading '" + buffer + "': only supporting double and missing values"};
                            }
                        }

                        // --- Check the delimiter
                        // Read an item, not the end of anything: continue
                        if (c == ',') { continue; }
                        // Mark the end of a dimension: check the expected_length if needed
                        if (c == ':' || c == '\n' || c == EOF) {
                            // After reading the first dimension, record the length_
                            if (cur_dim == 0) { length = series.size(); }
                            cur_dim++;
                            // Length check
                            if (has_equallength && length != expected_length) {
                                return {"Error reading the data: non matching expected_length "s +
                                        std::to_string(expected_length) + " vs " + std::to_string(length)};
                            }
                            // End of the dimension loop
                            loop_dimension = false;
                        }
                        // Mark the end of the current series: check the dimension, no label
                        if (c == '\n' || c == EOF) {
                            if (has_labels) { return {"Error reading the data: missing get_label"}; }
                            else if (series.size() != cur_dim * length) {
                                return {"Error reading the data: non matching dimension"};
                            }
                            // Ok, store the series in the dataset
                            dataset.push_back(TSData::TS(std::move(series), ndim, has_missing, {}));
                            // Update min/max length
                            data.shortest_length = std::min(data.shortest_length, length);
                            data.longest_length = std::max(data.longest_length, length);
                            // Stop the series loop, stop the data loop if EOF
                            loop_series = false;
                            loop_data = c != EOF;
                        }
                    } while (loop_dimension);
                } // End of if - read labels or read band
            } while (loop_series);
        } while (loop_data);

        // All good!
        return {};
    } // End of read_data


    std::variant <std::string, TSData> TSReader::read(std::istream &input) {
        TSReader reader(input);
        return reader.read();
    }

} // End of namespace tempo::reader