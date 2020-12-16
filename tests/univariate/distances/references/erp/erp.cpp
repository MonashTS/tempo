#include "erp.hpp"

namespace reference {

    using namespace std;
    using namespace tempo;
    using namespace tempo::univariate;

    double erp_matrix(
            const double *series1, size_t length1_,
            const double *series2, size_t length2_,
            double gValue,
            size_t w_
    ) {
        constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

        long length1 = to_signed(length1_);
        long length2 = to_signed(length2_);
        long w = to_signed(w_);

        // Check lengths. Be explicit in the conditions.
        if (length1 == 0 && length2 == 0) { return 0; }
        if (length1 == 0 && length2 != 0) { return POSITIVE_INFINITY; }
        if (length1 != 0 && length2 == 0) { return POSITIVE_INFINITY; }

        // We will only allocate a double-row buffer: use the smallest possible dimension as the columns.
        const double *cols = (length1 < length2) ? series1 : series2;
        const double *lines = (length1 < length2) ? series2 : series1;
        const long nbcols = min(length1, length2);
        const long nblines = max(length1, length2);

        // Cap the windows
        if (w > nblines) { w = nblines; }

        // Check if, given the constralong w, we can have an alignment.
        if (nblines - nbcols > w) { return POSITIVE_INFINITY; }

        // Allocate a double buffer for the columns. Declare the index of the 'c'urrent and 'p'revious buffer.
        // Note: we use a vector as a way to initialize the buffer with POSITIVE_INFINITY
        vector<std::vector<double>> matrix(nblines + 1, std::vector<double>(nbcols + 1, POSITIVE_INFINITY));

        // Initialisation of the first line and column
        matrix[0][0] = 0;
        for (long j{1}; j < nbcols + 1; j++) {
            matrix[0][j] = matrix[0][j - 1] + square_dist(gValue, cols[j - 1]);
        }
        for (long i{1}; i < nblines + 1; i++) {
            matrix[i][0] = matrix[i - 1][0] + square_dist(lines[i - 1], gValue);
        }

        // Iterate over the lines
        for (long i{1}; i < nblines + 1; ++i) {
            const double li = lines[i - 1];
            long l = max<long>(i - w, 1);
            long r = min<long>(i + w + 1, nbcols + 1);

            // Iterate through the rest of the columns
            for (long j{l}; j < r; ++j) {
                matrix[i][j] = min(
                        matrix[i][j - 1] + square_dist(gValue, cols[j - 1]),        // Previous
                        min(matrix[i - 1][j - 1] + square_dist(li, cols[j - 1]),    // Diagonal
                            matrix[i - 1][j] + square_dist(li, gValue)              // Above
                        )
                );
            }
        } // End of for over lines

        return matrix[nblines][nbcols];
    }

}