#include "wdtw.hpp"

namespace reference {

    using namespace std;
    using namespace tempo;
    using namespace tempo::univariate;

    /// Reference implementation on a matrix
    double wdtw_matrix(const double *series1, size_t length1_, const double *series2, size_t length2_, const double *weights) {
        const long length1 = to_signed(length1_);
        const long length2 = to_signed(length2_);

        constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

        // Check lengths. Be explicit in the conditions.
        if (length1 == 0 && length2 == 0) { return 0; }
        if (length1 == 0 && length2 != 0) { return POSITIVE_INFINITY; }
        if (length1 != 0 && length2 == 0) { return POSITIVE_INFINITY; }
        // Matrix
        vector<std::vector<double>> matrix(length1, std::vector<double>(length2, 0));
        // First value
        matrix[0][0] = weights[0] * square_dist(series1[0], series2[0]);
        // First line
        for (long i = 1; i < length2; i++) {
            matrix[0][i] = matrix[0][i - 1] + weights[i] * square_dist(series1[0], series2[i]);
        }
        // First column
        for (long i = 1; i < length1; i++) {
            matrix[i][0] = matrix[i - 1][0] + weights[i] * square_dist(series1[i], series2[0]);
        }
        // Matrix computation
        for (long i = 1; i < length1; i++) {
            for (long j = 1; j < length2; j++) {
                const auto d = weights[abs(i - j)] * square_dist(series1[i], series2[j]);
                const auto v = min(matrix[i][j - 1], std::min(matrix[i - 1][j], matrix[i - 1][j - 1])) + d;
                matrix[i][j] = v;
            }
        }
        return matrix[length1 - 1][length2 - 1];
    }


} // End of namespace references
