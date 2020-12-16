#include "msm.hpp"

namespace reference {

    using namespace std;
    using namespace tempo;
    using namespace tempo::univariate;

    /// Reference code on matrix
    double msm_matrix(
            const double *series1, ssize_t length1,
            const double *series2, ssize_t length2,
            double c) {

        constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

        // Check lengths. Be explicit in the conditions
        if (length1 == 0 && length2 == 0) { return 0; }
        if (length1 == 0 && length2 != 0) { return POSITIVE_INFINITY; }
        if (length1 != 0 && length2 == 0) { return POSITIVE_INFINITY; }

        const long maxLength = max(length1, length2);
        vector<std::vector<double>> cost(maxLength, std::vector<double>(maxLength, POSITIVE_INFINITY));

        // Initialization
        cost[0][0] = abs(series1[0] - series2[0]);
        for (long i = 1; i < length1; i++) {
            cost[i][0] = cost[i - 1][0] + get_cost(series1[i], series1[i - 1], series2[0], c);
        }
        for (long i = 1; i < length2; i++) {
            cost[0][i] = cost[0][i - 1] + get_cost(series2[i], series1[0], series2[i - 1], c);
        }

        // Main Loop
        for (long i = 1; i < length1; i++) {
            for (long j = 1; j < length2; j++) {
                double d1, d2, d3;
                d1 = cost[i - 1][j - 1] + abs(series1[i] - series2[j]);                         // Diag
                d2 = cost[i - 1][j] + get_cost(series1[i], series1[i - 1], series2[j], c);     // Prev
                d3 = cost[i][j - 1] + get_cost(series2[j], series1[i], series2[j - 1], c);      // Top
                cost[i][j] = min(d1, std::min(d2, d3));
            }
        }

        // Output
        return cost[length1 - 1][length2 - 1];
    }

} // End of namespace reference
