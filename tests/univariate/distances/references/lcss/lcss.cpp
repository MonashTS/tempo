#include "lcss.hpp"

namespace reference {

    using namespace std;
    using namespace tempo;
    using namespace tempo::univariate;

    double lcss_matrix(
            const double *series1, ssize_t length1,
            const double *series2, ssize_t length2,
            double epsilon,
            size_t w){

        constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

        // Check lengths. Be explicit in the conditions
        if (length1 == 0 && length2 == 0) { return 0; }
        if (length1 == 0 && length2 != 0) { return POSITIVE_INFINITY; }
        if (length1 != 0 && length2 == 0) { return POSITIVE_INFINITY; }

        // Allocate the working space: full matrix + space for borders (first column / first line)
        int maxLength = max<int>(length1, length2);
        int minLength = min<int>(length1, length2);
        vector<std::vector<int>> matrix(maxLength + 1, std::vector<int>(maxLength + 1, 0));

        // Marker for final point
        matrix[length1][length2] = -1;

        // LCSS
        // Note: series1 and series2 are 0-indexed while the matrix is 1-indexed (0 being the borders)
        //       hence, we have i-1 and j-1 when accessing series1 and series2
        for (long i = 1; i <= length1; ++i) {
            auto series1_i = series1[i - 1];
            long jStart = max<int>(1, i - w);
            long jStop = min<int>(i + w, length2);
            for (long j = jStart; j <= jStop; ++j) {
                if (fabs(series1_i - series2[j - 1])<epsilon) {
                    matrix[i][j] = matrix[i - 1][j - 1] + 1;
                } else {
                    // Because of the window, we MUST include the diagonal! Imagine the case with w=0,
                    // the cost could not be propagated without looking at the diagonal.
                    matrix[i][j] = std::max(matrix[i - 1][j - 1], std::max(matrix[i][j - 1], matrix[i - 1][j]));
                }
            }
        }

        // Check if we have an alignment
        if( matrix[length1][length2] == -1){ return POSITIVE_INFINITY; }

        // Convert in range [0-1]
        return 1.0 - (((double) matrix[length1][length2]) / (double) minLength);
    }


} // End of namespace reference
