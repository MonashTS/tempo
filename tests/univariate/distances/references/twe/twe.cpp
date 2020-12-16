#include "twe.hpp"

namespace reference {

    using namespace std;
    using namespace tempo;
    using namespace tempo::univariate;

    /// Based on reference implementation by Pierre-François Marteau.
    /// * Added some checks.
    /// * Fix degree to 2 when evaluating the distance
    /// * Return TWE instead of modifying a pointed variable
    double twe_Marteau(const double *ta, ssize_t r, const double *tb, ssize_t c, double nu, double lambda) {

        constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

        const double deg = 2;
        double disti1, distj1, dist;
        long i, j;

        // Check lengths. Be explicit in the conditions.
        if (r == 0 && c == 0) { return 0; }
        if (r == 0 && c != 0) { return POSITIVE_INFINITY; }
        if (r != 0 && c == 0) { return POSITIVE_INFINITY; }

        // allocations
        double **D = (double **) calloc(r + 1, sizeof(double *));
        double *Di1 = (double *) calloc(r + 1, sizeof(double));
        double *Dj1 = (double *) calloc(c + 1, sizeof(double));

        double dmin, htrans, dist0;

        for (i = 0; i <= r; i++) { D[i] = (double *) calloc(c + 1, sizeof(double)); }

        // local costs initializations
        for (j = 1; j <= c; j++) {
            distj1 = 0;
            if (j > 1) {
                distj1 += pow(fabs(tb[j - 2] - tb[j - 1]), deg);
            } else { distj1 += pow(fabs(tb[j - 1]), deg); }

            Dj1[j] = distj1;
        }

        for (i = 1; i <= r; i++) {
            disti1 = 0;
            if (i > 1) {
                disti1 += pow(fabs(ta[i - 2] - ta[i - 1]), deg);
            } else { disti1 += pow(fabs(ta[i - 1]), deg); }

            Di1[i] = disti1;

            for (j = 1; j <= c; j++) {
                (dist) = 0;
                (dist) += pow(fabs(ta[i - 1] - tb[j - 1]), deg);
                if (i > 1 && j > 1) {
                    (dist) += pow(fabs(ta[i - 2] - tb[j - 2]), deg);
                }

                D[i][j] = dist;
            }
        }// for i

        // border of the cost matrix initialization
        D[0][0] = 0;
        for (i = 1; i <= r; i++) { D[i][0] = POSITIVE_INFINITY; }
        for (j = 1; j <= c; j++) { D[0][j] = POSITIVE_INFINITY; }

        for (i = 1; i <= r; i++) {
            for (j = 1; j <= c; j++) {
                htrans = fabs((double) ((i - 1) - (j - 1)));
                if (j > 1 && i > 1) {
                    htrans += fabs((double) ((i - 2) - (j - 2)));
                }
                dist0 = D[i - 1][j - 1] + D[i][j] + (nu) * htrans;
                dmin = dist0;
                if (i > 1) {
                    htrans = ((double) ((i - 1) - (i - 2)));
                } else { htrans = (double) 1; }
                (dist) = Di1[i] + D[i - 1][j] + (lambda) + (nu) * htrans;
                if (dmin > (dist)) {
                    dmin = (dist);
                }
                if (j > 1) {
                    htrans = ((double) ((j - 1) - (j - 2)));
                } else { htrans = (double) 1; }
                (dist) = Dj1[j] + D[i][j - 1] + (lambda) + (nu) * htrans;
                if (dmin > (dist)) {
                    dmin = (dist);
                }
                D[i][j] = dmin;
            }
        }

        dist = D[r][c];

        // freeing
        for (i = 0; i <= r; i++) { free(D[i]); }
        free(D);
        free(Di1);
        free(Dj1);

        return dist;

    }
}