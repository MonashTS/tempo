package tempo.univariate.distances.lowerbounds;

import static tempo.univariate.distances.utils.dist;
import static tempo.univariate.distances.utils.min;
import static tempo.univariate.distances.utils.EPSILON;

import static java.lang.Double.POSITIVE_INFINITY;
import static java.lang.Double.min;
import static java.lang.Double.max;
import static java.lang.Integer.max;
import static java.lang.Integer.min;

public class LbEnhanced {

	public static double lb_Enhanced2j(
			double[] query, double[] qup, double[] qlo,
			double[] candidate, double[] cup, double[] clo,
			int v, int w, double cutoff) {

		int lq = query.length;
		int nbands = min(lq / 2, v);

		double lb1 = -EPSILON; // Init with a small negative value: handle numerical instability

		// --- --- --- Do L & R bands
		// First alignment
		lb1 += dist(query[0], candidate[0]);
		// Manage the case of series of length 1
		if (lq == 1) {
			return (lb1 > cutoff) ? POSITIVE_INFINITY : lb1;
		}
		int last = lq - 1;
		// Last alignment
		lb1 += dist(query[last], candidate[last]);
		// L & R bands
		for (int i = 1; i < nbands && lb1 <= cutoff; ++i) {
			int fixR = last - i;
			double minL = dist(query[i], candidate[i]);
			double minR = dist(query[fixR], candidate[fixR]);
			for (int j = max(i - w, 0); j < i; ++j) {
				int movR = last - j;
				minL = min(minL, dist(query[i], candidate[j]), dist(query[j], candidate[i]));
				minR = min(minR, dist(query[fixR], candidate[movR]), dist(query[movR], candidate[fixR]));
			}
			lb1 = lb1 + minL + minR;
		}
		// --- --- ---
		if (lb1 > cutoff) {
			return POSITIVE_INFINITY;
		}

		// --- --- --- Bridge with LB Keogh, continue while we are <= cutoff
		double lb2 = lb1;
		int end = lq - nbands;
		for (int i = nbands; i < end && lb1 <= cutoff && lb2 <= cutoff; ++i) {
			// Query - envelope candidate
			{
				double qi = query[i];
				double ui = cup[i];
				if (qi > ui) {
					lb1 += dist(qi, ui);
				} else {
					double li = clo[i];
					if (qi < li) {
						lb1 += dist(qi, li);
					}
				}
			}
			// Candidate - envelope query
			{
				double ci = candidate[i];
				double ui = qup[i];
				if (ci > ui) {
					lb2 += dist(ci, ui);
				} else {
					double li = qlo[i];
					if (ci < li) {
						lb2 += dist(ci, li);
					}
				}
			}
		}

		lb1 = max(lb1, lb2);
		return (lb1 > cutoff) ? POSITIVE_INFINITY : lb1;

	}
}
