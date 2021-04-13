package tempo.univariate.distances;

import static tempo.univariate.distances.utils.max;

import static java.lang.Double.POSITIVE_INFINITY;
import static java.lang.Double.min;
import static java.lang.Integer.max;
import static java.lang.Integer.min;
import static java.lang.Math.abs;

public class LCSS {

	/// Check if two numbers are within epsilon (1 = similar, 0 = not similar)
	static boolean sim(double a, double b, double epsilon) {
		return abs(a - b) < epsilon;
	}

	static double distance(double[] lines, double[] cols, double epsilon, int w, double cutoff) {

		// Ensure that lines are longer than columns
		if (lines.length < cols.length) {
			double[] swap = lines;
			lines = cols;
			cols = swap;
		}

		// Cap the windows and check that, given the constraint, an alignment is
		// possible
		if (w > lines.length) {
			w = lines.length;
		}
		if (lines.length - cols.length > w) {
			return POSITIVE_INFINITY;
		}

		// --- --- --- Declarations
		int nblines = lines.length;
		int nbcols = cols.length;

		// Setup buffers - get an extra cell for border condition. Init to 0 (done by
		// Java)
		int[] buffers = new int[(1 + nbcols) * 2];
		int c = 0 + 1; // Start of current line in buffer - account for the extra cell
		int p = nbcols + 2; // Start of previous line in buffer - account for two extra cells

		// Line & columns indices
		int i = 0;
		int j = 0;

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Score to reach to equal ub, to beat to do better
		if (cutoff > 1) {
			cutoff = 1;
		}
		int to_reach = (int) Math.ceil((1 - cutoff) * nbcols);
		int current_max = 0;

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Initialisation: OK, border line and "first diag" init to 0

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Main loop
		for (; i < nblines; ++i) {
			// --- --- --- Stop if not enough remaining lines to reach the target (by taking the diagonal)
			int lines_left = nblines - i;
			if (current_max + lines_left < to_reach) {
				return POSITIVE_INFINITY;
			}
			// --- --- --- Swap and variables init
			int swap = c;
			c = p;
			p = swap;
			double li = lines[i];
			int jStart = max(i - w, 0);
			int jStop = min(i + w + 1, nbcols);
			// --- --- --- Init the border (very first column)
			buffers[c + jStart - 1] = 0;
			// --- --- --- Iterate through the columns
			for (j = jStart; j < jStop; ++j) {
				if (sim(li, cols[j], epsilon)) {
					int cost = buffers[p + j - 1] + 1; // Diag + 1
					current_max = max(current_max, cost);
					buffers[c + j] = cost;
				} else { // Note: Diagonal lookup required, e.g. when w=0
					buffers[c + j] = max(buffers[c + j - 1], buffers[p + j - 1], buffers[p + j]);
				}
			}
		}

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Finalisation: put the result on a [0 - 1] range
		return 1.0 - (double) (buffers[c + nbcols - 1]) / (double) nbcols;

	}

}
