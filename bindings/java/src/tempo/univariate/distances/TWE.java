package tempo.univariate.distances;

import static tempo.univariate.distances.utils.dist;
import static tempo.univariate.distances.utils.min;
import static java.lang.Double.POSITIVE_INFINITY;
import static java.lang.Double.min;
import static java.lang.Math.abs;

public class TWE {

	static double distance(double[] lines, double[] cols, double nu, double lambda, double cutoff) {
		// Ensure that lines are longer than columns
		if (lines.length < cols.length) {
			double[] swap = lines;
			lines = cols;
			cols = swap;
		}

		// --- --- --- Declarations
		int nblines = lines.length;
		int nbcols = cols.length;

		// Setup buffers - no extra initialization required - border condition managed in the code.
		double[] buffers = new double[2 * nbcols];
		int c = 0;
		int p = nbcols;

		// Buffer holding precomputed distance between columns
		double[] distcol = new double[nbcols];

		// Line & columns indices
		int i = 0;
		int j = 0;

		// Cost accumulator in a line, also used as the "left neighbor"
		double cost = 0;

		// EAP variable: track where to start the next line, and the position of the previous pruning point.
		// Must be init to 0: index 0 is the next starting point and also the "previous pruning point"
		int next_start = 0;
		int prev_pp = 0;

		// Constants: we only consider timestamp spaced by 1, so:
		// In the "delete" case, we always have a time difference of 1, so we always have 1*nu+lambda
		final double nu_lambda = nu + lambda;
		// In the "match" case, we always have nu*(|i-j|+|(i-1)-(j-1)|) == 2*nu*|i-j|
		final double nu2 = 2 * nu;

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Create a new tighter upper bounds using the last alignment.
		// The last alignment can only be computed if we have nbcols >= 2
		double ub = cutoff;
		if (nbcols >= 2) {
			double li = lines[nblines - 1];
			double li1 = lines[nblines - 2];
			double co = cols[nbcols - 1];
			double co1 = cols[nbcols - 2];
			double distli = dist(li1, li);
			double distco = dist(co1, co);
			double la = min(distco + nu_lambda, // "Delete_B": over the columns / Prev
					dist(li, co) + dist(li1, co1) + nu2 * (nblines - nbcols), // Match: Diag. Ok: nblines >= nbcols
					distli + nu_lambda // "Delete_A": over the lines / Top
			);
			ub = (cutoff + utils.EPSILON) - la;
		}

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Initialisation of the first line. Deal with the line top border condition.
		{
			// Case [0,0]: special "Match case"
			cost = dist(lines[0], cols[0]);
			buffers[c + 0] = cost;
			// Distance for the first column is relative to 0 "by conventions" (from the paper, section 4.2)
			distcol[0] = dist(0, cols[0]);
			// Rest of the line: [i==0, j>=1]: "Delete_B case" (prev). We also initialize 'distcol' here.
			for (j = 1; j < nbcols; ++j) {
				double d = dist(cols[j - 1], cols[j]);
				distcol[j] = d;
				cost = cost + d + nu_lambda;
				buffers[c + j] = cost;
				if (cost <= ub) {
					prev_pp = j + 1;
				} else {
					break;
				}
			}
			// Complete the initialisation of distcol
			for (; j < nbcols; ++j) {
				double d = dist(cols[j - 1], cols[j]);
				distcol[j] = d;
			}
			// Next line.
			++i;
		}

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Main loop, starts at the second line
		for (; i < nblines; ++i) {
			// --- --- --- Swap and variables init
			int swap = c;
			c = p;
			p = swap;
			double li = lines[i];
			double li1 = lines[i - 1];
			double distli = dist(li1, li);
			int curr_pp = next_start; // Next pruning point init at the start of the line
			j = next_start;
			// --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
			{
				cost = buffers[p + j] + distli + nu_lambda; // "Delete_A" / Top
				buffers[c + j] = cost;
				if (cost <= ub) {
					curr_pp = j + 1;
				} else {
					++next_start;
				}
				++j;
			}
			// --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
			for (; j == next_start && j < prev_pp; ++j) {
				cost = min(buffers[p + j - 1] + dist(li, cols[j]) + dist(li1, cols[j - 1]) + nu2 * abs(i - j), // "Match": Diag
						buffers[p + j] + distli + nu_lambda // "Delete_A" / Top
				);
				buffers[c + j] = cost;
				if (cost <= ub) {
					curr_pp = j + 1;
				} else {
					++next_start;
				}
			}
			// --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
			for (; j < prev_pp; ++j) {
				cost = min(cost + distcol[j] + nu_lambda, // "Delete_B": over the columns / Prev
						buffers[p + j - 1] + dist(li, cols[j]) + dist(li1, cols[j - 1]) + nu2 * abs(i - j), // "Matc"h: Diag
						buffers[p + j] + distli + nu_lambda // "Delete_A": over the lines / Top
				);
				buffers[c + j] = cost;
				if (cost <= ub) {
					curr_pp = j + 1;
				}
			}
			// --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
			if (j < nbcols) { // If so, two cases.
				if (j == next_start) { // Case 1: Advancing next start: only diag.
					cost = buffers[p + j - 1] + dist(li, cols[j]) + dist(li1, cols[j - 1]) + nu2 * abs(i - j); // "Match": Diag
					buffers[c + j] = cost;
					if (cost <= ub) {
						curr_pp = j + 1;
					} else {
						// Special case if we are on the last alignment: return the actual cost if we are <= cutoff
						if (i == nblines - 1 && j == nbcols - 1 && cost <= cutoff) {
							return cost;
						} else {
							return POSITIVE_INFINITY;
						}
					}
				} else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
					cost = min(cost + distcol[j] + nu_lambda, // "Delete_B": over the columns / Prev
							buffers[p + j - 1] + dist(li, cols[j]) + dist(li1, cols[j - 1]) + nu2 * abs(i - j) // "Match": Diag
					);
					buffers[c + j] = cost;
					if (cost <= ub) {
						curr_pp = j + 1;
					}
				}
				++j;
			} else { // Previous pruning point is out of bound: exit if we extended next start up to here.
				if (j == next_start) {
					// But only if we are above the original UB
					// Else set the next starting point to the last valid column
					if (cost > cutoff) {
						return POSITIVE_INFINITY;
					} else {
						next_start = nbcols - 1;
					}
				}
			}
			// --- --- --- Stage 4: After the previous pruning point: only prev.
			// Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
			for (; j == curr_pp && j < nbcols; ++j) {
				cost = cost + distcol[j] + nu_lambda; // "Delete_B": over the columns / Prev
				buffers[c + j] = cost;
				if (cost <= ub) {
					++curr_pp;
				}
			}
			// --- --- ---
			prev_pp = curr_pp;
		} // End of main loop for(;i<nblines;++i)

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Finalization
		// Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols).
		// Cost must be <= original bound.
		if (j == nbcols && cost <= cutoff) {
			return cost;
		} else {
			return POSITIVE_INFINITY;
		}

	}

}
