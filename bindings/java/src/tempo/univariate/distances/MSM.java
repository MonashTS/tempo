package tempo.univariate.distances;

import static tempo.univariate.distances.utils.min;
import static java.lang.Double.POSITIVE_INFINITY;
import static java.lang.Double.min;
import static java.lang.Math.abs;

public class MSM {

	static double split_merge_cost(double new_point, double xi, double yj, double c) {
		if (((xi <= new_point) && (new_point <= yj)) || ((yj <= new_point) && (new_point <= xi))) {
			return c;
		} else {
			return c + min(abs(new_point - xi), abs(new_point - yj));
		}
	}

	static double distance(double[] lines, double[] cols, double co, double cutoff) {
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

		// Line & columns indices
		int i = 0;
		int j = 0;

		// Cost accumulator in a line, also used as the "left neighbor"
		double cost = 0;

		// EAP variable: track where to start the next line, and the position of the previous pruning point.
		// Must be init to 0: index 0 is the next starting point and also the "previous pruning point"
		int next_start = 0;
		int prev_pp = 0;

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Create a new tighter upper bounds using the last alignment.
		// The last alignment can only be computed if we have nbcols >= 2
		double ub = cutoff;
		if (nbcols >= 2) {
			double li = lines[nblines - 1];
			double li1 = lines[nblines - 2];
			double cj = cols[nbcols - 1];
			double cj1 = cols[nbcols - 2];
			double la = min(abs(li - cj), // Diag: Move
					split_merge_cost(cj, li, cj1, co), // Previous: Split/Merge
					split_merge_cost(li, li1, cj, co) // Above: Split/Merge
			);
			ub = (cutoff + utils.EPSILON) - la;
		}

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Initialisation: compute the first line. Required as the main loop starts at line=1, not 0.
		{
			double l0 = lines[0];
			// First cell (0,0) is a special case. Early abandon if above the cut-off point.
			{
				cost = abs(l0 - cols[0]); // Very first cell
				buffers[c + 0] = cost;
				if (cost <= ub) {
					prev_pp = 1;
				} else {
					return POSITIVE_INFINITY;
				}
			}
			// Rest of the line, a cell only depends on the previous cell. Stop when > ub, update prev_pp.
			for (j = 1; j < nbcols; ++j) {
				cost = cost + split_merge_cost(cols[j], l0, cols[j - 1], co);
				if (cost <= ub) {
					buffers[c + j] = cost;
					prev_pp = j + 1;
				} else {
					break;
				}
			}
			// Next line.
			++i;
		}

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Main loop
		for (; i < nblines; ++i) {
			// --- --- --- Swap and variables init
			int swap = c;
			c = p;
			p = swap;
			double li = lines[i];
			double li1 = lines[i - 1];
			int curr_pp = next_start; // Next pruning point init at the start of the line
			j = next_start;
			// --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
			{
				cost = buffers[p + j] + split_merge_cost(li, li1, cols[j], co);
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
				double cj = cols[j];
				cost = min(buffers[p + j - 1] + abs(li - cj), // Diag: Move
						buffers[p + j] + split_merge_cost(li, li1, cj, co) // Above: Split/Merge
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
				double cj = cols[j];
				cost = min(buffers[p + j - 1] + abs(li - cj), // Diag: Move
						cost + split_merge_cost(cj, li, cols[j - 1], co), // Previous: Split/Merge
						buffers[p + j] + split_merge_cost(li, li1, cj, co) // Above: Split/Merge
				);
				buffers[c + j] = cost;
				if (cost <= ub) {
					curr_pp = j + 1;
				}
			}
			// --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
			if (j < nbcols) { // If so, two cases.
				double cj = cols[j];
				if (j == next_start) { // Case 1: Advancing next start: only diag.
					cost = buffers[p + j - 1] + abs(li - cj); // Diag: Move
					buffers[c + j] = cost;
					if (cost <= ub) {
						curr_pp = j + 1;
					} else {
						// Special case if we are on the last alignment: return the actual cost if we
						// are <= cutoff
						if (i == nblines - 1 && j == nbcols - 1 && cost <= cutoff) {
							return cost;
						} else {
							return POSITIVE_INFINITY;
						}
					}
				} else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
					cost = min(buffers[p + j - 1] + abs(li - cj), // Diag: Move
							cost + split_merge_cost(cj, li, cols[j - 1], co) // Previous: Split/Merge
					);
					buffers[c + j] = cost;
					if (cost <= ub) {
						curr_pp = j + 1;
					}
				}
				++j;
			} else { // Previous pruning point is out of bound: exit if we extended next start up to here.
				if (j == next_start) {
					// But only if we are above the original UB. Else set the next starting point to the last valid column
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
				cost = cost + split_merge_cost(cols[j], li, cols[j - 1], co); // Previous: Split/Merge
				buffers[c + j] = cost;
				if (cost <= ub) {
					++curr_pp;
				}
			}
			// --- --- ---
			prev_pp = curr_pp;
		} // End of main loop for(;i<nblines;++i)

		// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
		// Finalisation
		// Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols).
		// Cost must be <= original bound.
		if (j == nbcols && cost <= cutoff) {
			return cost;
		} else {
			return POSITIVE_INFINITY;
		}

	}

}
